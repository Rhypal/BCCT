from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


_TIMM_MODEL_NAMES = {
    ('dinov3_vits16', 'LVD1689M'): 'vit_small_patch16_dinov3.lvd1689m',
    ('dinov3_vitb16', 'LVD1689M'): 'vit_base_patch16_dinov3.lvd1689m',
    ('dinov3_vitl16', 'LVD1689M'): 'vit_large_patch16_dinov3.lvd1689m',
    ('dinov3_vitl16', 'SAT493M'): 'vit_large_patch16_dinov3.sat493m',
}


def _unwrap_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    """Extract a state dict from common PyTorch checkpoint containers."""
    if not isinstance(checkpoint, dict):
        raise TypeError(
            'The local checkpoint must contain a state-dict dictionary, '
            'but got {}.'.format(type(checkpoint).__name__)
        )

    state_dict = checkpoint
    for key in ('state_dict', 'model', 'teacher'):
        value = state_dict.get(key)
        if isinstance(value, dict):
            state_dict = value
            break

    if not state_dict:
        raise RuntimeError('The local checkpoint contains an empty state dict.')

    # Remove wrappers introduced by DDP/torch.compile/training frameworks.
    for prefix in ('module.', '_orig_mod.', 'model.'):
        if state_dict and all(str(key).startswith(prefix) for key in state_dict):
            state_dict = {
                str(key)[len(prefix):]: value
                for key, value in state_dict.items()
            }

    if not all(torch.is_tensor(value) for value in state_dict.values()):
        raise RuntimeError(
            'The selected checkpoint dictionary contains non-tensor values and '
            'is not a model state dict.'
        )
    return state_dict


def _load_local_state_dict(weights_path: Path) -> Dict[str, torch.Tensor]:
    suffix = weights_path.suffix.lower()

    if suffix == '.safetensors':
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                'Loading a .safetensors checkpoint requires the safetensors '
                'package. Install it in the current environment.'
            ) from exc
        checkpoint = load_file(str(weights_path), device='cpu')
    elif suffix in {'.pth', '.pt', '.bin'}:
        try:
            checkpoint = torch.load(
                str(weights_path),
                map_location='cpu',
                weights_only=True,
            )
        except TypeError:
            # Compatibility with PyTorch versions that do not have weights_only.
            checkpoint = torch.load(str(weights_path), map_location='cpu')
    else:
        raise ValueError(
            'Unsupported DINOv3 checkpoint extension {!r}. Expected '
            '.safetensors, .pth, .pt, or .bin.'.format(suffix)
        )

    return _unwrap_state_dict(checkpoint)


class FrozenDINOv3(nn.Module):
    """Selectable frozen ViT-B/16-IN21K or DINOv3 feature extractor.

    The historical class name is retained so existing train.py/test.py imports
    remain valid. Select the actual backbone with ``backbone.type``.

    Returns:
        features: L2-normalized class-token features, or a class-token and
            mean-patch-token concatenation when selected by configuration.
        style: concatenated mean and standard deviation of final patch tokens.
    """

    def __init__(self, cfg: dict):
        super().__init__()

        root_cfg = dict(cfg)
        backbone_type = str(
            root_cfg.get('type', 'vit_b16_in21k')
        ).strip().lower().replace('-', '_')
        aliases = {
            'vit': 'vit_b16_in21k',
            'vit_b16': 'vit_b16_in21k',
            'vit_b16_in21k': 'vit_b16_in21k',
            'dinov3': 'dinov3',
            'dinov3_sat': 'dinov3',
        }
        if backbone_type not in aliases:
            raise ValueError(
                'Unsupported backbone.type={!r}. Expected vit_b16_in21k or '
                'dinov3.'.format(backbone_type)
            )
        self.backbone_type = aliases[backbone_type]

        section_name = (
            'vit_b16_in21k'
            if self.backbone_type == 'vit_b16_in21k'
            else 'dinov3'
        )
        selected_cfg = root_cfg.get(section_name, {}) or {}
        if not isinstance(selected_cfg, dict):
            raise TypeError('backbone.{} must be a mapping.'.format(section_name))
        # Nested values describe the selected implementation. Top-level values
        # remain supported for checkpoints/configs produced by older versions.
        cfg = {**root_cfg, **selected_cfg}

        if self.backbone_type == 'vit_b16_in21k':
            self._build_vit_b16_in21k(cfg)
            self.backbone.eval()
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self._configure_feature_layout(cfg)
            print(
                'Loaded frozen vision backbone: type={}, model={}, weights={}'
                .format(
                    self.backbone_type,
                    self.model_description,
                    self.weights_path,
                )
            )
            return

        weights_path_value = cfg.get('weights_path')
        if not weights_path_value:
            raise ValueError(
                'backbone.weights_path is required for local DINOv3 loading.'
            )

        weights_path = Path(str(weights_path_value)).expanduser()
        if not weights_path.is_file():
            raise FileNotFoundError(
                'DINOv3 weight file does not exist: {}'.format(weights_path)
            )
        weights_path = weights_path.resolve()

        state_dict = _load_local_state_dict(weights_path)

        # Official Hugging Face DINOv3 weights use keys such as
        # embeddings.cls_token and encoder.layer.0....  timm uses cls_token,
        # patch_embed.proj and blocks.0....
        is_transformers_checkpoint = (
            'embeddings.cls_token' in state_dict
            or any(key.startswith('encoder.layer.') for key in state_dict)
        )

        if is_transformers_checkpoint:
            self.backend = 'transformers'
            self._build_transformers_backbone(cfg, weights_path, state_dict)
        else:
            self.backend = 'timm'
            self._build_timm_backbone(cfg, state_dict)

        # Release the second reference to the large checkpoint before training.
        del state_dict

        self.weights_path = str(weights_path)
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self._configure_feature_layout(cfg)

        print(
            'Loaded frozen vision backbone: type={}, backend={}, model={}, '
            'weights={}'
            .format(
                self.backbone_type,
                self.backend,
                self.model_description,
                self.weights_path,
            )
        )

    def _build_vit_b16_in21k(self, cfg: dict) -> None:
        """Build exactly the frozen ViT-B/16-IN21K used by DSPCL."""
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                'ViT-B/16-IN21K requires timm. Install the same timm version '
                'used by the DSPCL comparison environment.'
            ) from exc

        model_name = str(
            cfg.get('model_name', 'vit_base_patch16_224.augreg_in21k')
        )
        checkpoint_value = cfg.get('weights_path')
        checkpoint_path = None
        if checkpoint_value:
            checkpoint_path = Path(str(checkpoint_value)).expanduser().resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    'ViT-B/16-IN21K checkpoint does not exist: {}'
                    .format(checkpoint_path)
                )

        use_pretrained = bool(cfg.get('pretrained', True)) and checkpoint_path is None
        self.backbone = timm.create_model(
            model_name,
            pretrained=use_pretrained,
            num_classes=0,
        )

        if checkpoint_path is not None:
            from tools.checkpoint import load_timm_feature_weights
            # IN21K checkpoints contain head.weight/head.bias. The feature-only
            # model intentionally has no classifier head, exactly as in DSPCL.
            load_timm_feature_weights(self.backbone, checkpoint_path)

        dim = getattr(self.backbone, 'num_features', None)
        if dim is None:
            dim = getattr(self.backbone, 'embed_dim', None)
        if dim is None:
            raise AttributeError('Could not infer ViT-B/16 feature dimension.')
        if int(dim) != 768:
            raise RuntimeError(
                'Expected ViT-B/16 feature_dim=768, got {} from {!r}.'
                .format(dim, model_name)
            )

        self.backend = 'timm_vit'
        self.feature_dim = int(dim)
        self.model_description = model_name
        self.weights_path = (
            str(checkpoint_path) if checkpoint_path is not None
            else 'timm-pretrained'
        )

    def _configure_feature_layout(self, cfg: dict) -> None:
        """Configure a classifier feature from frozen final-layer tokens."""
        self.token_dim = int(self.feature_dim)
        default_pooling = (
            'cls_patch_concat' if self.backbone_type == 'dinov3' else 'cls'
        )
        pooling = str(cfg.get('feature_pooling', default_pooling)).lower()
        aliases = {
            'cls': 'cls',
            'class': 'cls',
            'cls_patch_concat': 'cls_patch_concat',
            'cls_patch': 'cls_patch_concat',
        }
        if pooling not in aliases:
            raise ValueError(
                'Unsupported feature_pooling={!r}; use cls or '
                'cls_patch_concat.'.format(pooling)
            )
        self.feature_pooling = aliases[pooling]
        self.feature_dim = (
            self.token_dim * 2
            if self.feature_pooling == 'cls_patch_concat'
            else self.token_dim
        )
        # Style routing always describes patch-token distribution, independent
        # of whether patch means are also exposed to the classifier.
        self.style_dim = 2 * self.token_dim

    def _build_transformers_backbone(
        self,
        cfg: dict,
        weights_path: Path,
        state_dict: Dict[str, torch.Tensor],
    ) -> None:
        """Build DINOv3 from a local HF config and load local HF weights."""
        try:
            from transformers import AutoConfig, AutoModel
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                'This is a Hugging Face Transformers-format DINOv3 '
                'checkpoint. Install/upgrade transformers to a version that '
                'contains DINOv3 support (the official integration starts at '
                'transformers 4.56.0).'
            ) from exc

        config_value = cfg.get('config_path')
        config_path = (
            Path(str(config_value)).expanduser()
            if config_value
            else weights_path.with_name('config.json')
        )
        if not config_path.is_file():
            raise FileNotFoundError(
                'This checkpoint is in Hugging Face Transformers format, so '
                'its matching config.json is required. Put config.json next '
                'to the weight file or set backbone.config_path. Missing: {}'
                .format(config_path)
            )
        config_path = config_path.resolve()

        try:
            # local_files_only prevents any fallback to the Hugging Face Hub.
            config = AutoConfig.from_pretrained(
                str(config_path),
                local_files_only=True,
            )
            self.backbone = AutoModel.from_config(config)
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                'The installed transformers version cannot construct the '
                'DINOv3 architecture described by {}. Upgrade transformers '
                'to a version with DINOv3 support.'
                .format(config_path)
            ) from exc

        try:
            incompatible = self.backbone.load_state_dict(
                state_dict,
                strict=True,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                'The local config.json and weight file do not describe the '
                'same DINOv3 model. Make sure both files come from exactly the '
                'same Hugging Face repository.\nOriginal load error:\n{}'
                .format(exc)
            ) from exc

        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                'Incomplete Transformers DINOv3 load. Missing keys: {}; '
                'unexpected keys: {}.'
                .format(incompatible.missing_keys, incompatible.unexpected_keys)
            )

        dim = getattr(config, 'hidden_size', None)
        if dim is None:
            raise AttributeError(
                'Could not infer hidden_size from the DINOv3 config.'
            )
        self.feature_dim = int(dim)
        self.num_register_tokens = int(
            getattr(config, 'num_register_tokens', 4)
        )
        self.model_description = str(
            getattr(config, '_name_or_path', 'transformers-dinov3')
        )

    def _build_timm_backbone(
        self,
        cfg: dict,
        state_dict: Dict[str, torch.Tensor],
    ) -> None:
        """Build timm DINOv3 and load a local timm-format state dict."""
        model_name = str(cfg.get('model_name', 'dinov3_vitl16'))
        weights_tag = str(cfg.get('weights_tag', 'SAT493M')).upper()
        timm_model_name = cfg.get('timm_model_name')

        if not timm_model_name:
            if model_name.startswith('vit_') and '_dinov3.' in model_name:
                timm_model_name = model_name
            else:
                key = (model_name, weights_tag)
                if key not in _TIMM_MODEL_NAMES:
                    raise ValueError(
                        'Unsupported local DINOv3 model/tag combination: {} + '
                        '{}. Set backbone.timm_model_name explicitly if needed.'
                        .format(model_name, weights_tag)
                    )
                timm_model_name = _TIMM_MODEL_NAMES[key]

        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                'A timm-format DINOv3 checkpoint requires timm >= 1.0.20.'
            ) from exc

        if not timm.is_model(timm_model_name):
            raise RuntimeError(
                'The installed timm does not provide {!r}. Upgrade to '
                'timm >= 1.0.20.'.format(timm_model_name)
            )

        self.backbone = timm.create_model(
            timm_model_name,
            pretrained=False,
            num_classes=0,
        )
        try:
            incompatible = self.backbone.load_state_dict(
                state_dict,
                strict=True,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                'The checkpoint is not compatible with timm model {!r}.\n'
                'Original load error:\n{}'
                .format(timm_model_name, exc)
            ) from exc

        # With strict=True these lists should be empty.
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                'Incomplete timm DINOv3 load. Missing keys: {}; '
                'unexpected keys: {}.'
                .format(incompatible.missing_keys, incompatible.unexpected_keys)
            )

        dim = getattr(self.backbone, 'num_features', None)
        if dim is None:
            dim = getattr(self.backbone, 'embed_dim', None)
        if dim is None:
            raise AttributeError('Could not infer DINOv3 embedding dimension.')

        self.feature_dim = int(dim)
        self.model_description = str(timm_model_name)

    def train(self, mode: bool = True):
        # Keep the backbone frozen even when the outer model enters train mode.
        super().train(False)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        if self.backend == 'timm_vit':
            tokens = self.backbone.forward_features(x)
            if isinstance(tokens, dict):
                cls = tokens.get('x_norm_clstoken')
                patch = tokens.get('x_norm_patchtokens')
                if cls is None or patch is None:
                    raise RuntimeError(
                        'Unexpected dictionary returned by timm ViT-B/16.'
                    )
            else:
                if not torch.is_tensor(tokens) or tokens.ndim != 3:
                    raise RuntimeError(
                        'Expected timm ViT-B/16 forward_features() to return '
                        'tokens with shape [B, N, C].'
                    )
                cls = self.backbone.forward_head(tokens, pre_logits=True)
                num_prefix_tokens = int(
                    getattr(self.backbone, 'num_prefix_tokens', 1)
                )
                patch = tokens[:, num_prefix_tokens:]
        elif self.backend == 'transformers':
            outputs = self.backbone(
                pixel_values=x,
                return_dict=True,
            )
            tokens = outputs.last_hidden_state
            if not torch.is_tensor(tokens) or tokens.ndim != 3:
                raise RuntimeError(
                    'Expected Transformers DINOv3 last_hidden_state to have '
                    'shape [B, N, C].'
                )
            cls = tokens[:, 0]
            patch = tokens[:, 1 + self.num_register_tokens:]
        else:
            tokens = self.backbone.forward_features(x)

            # Compatibility with Meta's dictionary-style feature output.
            if isinstance(tokens, dict):
                cls = tokens['x_norm_clstoken']
                patch = tokens['x_norm_patchtokens']
            else:
                if not torch.is_tensor(tokens) or tokens.ndim != 3:
                    raise RuntimeError(
                        'Expected DINOv3 forward_features() to return a '
                        '[B, N, C] tensor or a feature dictionary.'
                    )
                num_prefix_tokens = int(
                    getattr(self.backbone, 'num_prefix_tokens', 1)
                )
                cls = tokens[:, 0]
                patch = tokens[:, num_prefix_tokens:]

        if patch.shape[1] == 0:
            raise RuntimeError('DINOv3 returned no patch tokens.')

        patch = patch.float()
        cls = F.normalize(cls.float(), dim=-1)
        mean = patch.mean(dim=1)
        patch_mean = F.normalize(mean, dim=-1)
        if self.feature_pooling == 'cls_patch_concat':
            features = F.normalize(torch.cat([cls, patch_mean], dim=-1), dim=-1)
        else:
            features = cls
        std = patch.std(dim=1, unbiased=False)
        style = torch.cat([mean, std], dim=-1)
        return features, style
