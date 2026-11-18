from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpandableCosineClassifier(nn.Module):
    def __init__(self, dim: int, num_classes: int = 0, scale: float = 20.0):
        super().__init__()
        self.dim = dim
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.empty(max(num_classes, 1), dim))
        nn.init.normal_(self.weight, std=0.02)
        self._num_classes = int(num_classes)

    @property
    def num_classes(self):
        return self._num_classes

    def expand(self, new_num_classes: int, init_vectors: torch.Tensor | None = None):
        if new_num_classes <= self._num_classes:
            return
        old_n = self._num_classes
        old_weight = self.weight.detach().clone()[:max(old_n, 0)]
        new_w = torch.empty(new_num_classes, self.dim, device=self.weight.device)
        nn.init.normal_(new_w, std=0.02)
        if old_n > 0:
            new_w[:old_n].copy_(old_weight[:old_n])
        if init_vectors is not None:
            init_vectors = F.normalize(init_vectors.to(new_w), dim=-1)
            expected = new_num_classes - old_n
            if init_vectors.shape != (expected, self.dim):
                raise ValueError(f'init_vectors must be {(expected, self.dim)}, got {tuple(init_vectors.shape)}')
            new_w[old_n:new_num_classes].copy_(init_vectors)
        self.weight = nn.Parameter(new_w)
        self._num_classes = int(new_num_classes)

    def forward(self, x):
        if self._num_classes <= 0:
            raise RuntimeError('Classifier has no classes yet.')
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight[:self._num_classes], dim=-1)
        return self.scale * (x @ w.T)

    def mask_gradient(self, trainable_class_ids):
        if self.weight.grad is None:
            return
        mask = torch.zeros(self._num_classes, dtype=torch.bool, device=self.weight.grad.device)
        if len(trainable_class_ids):
            mask[torch.as_tensor(trainable_class_ids, device=mask.device, dtype=torch.long)] = True
        self.weight.grad[:self._num_classes][~mask] = 0


class DomainEvidenceHead(nn.Module):
    """Frozen native-space evidence over labels observed in one domain.

    The default shrinkage-LDA mode fits empirical class means and one pooled
    full covariance, then stores only the solved linear discriminants. Cosine
    and diagonal-LDA heads remain available for ablations. All operate in the
    post-backbone native feature space.
    """

    def __init__(
        self,
        dim: int,
        class_ids,
        scale: float = 20.0,
        blend_weight: float = 0.0,
        hidden_dim: int = 0,
        dropout: float = 0.0,
        mode: str = 'cosine',
    ):
        super().__init__()
        ids = torch.as_tensor(class_ids, dtype=torch.long)
        if ids.ndim != 1 or ids.numel() == 0:
            raise ValueError('Domain evidence head requires at least one class id.')
        if ids.unique().numel() != ids.numel():
            raise ValueError('Domain evidence head class ids must be unique.')
        self.dim = int(dim)
        self.scale = float(scale)
        self.mode = str(mode).strip().lower()
        if self.mode not in {'cosine', 'diagonal_lda', 'shrinkage_lda'}:
            raise ValueError('Unknown domain evidence mode: ' + self.mode)
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim < 0:
            raise ValueError('Domain evidence hidden_dim must be non-negative.')
        self.register_buffer('class_ids', ids.contiguous())
        self.register_buffer(
            'blend_weight', torch.tensor(float(blend_weight), dtype=torch.float32)
        )
        # Shared within-domain precision.  In diagonal_lda mode this whitens
        # unreliable feature dimensions without storing a covariance per class.
        # Keeping one vector per domain also makes the auxiliary branch compact.
        self.register_buffer('precision', torch.ones(self.dim, dtype=torch.float32))
        if self.mode == 'shrinkage_lda':
            self.register_buffer('discriminant_weight', torch.zeros(len(ids), self.dim))
            self.register_buffer('discriminant_bias', torch.zeros(len(ids)))
            self.register_buffer('logit_temperature', torch.ones(()))
        self.weight = nn.Parameter(torch.empty(len(ids), self.dim))
        nn.init.normal_(self.weight, std=0.02)
        if self.hidden_dim > 0:
            self.residual_norm = nn.LayerNorm(self.dim)
            self.residual_down = nn.Linear(self.dim, self.hidden_dim)
            self.residual_dropout = nn.Dropout(float(dropout))
            self.residual_up = nn.Linear(self.hidden_dim, len(ids))
            nn.init.xavier_uniform_(self.residual_down.weight)
            nn.init.zeros_(self.residual_down.bias)
            # A small, non-zero output initialization lets both layers learn on
            # the first step while keeping the initial cosine solution dominant.
            nn.init.normal_(self.residual_up.weight, std=0.01)
            nn.init.zeros_(self.residual_up.bias)
        else:
            self.residual_norm = None
            self.residual_down = None
            self.residual_dropout = None
            self.residual_up = None

    def forward(self, x):
        if self.mode == 'shrinkage_lda':
            x = F.normalize(x.float(), dim=-1)
            return F.linear(x, self.discriminant_weight, self.discriminant_bias) / (
                self.logit_temperature.clamp_min(1e-6)
            )
        original = x
        x = F.normalize(original, dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        if self.mode == 'diagonal_lda':
            precision = self.precision.to(x).clamp_min(1e-6)
            precision = precision / precision.mean().clamp_min(1e-6)
            # The class-independent x^T Sigma^-1 x term can be omitted.  With
            # unit precision this reduces to the original cosine classifier
            # up to a class-independent constant.
            linear = (x * precision) @ weight.T
            bias = -0.5 * (weight.square() * precision).sum(dim=-1)
            logits = self.scale * (linear + bias.unsqueeze(0))
        else:
            logits = self.scale * (x @ weight.T)
        if self.residual_down is not None:
            residual = self.residual_down(self.residual_norm(original))
            residual = F.gelu(residual)
            residual = self.residual_dropout(residual)
            logits = logits + self.residual_up(residual)
        return logits

    @torch.no_grad()
    def lda_statistics(self, features, local_labels, chunk_size=2048):
        """Pooled within-class scatter, with empirical (not unit) class means."""
        if self.mode != 'shrinkage_lda':
            raise RuntimeError('Full covariance statistics require shrinkage_lda.')
        x = F.normalize(features.detach().float().cpu(), dim=-1)
        y = local_labels.detach().long().cpu()
        counts = torch.bincount(y, minlength=len(self.class_ids))
        if len(x) != len(y) or bool((counts == 0).any()):
            raise ValueError('LDA needs aligned samples and at least one per class.')
        means = torch.zeros(len(counts), self.dim, dtype=torch.float64)
        means.index_add_(0, y, x.double())
        means /= counts[:, None]
        means = means.to(self.weight.device)
        scatter = means.new_zeros(self.dim, self.dim)
        for start in range(0, len(x), chunk_size):
            xb = x[start:start + chunk_size].to(means)
            yb = y[start:start + chunk_size].to(means.device)
            residual = xb - means[yb]
            scatter.addmm_(residual.T, residual)
        return means, scatter / max(len(x) - len(counts), 1)

    @torch.no_grad()
    def fit_lda_statistics(self, means, covariance, shrinkage):
        shrinkage = float(shrinkage)
        if not 0.0 < shrinkage <= 1.0:
            raise ValueError('Covariance shrinkage must be in (0, 1].')
        covariance = covariance.to(device=self.weight.device, dtype=torch.float64)
        means = means.to(covariance)
        average_variance = covariance.diagonal().mean().clamp_min(1e-8)
        regularized = (1.0 - shrinkage) * covariance
        regularized.diagonal().add_(shrinkage * average_variance + 1e-8)
        discriminants = torch.linalg.solve(regularized, means.T).T
        self.weight.copy_(means.to(self.weight))
        self.discriminant_weight.copy_(discriminants.to(self.discriminant_weight))
        self.discriminant_bias.copy_(
            (-0.5 * (means * discriminants).sum(-1)).to(self.discriminant_bias)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Checkpoints produced before diagonal evidence did not contain this
        # buffer.  Unit precision exactly preserves their cosine behaviour.
        precision_key = prefix + 'precision'
        if precision_key not in state_dict:
            state_dict[precision_key] = torch.ones_like(self.precision)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
