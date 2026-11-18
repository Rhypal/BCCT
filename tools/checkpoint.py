from __future__ import annotations

from pathlib import Path
import torch



def load_timm_feature_weights(backbone, path):
    """Allow an unused pretraining head, but never silently omit encoder weights."""
    from timm.models import load_checkpoint
    incompatible = load_checkpoint(backbone, str(path), strict=False)
    if incompatible is not None:
        unexpected = [
            key for key in incompatible.unexpected_keys if not key.startswith('head.')
        ]
        if incompatible.missing_keys or unexpected:
            raise RuntimeError(
                'Incomplete ViT encoder checkpoint: '
                f'missing={incompatible.missing_keys}, unexpected={unexpected}'
            )


def save_checkpoint(path, *, task_id, model, stats_bank, style_bank, registry, cfg, metrics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        'task_id': int(task_id),
        'model_state': model.state_dict(),
        'operators': [op.state_dict() for op in model.operators],
        'stats_bank': stats_bank.state_dict(),
        'style_bank': style_bank.state_dict(),
        'registry': registry.state_dict(),
        'config': cfg,
        'metrics': metrics,
        'feature_dim': model.classifier.dim,
        'style_dim': model.router.input_dim,
        'num_classes': model.classifier.num_classes,
        'num_domains': len(model.operators),
        'domain_evidence_class_ids': model.domain_evidence_class_ids(),
        'domain_evidence_arch': {
            'mode': str(model.domain_evidence_mode),
            'hidden_dim': int(model.domain_evidence_hidden_dim),
            'dropout': float(model.domain_evidence_dropout),
        },
    }
    torch.save(state, path)


def load_model_state(path, model, stats_bank, style_bank, registry, map_location='cpu'):
    from model.operator import DomainOperator
    state = torch.load(path, map_location=map_location)
    if int(state['feature_dim']) != model.classifier.dim:
        raise ValueError(
            'Backbone feature layout differs from the checkpoint. The updated '
            'default uses CLS features; start a new training run or load the '
            'original checkpoint configuration for legacy evaluation.'
        )
    previous_data = state.get('config', {}).get('data', {})
    current_data = model.cfg.get('data', {})
    for key in ('mean', 'std', 'interpolation', 'preprocessing_version'):
        if previous_data.get(key) != current_data.get(key):
            raise ValueError(
                f'Checkpoint data.{key} differs from the active configuration. '
                'Evaluate with its original config, or retrain with the updated default.'
            )
    registry.load_state_dict(state['registry'])
    model.classifier.expand(int(state['num_classes']))
    model.router.expand(int(state['num_domains']))
    model.ensure_feature_adapters(int(state['num_domains']))
    evidence_arch = state.get(
        'domain_evidence_arch',
        # Old checkpoints contain the former cosine-only evidence head.
        {'hidden_dim': 0, 'dropout': 0.0},
    )
    model.set_domain_evidence_arch(
        int(evidence_arch.get('hidden_dim', 0)),
        float(evidence_arch.get('dropout', 0.0)),
        str(evidence_arch.get('mode', 'cosine')),
    )
    model.ensure_domain_evidence_heads(state.get('domain_evidence_class_ids', []))
    model.operators = [DomainOperator.from_state_dict(x) for x in state['operators']]
    model.load_state_dict(state['model_state'], strict=True)
    model.freeze_feature_adapters()
    model.freeze_domain_evidence_heads()
    stats_bank.load_state_dict(state['stats_bank'])
    style_bank.load_state_dict(state['style_bank'])
    return state
