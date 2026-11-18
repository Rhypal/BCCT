from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import tempfile
from engine.trainer import IncrementalTrainer
from dataset.incremental_manager import ClassRegistry
from tools.checkpoint import save_checkpoint, load_model_state

from model.operator import DomainOperator
from model.bridge_cdil import BridgeCDILModel
from model.classifier import DomainEvidenceHead
from model.style_router import DomainStyleBank
from tools.stats import ClassStatsBank
from tools.transport import estimate_bridge_operator
from tools.bures import bures_wasserstein_cov_distance_sq
from tools.losses import (
    angular_margin_cross_entropy,
    prepare_robust_boundary_stats,
    robust_boundary_expansion_loss,
    supervised_contrastive_loss,
)
from tools.metrics import recurring_novel_harmonic


def make_cfg():
    return {
        'seed': 1,
        'statistics': {'class_rank': 8},
        'transport': {
            'domain_rank': 6, 'probe_dim': 12, 'min_bridges': 2,
            'bw_steps': 10, 'bw_lr': 0.03, 'identity_reg': 0.01, 'eps': 1e-5,
        },
        'router': {
            'hidden_dim': 16,
            'temperature': 0.5,
            'hard_inference': False,
            'inference_mode': 'classwise_max',
        },
        'feature_adapter': {
            'enabled': True, 'bottleneck_dim': 8, 'stages': 2,
        },
        'classifier': {'scale': 10.0},
        'domain_evidence': {
            'enabled': True, 'scale': 10.0, 'blend_weight': 0.35,
            'hidden_dim': 8, 'dropout': 0.1,
        },
    }


def main():
    torch.manual_seed(1)
    cfg = make_cfg()
    d, sdim = 32, 20
    # Task 0: classes 0/1/2
    x0 = F.normalize(torch.randn(180, d), dim=-1)
    y0 = torch.arange(180) % 3
    bank = ClassStatsBank(rank=8)
    bank.update_from_features(x0, y0)
    assert all(
        torch.as_tensor(value['residual_var']).item() > 0
        for value in bank.stats.values()
    )
    replay_x, replay_y = bank.sample([0, 1, 2], 5, generator=torch.Generator().manual_seed(7))
    assert replay_x.shape == (15, d)
    assert replay_y.bincount(minlength=3).tolist() == [5, 5, 5]
    assert torch.allclose(replay_x.norm(dim=-1), torch.ones(15), atol=1e-5)

    # Task 1: recurring 1/2 + novel 3, with a synthetic domain transform.
    raw = F.normalize(torch.randn(180, d), dim=-1)
    y1 = (torch.arange(180) % 3) + 1
    raw = F.normalize(raw + 0.15 * torch.randn_like(raw), dim=-1)

    op = estimate_bridge_operator(
        task_id=1, features=raw, labels=y1, bridge_ids=[1, 2],
        stats_bank=bank, cfg=cfg['transport'], class_rank=8, seed=1,
    )
    assert op.bridge_count == 2
    z = op.apply(raw)
    assert z.shape == raw.shape
    assert op.mean_before is not None and op.mean_after is not None
    assert torch.isfinite(torch.tensor([op.mean_before, op.mean_after])).all()
    if op.A is not None:
        spectrum = torch.linalg.eigvalsh(op.A.double())
        assert float(spectrum.min()) >= 0.25 - 1e-4
        assert float(spectrum.max()) <= 4.0 + 1e-4

    model = BridgeCDILModel(d, sdim, cfg)
    model.add_feature_adapter(0)
    model.add_operator(DomainOperator(task_id=0))
    model.add_feature_adapter(1)
    model.add_operator(op)
    model.classifier.expand(4, init_vectors=F.normalize(torch.randn(4, d), dim=-1))
    model.add_domain_evidence_head(0, [0, 1, 2])
    model.add_domain_evidence_head(1, [1, 2, 3])
    styles = torch.randn(10, sdim)
    logits, alpha = model.forward_auto(torch.randn(10, d), styles)
    assert logits.shape == (10, 4)
    assert alpha.shape == (10, 2)

    adapter_features = torch.randn(12, d)
    adapter_labels = torch.arange(12) % 3
    adapted_a = model.feature_adapters[0](adapter_features)
    adapted_b = model.feature_adapters[0](adapter_features + 0.01 * torch.randn_like(adapter_features))
    adapter_loss = supervised_contrastive_loss(
        torch.stack([adapted_a, adapted_b], dim=1), adapter_labels
    )
    adapter_loss.backward()
    assert model.feature_adapters[0].up.weight.grad is not None
    assert torch.isfinite(model.feature_adapters[0].up.weight.grad).all()
    angular_loss = angular_margin_cross_entropy(
        torch.randn(12, 3), adapter_labels, scale=10.0, margin=0.1
    )
    assert torch.isfinite(angular_loss)

    restored = BridgeCDILModel(d, sdim, cfg)
    restored.classifier.expand(4)
    restored.router.expand(2)
    restored.ensure_feature_adapters(2)
    restored.ensure_domain_evidence_heads([[0, 1, 2], [1, 2, 3]])
    restored.load_state_dict(model.state_dict(), strict=True)
    assert len(restored.feature_adapters) == 2
    assert len(restored.domain_evidence_heads) == 2
    evidence_logits = model.domain_evidence_heads[0](adapter_features)
    evidence_loss = F.cross_entropy(evidence_logits, adapter_labels)
    evidence_loss.backward()
    assert model.domain_evidence_heads[0].residual_up.weight.grad is not None
    assert torch.isfinite(
        model.domain_evidence_heads[0].residual_up.weight.grad
    ).all()

    lda_head = DomainEvidenceHead(
        d, [0, 1, 2], scale=10.0, mode='diagonal_lda'
    )
    with torch.no_grad():
        lda_head.weight.copy_(F.normalize(torch.randn(3, d), dim=-1))
        lda_head.precision.copy_(torch.linspace(0.25, 2.0, d))
    lda_logits = lda_head(torch.randn(7, d))
    assert lda_logits.shape == (7, 3)
    assert torch.isfinite(lda_logits).all()
    # Full covariance discriminants must equal the Gaussian classifier with
    # its class-independent query quadratic term removed.
    full_lda = DomainEvidenceHead(d, [0, 1, 2], mode='shrinkage_lda')
    fit_x = F.normalize(torch.randn(60, d), dim=-1)
    fit_y = torch.arange(60) % 3
    means, covariance = full_lda.lda_statistics(fit_x, fit_y, chunk_size=11)
    reference_means = torch.stack([fit_x[fit_y == c].double().mean(0) for c in range(3)])
    residual = fit_x.double() - reference_means[fit_y]
    torch.testing.assert_close(means, reference_means, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(covariance, residual.T @ residual / 57, atol=1e-8, rtol=1e-6)
    full_lda.fit_lda_statistics(means, covariance, 0.1)
    regularized = 0.9 * covariance + torch.eye(d, dtype=torch.float64) * (
        0.1 * covariance.diagonal().mean() + 1e-8
    )
    queries = F.normalize(torch.randn(8, d), dim=-1)
    difference = queries.double()[:, None] - means[None]
    reference_logits = -0.5 * torch.einsum(
        'bcd,de,bce->bc', difference, torch.linalg.inv(regularized), difference
    )
    actual_logits = full_lda(queries).double()
    torch.testing.assert_close(
        actual_logits - actual_logits[:, :1],
        reference_logits - reference_logits[:, :1], atol=2e-5, rtol=2e-5,
    )
    # Strictly load a pre-diagonal checkpoint: the compatibility hook supplies
    # unit precision and therefore preserves the former cosine behaviour.
    legacy_state = {
        key: value for key, value in lda_head.state_dict().items()
        if key != 'precision'
    }
    legacy_head = DomainEvidenceHead(d, [0, 1, 2], scale=10.0, mode='cosine')
    legacy_head.load_state_dict(legacy_state, strict=True)
    assert torch.allclose(legacy_head.precision, torch.ones(d))

    style_bank = DomainStyleBank()
    style_bank.update(
        0,
        torch.randn(40, sdim),
        F.normalize(torch.randn(40, d) + 1.0, dim=-1),
        key_count=4,
        key_iterations=3,
        key_max_samples=40,
        seed=1,
    )
    style_bank.update(
        1,
        torch.randn(40, sdim),
        F.normalize(torch.randn(40, d) - 1.0, dim=-1),
        key_count=4,
        key_iterations=3,
        key_max_samples=40,
        seed=1,
    )
    domain_probability = style_bank.domain_probabilities(
        F.normalize(torch.randn(6, d) + 1.0, dim=-1),
        num_domains=2,
        temperature=0.02,
        neighbors=2,
    )
    assert domain_probability.shape == (6, 2)
    assert torch.allclose(domain_probability.sum(dim=-1), torch.ones(6), atol=1e-6)
    mixture_logits, mixture_alpha = model.forward_domain_mixture(
        torch.randn(6, d), domain_probability
    )
    assert mixture_logits.shape == (6, 4)
    assert mixture_alpha.shape == (6, 2)
    reloaded_style_bank = DomainStyleBank()
    reloaded_style_bank.load_state_dict(style_bank.state_dict())
    assert set(reloaded_style_bank.feature_keys) == {0, 1}
    for task_id, class_ids in enumerate(([0, 1, 2], [1, 2, 3])):
        key_y = torch.tensor(class_ids * 6)
        style_bank.fit_class_keys(
            task_id, torch.randn(18, d), key_y,
            keys_per_class=2, iterations=2, max_samples=18,
        )
    semantic_scores = style_bank.class_domain_scores(torch.randn(6, d), 2, 4)
    assert semantic_scores.shape == (6, 2, 4)
    assert torch.isneginf(semantic_scores[:, 0, 3]).all()
    assert torch.isneginf(semantic_scores[:, 1, 0]).all()
    semantic_logits, semantic_alpha = model.forward_domain_mixture(
        torch.randn(6, d), domain_probability, class_domain_scores=semantic_scores
    )
    torch.testing.assert_close(semantic_logits.exp().sum(1), torch.ones(6))
    torch.testing.assert_close(semantic_alpha.sum(1), torch.ones(6))
    reloaded_style_bank.load_state_dict(style_bank.state_dict())
    queries = torch.randn(4, d)
    torch.testing.assert_close(
        reloaded_style_bank.class_domain_scores(queries, 2, 4),
        style_bank.class_domain_scores(queries, 2, 4),
    )
    # An erroneous dataset-level prior must not irreversibly mask the correct
    # class/domain pair when its raw semantic evidence is strong.
    semantic_cfg = {
        'classifier': {'scale': 1.0}, 'router': {'hidden_dim': 4},
        'domain_evidence': {'enabled': True, 'blend_weight': 0.999},
    }
    semantic_model = BridgeCDILModel(2, 2, semantic_cfg)
    for task_id in range(2):
        semantic_model.add_operator(DomainOperator(task_id=task_id))
        semantic_model.add_domain_evidence_head(task_id, [task_id])
    semantic_model.classifier.expand(2, init_vectors=torch.eye(2))
    prior = torch.tensor([[0.999, 0.001]])
    query = torch.tensor([[0.0, 1.0]])
    wrong_logits, _ = semantic_model.forward_domain_mixture(query, prior)
    corrected_logits, _ = semantic_model.forward_domain_mixture(
        query, prior, class_domain_scores=torch.tensor([
            [[0.0, float('-inf')], [float('-inf'), 1.0]]
        ]),
    )
    assert wrong_logits.argmax(-1).item() == 0
    assert corrected_logits.argmax(-1).item() == 1
    equal_model = BridgeCDILModel(2, 2, {
        'classifier': {'scale': 1.0}, 'router': {'hidden_dim': 4},
    })
    equal_model.classifier.expand(2)
    with torch.no_grad():
        equal_model.classifier.weight.zero_()
    for task_id in range(2):
        equal_model.add_operator(DomainOperator(task_id=task_id))
    equal_logits, _ = equal_model.forward_domain_mixture(
        query, torch.tensor([[0.5, 0.5]]),
        class_domain_scores=torch.tensor([[[0.0, 0.0], [0.0, float('-inf')]]]),
    )
    # Class 0 occurs twice, class 1 once; equal evidence gives equal priors.
    torch.testing.assert_close(equal_logits.exp(), torch.tensor([[0.5, 0.5]]))

    native_style_bank = DomainStyleBank()
    native_style_bank.update(
        0,
        torch.randn(20, sdim),
        F.normalize(torch.randn(20, d) + 1.0, dim=-1),
        key_count=4,
        key_iterations=2,
        key_max_samples=20,
        feature_key_space='native',
    )
    native_style_bank.update(
        1,
        torch.randn(20, sdim),
        F.normalize(torch.randn(20, d) - 1.0, dim=-1),
        key_count=4,
        key_iterations=2,
        key_max_samples=20,
        feature_key_space='native',
    )
    native_queries = torch.stack(
        [
            F.normalize(torch.randn(5, d) + 1.0, dim=-1),
            F.normalize(torch.randn(5, d) - 1.0, dim=-1),
        ],
        dim=1,
    )
    native_probability = native_style_bank.domain_probabilities(
        native_queries, num_domains=2, temperature=0.05, neighbors=2
    )
    assert native_probability.shape == (5, 2)
    assert torch.allclose(
        native_probability.sum(dim=-1), torch.ones(5), atol=1e-6
    )

    # Regression test: small covariances must not collapse to zero Bures
    # distance (the old absolute eigenvalue floor disabled covariance transport).
    cov_a = torch.diag(torch.tensor([1.0e-4, 4.0e-4], dtype=torch.float64))
    cov_b = torch.diag(torch.tensor([4.0e-4, 9.0e-4], dtype=torch.float64))
    bw = bures_wasserstein_cov_distance_sq(cov_a, cov_b, eps=1e-5)
    assert torch.allclose(bw, torch.tensor(2.0e-4, dtype=bw.dtype), atol=1e-10)

    rb = robust_boundary_expansion_loss(
        model.classifier.weight[:4], bank, old_class_ids=[0, 1, 2], new_class_ids=[3]
    )
    assert torch.isfinite(rb)
    packed = prepare_robust_boundary_stats(
        bank, [0, 1, 2], device=model.classifier.weight.device
    )
    rb_packed = robust_boundary_expansion_loss(
        model.classifier.weight[:4],
        bank,
        old_class_ids=[0, 1, 2],
        new_class_ids=[3],
        prepared_stats=packed,
    )
    assert torch.allclose(rb, rb_packed, atol=1e-6)

    rn = recurring_novel_harmonic(
        pred=torch.tensor([0, 1, 1, 0, 2, 0]),
        target=torch.tensor([0, 1, 1, 1, 2, 2]),
        task_ids=torch.tensor([0, 0, 1, 1, 1, 1]),
        class_groups={
            0: {'recurring_ids': [], 'novel_ids': [0, 1]},
            1: {'recurring_ids': [1], 'novel_ids': [2]},
        },
    )
    assert rn['recurring_count'] == 2 and rn['novel_count'] == 4
    assert abs(rn['h_rec'] - 0.5) < 1e-8
    assert abs(rn['h_nov'] - 0.75) < 1e-8
    assert abs(rn['h_rn'] - 0.6) < 1e-8

    # Exercise actual evidence fitting/calibration/refit and checkpoint loading,
    # using synthetic features only (no backbone or dataset training).
    fit_cfg = make_cfg()
    fit_cfg['feature_adapter']['enabled'] = False
    fit_cfg['domain_evidence'].update({
        'mode': 'shrinkage_lda', 'hidden_dim': 0,
        'covariance_shrinkage_candidates': [0.1, 0.3],
        'temperature_candidates': [1.0, 4.0], 'calibration_fraction': 0.25,
        'blend_candidates': [0.0, 0.5, 0.9], 'calibration_metric': 'nll',
    })
    fit_model = BridgeCDILModel(8, 4, fit_cfg)
    fit_model.classifier.expand(2)
    fit_model.add_operator(DomainOperator(task_id=0))
    fit_model.add_domain_evidence_head(0, [0, 1])
    trainer = IncrementalTrainer.__new__(IncrementalTrainer)
    trainer.cfg, trainer.model, trainer.device = fit_cfg, fit_model, torch.device('cpu')
    base = F.normalize(torch.randn(24, 8), dim=-1)
    fit_features = torch.cat([base, F.normalize(base + 0.01 * torch.randn_like(base), dim=-1)])
    fit_labels = (torch.arange(24) % 2).repeat(2)
    info = {'task_id': 0, 'current_ids': [0, 1]}
    train_indices, calibration_indices = trainer._evidence_split_indices(info, fit_labels, 24)
    assert len(calibration_indices) == 6 and len(train_indices) == 36
    assert not set((train_indices % 24).tolist()) & set(calibration_indices.tolist())
    trainer.train_domain_evidence_head(
        info, fit_features, fit_features, fit_labels, samples_per_view=24
    )
    fitted_head = fit_model.domain_evidence_heads[0]
    assert all(not p.requires_grad for p in fitted_head.parameters())
    assert torch.isfinite(fitted_head(base)).all()
    registry = ClassRegistry()
    registry.add_classes(['a', 'b'])
    with tempfile.TemporaryDirectory(prefix='bridge_unit_') as directory:
        path = Path(directory) / 'checkpoint.pt'
        save_checkpoint(
            path, task_id=0, model=fit_model, stats_bank=ClassStatsBank(),
            style_bank=DomainStyleBank(), registry=registry, cfg=fit_cfg,
        )
        loaded_model = BridgeCDILModel(8, 4, fit_cfg)
        load_model_state(
            path, loaded_model, ClassStatsBank(), DomainStyleBank(), ClassRegistry()
        )
        torch.testing.assert_close(
            loaded_model.forward_known_domain(base, 0),
            fit_model.forward_known_domain(base, 0),
        )

    print('Smoke test passed.')


if __name__ == '__main__':
    main()
