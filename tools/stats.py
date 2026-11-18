from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def fit_lowrank_gaussian(x: torch.Tensor, rank: int = 32, eps: float = 1e-5):
    x = x.float().cpu()
    n, d = x.shape
    mean = x.mean(0)
    xc = x - mean
    if n <= 1:
        basis = torch.eye(d)[:, :1]
        eigvals = torch.full((1,), eps)
        return {
            'count': n,
            'mean': mean,
            'basis': basis,
            'eigvals': eigvals,
            'residual_var': torch.tensor(eps),
        }
    q = min(rank, d, max(1, n - 1))
    # PCA via SVD. For dataset sizes used here this is stable and deterministic under the global seed.
    _, s, v = torch.pca_lowrank(xc, q=q, center=False)
    eigvals = (s.square() / max(n - 1, 1)).clamp_min(eps)
    # Keeping only rank principal directions and treating every omitted
    # direction as zero variance makes feature replay collapse to a thin
    # subspace.  A factor-analysis tail retains their average variance using
    # one scalar rather than D additional values.
    total_variance = xc.square().sum() / max(n - 1, 1)
    residual_dim = max(d - q, 1)
    residual_var = (
        (total_variance - eigvals[:q].sum()) / residual_dim
    ).clamp_min(eps)
    return {
        'count': n,
        'mean': mean,
        'basis': v[:, :q].contiguous(),
        'eigvals': eigvals[:q].contiguous(),
        'residual_var': residual_var.contiguous(),
    }


def covariance_action(stats: dict, q: torch.Tensor):
    U = stats['basis'].to(q)
    e = stats['eigvals'].to(q)
    # This action is used only to *discover* a low-rank transport subspace.
    # Excluding the isotropic tail prevents a global variance difference from
    # injecting arbitrary probe directions into that subspace.  The full
    # factor-analysis covariance, including its tail, is used by the Bures
    # objective in projected_covariance below.
    return U @ (e.unsqueeze(-1) * (U.T @ q))


def projected_covariance(stats: dict, Uproj: torch.Tensor, eps: float = 1e-5):
    basis = stats['basis'].to(Uproj)
    eigvals = stats['eigvals'].to(Uproj)
    residual_var = torch.as_tensor(
        stats.get('residual_var', 0.0),
        device=Uproj.device,
        dtype=Uproj.dtype,
    ).clamp_min(0.0)
    P = Uproj.T @ basis
    cov = residual_var * torch.eye(
        Uproj.shape[1], device=Uproj.device, dtype=Uproj.dtype
    )
    cov = cov + (P * (eigvals - residual_var).clamp_min(0).unsqueeze(0)) @ P.T
    cov = 0.5 * (cov + cov.T)
    cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    return cov


def merge_lowrank_gaussians(a: dict, b: dict, rank: int = 32, eps: float = 1e-5):
    n1, n2 = int(a['count']), int(b['count'])
    if n1 <= 0:
        return b
    if n2 <= 0:
        return a
    n = n1 + n2
    m1, m2 = a['mean'].float(), b['mean'].float()
    mean = (n1 * m1 + n2 * m2) / n
    residual_a = torch.as_tensor(a.get('residual_var', eps)).float()
    residual_b = torch.as_tensor(b.get('residual_var', eps)).float()
    residual_var = (
        ((n1 - 1) * residual_a + (n2 - 1) * residual_b)
        / max(n - 1, 1)
    ).clamp_min(eps)
    cols = []
    if n1 > 1:
        c1 = ((n1 - 1) / max(n - 1, 1)) ** 0.5
        excess_a = (a['eigvals'].float() - residual_a).clamp_min(0.0)
        cols.append(a['basis'].float() * (excess_a.sqrt() * c1).unsqueeze(0))
    if n2 > 1:
        c2 = ((n2 - 1) / max(n - 1, 1)) ** 0.5
        excess_b = (b['eigvals'].float() - residual_b).clamp_min(0.0)
        cols.append(b['basis'].float() * (excess_b.sqrt() * c2).unsqueeze(0))
    delta = (m1 - m2).unsqueeze(1)
    cb = ((n1 * n2 / n) / max(n - 1, 1)) ** 0.5
    cols.append(delta * cb)
    B = torch.cat(cols, dim=1)
    U, S, _ = torch.linalg.svd(B, full_matrices=False)
    q = min(rank, U.shape[1])
    return {
        'count': n,
        'mean': mean,
        'basis': U[:, :q].contiguous(),
        'eigvals': (S[:q].square() + residual_var).clamp_min(eps).contiguous(),
        'residual_var': residual_var.contiguous(),
    }


class ClassStatsBank:
    def __init__(self, rank: int = 32, eps: float = 1e-5):
        self.rank = int(rank)
        self.eps = float(eps)
        self.stats: Dict[int, dict] = {}

    def has(self, class_id: int):
        return int(class_id) in self.stats

    def get(self, class_id: int):
        return self.stats[int(class_id)]

    def update_from_features(self, features: torch.Tensor, labels: torch.Tensor, class_ids=None):
        features = features.float().cpu()
        labels = labels.long().cpu()
        ids = sorted(labels.unique().tolist()) if class_ids is None else sorted(class_ids)
        for cid in ids:
            mask = labels == int(cid)
            if not mask.any():
                continue
            new = fit_lowrank_gaussian(features[mask], self.rank, self.eps)
            if int(cid) in self.stats:
                self.stats[int(cid)] = merge_lowrank_gaussians(self.stats[int(cid)], new, self.rank, self.eps)
            else:
                self.stats[int(cid)] = new

    def means_tensor(self, class_ids):
        return torch.stack([self.stats[int(c)]['mean'] for c in class_ids], 0)

    def sample(self, class_ids, samples_per_class: int, *, generator=None):
        """Sample normalized features from the stored low-rank Gaussians.

        This reuses only compact sufficient statistics; no historical image or
        per-image feature is retained.
        """
        samples_per_class = int(samples_per_class)
        if samples_per_class <= 0:
            return torch.empty(0, 0), torch.empty(0, dtype=torch.long)
        features, labels = [], []
        for cid in class_ids:
            if not self.has(cid):
                continue
            st = self.get(cid)
            mean = st['mean'].float().cpu()
            basis = st['basis'].float().cpu()
            eigvals = st['eigvals'].float().cpu()
            residual_var = torch.as_tensor(
                st.get('residual_var', self.eps), dtype=mean.dtype
            ).clamp_min(self.eps)
            principal_std = (eigvals - residual_var).clamp_min(0.0).sqrt()
            principal_noise = torch.randn(
                samples_per_class,
                basis.shape[1],
                generator=generator,
            )
            tail_noise = torch.randn(
                samples_per_class,
                mean.numel(),
                generator=generator,
            ) * residual_var.sqrt()
            sampled = (
                mean.unsqueeze(0)
                + (principal_noise * principal_std.unsqueeze(0)) @ basis.T
                + tail_noise
            )
            features.append(F.normalize(sampled, dim=-1))
            labels.append(torch.full((samples_per_class,), int(cid), dtype=torch.long))
        if not features:
            feature_dim = 0
            if self.stats:
                feature_dim = int(next(iter(self.stats.values()))['mean'].numel())
            return torch.empty(0, feature_dim), torch.empty(0, dtype=torch.long)
        return torch.cat(features, dim=0), torch.cat(labels, dim=0)

    def state_dict(self):
        return {
            'rank': self.rank,
            'eps': self.eps,
            'stats': {int(k): {kk: (vv.cpu() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} for k, v in self.stats.items()},
        }

    def load_state_dict(self, state):
        self.rank = int(state['rank'])
        self.eps = float(state['eps'])
        self.stats = {int(k): v for k, v in state['stats'].items()}
        # Old checkpoints contained a strictly low-rank covariance.  Preserve
        # their behaviour on load, while every newly fitted statistic uses the
        # factor-analysis tail above.
        for value in self.stats.values():
            value.setdefault('residual_var', torch.tensor(self.eps))


def class_means(features: torch.Tensor, labels: torch.Tensor, class_ids):
    out = []
    for cid in class_ids:
        x = features[labels == int(cid)]
        if x.numel() == 0:
            raise ValueError(f'Class {cid} has no features.')
        out.append(F.normalize(x.mean(0), dim=0))
    return torch.stack(out, 0)
