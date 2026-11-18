from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from model.operator import DomainOperator
from .stats import fit_lowrank_gaussian, covariance_action, projected_covariance
from .bures import bures_wasserstein_cov_distance_sq


def estimate_domain_subspace(current_stats, reference_stats, bridge_ids, feature_dim, rank, probe_dim, seed=42):
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    q = torch.randn(feature_dim, probe_dim, generator=g)
    q = torch.linalg.qr(q, mode='reduced').Q
    columns = []
    for cid in bridge_ids:
        cur = current_stats[int(cid)]
        ref = reference_stats.get(int(cid))
        dcov_q = covariance_action(cur, q) - covariance_action(ref, q)
        dmean = (cur['mean'] - ref['mean']).unsqueeze(1)
        columns.extend([dcov_q, dmean])
    M = torch.cat(columns, dim=1)
    if torch.linalg.norm(M) < 1e-10:
        return torch.empty(feature_dim, 0)
    U, _, _ = torch.linalg.svd(M, full_matrices=False)
    r = min(rank, U.shape[1])
    return U[:, :r].contiguous()


def estimate_bridge_operator(
    task_id: int,
    features: torch.Tensor,
    labels: torch.Tensor,
    bridge_ids,
    stats_bank,
    cfg: dict,
    class_rank: int,
    seed: int = 42,
    device=None,
):
    bridge_ids = [int(c) for c in bridge_ids if stats_bank.has(int(c))]
    min_bridges = int(cfg.get('min_bridges', 2))
    if len(bridge_ids) < min_bridges:
        return DomainOperator(task_id=task_id, bridge_count=len(bridge_ids))

    eps = float(cfg.get('eps', 1e-5))
    current = {}
    for cid in bridge_ids:
        x = features[labels == cid]
        current[cid] = fit_lowrank_gaussian(x, class_rank, eps)

    feature_dim = features.shape[1]
    U = estimate_domain_subspace(
        current_stats=current,
        reference_stats=stats_bank,
        bridge_ids=bridge_ids,
        feature_dim=feature_dim,
        rank=int(cfg.get('domain_rank', 24)),
        probe_dim=int(cfg.get('probe_dim', 48)),
        seed=seed + task_id,
    )
    if U.numel() == 0:
        return DomainOperator(task_id=task_id, bridge_count=len(bridge_ids))

    r = U.shape[1]
    device = torch.device(
        device
        if device is not None
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    # The nested matrix square roots in the Bures objective operate on the
    # covariance-squared scale. Float64 is inexpensive at rank 24 and avoids
    # erasing the transport signal through float32 cancellation.
    Udev = U.to(device=device, dtype=torch.float64)
    proj_pairs = []
    for cid in bridge_ids:
        cur = current[cid]
        ref = stats_bank.get(cid)
        ccur = projected_covariance(cur, Udev, eps)
        cref = projected_covariance(ref, Udev, eps)
        mcur = cur['mean'].to(device=device, dtype=Udev.dtype) @ Udev
        mref = ref['mean'].to(device=device, dtype=Udev.dtype) @ Udev
        proj_pairs.append((ccur, cref, mcur, mref))

    H = torch.nn.Parameter(torch.zeros(r, r, device=device, dtype=Udev.dtype))
    opt = torch.optim.Adam([H], lr=float(cfg.get('bw_lr', 0.03)))
    steps = int(cfg.get('bw_steps', 120))
    reg = float(cfg.get('identity_reg', 0.01))
    mean_geometry_weight = float(cfg.get('mean_geometry_weight', 0.1))
    min_scale = float(cfg.get('min_scale', 0.25))
    max_scale = float(cfg.get('max_scale', 4.0))
    if not 0 < min_scale <= 1.0 <= max_scale:
        raise ValueError(
            'transport scales must satisfy 0 < min_scale <= 1 <= max_scale.'
        )
    log_min_scale = math.log(min_scale)
    log_max_scale = math.log(max_scale)
    log_scale_limit = max(abs(log_min_scale), abs(log_max_scale))
    I = torch.eye(r, device=device, dtype=Udev.dtype)
    current_means = torch.stack([item[2] for item in proj_pairs], dim=0)
    reference_means = torch.stack([item[3] for item in proj_pairs], dim=0)

    def conditioned_map(parameter):
        symmetric = 0.5 * (parameter + parameter.T)
        # Bounding ||S||_F also bounds every eigenvalue of symmetric S. This
        # avoids differentiating through eigendecomposition at repeated
        # eigenvalues, which is numerically unstable at the identity start.
        frobenius = torch.linalg.vector_norm(symmetric)
        divisor = (frobenius / log_scale_limit).clamp_min(1.0)
        return torch.matrix_exp(symmetric / divisor)

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        A = conditioned_map(H)
        loss = 0.0
        for ccur, cref, _, _ in proj_pairs:
            mapped = A @ ccur @ A.T
            loss = loss + bures_wasserstein_cov_distance_sq(mapped, cref, eps)
        # The optimal shared translation is analytic. The remaining residual
        # aligns relative bridge-class geometry rather than merely matching one
        # global domain mean after covariance fitting.
        b_step = (reference_means - current_means @ A.T).mean(dim=0)
        mapped_means = current_means @ A.T + b_step
        mean_loss = F.smooth_l1_loss(
            mapped_means,
            reference_means,
            reduction='sum',
        ) / len(proj_pairs)
        identity_loss = (A - I).square().sum() / r
        loss = (
            loss / len(proj_pairs)
            + mean_geometry_weight * mean_loss
            + reg * identity_loss
        )
        if not bool(torch.isfinite(loss)):
            break
        loss.backward()
        if H.grad is None or not bool(torch.isfinite(H.grad).all()):
            H.grad = None
            break
        torch.nn.utils.clip_grad_norm_([H], 10.0)
        opt.step()

    with torch.no_grad():
        A = conditioned_map(H)
        b = (reference_means - current_means @ A.T).mean(dim=0)
        bw_before = torch.stack([bures_wasserstein_cov_distance_sq(ccur, cref, eps) for ccur, cref, _, _ in proj_pairs]).mean().item()
        bw_after = torch.stack([bures_wasserstein_cov_distance_sq(A @ ccur @ A.T, cref, eps) for ccur, cref, _, _ in proj_pairs]).mean().item()
        mean_before = (
            current_means
            + (reference_means - current_means).mean(dim=0)
            - reference_means
        ).square().sum(dim=-1).mean().item()
        mean_after = (
            current_means @ A.T + b - reference_means
        ).square().sum(dim=-1).mean().item()
    return DomainOperator(
        task_id=task_id,
        U=U.float().cpu(),
        A=A.float().cpu(),
        b=b.float().cpu(),
        bridge_count=len(bridge_ids),
        bw_before=float(bw_before),
        bw_after=float(bw_after),
        mean_before=float(mean_before),
        mean_after=float(mean_after),
    )
