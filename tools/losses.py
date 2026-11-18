from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def angular_margin_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    scale: float,
    margin: float = 0.0,
):
    """Cross-entropy on cosine logits with an optional target angular margin.

    The margin is applied only while training a domain adapter.  It contracts
    each current-domain class on the hypersphere without changing the common
    cosine classifier used at inference.
    """
    margin = float(margin)
    if margin <= 0:
        return F.cross_entropy(logits, targets)
    scale = float(scale)
    if scale <= 0:
        raise ValueError('Cosine-logit scale must be positive.')
    cosine = (logits / scale).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    target_cosine = cosine.gather(1, targets[:, None]).squeeze(1)
    target_margin_cosine = (
        target_cosine * math.cos(margin)
        - (1.0 - target_cosine.square()).clamp_min(0.0).sqrt()
        * math.sin(margin)
    )
    adjusted = logits.clone()
    adjusted.scatter_(1, targets[:, None], (scale * target_margin_cosine)[:, None])
    return F.cross_entropy(adjusted, targets)


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
):
    """Multi-positive supervised contrastive loss for feature-level views."""
    if features.ndim != 3:
        raise ValueError('features must have shape [batch, views, dim].')
    batch, views, dim = features.shape
    flattened = F.normalize(features.reshape(batch * views, dim), dim=-1)
    repeated_labels = labels.repeat_interleave(views)
    logits = flattened @ flattened.T / float(temperature)
    self_mask = torch.eye(
        len(flattened), dtype=torch.bool, device=flattened.device
    )
    logits = logits.masked_fill(self_mask, float('-inf'))
    positives = repeated_labels[:, None].eq(repeated_labels[None, :]) & ~self_mask
    valid = positives.any(dim=1)
    if not bool(valid.any()):
        return flattened.reshape(-1)[:0].sum()
    logits = logits[valid]
    positives = positives[valid]
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_prob = torch.where(
        positives, log_prob, torch.zeros_like(log_prob)
    ).sum(dim=1) / positives.sum(dim=1).clamp_min(1)
    return -positive_log_prob.mean()


def prepare_robust_boundary_stats(
    stats_bank,
    old_class_ids,
    *,
    device,
    dtype=torch.float32,
):
    """Pack variable-rank historical statistics once for vectorized reuse."""
    entries = [
        (int(class_id), stats_bank.get(class_id))
        for class_id in old_class_ids
        if stats_bank.has(class_id)
    ]
    if not entries:
        return None

    feature_dim = int(entries[0][1]['mean'].numel())
    max_rank = max(int(stats['basis'].shape[1]) for _, stats in entries)
    means = torch.empty(len(entries), feature_dim, device=device, dtype=dtype)
    bases = torch.zeros(
        len(entries), feature_dim, max_rank, device=device, dtype=dtype
    )
    eigvals = torch.zeros(len(entries), max_rank, device=device, dtype=dtype)
    class_ids = torch.empty(len(entries), device=device, dtype=torch.long)

    for row, (class_id, stats) in enumerate(entries):
        rank = int(stats['basis'].shape[1])
        class_ids[row] = class_id
        means[row] = F.normalize(
            stats['mean'].to(device=device, dtype=dtype), dim=0
        )
        bases[row, :, :rank] = stats['basis'].to(device=device, dtype=dtype)
        eigvals[row, :rank] = stats['eigvals'].to(device=device, dtype=dtype)

    return {
        'class_ids': class_ids,
        'means': means,
        'bases': bases,
        'eigvals': eigvals,
    }


def robust_boundary_expansion_loss(
    classifier_weight: torch.Tensor,
    stats_bank,
    old_class_ids,
    new_class_ids,
    margin: float = 0.2,
    rho: float = 1.0,
    eps: float = 1e-8,
    prepared_stats=None,
):
    """Worst-case margin against old-class low-rank uncertainty ellipsoids.

    Only new-class weights receive gradient; old-class weights/statistics are detached.
    """
    if not old_class_ids or not new_class_ids:
        return classifier_weight.new_zeros(())
    w = F.normalize(classifier_weight, dim=-1)
    packed = prepared_stats
    if packed is None:
        packed = prepare_robust_boundary_stats(
            stats_bank,
            old_class_ids,
            device=w.device,
            dtype=w.dtype,
        )
    if packed is None:
        return classifier_weight.new_zeros(())

    old_indices = packed['class_ids'].to(w.device)
    new_indices = torch.as_tensor(new_class_ids, device=w.device, dtype=torch.long)
    means = packed['means'].to(w.device, w.dtype).detach()
    bases = packed['bases'].to(w.device, w.dtype).detach()
    eigvals = packed['eigvals'].to(w.device, w.dtype).clamp_min(0).detach()

    # [old, new, feature]. Old weights/statistics are detached so this term
    # changes only newly introduced classifier weights.
    diff = w[old_indices].detach().unsqueeze(1) - w[new_indices].unsqueeze(0)
    nominal = torch.einsum('ond,od->on', diff, means)
    projected = torch.einsum('odr,ond->onr', bases, diff)
    uncertainty = float(rho) * torch.sqrt(
        (eigvals.unsqueeze(1) * projected.square()).sum(dim=-1) + eps
    )
    worst_margin = nominal - uncertainty
    return F.relu(float(margin) - worst_margin).mean()
