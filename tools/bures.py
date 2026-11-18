from __future__ import annotations

import torch


def symmetrize(x):
    return 0.5 * (x + x.transpose(-1, -2))


def sqrtm_psd(x: torch.Tensor, eps: float = 1e-6):
    x = symmetrize(x)
    e, v = torch.linalg.eigh(x)
    # Do not clamp eigenvalues to an absolute ``eps`` here. For Bures distance
    # the inner matrix has eigenvalues on the covariance-squared scale. With
    # normalized visual features these are commonly below 1e-5; flooring them
    # to 1e-5 makes the distance negative and the final clamp returns zero.
    # Covariance inputs are already regularized, so only numerical negatives
    # need to be removed.
    e = e.clamp_min(0.0)
    return (v * e.sqrt().unsqueeze(-2)) @ v.transpose(-1, -2)


def bures_wasserstein_cov_distance_sq(cov_a: torch.Tensor, cov_b: torch.Tensor, eps: float = 1e-6):
    cov_a = symmetrize(cov_a)
    cov_b = symmetrize(cov_b)
    sqrt_a = sqrtm_psd(cov_a, eps)
    middle = symmetrize(sqrt_a @ cov_b @ sqrt_a)
    sqrt_middle = sqrtm_psd(middle, eps)
    val = torch.trace(cov_a) + torch.trace(cov_b) - 2.0 * torch.trace(sqrt_middle)
    return val.clamp_min(0.0)
