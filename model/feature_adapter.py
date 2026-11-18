from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualFeatureAdapter(nn.Module):
    """Compact cascaded post-backbone calibration with an identity start.

    A second residual stage increases within-domain fitting capacity without
    modifying any transformer block.  Keeping this post-backbone is distinct
    from inserting adapters throughout a pretrained ViT and keeps cached
    frozen features reusable.
    """

    def __init__(self, dim: int, bottleneck_dim: int = 128, stages: int = 1):
        super().__init__()
        stages = int(stages)
        if stages not in {1, 2}:
            raise ValueError('Residual feature adapter stages must be 1 or 2.')
        self.stages = stages
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, dim)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        if self.stages == 2:
            self.refine_norm = nn.LayerNorm(dim)
            self.refine_down = nn.Linear(dim, bottleneck_dim)
            self.refine_up = nn.Linear(bottleneck_dim, dim)
            nn.init.xavier_uniform_(self.refine_down.weight)
            nn.init.zeros_(self.refine_down.bias)
            nn.init.zeros_(self.refine_up.weight)
            nn.init.zeros_(self.refine_up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.up(F.gelu(self.down(self.norm(features))))
        calibrated = features + residual
        if self.stages == 2:
            refinement = self.refine_up(
                F.gelu(self.refine_down(self.refine_norm(calibrated)))
            )
            calibrated = calibrated + refinement
        return F.normalize(calibrated, dim=-1)
