from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DomainOperator:
    """Low-rank selective canonicalization operator.

    z_can = z + ((zU)A^T + b - zU)U^T
    """

    task_id: int
    U: torch.Tensor | None = None   # [D, r]
    A: torch.Tensor | None = None   # [r, r]
    b: torch.Tensor | None = None   # [r]
    bridge_count: int = 0
    bw_before: float | None = None
    bw_after: float | None = None
    mean_before: float | None = None
    mean_after: float | None = None

    @property
    def is_identity(self):
        return self.U is None or self.U.numel() == 0

    def apply(self, z: torch.Tensor) -> torch.Tensor:
        if self.is_identity:
            return z
        U = self.U.to(z.device, z.dtype)
        A = self.A.to(z.device, z.dtype)
        b = self.b.to(z.device, z.dtype)
        p = z @ U
        p2 = p @ A.T + b
        return z + (p2 - p) @ U.T

    def state_dict(self):
        return {
            'task_id': self.task_id,
            'U': None if self.U is None else self.U.cpu(),
            'A': None if self.A is None else self.A.cpu(),
            'b': None if self.b is None else self.b.cpu(),
            'bridge_count': self.bridge_count,
            'bw_before': self.bw_before,
            'bw_after': self.bw_after,
            'mean_before': self.mean_before,
            'mean_after': self.mean_after,
        }

    @classmethod
    def from_state_dict(cls, state):
        return cls(
            task_id=int(state['task_id']),
            U=state.get('U'),
            A=state.get('A'),
            b=state.get('b'),
            bridge_count=int(state.get('bridge_count', 0)),
            bw_before=state.get('bw_before'),
            bw_after=state.get('bw_after'),
            mean_before=state.get('mean_before'),
            mean_after=state.get('mean_after'),
        )
