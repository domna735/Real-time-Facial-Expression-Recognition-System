from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GradAccumState:
    accum_steps: int
    step_in_accum: int = 0

    def should_step(self) -> bool:
        return (self.step_in_accum + 1) >= int(self.accum_steps)

    def next(self) -> "GradAccumState":
        n = int(self.accum_steps)
        if n <= 1:
            return GradAccumState(accum_steps=1, step_in_accum=0)
        s = (self.step_in_accum + 1) % n
        return GradAccumState(accum_steps=n, step_in_accum=int(s))


def scale_loss_for_accum(loss: torch.Tensor, *, accum_steps: int) -> torch.Tensor:
    """Scale loss so gradients match a larger effective batch size."""
    n = max(1, int(accum_steps))
    return loss / float(n)
