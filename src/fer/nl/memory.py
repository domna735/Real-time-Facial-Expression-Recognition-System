from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


class AssociativeMemory(nn.Module):
    """Minimal learnable memory module for NL scaffolding.

    This is a small, safe building block that can be plugged into a meta-optimizer later.
    It intentionally does NOT implement full second-order NL training yet.
    """

    def __init__(self, *, hidden_dim: int = 32, layers: int = 1, input_dim: int = 4) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")

        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.input_dim = int(input_dim)

        mods = []
        d = self.input_dim
        for _ in range(self.layers):
            mods.append(nn.Linear(d, self.hidden_dim))
            mods.append(nn.SiLU())
            d = self.hidden_dim
        mods.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return a per-parameter gate in [0,1].

        features: [..., input_dim]
        """
        x = self.net(features)
        return torch.sigmoid(x)


@dataclass
class NLFeatures:
    """A tiny feature pack used by NL-style gates.

    Later we can extend this with:
    - gradient norm statistics
    - EMA momentum statistics
    - teacher-student consistency scores
    """

    grad_abs_mean: torch.Tensor
    grad_abs_max: torch.Tensor
    param_abs_mean: torch.Tensor
    step_frac: torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        return torch.stack(
            [
                self.grad_abs_mean,
                self.grad_abs_max,
                self.param_abs_mean,
                self.step_frac,
            ],
            dim=-1,
        )


def build_nl_features(
    *,
    grad: torch.Tensor,
    param: torch.Tensor,
    step: int,
    total_steps: int,
) -> NLFeatures:
    g = grad.detach()
    p = param.detach()
    step_frac = torch.tensor(float(step) / max(1, int(total_steps)), device=g.device, dtype=g.dtype)
    return NLFeatures(
        grad_abs_mean=g.abs().mean(),
        grad_abs_max=g.abs().max(),
        param_abs_mean=p.abs().mean(),
        step_frac=step_frac,
    )


@torch.no_grad()
def apply_memory_gate(
    *,
    grad: torch.Tensor,
    param: torch.Tensor,
    memory: AssociativeMemory,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    """Return a gated gradient (scaffold only)."""

    feats = build_nl_features(grad=grad, param=param, step=step, total_steps=total_steps).as_tensor()
    gate = memory(feats)
    return grad * gate
