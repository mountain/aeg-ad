from __future__ import annotations

from typing import Any

import torch

from aegad.core.dual import DualTensor


def lift(x: torch.Tensor) -> DualTensor:
    z = torch.zeros_like(x)
    return DualTensor(x, z, z)


def const(x: torch.Tensor) -> DualTensor:
    return lift(x)


def seed_u(x: torch.Tensor) -> DualTensor:
    one = torch.ones_like(x)
    zero = torch.zeros_like(x)
    return DualTensor(x, one, zero)


def seed_v(x: torch.Tensor) -> DualTensor:
    one = torch.ones_like(x)
    zero = torch.zeros_like(x)
    return DualTensor(x, zero, one)


def seed_a(x: torch.Tensor, mu: Any, lam: Any) -> DualTensor:
    val, du, dv = seed_a_components(x, mu, lam)
    return DualTensor(val, du, dv)


def seed_a_components(
    x: torch.Tensor, mu: Any, lam: Any
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    du = torch.ones_like(x) * mu
    dv = lam * x
    return x, du, dv
