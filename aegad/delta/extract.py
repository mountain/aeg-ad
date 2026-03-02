from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.func import jvp

from aegad.core.seed import seed_a


@dataclass(frozen=True)
class Delta:
    y: torch.Tensor
    yu: torch.Tensor
    yv: torch.Tensor


def extract(x: Any) -> Delta:
    if hasattr(x, "val") and hasattr(x, "du") and hasattr(x, "dv"):
        y = _as_tensor(x.val)
        yu = _as_tensor(x.du)
        yv = _as_tensor(x.dv)
        return Delta(y=y, yu=yu, yv=yv)

    if isinstance(x, (tuple, list)) and len(x) == 3:
        y, yu, yv = x
        return Delta(y=_as_tensor(y), yu=_as_tensor(yu), yv=_as_tensor(yv))

    raise TypeError("expected DualTensor-like or (y, yu, yv)")


def from_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    mu: torch.Tensor,
    lam: torch.Tensor,
    method: str,
) -> Delta:
    m = method.lower()
    if m in {"dual", "forward_dual", "explicit"}:
        fd = getattr(model, "forward_dual", None)
        if callable(fd):
            y, yu, yv = fd(x, mu, lam)
            return Delta(y=_as_tensor(y), yu=_as_tensor(yu), yv=_as_tensor(yv))
        y_dual = model(seed_a(x, mu, lam))
        return extract(y_dual)

    if m in {"jvp", "torch.func.jvp"}:
        y = model(x)
        dx_u = torch.ones_like(x) * mu
        dx_v = lam * x
        _, yu = jvp(model, (x,), (dx_u,))
        _, yv = jvp(model, (x,), (dx_v,))
        return Delta(y=y, yu=yu, yv=yv)

    raise ValueError(f"unknown method: {method}")


def _as_tensor(x: Any) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError("expected torch.Tensor")
    return x
