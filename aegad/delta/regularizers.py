from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from torch.func import jvp

from aegad.delta.extract import Delta

_Reduce = Literal["mean", "sum"]


def energy_l2(yu: torch.Tensor, yv: torch.Tensor, *, reduce: _Reduce = "mean") -> torch.Tensor:
    return _reduce(yu.square() + yv.square(), reduce=reduce)


def relative_energy_l2(
    y: torch.Tensor,
    yu: torch.Tensor,
    yv: torch.Tensor,
    *,
    eps: float = 1e-6,
    reduce: _Reduce = "mean",
) -> torch.Tensor:
    denom = y.square() + eps
    return _reduce((yu.square() + yv.square()) / denom, reduce=reduce)


def balance_l2(
    yu: torch.Tensor, yv: torch.Tensor, *, c: float = 1.0, reduce: _Reduce = "mean"
) -> torch.Tensor:
    return _reduce((yu - c * yv).square(), reduce=reduce)


def pushforward(
    f: Callable[[torch.Tensor], torch.Tensor],
    d: Delta,
) -> Delta:
    y = f(d.y)
    _, yu = jvp(f, (d.y,), (d.yu,))
    _, yv = jvp(f, (d.y,), (d.yv,))
    return Delta(y=y, yu=yu, yv=yv)


def invariant_l2(
    d: Delta,
    *,
    f: Callable[[torch.Tensor], torch.Tensor] | None = None,
    reduce: _Reduce = "mean",
) -> torch.Tensor:
    dd = pushforward(f, d) if f is not None else d
    return energy_l2(dd.yu, dd.yv, reduce=reduce)


def _reduce(x: torch.Tensor, *, reduce: _Reduce) -> torch.Tensor:
    if reduce == "mean":
        return x.mean()
    if reduce == "sum":
        return x.sum()
    raise ValueError(f"unknown reduce: {reduce}")
