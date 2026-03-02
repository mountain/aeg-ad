from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from aegad.delta.extract import Delta


@dataclass(frozen=True)
class DeltaLoss:
    total: torch.Tensor
    data: torch.Tensor
    reg: torch.Tensor


def delta_loss(
    *,
    data_loss: torch.Tensor,
    d: Delta,
    alpha: float,
    reg_fn: Callable[[Delta], torch.Tensor],
) -> DeltaLoss:
    reg = reg_fn(d)
    total = data_loss + alpha * reg
    return DeltaLoss(total=total, data=data_loss, reg=reg)
