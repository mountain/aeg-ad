from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch


def as_forward_ad_dual(val: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    if val.shape != tangent.shape:
        raise ValueError("tangent.shape must equal val.shape")
    if val.dtype != tangent.dtype:
        raise ValueError("tangent.dtype must equal val.dtype")
    if val.device != tangent.device:
        raise ValueError("tangent.device must equal val.device")
    make_dual = cast(
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        torch.autograd.forward_ad.make_dual,
    )
    return make_dual(val, tangent)
