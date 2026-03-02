from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, overload

import torch


@dataclass(frozen=True)
class AEGStepStats:
    beta_effective: float
    grad_norm: float
    jgrad_norm: float
    rot_update_norm: float


class AEGOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        base: torch.optim.Optimizer,
        *,
        beta: float,
        max_rotation_ratio: float = 0.3,
        eps: float = 1e-12,
    ) -> None:
        if beta < 0:
            raise ValueError("beta must be >= 0")
        if max_rotation_ratio < 0:
            raise ValueError("max_rotation_ratio must be >= 0")
        if eps <= 0:
            raise ValueError("eps must be > 0")

        self.base = base
        self.beta = float(beta)
        self.max_rotation_ratio = float(max_rotation_ratio)
        self.eps = float(eps)

        self.param_groups = base.param_groups
        self.defaults = base.defaults
        self.state = base.state

        self._last_stats: AEGStepStats | None = None

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = self.base.step(closure)

        beta_eff, grad_norm, jgrad_norm, rot_update_norm = self._apply_rotation_update()
        self._last_stats = AEGStepStats(
            beta_effective=beta_eff,
            grad_norm=grad_norm,
            jgrad_norm=jgrad_norm,
            rot_update_norm=rot_update_norm,
        )
        if closure is None:
            return None
        if loss is None:
            raise RuntimeError("base optimizer returned None for closure step")
        return float(loss)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        sd: dict[str, Any] = self.base.state_dict()
        sd["aeg"] = {
            "beta": self.beta,
            "max_rotation_ratio": self.max_rotation_ratio,
            "eps": self.eps,
        }
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        aeg = state_dict.get("aeg")
        if isinstance(aeg, dict):
            beta = aeg.get("beta")
            max_rotation_ratio = aeg.get("max_rotation_ratio")
            eps = aeg.get("eps")
            if isinstance(beta, (int, float)):
                self.beta = float(beta)
            if isinstance(max_rotation_ratio, (int, float)):
                self.max_rotation_ratio = float(max_rotation_ratio)
            if isinstance(eps, (int, float)) and eps > 0:
                self.eps = float(eps)
        self.base.load_state_dict(state_dict)

    @property
    def last_stats(self) -> AEGStepStats | None:
        return self._last_stats

    @torch.no_grad()
    def _apply_rotation_update(self) -> tuple[float, float, float, float]:
        g2 = 0.0
        j2 = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if not torch.isfinite(g).all():
                    raise RuntimeError("non-finite gradient encountered")
                g2 += float(g.detach().pow(2).sum().item())
                jg = _j_apply(g)
                j2 += float(jg.detach().pow(2).sum().item())

        grad_norm = g2**0.5
        jgrad_norm = j2**0.5
        if grad_norm == 0.0 or self.beta == 0.0 or self.max_rotation_ratio == 0.0:
            return 0.0, grad_norm, jgrad_norm, 0.0

        beta_eff = min(self.beta, (self.max_rotation_ratio * grad_norm) / (jgrad_norm + self.eps))

        rot_update2 = 0.0
        for group in self.param_groups:
            lr = float(group.get("lr", 0.0))
            for p in group["params"]:
                if p.grad is None:
                    continue
                jg = _j_apply(p.grad)
                upd = lr * beta_eff * jg
                p.add_(upd)
                rot_update2 += float(upd.detach().pow(2).sum().item())

        rot_update_norm = rot_update2**0.5
        if not torch.isfinite(torch.tensor(rot_update_norm)):
            raise RuntimeError("non-finite rotation update encountered")
        return beta_eff, grad_norm, jgrad_norm, rot_update_norm


def _j_apply(g: torch.Tensor) -> torch.Tensor:
    flat = g.reshape(-1)
    n = flat.numel()
    if n < 2:
        return torch.zeros_like(g)
    a = flat[0 : n - 1 : 2]
    b = flat[1:n:2]
    jflat = torch.zeros_like(flat)
    jflat[0 : n - 1 : 2] = b
    jflat[1:n:2] = -a
    return jflat.reshape_as(g)
