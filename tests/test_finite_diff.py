from typing import cast

import torch

from aegad import DualTensor, seed_a


def _f(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x) * torch.sigmoid(x) + x * x


def test_dual_matches_finite_difference() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 4, dtype=torch.float64)
    mu = torch.tensor(2.0, dtype=torch.float64)
    lam = torch.tensor(0.5, dtype=torch.float64)
    eps = 1e-6

    dx_u = torch.ones_like(x) * mu
    dx_v = lam * x

    y_u_fd = (_f(x + eps * dx_u) - _f(x - eps * dx_u)) / (2 * eps)
    y_v_fd = (_f(x + eps * dx_v) - _f(x - eps * dx_v)) / (2 * eps)

    X = seed_a(x, mu, lam)
    Y = cast(DualTensor, _f(X))
    torch.testing.assert_close(Y.du, y_u_fd, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(Y.dv, y_v_fd, rtol=1e-5, atol=1e-5)
