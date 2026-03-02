import torch

from aegad import seed_a


def test_seed_a_channels() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 5)
    mu = torch.tensor(2.0)
    lam = torch.tensor(0.5)

    X = seed_a(x, mu, lam)
    torch.testing.assert_close(X.du, torch.ones_like(x) * mu)
    torch.testing.assert_close(X.dv, lam * x)


def test_seed_a_vjp_transpose_rule() -> None:
    torch.manual_seed(0)
    x = torch.randn(7, requires_grad=True)
    mu = torch.tensor(2.0)
    lam = torch.tensor(0.5)

    X = seed_a(x, mu, lam)
    g_val = torch.randn_like(X.val)
    g_du = torch.randn_like(X.du)
    g_dv = torch.randn_like(X.dv)
    L = (X.val * g_val + X.du * g_du + X.dv * g_dv).sum()
    L.backward()

    torch.testing.assert_close(x.grad, g_val + lam * g_dv)
