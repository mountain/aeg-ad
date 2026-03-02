import torch
from torch.func import jvp, vjp, vmap

from aegad import seed_a_components


def test_seed_a_components_jvp_matches_reference() -> None:
    x = torch.randn(3, 5, dtype=torch.float64)
    t = torch.randn_like(x)
    mu = 2.0
    lam = 0.3

    (y_val, y_du, y_dv), (t_val, t_du, t_dv) = jvp(
        lambda z: seed_a_components(z, mu, lam), (x,), (t,)
    )
    ref_val, ref_du, ref_dv = seed_a_components(x, mu, lam)
    assert torch.allclose(y_val, ref_val)
    assert torch.allclose(y_du, ref_du)
    assert torch.allclose(y_dv, ref_dv)

    assert torch.allclose(t_val, t)
    assert torch.allclose(t_du, torch.zeros_like(x))
    assert torch.allclose(t_dv, lam * t)


def test_seed_a_components_vjp_matches_reference() -> None:
    x = torch.randn(4, 7, dtype=torch.float64, requires_grad=True)
    mu = 2.0
    lam = 0.3

    y, pullback = vjp(lambda z: seed_a_components(z, mu, lam), x)

    g_val = torch.randn_like(y[0])
    g_du = torch.randn_like(y[1])
    g_dv = torch.randn_like(y[2])

    (gx,) = pullback((g_val, g_du, g_dv))
    ref = g_val + lam * g_dv
    assert torch.allclose(gx, ref)


def test_seed_a_components_vmap_smoke() -> None:
    x = torch.randn(2, 3, 5)
    mu = 2.0
    lam = 0.3
    y_val, y_du, y_dv = vmap(lambda z: seed_a_components(z, mu, lam))(x)
    assert y_val.shape == x.shape
    assert y_du.shape == x.shape
    assert y_dv.shape == x.shape


def test_forward_ad_adapter_smoke() -> None:
    from aegad import as_forward_ad_dual

    x = torch.randn(3, 5)
    t = torch.randn_like(x)
    with torch.autograd.forward_ad.dual_level():
        d = as_forward_ad_dual(x, t)
        y, tt = torch.autograd.forward_ad.unpack_dual(d)
        assert isinstance(d, torch.Tensor)
        assert torch.allclose(y, x)
        assert torch.allclose(tt, t)
