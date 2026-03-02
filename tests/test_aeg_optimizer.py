import torch

from aegad.optim import AEGOptimizer


def test_aeg_optimizer_rotation_update_matches_formula() -> None:
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    base = torch.optim.SGD([p], lr=1.0, momentum=0.0)
    opt = AEGOptimizer(base, beta=0.5, max_rotation_ratio=10.0)

    g = torch.tensor([0.1, -0.2, 0.3, -0.4])
    p.grad = g.clone()

    opt.step()

    jg = torch.tensor([-0.2, -0.1, -0.4, -0.3])
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0]) - g + 1.0 * 0.5 * jg
    assert torch.allclose(p.detach(), expected)


def test_aeg_optimizer_beta_is_clamped() -> None:
    p = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0]))
    base = torch.optim.SGD([p], lr=2.0, momentum=0.0)
    opt = AEGOptimizer(base, beta=10.0, max_rotation_ratio=0.1)

    g = torch.tensor([1.0, 2.0, 3.0, 4.0])
    p.grad = g.clone()
    opt.step()

    stats = opt.last_stats
    assert stats is not None
    assert stats.beta_effective <= 10.0

    rot_ratio = stats.rot_update_norm / (2.0 * stats.grad_norm + 1e-12)
    assert rot_ratio <= 0.1001
