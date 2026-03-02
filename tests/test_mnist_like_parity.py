import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp

from aegad import seed_a


class _TinyMnistCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.fc1 = nn.Linear(16 * 5 * 5, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def test_param_grads_match_torch_func_jvp() -> None:
    torch.manual_seed(0)

    model_dual = _TinyMnistCnn()
    model_ref = _TinyMnistCnn()
    model_ref.load_state_dict(model_dual.state_dict())

    x = torch.randn(8, 1, 28, 28)
    target = torch.randint(0, 10, (8,))
    mu = torch.tensor(1.0)
    lam = torch.tensor(0.1)
    alpha = 0.01

    logits = model_dual(x)
    data_loss = F.cross_entropy(logits, target)
    y_dual = model_dual(seed_a(x, mu, lam))
    reg = y_dual.du.square().mean() + y_dual.dv.square().mean()
    loss = data_loss + alpha * reg
    loss.backward()

    logits_ref = model_ref(x)
    data_loss_ref = F.cross_entropy(logits_ref, target)
    dx_u = torch.ones_like(x) * mu
    dx_v = lam * x
    _, y_u = jvp(model_ref, (x,), (dx_u,))
    _, y_v = jvp(model_ref, (x,), (dx_v,))
    reg_ref = y_u.square().mean() + y_v.square().mean()
    loss_ref = data_loss_ref + alpha * reg_ref
    loss_ref.backward()

    for p_dual, p_ref in zip(model_dual.parameters(), model_ref.parameters(), strict=True):
        if p_dual.grad is None or p_ref.grad is None:
            raise AssertionError("missing gradients")
        torch.testing.assert_close(p_dual.grad, p_ref.grad, rtol=1e-3, atol=1e-4)
