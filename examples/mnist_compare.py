import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp
from torch.utils.data import DataLoader

from aegad import seed_a


class TinyMnistCnn(nn.Module):
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    import torchvision
    from torchvision import transforms

    device = torch.device(args.device)
    torch.manual_seed(0)

    ds = torchvision.datasets.MNIST(
        root=".data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    x, y = next(iter(dl))
    x = x.to(device)
    y = y.to(device)

    mu = torch.tensor(args.mu, device=device)
    lam = torch.tensor(args.lam, device=device)
    alpha = float(args.alpha)

    model_dual = TinyMnistCnn().to(device)
    model_ref = TinyMnistCnn().to(device)
    model_ref.load_state_dict(model_dual.state_dict())

    logits = model_dual(x)
    data_loss = F.cross_entropy(logits, y)
    y_dual = model_dual(seed_a(x, mu, lam))
    reg = y_dual.du.square().mean() + y_dual.dv.square().mean()
    loss = data_loss + alpha * reg
    loss.backward()

    logits_ref = model_ref(x)
    data_loss_ref = F.cross_entropy(logits_ref, y)
    dx_u = torch.ones_like(x) * mu
    dx_v = lam * x
    _, y_u = jvp(model_ref, (x,), (dx_u,))
    _, y_v = jvp(model_ref, (x,), (dx_v,))
    reg_ref = y_u.square().mean() + y_v.square().mean()
    loss_ref = data_loss_ref + alpha * reg_ref
    loss_ref.backward()

    diffs = []
    for p_dual, p_ref in zip(model_dual.parameters(), model_ref.parameters(), strict=True):
        diffs.append((p_dual.grad - p_ref.grad).abs().max().item())
    print(
        {
            "loss_dual": float(loss.detach().cpu()),
            "loss_ref": float(loss_ref.detach().cpu()),
            "max_param_grad_abs_diff": max(diffs),
        }
    )


if __name__ == "__main__":
    main()
