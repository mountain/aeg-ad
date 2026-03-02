import argparse
import json
import sys
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aegad.delta import delta_loss, energy_l2, from_model


class TinyMlp(nn.Module):
    def __init__(self, in_dim: int = 28 * 28, hidden: int = 64, out_dim: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def forward_dual(
        self, x: torch.Tensor, mu: torch.Tensor, lam: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        val = x.flatten(1)
        du = (torch.ones_like(x) * mu).flatten(1)
        dv = (lam * x).flatten(1)

        val = self.fc1(val)
        du = F.linear(du, self.fc1.weight, None)
        dv = F.linear(dv, self.fc1.weight, None)

        val, du, dv = _relu_dual(val, du, dv)

        val = self.fc2(val)
        du = F.linear(du, self.fc2.weight, None)
        dv = F.linear(dv, self.fc2.weight, None)
        return val, du, dv


def _relu_dual(
    val: torch.Tensor, du: torch.Tensor, dv: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = F.relu(val)
    mask = (val > 0).to(dtype=val.dtype)
    return out, du * mask, dv * mask


@torch.no_grad()
def _accuracy(model: nn.Module, dl: DataLoader[Any], device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    model.train()
    return correct / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--method", type=str, default="dual")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    import torchvision
    from torchvision import transforms

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(root=".data", train=True, download=True, transform=tfm)
    test_ds = torchvision.datasets.MNIST(root=".data", train=False, download=True, transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=0)

    model = TinyMlp().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    mu = torch.tensor(args.mu, device=device)
    lam = torch.tensor(args.lam, device=device)
    alpha = float(args.alpha)

    rows: list[dict[str, Any]] = []
    for ep in range(1, args.epochs + 1):
        seen = 0
        correct = 0
        data_loss_sum = 0.0
        reg_sum = 0.0
        t0 = time.perf_counter()
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            d = from_model(model, x, mu=mu, lam=lam, method=args.method)
            data_loss = F.cross_entropy(d.y, y)
            dl = delta_loss(
                data_loss=data_loss,
                d=d,
                alpha=alpha,
                reg_fn=lambda dd: energy_l2(dd.yu, dd.yv),
            )
            torch.autograd.backward(dl.total)
            opt.step()

            with torch.no_grad():
                b = int(y.numel())
                seen += b
                data_loss_sum += float(dl.data.detach().item()) * b
                reg_sum += float(dl.reg.detach().item()) * b
                pred = d.y.argmax(dim=1)
                correct += int((pred == y).sum().item())

        elapsed = time.perf_counter() - t0
        train_acc = correct / max(seen, 1)
        train_data_loss = data_loss_sum / max(seen, 1)
        train_reg = reg_sum / max(seen, 1)
        test_acc = _accuracy(model, test_dl, device)

        row: dict[str, Any] = {
            "epoch": ep,
            "train_data_loss": train_data_loss,
            "train_reg": train_reg,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "elapsed_s": elapsed,
        }
        rows.append(row)
        print(
            f"epoch={ep}/{args.epochs} | method={args.method} | "
            f"train_loss={train_data_loss:.6g} | reg={train_reg:.6g} | "
            f"train_acc={train_acc:.4f} | test_acc={test_acc:.4f} | sec={elapsed:.2f}",
            file=sys.stderr,
            flush=True,
        )

    payload = {
        "model": "TinyMlp",
        "method": args.method,
        "alpha": alpha,
        "mu": args.mu,
        "lam": args.lam,
        "rows": rows,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
