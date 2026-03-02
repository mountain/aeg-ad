import argparse
import json
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from aegad.delta import delta_loss, energy_l2, from_model


class TinyBlock(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, d_ff: int = 128) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim).transpose(1, 3)
        q, k, v = qkv.unbind(dim=2)
        scale = self.head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        a = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, c)
        a = self.proj(a)
        x = self.ln1(x + a)
        y = self.ff2(F.relu(self.ff1(x)))
        x = self.ln2(x + y)
        return x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=32)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.1)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    model = TinyBlock(d_model=args.d_model).to(device)

    x = torch.randn(args.batch, args.seq, args.d_model, device=device)
    mu = torch.tensor(args.mu, device=device)
    lam = torch.tensor(args.lam, device=device)
    alpha = float(args.alpha)

    t0 = time.perf_counter()
    d = from_model(model, x, mu=mu, lam=lam, method="jvp")
    data_loss = d.y.square().mean()
    dl = delta_loss(
        data_loss=data_loss,
        d=d,
        alpha=alpha,
        reg_fn=lambda dd: energy_l2(dd.yu, dd.yv),
    )
    dl.total.backward()
    elapsed = time.perf_counter() - t0

    gnorm = 0.0
    for p_ in model.parameters():
        if p_.grad is None:
            continue
        gnorm += float(p_.grad.detach().pow(2).sum().item())
    gnorm = gnorm**0.5

    payload = {
        "model": "TinyBlock",
        "method": "jvp",
        "alpha": alpha,
        "mu": args.mu,
        "lam": args.lam,
        "data_loss": float(dl.data.detach().cpu()),
        "reg": float(dl.reg.detach().cpu()),
        "total": float(dl.total.detach().cpu()),
        "grad_norm": gnorm,
        "elapsed_s": elapsed,
    }
    print(json.dumps(payload, indent=2))
    print(f"elapsed_s={elapsed:.4f}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
