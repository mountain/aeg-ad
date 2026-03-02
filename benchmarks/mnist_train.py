import argparse
import json
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aegad.delta import delta_loss, energy_l2, from_model
from aegad.optim import AEGOptimizer


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

    def forward_dual(
        self, x: torch.Tensor, mu: torch.Tensor, lam: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        val = x
        du = torch.ones_like(x) * mu
        dv = lam * x

        val, du, dv = _conv2d_dual(val, du, dv, self.conv1)
        val, du, dv = _relu_dual(val, du, dv)
        val, du, dv = _max_pool2d_dual(val, du, dv, kernel_size=2)

        val, du, dv = _conv2d_dual(val, du, dv, self.conv2)
        val, du, dv = _relu_dual(val, du, dv)
        val, du, dv = _max_pool2d_dual(val, du, dv, kernel_size=2)

        val = torch.flatten(val, 1)
        du = torch.flatten(du, 1)
        dv = torch.flatten(dv, 1)

        val, du, dv = _linear_dual(val, du, dv, self.fc1)
        val, du, dv = _relu_dual(val, du, dv)
        val, du, dv = _linear_dual(val, du, dv, self.fc2)
        return val, du, dv


class OptAEGV3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ux = nn.Parameter(torch.tensor(0.0))
        self.uy = nn.Parameter(torch.tensor(1.0))
        self.vx = nn.Parameter(torch.tensor(0.0))
        self.vy = nn.Parameter(torch.tensor(1.0))
        self.afactor = nn.Parameter(torch.tensor(1.0))
        self.mfactor = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x * (1 + self.uy) + self.ux
        v = x * (1 + self.vy) + self.vx

        dx = self.afactor * F.relu(u)
        dy = self.mfactor * v * torch.tanh(x)
        return x * (1 + dy) + dx

    def forward_dual(
        self, x: torch.Tensor, du: torch.Tensor, dv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uy = 1 + self.uy
        vy = 1 + self.vy

        u = x * uy + self.ux
        u_du = du * uy
        u_dv = dv * uy

        v = x * vy + self.vx
        v_du = du * vy
        v_dv = dv * vy

        relu_u, relu_u_du, relu_u_dv = _relu_dual(u, u_du, u_dv)
        dx = self.afactor * relu_u
        dx_du = self.afactor * relu_u_du
        dx_dv = self.afactor * relu_u_dv

        th = torch.tanh(x)
        dth = 1 - th * th
        th_du = dth * du
        th_dv = dth * dv

        v_th = v * th
        v_th_du = v_du * th + v * th_du
        v_th_dv = v_dv * th + v * th_dv

        dy = self.mfactor * v_th
        dy_du = self.mfactor * v_th_du
        dy_dv = self.mfactor * v_th_dv

        one_plus_dy = 1 + dy
        out = x * one_plus_dy + dx
        out_du = du * one_plus_dy + x * dy_du + dx_du
        out_dv = dv * one_plus_dy + x * dy_dv + dx_dv
        return out, out_du, out_dv


class MnistCnn700(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv0 = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False)
        self.lnon0 = OptAEGV3()
        self.conv1 = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False)
        self.lnon1 = OptAEGV3()
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False)
        self.lnon2 = OptAEGV3()
        self.fc = nn.Linear(4 * 3 * 3, 10, bias=False)
        self.norm0 = nn.RMSNorm([4, 28, 28], elementwise_affine=False)
        self.norm1 = nn.RMSNorm([4, 14, 14], elementwise_affine=False)
        self.norm2 = nn.RMSNorm([4, 7, 7], elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.lnon0(x)
        x = self.pool(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.lnon1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.lnon2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward_dual(
        self, x: torch.Tensor, mu: torch.Tensor, lam: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        val = x
        du = torch.ones_like(x) * mu
        dv = lam * x

        val, du, dv = _conv2d_dual(val, du, dv, self.conv0)
        val, du, dv = _rmsnorm_dual(val, du, dv, eps=1e-6)
        val, du, dv = self.lnon0.forward_dual(val, du, dv)
        val, du, dv = _max_pool2d_dual(val, du, dv, kernel_size=2)

        val, du, dv = _conv2d_dual(val, du, dv, self.conv1)
        val, du, dv = _rmsnorm_dual(val, du, dv, eps=1e-6)
        val, du, dv = self.lnon1.forward_dual(val, du, dv)
        val, du, dv = _max_pool2d_dual(val, du, dv, kernel_size=2)

        val, du, dv = _conv2d_dual(val, du, dv, self.conv2)
        val, du, dv = _rmsnorm_dual(val, du, dv, eps=1e-6)
        val, du, dv = self.lnon2.forward_dual(val, du, dv)
        val, du, dv = _max_pool2d_dual(val, du, dv, kernel_size=2)

        val = torch.flatten(val, 1)
        du = torch.flatten(du, 1)
        dv = torch.flatten(dv, 1)
        val, du, dv = _linear_dual(val, du, dv, self.fc)
        return val, du, dv


def _relu_dual(
    val: torch.Tensor, du: torch.Tensor, dv: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = F.relu(val)
    mask = (val > 0).to(dtype=val.dtype)
    return out, du * mask, dv * mask


def _max_pool2d_dual(
    val: torch.Tensor,
    du: torch.Tensor,
    dv: torch.Tensor,
    *,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    ceil_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out, idx = F.max_pool2d(
        val,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True,
    )
    flat_idx = idx.reshape(idx.shape[0], idx.shape[1], -1)
    out_du = du.reshape(du.shape[0], du.shape[1], -1).gather(2, flat_idx).reshape_as(out)
    out_dv = dv.reshape(dv.shape[0], dv.shape[1], -1).gather(2, flat_idx).reshape_as(out)
    return out, out_du, out_dv


def _conv2d_dual(
    val: torch.Tensor,
    du: torch.Tensor,
    dv: torch.Tensor,
    layer: nn.Conv2d,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = F.conv2d(
        val,
        layer.weight,
        layer.bias,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
    )
    out_du = F.conv2d(
        du,
        layer.weight,
        None,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
    )
    out_dv = F.conv2d(
        dv,
        layer.weight,
        None,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
    )
    return out, out_du, out_dv


def _linear_dual(
    val: torch.Tensor,
    du: torch.Tensor,
    dv: torch.Tensor,
    layer: nn.Linear,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = F.linear(val, layer.weight, layer.bias)
    out_du = F.linear(du, layer.weight, None)
    out_dv = F.linear(dv, layer.weight, None)
    return out, out_du, out_dv


def _rmsnorm_dual(
    val: torch.Tensor,
    du: torch.Tensor,
    dv: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dims = tuple(range(1, val.ndim))
    m = val.pow(2).mean(dim=dims, keepdim=True)
    m_sqrt = m.sqrt()
    denom = m_sqrt + eps

    m_du = (2 * val * du).mean(dim=dims, keepdim=True)
    m_dv = (2 * val * dv).mean(dim=dims, keepdim=True)
    m_sqrt_du = 0.5 * m_du / m_sqrt
    m_sqrt_dv = 0.5 * m_dv / m_sqrt

    out = val / denom
    out_du = (du * denom - val * m_sqrt_du) / (denom * denom)
    out_dv = (dv * denom - val * m_sqrt_dv) / (denom * denom)
    return out, out_du, out_dv


@dataclass(frozen=True)
class EpochStats:
    mode: str
    model: str
    params: int
    epoch: int
    train_total_loss: float
    train_data_loss: float
    train_reg: float
    train_acc: float
    test_acc: float
    elapsed_s: float
    images_per_s: float
    lr: float


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _iter_limited(dl: Iterable[tuple[torch.Tensor, torch.Tensor]], max_batches: int | None):
    if max_batches is None:
        yield from dl
        return
    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        yield batch


@torch.no_grad()
def _accuracy(model: nn.Module, dl: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    model.train()
    return correct / max(total, 1)


def _train_one_epoch(
    *,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    train_dl: DataLoader,
    device: torch.device,
    mode: str,
    mu: torch.Tensor,
    lam: torch.Tensor,
    alpha: float,
    max_train_batches: int | None,
    clip_grad_norm: float | None,
) -> tuple[float, float, float, float, float, float]:
    model.train()
    seen = 0
    correct = 0
    total_loss_sum = 0.0
    data_loss_sum = 0.0
    reg_sum = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    t0 = time.perf_counter()

    for x, y in _iter_limited(train_dl, max_train_batches):
        x = x.to(device)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)

        if mode == "baseline":
            logits = model(x)
            data_loss = F.cross_entropy(logits, y)
            reg = torch.zeros((), device=device, dtype=data_loss.dtype)
            loss = data_loss
        elif mode in {"dual", "jvp"}:
            d = from_model(model, x, mu=mu, lam=lam, method=mode)
            logits = d.y
            data_loss = F.cross_entropy(logits, y)
            dl = delta_loss(
                data_loss=data_loss,
                d=d,
                alpha=alpha,
                reg_fn=lambda dd: energy_l2(dd.yu, dd.yv),
            )
            reg = dl.reg
            loss = dl.total
        else:
            raise ValueError(f"unknown mode: {mode}")

        loss.backward()
        if clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        opt.step()

        with torch.no_grad():
            batch = int(y.numel())
            seen += batch
            total_loss_sum += float(loss.detach().item()) * batch
            data_loss_sum += float(data_loss.detach().item()) * batch
            reg_sum += float(reg.detach().item()) * batch
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())

    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    train_total_loss = total_loss_sum / max(seen, 1)
    train_data_loss = data_loss_sum / max(seen, 1)
    train_reg = reg_sum / max(seen, 1)
    train_acc = correct / max(seen, 1)
    images_per_s = seen / max(elapsed, 1e-9)
    return train_total_loss, train_data_loss, train_reg, train_acc, elapsed, images_per_s


def _run(
    *,
    mode: str,
    dataset: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    optimizer: str,
    momentum: float,
    nesterov: bool,
    scheduler: str,
    step_size: int,
    gamma: float,
    min_lr: float,
    cosine_tmax: int,
    clip_grad_norm: float | None,
    device: torch.device,
    seed: int,
    mu: float,
    lam: float,
    alpha: float,
    geom_beta: float,
    geom_max_rotation_ratio: float,
    max_train_batches: int | None,
    max_test_batches: int | None,
    train_rotate: float,
    progress: bool,
) -> list[EpochStats]:
    import torchvision
    from torchvision import transforms

    _set_seed(seed)

    ds = dataset.lower()
    if ds == "mnist":
        ds_cls = torchvision.datasets.MNIST
    elif ds in {"fashionmnist", "fashion"}:
        ds_cls = torchvision.datasets.FashionMNIST
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    train_tfm = (
        transforms.Compose([transforms.RandomRotation(train_rotate), transforms.ToTensor()])
        if train_rotate > 0
        else transforms.Compose([transforms.ToTensor()])
    )
    test_tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = ds_cls(root=".data", train=True, download=True, transform=train_tfm)
    test_ds = ds_cls(root=".data", train=False, download=True, transform=test_tfm)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=0)

    model_label = f"{ds}:{model_name}"
    model = _make_model(model_name).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = _make_optimizer(
        model,
        optimizer=optimizer,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        geom_beta=geom_beta,
        geom_max_rotation_ratio=geom_max_rotation_ratio,
    )
    sched = _make_scheduler(
        opt,
        scheduler=scheduler,
        epochs=epochs,
        step_size=step_size,
        gamma=gamma,
        min_lr=min_lr,
        cosine_tmax=cosine_tmax,
    )

    mu_t = torch.tensor(mu, device=device)
    lam_t = torch.tensor(lam, device=device)

    mode_label = mode if geom_beta <= 0 else f"{mode}+geom"
    out: list[EpochStats] = []
    for ep in range(1, epochs + 1):
        train_total_loss, train_data_loss, train_reg, train_acc, elapsed, ips = _train_one_epoch(
            model=model,
            opt=opt,
            train_dl=train_dl,
            device=device,
            mode=mode,
            mu=mu_t,
            lam=lam_t,
            alpha=alpha,
            max_train_batches=max_train_batches,
            clip_grad_norm=clip_grad_norm,
        )

        test_acc = _accuracy(model, _limited_dataloader(test_dl, max_test_batches), device)
        lr_used = float(opt.param_groups[0]["lr"])
        if progress:
            print(
                " | ".join(
                    [
                        f"model={model_label}",
                        f"params={params}",
                        f"mode={mode_label}",
                        f"epoch={ep}/{epochs}",
                        f"lr={lr_used:.6g}",
                        f"train_data_loss={train_data_loss:.6g}",
                        f"train_reg={train_reg:.6g}",
                        f"train_acc={train_acc:.4f}",
                        f"test_acc={test_acc:.4f}",
                        f"img/s={ips:.1f}",
                        f"sec={elapsed:.2f}",
                    ]
                ),
                file=sys.stderr,
                flush=True,
            )
        out.append(
            EpochStats(
                mode=mode_label,
                model=model_label,
                params=params,
                epoch=ep,
                train_total_loss=train_total_loss,
                train_data_loss=train_data_loss,
                train_reg=train_reg,
                train_acc=train_acc,
                test_acc=test_acc,
                elapsed_s=elapsed,
                images_per_s=ips,
                lr=lr_used,
            )
        )
        if sched is not None:
            sched.step()
    return out


def _make_model(name: str) -> nn.Module:
    n = name.lower()
    if n in {"tiny", "tiny_mnist_cnn"}:
        return TinyMnistCnn()
    if n in {"mnist_cnn_700", "cnn700", "opt_aeg_v3"}:
        return MnistCnn700()
    raise ValueError(f"unknown model: {name}")


def _make_optimizer(
    model: nn.Module,
    *,
    optimizer: str,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    geom_beta: float,
    geom_max_rotation_ratio: float,
) -> torch.optim.Optimizer:
    opt = optimizer.lower()
    base: torch.optim.Optimizer
    if opt == "sgd":
        base = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    elif opt == "adamw":
        base = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"unknown optimizer: {optimizer}")

    if geom_beta > 0:
        return AEGOptimizer(base, beta=geom_beta, max_rotation_ratio=geom_max_rotation_ratio)
    return base


def _make_scheduler(
    opt: torch.optim.Optimizer,
    *,
    scheduler: str,
    epochs: int,
    step_size: int,
    gamma: float,
    min_lr: float,
    cosine_tmax: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    sch = scheduler.lower()
    if sch in {"", "none"}:
        return None
    if sch == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    if sch == "cosine":
        tmax = cosine_tmax if cosine_tmax > 0 else epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tmax, eta_min=min_lr)
    raise ValueError(f"unknown scheduler: {scheduler}")


def _limited_dataloader(dl: DataLoader, max_batches: int | None):
    if max_batches is None:
        return dl

    class _Wrapper:
        def __iter__(self):
            return _iter_limited(iter(dl), max_batches)

    return _Wrapper()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dataset", type=str, default="mnist")
    p.add_argument("--model", type=str, default="tiny")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--optimizer", type=str, default="sgd")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nesterov", action="store_true")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--scheduler", type=str, default="none")
    p.add_argument("--step-size", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--cosine-tmax", type=int, default=0)
    p.add_argument("--clip-grad-norm", type=float, default=0.0)
    p.add_argument("--train-rotate", type=float, default=0.0)
    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--geom-beta", type=float, default=0.0)
    p.add_argument("--geom-max-rotation-ratio", type=float, default=0.3)
    p.add_argument("--modes", type=str, default="baseline,dual")
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-test-batches", type=int, default=0)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    device = torch.device(args.device)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    max_train_batches = None if args.max_train_batches <= 0 else args.max_train_batches
    max_test_batches = None if args.max_test_batches <= 0 else args.max_test_batches
    clip_grad_norm = None if args.clip_grad_norm <= 0 else float(args.clip_grad_norm)

    all_stats: list[EpochStats] = []
    for mode in modes:
        all_stats.extend(
            _run(
                mode=mode,
                dataset=args.dataset,
                model_name=args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                optimizer=args.optimizer,
                momentum=args.momentum,
                nesterov=bool(args.nesterov),
                scheduler=args.scheduler,
                step_size=args.step_size,
                gamma=args.gamma,
                min_lr=args.min_lr,
                cosine_tmax=args.cosine_tmax,
                clip_grad_norm=clip_grad_norm,
                device=device,
                seed=args.seed,
                mu=args.mu,
                lam=args.lam,
                alpha=args.alpha,
                geom_beta=float(args.geom_beta),
                geom_max_rotation_ratio=float(args.geom_max_rotation_ratio),
                max_train_batches=max_train_batches,
                max_test_batches=max_test_batches,
                train_rotate=args.train_rotate,
                progress=bool(args.progress),
            )
        )

    payload = [asdict(s) for s in all_stats]
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
