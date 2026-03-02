# Phase 4（几何化优化 / B2）

Phase 4 的目标是在不引入二阶方法的前提下，把几何结构放进优化器更新里，形成可控的 “耗散 + 旋转” 更新：

`Δθ = -η M g + β J g`

其中 `M ⪰ 0`（耗散项），`J` 反对称（旋转项）。

## 最小实现（本仓库）

- `aegad.optim.AEGOptimizer(base=..., beta=..., max_rotation_ratio=...)`
  - 先执行 `base.step()`（耗散项）
  - 再执行按 2×2 block 的简单旋转更新 `lr * beta * Jg`
  - `beta` 自动 clamp：旋转更新的范数不超过 `max_rotation_ratio * (lr * ||g||)` 量级

## 试跑（MNIST tiny）

对照：baseline（无几何项）

```bash
.venv/bin/python benchmarks/mnist_train.py --model tiny --modes baseline --epochs 20 --out benchmarks/out_tiny_baseline_e20.json
```

几何化：baseline+geom（同 seed）

```bash
.venv/bin/python benchmarks/mnist_train.py --model tiny --modes baseline --epochs 20 --geom-beta 0.05 --geom-max-rotation-ratio 0.1 --out benchmarks/out_tiny_baseline_geom_e20.json
```
