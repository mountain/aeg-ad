# Phase 2（Delta-Loss 工具包）

Phase 2 的目标是在 Scenario A（把 δ 派生量注入 loss）的基础上，提供可复用的抽取接口、正则模板与示例，便于在不同模型/任务上快速试验与复现。

## 核心概念

对任意模型 `y = f(x)`，Phase 2 关注两条方向上的输出变化量：

- `y_u = J_x f(x) · u`
- `y_v = J_x f(x) · v`

并把它们写入训练目标：

`loss = data_loss(y, target) + alpha * reg(y, y_u, y_v)`

## API

`aegad.delta` 提供：

- `Delta(y, yu, yv)`：统一的三元输出结构
- `from_model(model, x, mu, lam, method=...)`：从模型与输入得到 `(y, yu, yv)`
  - `method="dual"`：优先调用 `model.forward_dual(x, mu, lam)`，否则尝试 `model(seed_a(...))`
  - `method="jvp"`：用 `torch.func.jvp` 计算 `y_u/y_v`（reference）
- `energy_l2 / balance_l2 / relative_energy_l2 / invariant_l2`：常用 δ 正则模板
- `delta_loss(data_loss, d, alpha, reg_fn)`：把 data loss 与 δ 正则组合成 `total/data/reg`

## 最小示例（MLP + MNIST）

运行 dual 版本（每个 epoch 会打印进度；stdout 输出 JSON）：

```bash
.venv/bin/python examples/phase2_mlp_mnist.py --method dual --epochs 10 --alpha 0.1
```

运行 jvp 参考版本：

```bash
.venv/bin/python examples/phase2_mlp_mnist.py --method jvp --epochs 10 --alpha 0.1
```

## 示例（Transformer Block）

该示例默认使用 `jvp` 抽取 `(y, yu, yv)`，用于展示模板与组合方式：

```bash
.venv/bin/python examples/phase2_transformer_block.py --device cpu
```
