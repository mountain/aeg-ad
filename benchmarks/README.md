# Benchmarks

## MNIST 训练：精度与速度对比

对比同一网络在不同模式下的训练精度与训练吞吐：

- `baseline`：标准 MNIST 训练（只做普通 forward）
- `dual`：Scenario A（roadmap Phase 1）：`loss = data_loss + α * (||y_u||^2 + ||y_v||^2)`，其中 `(y, y_u, y_v)` 由 compositional dual forward 计算
- `jvp`：用 `torch.func.jvp` 计算同样的 `(y_u, y_v)` 作为参考路径（通常更慢，但可做 gold baseline）

脚本输出每个 epoch 的：

- `train_total_loss/train_data_loss/train_reg/train_acc/test_acc`
- `elapsed_s/images_per_s`

此外会默认每个 epoch 打印一行进度到 stderr，方便观察训练过程；如需安静模式可传 `--no-progress`（stdout 仍会输出最终 JSON）。

## 运行

安装依赖（已支持 uv + 清华镜像）：

```bash
uv venv
uv pip install -e '.[dev]'
```

CPU 上跑一个小规模 smoke：

```bash
.venv/bin/python benchmarks/mnist_train.py --epochs 1 --modes baseline,dual --max-train-batches 10 --max-test-batches 2
```

切换到 700 参数模型（OptAEGV3 + RMSNorm + 3x(Conv+Pool)）：

```bash
.venv/bin/python benchmarks/mnist_train.py --model opt_aeg_v3 --epochs 53 --batch-size 512 --optimizer adamw --lr 0.001 --weight-decay 0.2 --scheduler cosine --cosine-tmax 53 --min-lr 0.00001 --train-rotate 10 --modes baseline,dual --alpha 0.1
```

完整训练（示例）：

```bash
.venv/bin/python benchmarks/mnist_train.py --device cpu --epochs 5 --batch-size 128 --lr 0.1 --modes baseline,dual
```

包含 jvp 参考（更慢）：

```bash
.venv/bin/python benchmarks/mnist_train.py --device cpu --epochs 3 --modes baseline,dual,jvp
```

带 LR 衰减与权重衰减（更容易到 99%+）：

```bash
.venv/bin/python benchmarks/mnist_train.py --device cpu --epochs 20 --batch-size 128 --lr 0.1 --weight-decay 0.0005 --scheduler step --step-size 10 --gamma 0.1 --modes baseline,dual --alpha 0.1
```

AdamW + cosine（常更稳，但更慢）：

```bash
.venv/bin/python benchmarks/mnist_train.py --device cpu --epochs 20 --optimizer adamw --lr 0.001 --weight-decay 0.01 --scheduler cosine --min-lr 0.00001 --modes baseline,dual --alpha 0.1
```

写出 JSON 结果：

```bash
.venv/bin/python benchmarks/mnist_train.py --epochs 5 --modes baseline,dual --out benchmarks/out_mnist.json
```
