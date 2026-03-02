# aeg-ad
AEG-based AD and neural network framework

## Phase 2（Delta-Loss）

Phase 2 提供 delta-loss 工具包：抽取 `(y, y_u, y_v)` 并用模板正则快速组合训练目标。

- 说明文档：[phase2.md](file:///Users/mingli/AEG/aeg-ad/docs/phase2.md)

### 快速开始

```bash
uv venv
uv pip install -e '.[dev]'
```

MLP + MNIST（dual vs jvp）：

```bash
.venv/bin/python examples/phase2_mlp_mnist.py --method dual --epochs 10 --alpha 0.1
.venv/bin/python examples/phase2_mlp_mnist.py --method jvp --epochs 10 --alpha 0.1
```

## Phase 3（PyTorch AD Ecosystem Profile）

- 说明文档：[phase3.md](file:///Users/mingli/AEG/aeg-ad/docs/phase3.md)

## Phase 4（几何化优化）

- 说明文档：[phase4.md](file:///Users/mingli/AEG/aeg-ad/docs/phase4.md)
