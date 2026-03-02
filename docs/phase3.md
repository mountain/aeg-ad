# Phase 3（PyTorch AD Ecosystem Profile）

Phase 3 的目标是把 “δ 训练（Scenario A）” 放到 PyTorch AD 生态中验证：`torch.func.jvp/vjp/vmap` 与 `forward_ad` 的互操作边界明确、可复现、可回归。

## 核心结论（当前实现）

- `DualTensor` 作为三通道载体，通道数据存储在属性（`val/du/dv`）中；因此对 `torch.func` 来说它只是一条 “primal Tensor”，不会把 `du/dv` 当作 pytree 输出参与变换。
- 对 `torch.func` 的互操作入口统一为 pytree 形式：`seed_a_components(x, mu, lam) -> (val, du, dv)`。

## 支持面

当前保证支持并在测试中覆盖：

- `torch.func.jvp(seed_a_components, ...)`
- `torch.func.vjp(seed_a_components, ...)`
- `torch.func.vmap(seed_a_components, ...)`（smoke）

同时提供（单向）`forward_ad` 适配器：

- `as_forward_ad_dual(val, tangent)`：返回 `torch.autograd.forward_ad.make_dual(val, tangent)`，需在 `torch.autograd.forward_ad.dual_level()` 作用域内使用。

## 不在 Phase 3 承诺范围内

- 直接对 `seed_a`（返回 `DualTensor`）做 `torch.func.jvp/vjp` 并得到三通道 pytree 语义。
- 直接对包含 `DualTensor` 的完整模型做 `torch.func.vmap` 语义保证。
