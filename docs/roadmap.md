## 0. Scenarios and Layering: Put “Different Tracks” on One Map First

### Scenario Definitions

* **Scenario A (recommended baseline):** Do not change the network definition source code. Write δ-derived quantities (e.g. `(y_u, y_v)`) into the loss (regularizers/constraints), and still update parameters via standard `loss.backward()`.
* **Scenario B (with dissipation):** Explicitly add geometry (metric/projection/Hamiltonian rotation + dissipation) into the update rule, forming a “geometrized optimizer step”.

  * **B1:** Dissipation still lives in a scalar loss (essentially a variant of A; the change is on the loss side).
  * **B2:** Dissipation lives in the update operator (the “real” B; the change is on the optimizer side).

### Layered Architecture (to ensure A and B2 do not conflict)

* **L1 Core (semantics layer):** `DualTensor` representation + seeding + lifted ops (δ forward semantics; the foundation for JVP/VJP)
* **L2 Delta (observation layer):** Extract `(y_u, y_v)` from a Dual forward pass, and build energy/constraint terms
* **L3 Optim (update layer):** Geometrized optimizer (consumes only `param.grad` and optional geometric statistics)

> Conclusion: **A and B2 share L1/L2; they differ only at L3.** Therefore the tracks do not cause bottom-layer inconsistencies; they only add modules above.

---

## 1) Phase 1: Reference-First (S1 pure PyTorch composition) — Lock Semantics

**Goal:** Get a correctness gold standard as fast as possible, and make any ordinary `nn.Module` usable without modifying its source code.

### Deliverables

1. **`aegad/core`**

   * `DualTensor(val, du, dv)`: three-channel SoA container + invariant checks (shape/device/dtype/stride/broadcast)
   * `seed_a(x, mu, lambda)`: inject `du = μ`, `dv = λ·x`; supports scalar `μ/λ` or broadcastable tensors
   * `lifted ops (compositional implementation)`: cover a minimal closed set first
     `add/sub/mul/div`, `neg`, `exp/log/tanh/sigmoid/relu`, `pow`, `matmul/linear`, `conv2d` (bias is added to `val` only)
2. **PyTorch “zero-change network” integration (choose one or run both)**

   * **Option A: `__torch_dispatch__` route (recommended)**
     Make `DualTensor` a custom tensor-like type, intercept PyTorch ops, and apply three-channel rules automatically. No changes to the network source.
   * **Option B: explicit functional API (fallback)**
     `aegad.dual(model)(x_dual)` or `aegad.functional.*`, for corners that dispatch cannot easily cover.
3. **`aegad/tests` (hard gate)**

   * Numerical parity: composite Dual vs finite differences (small tensors)
   * Backward parity: `gradcheck` (for `seed_a` + composite networks)
   * Key identity regression: `seed_a` VJP transpose rule (`bar_a += bar_val + λ*bar_dv`)

### Exit Criteria (must pass to enter the next phase)

* Any common network (MLP / small CNN) runs end-to-end without source changes:

  * `y = model(x)` (normal)
  * `y_dual = model(seed_a(x))` (dual)
  * `loss = data_loss(y) + α*R(y_dual.du, y_dual.dv)`
  * `loss.backward()` updates parameters normally

---

## 2) Phase 2: Delta-Loss Pack (productizing Scenario A) — Turn δ into Reusable Training Pieces

**Goal:** Make A a plug-and-play training paradigm: users only choose regularizer/constraint templates.

### Deliverables

1. **`aegad/delta`**

   * `extract(y_dual) -> (y, yu, yv)` plus required derived quantities
   * Regularizer template library (start with 3–5 expressive primitives)

     * `R_smooth = ||yu||^2 + ||yv||^2` (basic δ smoothing)
     * `R_balance = ||yu - c*yv||^2` (direction coupling)
     * `R_invariant = ||f(y, yu, yv)||` (hooks reserved for future Hamilton/contact-structure terms)
2. **`aegad/examples`**

   * A minimal example (MLP classification)
   * A “real” example (small UNet or a transformer block) showing how δ regularization affects generalization/stability

### Exit Criteria

* Scenario A is reproducible on 2–3 tasks (not aiming for best scores; aiming for “stable + interpretable + CI-runnable”)
* At least one example runs on CPU (CI-friendly)

---

## 3) Phase 3: PyTorch AD Ecosystem Profile (JVP/VJP/torch.func) — Make Interop Real

**Goal:** Beyond `backward()`, also:

* Use `torch.func.jvp / vjp` to transform Dual components
* Establish interface constraints early for future S2 fused ops

### Deliverables

1. **JVP/VJP compliance test matrix**

   * `torch.func.jvp(seed_a, ...)` vs a hand-written JVP reference
   * `torch.func.vjp(seed_a, ...)` vs a hand-written VJP reference
   * `vmap` (if you expect batched transforms later): at least “does not crash”, and document limitations if needed
2. **`forward_ad` interop (optional, but recommended to add an adapter layer)**

   * Provide a bridge `as_forward_ad_dual(val, tangent)` (one direction is enough)
3. **Concrete rules for in-place/view**

   * Add a CI “in-place denylist” or runtime checks: if an in-place op breaks three-channel consistency, raise a clear error

### Exit Criteria

* `gradcheck + func.jvp + func.vjp` pass on the core operator set
* Docs clearly state which `torch.func` transforms are supported vs not supported (avoid surprises when fused ops arrive)

---

## 4) Phase 4: Geometry Optim (Scenario B2: B with dissipation) — Put Geometry into the Optimizer, Not the Base

**Goal:** Implement a “dissipative geometric update” while avoiding second-order methods and preserving controllability.

### Recommended Minimal Viable B2 Form (engineering-friendly)

Given parameter gradients `g = ∇_θ L`, construct the update:

```math
\Delta\theta = -\eta\, M g \;+\; \beta\, J g
```

* `M \succeq 0`: symmetric positive (preconditioner/metric), ensures dissipative descent
* `J`: skew-symmetric (Hamiltonian rotation / approximate symplectic structure), provides a “move along level sets” component
* **Key:** start with the simplest block-diagonal / layer-wise scaling forms, and avoid Hessians/HVPs

### Deliverables

1. **`aegad/optim`**

   * `AEGOptimizer(base=AdamW, M=..., J=...)`: wrap an existing optimizer and insert the transforms
   * Default `M`: scalar/vector scaling per parameter-group (learnable or configurable)
   * Default `J`: simple 2×2 block rotations within a group (very conservative)
2. **Stability guardrails**

   * Automatic annealing/clamping for `β` (prevent rotation dominating dissipation)
   * Gradient-norm / update-norm monitoring (logging + early abort)
3. **Comparison experiment scaffolding**

   * A vs A+B2 (same task, same seed): observe convergence speed/oscillation/generalization

### Exit Criteria

* On at least one task, B2 is “not obviously worse and remains controllable”, and your monitoring strategy supports fast diagnosis

---

## 5) Phase 5: Performance Layer (S2: C++/CUDA fused) — Sink Only the Hot Spots

**Goal:** Speed up without changing semantics; fused ops are “implementation replacement”, not “semantic replacement”.

### Strategy

* Profile first: optimize only the top 3–5 hot spots
* Each fused op must provide:

  * **forward**
  * **backward (VJP)**
  * **jvp (strongly recommended):** otherwise you break `torch.func` / forward-mode interop

### Deliverables

1. `aegad_kernels` extension package (C++/CUDA, or CPU SIMD + CUDA)
2. Coverage list (by priority)

   * `seed_a`
   * elementwise (fuse common chained patterns)
   * `linear/matmul` (only if truly hot)
   * `conv2d` (CNN workloads)
3. Consistency tests

   * fused vs composite: match within error tolerances
   * backward/jvp parity checks (required)

### Exit Criteria

* With `USE_FUSED=1`, performance gains are clear (e.g. wall-clock / step meaningfully improves)
* With fused off, the composite fallback still works normally (maintainability baseline)

---

## 6) Phase 6: Release Engineering (enable long-term iteration)

### Deliverables

* `README`: the three paths are obvious at a glance

  * A: δ-loss training
  * B2: geometry optimizer
  * fused: performance switches and limitations
* `COMPAT.md`: PyTorch version matrix, CPU/CUDA/MPS support, `torch.func` support surface
* CI: at minimum includes

  * CPU quick tests (every PR)
  * GPU nightly (if using CUDA)
  * correctness battery (`gradcheck` / `jvp` / `vjp` / fused parity)

---

## Suggested Sequencing (for staffing/planning)

1. **P1 (lock semantics):** get S1 running end-to-end (the project’s source of truth)
2. **P2 (productize A):** turn A into reusable training pieces (immediately usable)
3. **P3 (ecosystem compliance):** fill the `jvp/vjp/func` holes early
4. **P4 (B2 plug-in):** implement geometric updates as an optimizer plug-in; do not touch the core
5. **P5 (fuse hot spots):** only then drop into C++/CUDA, backed by parity tests
