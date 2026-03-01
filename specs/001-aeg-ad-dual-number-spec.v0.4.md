# AEG-AD Dual Number (Dual Carrier) — Specification
**Spec ID:** 001  
**Version:** v0.4 (adds Appendix D: PyTorch reference skeleton; non-normative code templates)
**Status:** Draft  
**Last updated:** 2026-03-01  
**Scope:** First-order automatic differentiation (forward/JVP and reverse/VJP) for AEG δ-differentials, expressed as implementable engineering primitives and rule tables.  
**Goal:** Make AEG’s δ-differential behave like a dual-number-style carrier that is implementable, testable, and embeddable into neural-network operator libraries, consistent with the contact-structure formulation in the paper.

---

## Conformance Language
The key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are to be interpreted as described in RFC 2119.

---

## 0. Overview / Design Thesis
In AEG, the “differential” of a scalar field \(F\) is a horizontal differential along two directions:

\[
\delta F = (D_u F)\,du + (D_v F)\,dv,
\]

where the horizontal derivatives are

\[
D_u = \partial_u + \mu\,\partial_a,\qquad
D_v = \partial_v + \lambda a\,\partial_a,
\]

and the core flow (a 1-form) is

\[
\delta a = \omega = \mu\,du + \lambda a\,dv.
\]

**Reference identity (contact form):** In the AEG contact-structure formulation, one defines
\(\omega := \mu\,du + \lambda a\,dv\) and \(\alpha := da - \omega\). For any scalar field \(F(u,v,a)\),
the expression differential satisfies:
\[
\delta F = dF - (\partial_a F)\,\alpha
\]
and expands to:
\[
\delta F = (F_u + \mu F_a)\,du + (F_v + \lambda a F_a)\,dv.
\]
This provides an implementation-independent cross-check for the meaning of the `du/dv` channels.



Therefore, the natural first-order AD carrier is a value bundled with two directional derivatives:

\[
\boxed{\ \langle x;\ x_u,\ x_v\rangle\ }\quad \text{with}\quad x_u := D_u x,\;\; x_v := D_v x.
\]

This specification defines:
- The carrier types at scalar- and tensor-level.
- Forward-mode (JVP) seeding primitives and propagation rules.
- Reverse-mode (VJP) rules, emphasizing the AEG-specific VJP of the seeding map \(a \mapsto \langle a;\mu,\lambda a\rangle\).

---

## 1. Notation
- \((u,v,a)\): local AEG coordinates (contact-structure chart).
- \(\mu,\lambda\): context parameters (configuration/runtime injected).
- AEG Dual carrier: \(\mathrm{Dual}(x)=\langle x;\ x_u,x_v\rangle\).
- δ-reading: \(\delta x = x_u\,du + x_v\,dv\).

---

## 2. Data Model

### 2.1 Scalar-level `Dual<T>` (AoS semantics)
The scalar carrier is:

\[
\mathrm{Dual}(x) := \langle x;\ x_u,\ x_v\rangle.
\]

An implementation MUST store three scalars:

```text
struct Dual<T> {
  T val;  // x
  T du;   // D_u x
  T dv;   // D_v x
}
```

### 2.2 Tensor-level `DualTensor<T>` (recommended SoA storage)
For neural-network operator libraries, implementations SHOULD store `DualTensor<T>` as a struct-of-arrays (SoA):

```text
struct DualTensor<T> {
  Tensor<T> val;  // shape S
  Tensor<T> du;   // shape S
  Tensor<T> dv;   // shape S
}
```

Benefits:
- Large operators (matmul/conv) can reuse float kernels by running the same kernel independently on `val/du/dv`.
- Better SIMD/parallel behavior than AoS for many kernels.
- Reverse-mode adjoints naturally use the same three-channel container.

### 2.3 Invariants and mixed-mode lifting (required)
For tensor-level carriers, implementations MUST enforce the following invariants:
- `val`, `du`, `dv` MUST have identical shape.
- they MUST live on the same device/backend.
- they MUST have consistent dtype and layout/stride semantics.

When mixing a base scalar/tensor `c` with a dual carrier `X`, the base value MUST be lifted as:
\[
\operatorname{lift}(c)=\langle c;\ 0,\ 0\rangle.
\]
All mixed-mode operator overloads MUST behave as if this lifting was performed explicitly.


---

## 3. AEG Seed Primitives (JVP primitives)
Seeding defines how independent variables inject δ-structure into forward-mode AD.

### 3.1 Constant seed
\[
\mathrm{const}(c) = \langle c;\ 0,\ 0\rangle.
\]

### 3.2 Coordinate seeds
\[
\mathrm{seed}_u(u)=\langle u;\ 1,\ 0\rangle,\qquad
\mathrm{seed}_v(v)=\langle v;\ 0,\ 1\rangle.
\]

### 3.3 Flow seed (AEG-specific)
\[
\boxed{\ \mathrm{seed}_a(a)=\langle a;\ \mu,\ \lambda a\rangle\ }.
\]

This injects the AEG flow \(\delta a=\mu\,du+\lambda a\,dv\) into forward-mode AD.

### 3.4 Tensor-level seeding and broadcasting (required)
Implementations MUST provide tensor-level versions of `const`, `seed_u`, `seed_v`, and `seed_a` (see Appendix B.5).

For `seed_a(X, mu, lambda)`, `mu` and `lambda` MAY be:
- Python scalars (treated as constants), or
- tensors broadcastable to the shape of `X`.

Broadcasting MUST be applied consistently across `val/du/dv`, i.e.:
- `du = broadcast(mu)`
- `dv = broadcast(lambda) * X` (elementwise)

### 3.5 Optional: treating \(\mu,\lambda\) as differentiable parameters
If `mu` and/or `lambda` are treated as differentiable inputs (e.g. trainable parameters), implementations MUST define their VJP contributions for `seed_a` as specified in Section 8.3 (optional extension).


---

## 4. Forward-mode Rules (JVP propagation; scalar semantics)
Let
\[
X=\langle x;\ x_u,\ x_v\rangle,\quad Y=\langle y;\ y_u,\ y_v\rangle.
\]

### 4.1 Addition / subtraction
\[
X\pm Y=\langle x\pm y;\ x_u\pm y_u,\ x_v\pm y_v\rangle.
\]

### 4.2 Multiplication (Leibniz)
\[
X\cdot Y=\langle xy;\ x_u y + x y_u,\ x_v y + x y_v\rangle.
\]

### 4.3 Division (\(y\neq 0\))
\[
\frac{X}{Y}
=
\left\langle
\frac{x}{y};
\ \frac{x_u y-x y_u}{y^2},
\ \frac{x_v y-x y_v}{y^2}
\right\rangle.
\]

### 4.4 Unary function (chain rule)
For differentiable \(\Phi:\mathbb R\to\mathbb R\):

\[
\Phi(X)=\langle \Phi(x);\ \Phi'(x)\,x_u,\ \Phi'(x)\,x_v\rangle.
\]

---

## 5. Elementary Function Table (ready to implement)
All formulas are direct applications of the chain rule above.

### 5.1 Integer power
\[
X^n=\langle x^n;\ n x^{n-1}x_u,\ n x^{n-1}x_v\rangle.
\]

### 5.2 exp / log
- exp:
  \[
  e^X=\langle e^x;\ e^x x_u,\ e^x x_v\rangle.
  \]
- log (domain \(x>0\)):
  \[
  \ln X=\langle \ln x;\ x_u/x,\ x_v/x\rangle.
  \]

### 5.3 Trigonometric
- sin:
  \[
  \sin X=\langle \sin x;\ \cos x\,x_u,\ \cos x\,x_v\rangle.
  \]
- cos:
  \[
  \cos X=\langle \cos x;\ -\sin x\,x_u,\ -\sin x\,x_v\rangle.
  \]

### 5.4 Hyperbolic (optional)
- sinh:
  \[
  \sinh X=\langle \sinh x;\ \cosh x\,x_u,\ \cosh x\,x_v\rangle.
  \]
- cosh:
  \[
  \cosh X=\langle \cosh x;\ \sinh x\,x_u,\ \sinh x\,x_v\rangle.
  \]
- arcsinh:
  \[
  \operatorname{arcsinh}(X)
  =
  \left\langle
  \operatorname{arcsinh}(x);
  \ \frac{x_u}{\sqrt{1+x^2}},
  \ \frac{x_v}{\sqrt{1+x^2}}
  \right\rangle.
  \]

---

## 6. Linear Operators (NN kernels) as Dual-friendly primitives
For linear operators \(L\) (linear in the input), Dual propagation is “same kernel, three channels”.

### 6.1 Linear / matmul
If
\[
Y = W X + b,
\]
then implementations MUST compute:

\[
\begin{aligned}
Y_{\mathrm{val}} &= W X_{\mathrm{val}} + b,\\
Y_{u} &= W X_{u},\\
Y_{v} &= W X_{v}.
\end{aligned}
\]

### 6.2 Convolution
If
\[
Y = \mathrm{conv}(W, X) + b,
\]
then implementations MUST compute:

\[
\begin{aligned}
Y_{\mathrm{val}} &= \mathrm{conv}(W, X_{\mathrm{val}}) + b,\\
Y_{u} &= \mathrm{conv}(W, X_{u}),\\
Y_{v} &= \mathrm{conv}(W, X_{v}).
\end{aligned}
\]

---

## 7. Reverse-mode (VJP) in AEG: conventions

### 7.1 VJP definition
For a graph node \(Z = f(X,Y,\dots)\), VJP takes upstream cotangent \(\bar Z\) and accumulates input cotangents \(\bar X,\bar Y,\dots\).

For Dual nodes, the cotangent container MUST have three channels:

\[
\bar X = \langle \bar x,\ \bar x_u,\ \bar x_v\rangle,
\]

corresponding to adjoints of the `val/du/dv` channels.

### 7.2 Key claim (engineering guidance)
Except for the AEG seeding primitives, Dual VJPs do not require AEG-specific math: the three channels form ordinary computation graphs composed of standard arithmetic and unary ops. Implementations SHOULD reuse standard VJP rules channel-wise.

Therefore, the main AEG specialization in reverse mode is the transpose/VJP of the AEG seed map.

---

## 8. AEG VJP Primitives (crucial)

### 8.1 VJP for `const`
Forward:
\[
c \mapsto \langle c;\ 0,\ 0\rangle.
\]
Backward: no update is required unless `c` is treated as a differentiable parameter; in that case, the implementation MAY apply \(\bar c {+}{=} \bar x\).

### 8.2 VJP for `seed_u` / `seed_v`
Forward:
\[
u\mapsto \langle u;\ 1,\ 0\rangle,\qquad v\mapsto \langle v;\ 0,\ 1\rangle.
\]

Backward:
\[
\boxed{\ \bar u {+}{=} \bar x\ },\qquad \boxed{\ \bar v {+}{=} \bar x\ }.
\]

The constant derivative channels do not backpropagate into the input.

### 8.3 VJP for `seed_a` (AEG-specific primitive)
Forward:
\[
a \mapsto \langle a;\ \mu,\ \lambda a\rangle.
\]

Backward (VJP of the seeding map):
\[
\boxed{\ \bar a {+}{=} \bar x\ +\ \lambda\,\bar x_v\ }.
\]

Explanation:
- `val = a` contributes \(\bar x\) to \(\bar a\).
- `dv = \lambda a` contributes \(\lambda \bar x_v\) to \(\bar a\).
- `du = \mu` is constant w.r.t. \(a\), so it contributes nothing.

Optional (if `mu` and/or `lambda` are differentiable parameters):
- since `du = mu`, the transpose contribution is: \(\boxed{\ \bar\mu {+}{=} \bar x_u\ }\) (with a reduction consistent with broadcasting for tensors).
- since `dv = lambda * a`, the transpose contribution is: \(\boxed{\ \bar\lambda {+}{=} \bar x_v \cdot a\ }\) (again broadcasting-aware; for tensor `lambda` this is elementwise, otherwise reduce-sum).

These optional rules allow learning \(\mu,\lambda\) inside a PyTorch training loop when desired.



---

## 9. Standard VJP Rules (apply to all non-seed ops)
These are standard reverse-mode rules. For Dual graphs, implementations MUST apply them independently to each channel (`val`, `du`, `dv`) unless otherwise specified.

### 9.1 Addition
Forward: \(z=x+y\)  
Backward:
\[
\bar x {+}{=} \bar z,\qquad \bar y {+}{=} \bar z.
\]

### 9.2 Multiplication
Forward: \(z=x\cdot y\)  
Backward:
\[
\bar x {+}{=} \bar z\cdot y,\qquad \bar y {+}{=} \bar z\cdot x.
\]

### 9.3 Division
Forward: \(z=x/y\)  
Backward:
\[
\bar x {+}{=} \bar z\cdot \frac{1}{y},\qquad
\bar y {+}{=} \bar z\cdot \left(-\frac{x}{y^2}\right).
\]

### 9.4 Unary function
Forward: \(y=\Phi(x)\)  
Backward:
\[
\bar x {+}{=} \bar y \cdot \Phi'(x).
\]

---

## 10. Recommended Usage in Neural-Network Training
Dual/JVP is directional-derivative evaluation and SHOULD NOT replace standard backpropagation for training. A typical pattern is:
1. Run standard forward for the data loss term.
2. Run Dual forward to obtain \((y_u,y_v)\) for AEG geometric regularizers/constraints.
3. Backpropagate through the total loss using reverse mode.

Example structure:

```text
y      = model(x)                       // standard forward
y_dual = model(seed_a(x, mu, lambda))   // dual forward => (y, y_u, y_v)
loss   = data_loss(y) + alpha * reg(y_dual.du, y_dual.dv, y)
grad   = backprop(loss)
```

---

## 11. Worked Examples (sanity)

### 11.1 \(F(a)=a^2\)
Input:
\[
A=\langle a;\ \mu,\ \lambda a\rangle.
\]
Compute:
\[
F=A\cdot A=\langle a^2;\ 2a\mu,\ 2\lambda a^2\rangle.
\]
Thus:
\[
\delta F = (2a\mu)\,du + (2\lambda a^2)\,dv.
\]

### 11.2 \(G(u,v,a)=u+\ln a\)
\[
U=\langle u;\ 1,\ 0\rangle,\quad A=\langle a;\ \mu,\ \lambda a\rangle.
\]
\[
\ln A=\langle \ln a;\ \mu/a,\ \lambda\rangle.
\]
\[
G=U+\ln A=\langle u+\ln a;\ 1+\mu/a,\ \lambda\rangle.
\]

---

## 12. Consistency Checks (test requirements)
Implementations MUST include unit tests covering:

### 12.1 Algebraic identities
- `const(c)` has zero `du/dv`.
- Add/mul/div rules satisfy algebraic identities within floating error.

### 12.2 Seed correctness (AEG-specific)
For arbitrary \(a\):
- `seed_a(a).du == mu`
- `seed_a(a).dv == lambda * a`

### 12.3 Analytic cross-check (optional but recommended)
For known \(E(u,v,a)\), compare AD output with:
\[
x_u = E_u + \mu E_a,\qquad x_v = E_v + \lambda a E_a.
\]

### 12.4 VJP seed transpose test (AEG-specific)
Given upstream cotangent \(\bar X=\langle \bar x,\bar x_u,\bar x_v\rangle\) at the output of `seed_a(a)`, verify:
\[
\Delta \bar a = \bar x + \lambda \bar x_v.
\]

---

## 13. Non-goals / Out of Scope
- Full coordinate-covariant treatment of δ under arbitrary chart changes.
- Second-order \(\delta^2\) and higher-order AD (may be added later via hyper-dual style).
- Complexification / tube extension unless specified separately.

---

## 14. Roadmap (non-normative)
- v0.3: optional second-order seed and minimal hyper-dual extension.
- v0.4: operator-level JVP/VJP primitives for softmax/layernorm/attention with numerically stable formulas.
- v1.0: lock ABI/API for Python bindings (e.g., buffer protocol / DLPack) and publish a reference implementation.

---

## Appendix A — `Scalar` Contract (Minimum Engineering Interface)
Goal: define a minimal generic interface so the same tensor/kernels can run on:
- base scalars: `float32/float16` (and optionally `bf16/f64`)
- dual scalars: `Dual<T>`

### A.1 Taxonomy
- Base scalar: `f32`, `f16` (optionally `bf16`, `f64`)
- Dual scalar: `Dual<T>` where `T` is a base scalar
- Tensor scalar: the scalar type parameter of `Tensor<T>`

### A.2 Required operations by kernel class
To avoid over-constraining, the contract is layered by operator family.

#### A.2.1 Core arithmetic (required everywhere)
Any `Scalar` MUST support:
- `zero() -> T`
- `one() -> T`
- `add(a,b) -> T`
- `sub(a,b) -> T`
- `mul(a,b) -> T`
- `neg(a) -> T` (derivable from `sub(zero,a)` but often useful)
- `fma(a,b,c) -> T` (optional but strongly recommended; if missing, emulate)

Comparisons are NOT required for differentiable math, but MAY be needed for piecewise operators:
- `gt(a, 0)` / `max(a,b)` (optional; see A.4)

#### A.2.2 Transcendentals (required if you expose these ops)
For unary elementwise ops, a scalar SHOULD support the corresponding functions:
- `exp(x)`, `log(x)`, `sin(x)`, `cos(x)`
- `sqrt(x)` (needed for norms/layernorm; also useful for `rsqrt`)

For numerical stability, implementations often also want:
- `abs(x)` (robust tests, norms, clipping)
- `tanh(x)` / `sigmoid(x)` (if provided)

#### A.2.3 Reductions and normalization (required if you expose these ops)
If the library provides `reduce_sum`, `mean`, `variance`, `layernorm`, etc., the scalar MUST support:
- `add` and `mul`
- `div(a,b)` (or `mul(a, inv(b))`)
- `sqrt` (or `rsqrt`)

#### A.2.4 Dual lifting (internal requirement)
`Dual<T>` MUST be a `Scalar` whenever `T` is a `Scalar`. Implementations MUST lift:
- `Dual.zero() = <0;0,0>`
- `Dual.one()  = <1;0,0>`
- binary ops via the scalar rules in Sections 4–5
- unary ops via the chain rule in Section 4.4

### A.3 Minimal trait / concept sketch (language-agnostic)
```text
trait Scalar {
  fn zero() -> Self;
  fn one()  -> Self;

  fn add(a: Self, b: Self) -> Self;
  fn sub(a: Self, b: Self) -> Self;
  fn mul(a: Self, b: Self) -> Self;
  fn neg(a: Self) -> Self;

  // recommended
  fn fma(a: Self, b: Self, c: Self) -> Self;

  // optional families (gate by feature flags)
  fn div(a: Self, b: Self) -> Self;
  fn exp(x: Self) -> Self;
  fn log(x: Self) -> Self;
  fn sqrt(x: Self) -> Self;
  fn sin(x: Self) -> Self;
  fn cos(x: Self) -> Self;

  // optional for piecewise ops
  fn max(a: Self, b: Self) -> Self;
  fn cmp_gt_zero(x: Self) -> bool;  // only if scalar is ordered
}
```

### A.4 Piecewise nonlinearities policy
For `relu/max/leaky_relu`, implementations MUST choose a policy:
- Policy P1 (recommended for NN): allow piecewise ops only for ordered base scalars; lift Dual by branching on `val` and masking `du/dv`.
- Policy P2 (strict): disallow piecewise ops for Dual unless a subgradient convention is explicitly defined.

### A.5 Accumulation dtype policy
For reductions and matmul/conv accumulators:
- Base `f16/bf16` SHOULD accumulate in `f32`.
- Dual `Dual<f16>` SHOULD accumulate `val/du/dv` each in `f32`.

An implementation MAY encode this via an associated accumulation type `AccT`.

---

## Appendix B — `DualTensor` Conventions (SoA), Signatures, and VJP Seed Transpose

### B.1 SoA invariant
`DualTensor<T>` MUST maintain identical shape/dtype/layout across `val/du/dv`:
- same shape
- same dtype `T`
- same layout tag (row-major, channels-last, etc.)
- same stride semantics (or explicit contiguous marker)

### B.2 Layout tags (recommended minimal set)
Each `Tensor<T>` includes `shape`, `strides`, and a `layout_tag`, such as:
- `contiguous_row_major`
- `contiguous_col_major` (optional)
- `channels_last` (optional; for conv)
- `strided` (general case)

Implementations SHOULD start with `contiguous_row_major` + `strided` and add `channels_last` only when conv performance becomes a priority.

### B.3 Broadcasting semantics
Elementwise broadcasting MUST follow NumPy rules and be applied identically to `val/du/dv`.

### B.4 Core operator signatures (primal)
Elementwise unary:

```text
DualTensor<T> unary(phi, dphi, X: DualTensor<T>)
```

Semantics:
- `Y.val = phi(X.val)`
- `Y.du  = dphi(X.val) * X.du`
- `Y.dv  = dphi(X.val) * X.dv`

Elementwise binary:

```text
DualTensor<T> add(X, Y)
DualTensor<T> sub(X, Y)
DualTensor<T> mul(X, Y)
DualTensor<T> div(X, Y)  // if supported
```

Reductions:

```text
DualTensor<T> reduce_sum(X, axes, keepdim)
DualTensor<T> mean(X, axes, keepdim)
```

Semantics:
- `val/du/dv` MUST each be reduced in the same way.
- Accumulation SHOULD follow Appendix A.5 (`AccT`).

Matmul / linear:

```text
DualTensor<T> matmul(A: Tensor<T>, X: DualTensor<T>)                 // Y = A @ X
DualTensor<T> linear(W: Tensor<T>, b: Tensor<T>, X: DualTensor<T>)   // Y = W @ X + b
```

Bias MUST be added only to `val` by default.

Conv2d:

```text
DualTensor<T> conv2d(W: Tensor<T>, b: Tensor<T>?, X: DualTensor<T>, params)
```

### B.5 Tensor-level seeding ops
Implementations MUST provide:

```text
DualTensor<T> seed_a(X: Tensor<T>, mu: T, lambda: T)
DualTensor<T> seed_u(X: Tensor<T>)
DualTensor<T> seed_v(X: Tensor<T>)
DualTensor<T> const_tensor(X: Tensor<T>)
```

With semantics:
- `const_tensor`: `val=X`, `du=0`, `dv=0`
- `seed_u`: `val=X`, `du=1`, `dv=0`
- `seed_v`: `val=X`, `du=0`, `dv=1`
- `seed_a`: `val=X`, `du=mu`, `dv=lambda*X` (elementwise)

### B.6 Cotangent container for VJP
Adjoints SHOULD use the same SoA structure:

```text
DualTensor<T> BarX { val: Tensor<T>, du: Tensor<T>, dv: Tensor<T> }
```

This is a cotangent container, not a primal DualTensor.

### B.7 AEG-specific tensor VJP primitive: `seed_a` transpose
Forward:
- `X -> (val=X, du=mu, dv=lambda*X)`

Given upstream adjoint `BarY = <bar_val, bar_du, bar_dv>`, the VJP MUST update `BarX` as:

```text
BarX.val += BarY.val + lambda * BarY.dv
```

There is no contribution from `BarY.du` because `du=mu` is constant w.r.t. `X`.

### B.8 Python interop (non-normative guidance)
If Python is the host language, `Tensor<T>` SHOULD support at least one of:
- buffer protocol (NumPy view)
- DLPack capsule (interop with PyTorch/JAX)

`DualTensor<T>` MAY be exposed as an object with `.val/.du/.dv` fields or as a tuple `(val, du, dv)`. Exposing `.val/.du/.dv` is recommended.

### B.9 Minimal operator set for a first implementation
To validate the end-to-end stack (AEG seed + NN kernels), the minimal set is:
- elementwise: `add`, `mul`, `exp`, `log`, `sqrt`, `relu`
- reduce: `sum`, `mean`
- linear: `matmul` (or `linear`)
- plus: `seed_a`, `const_tensor`


---

## Appendix C — PyTorch Integration Profile (normative)

This appendix specifies how a conforming AEG-AD Dual library SHOULD embed into the PyTorch ecosystem, especially with respect to **VJP/backward** and **JVP/forward-mode** mechanisms.

### C.1 Representation in Python (required)
A PyTorch-facing implementation MUST represent a tensor-level dual carrier as either:
- an object with `.val/.du/.dv` fields that are `torch.Tensor`, or
- a plain tuple `(val, du, dv)` of `torch.Tensor`.

In both cases, the SoA invariants from Appendix B.1 MUST hold:
- same shape, dtype, device, and layout/stride semantics across the three channels.

### C.2 Autograd strategy (two interoperable options)

#### C.2.1 Option S1 — “Pure PyTorch composition” (recommended baseline)
Implement each Dual operator rule using only standard PyTorch tensor operations:
- `val` computed with ordinary ops
- `du/dv` computed with the JVP rules (Sections 4–6)

In this option:
- **Reverse-mode (VJP)** through the Dual computation graph is automatically provided by PyTorch autograd.
- The AEG-specific VJP behavior for `seed_a` arises automatically if `seed_a` is expressed as:
  - `val = a`
  - `du  = mu * ones_like(a)` (or broadcast)
  - `dv  = lambda * a`

This baseline should be used as the reference implementation for correctness tests.

#### C.2.2 Option S2 — “Fused custom ops” (for performance)
If an operator is implemented as a fused C++/CUDA kernel exposed to PyTorch (e.g. via `torch.autograd.Function` or `torch.library` custom ops), then the implementation MUST provide:

- a correct **backward/VJP** rule matching Sections 8–9 (and Appendix B.7 for `seed_a`).
- a correct **forward-mode/JVP** rule matching Sections 4–6.

If the fused op is used inside `torch.func` transforms (`torch.func.jvp`, `torch.func.vjp`, etc.), the implementation MUST follow PyTorch’s transform-compatibility constraints:
- prefer a functional `forward()` + `setup_context()` design (no `ctx` argument in `forward`)
- provide `jvp()` for forward-mode transforms
- provide `backward()` for reverse-mode transforms
- provide `vmap()` (or request generated rules) if you need `torch.vmap` compatibility.

### C.3 Required primitive: `seed_a` (AEG specialization point)
A PyTorch integration MUST expose `seed_a` as a callable (Python) function.

If `seed_a` is implemented as a fused kernel, its rules MUST be:

**Forward:**
- `val = a`
- `du  = mu`
- `dv  = lambda * a`

**Backward (VJP):**
- `bar_a += bar_val + lambda * bar_dv`  (Section 8.3 / Appendix B.7)
- optional: if `mu, lambda` are differentiable, accumulate `bar_mu` and `bar_lambda` as specified in Section 8.3.

**Forward-mode (JVP):**
Given tangents `(ȧ, μ̇, λ̇)` (usually `μ̇=λ̇=0`), the JVP outputs MUST be:
- `val̇ = ȧ`
- `du̇  = μ̇`
- `dv̇  = λ̇ * a + lambda * ȧ`

### C.4 Interop with `torch.func.jvp/vjp` (recommended)
If the library wants to participate in PyTorch’s function transforms:

- `torch.func.vjp(f, *primals)` requires reverse-mode support; ensure any custom ops provide `backward()`.
- `torch.func.jvp(f, primals, tangents)` requires forward-mode support; ensure any custom ops provide `jvp()`.

For AEG’s two directions, a pragmatic interop strategy is:
- keep the library’s internal Dual carrier as the primary JVP mechanism (computing `(du,dv)` in one pass), and
- if you need PyTorch’s transforms directly, compute two JVPs by calling `torch.func.jvp` twice with two different tangents.

### C.5 Interop with `torch.autograd.forward_ad` (optional)
PyTorch also provides a low-level forward-mode API (`torch.autograd.forward_ad.make_dual` / `unpack_dual`).

A library MAY provide adapters that map:
- `(val, du)` to a PyTorch dual tensor for u-direction, and
- `(val, dv)` to a PyTorch dual tensor for v-direction,

so users can run a PyTorch function under forward AD and recover tangents with `unpack_dual`.

### C.6 PyTorch-facing test checklist (required for PyTorch profile)
If Appendix C conformance is claimed, the implementation MUST include tests that:

1. Verify `seed_a` backward rule against analytic transpose (Appendix B.7).
2. If any fused custom op exists:
   - run PyTorch `gradcheck`-style numerical checks for its backward rule.
   - verify its `jvp()` rule by comparing against finite differences or a reference implementation.
---

## Appendix D — PyTorch Reference Skeleton (non-normative)

This appendix provides a **reference skeleton** showing how an implementation can embed the AEG Dual carrier into the **PyTorch** ecosystem, while supporting:

- **Reverse-mode / VJP** via `torch.autograd` (`backward()`).
- **Forward-mode / JVP** via `torch.func.jvp` (and, optionally, `torch.autograd.forward_ad`).

Everything in this appendix is **non-normative**: it is a *suggested* starting point.  
A conforming implementation MAY diverge as long as it satisfies the normative requirements of the main specification and Appendix C.

### D.1 Suggested Python package layout

```text
aegdual/
  __init__.py
  dual.py          # DualTensor carrier + helpers
  ops.py           # elementwise/unary/binary rules
  nn.py            # linear / conv wrappers (same-kernel-three-channels)
  torchfunc.py     # optional adapters to torch.func / forward_ad
  testing.py       # spec conformance tests for seed_a JVP/VJP, etc.
examples/
  sanity_seed.py
  mlp_reg.py
```

### D.2 Core carrier: `DualTensor` (SoA wrapper)

A minimal wrapper that carries `(val, du, dv)` as three `torch.Tensor` objects:

```python
# aegdual/dual.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple
import torch

@dataclass(frozen=True)
class DualTensor:
    """
    Non-normative reference carrier.
    Invariants (normative in Appendix B.1):
      - val/du/dv have identical shape, dtype, device, strides/layout semantics.
    """
    val: torch.Tensor
    du: torch.Tensor
    dv: torch.Tensor

    def as_tuple(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.val, self.du, self.dv)

    @property
    def shape(self): return self.val.shape
    @property
    def dtype(self): return self.val.dtype
    @property
    def device(self): return self.val.device

    def detach(self) -> "DualTensor":
        return DualTensor(self.val.detach(), self.du.detach(), self.dv.detach())

    def to(self, *args: Any, **kwargs: Any) -> "DualTensor":
        return DualTensor(self.val.to(*args, **kwargs),
                          self.du.to(*args, **kwargs),
                          self.dv.to(*args, **kwargs))

def lift(x: torch.Tensor) -> DualTensor:
    """Lift a primal tensor into DualTensor with zero tangents."""
    z = torch.zeros_like(x)
    return DualTensor(x, z, z)
```

### D.3 Seeding primitives (`const_tensor`, `seed_u`, `seed_v`, `seed_a`)

The core AEG seed is:

- `seed_a(x, mu, lambda): (val=x, du=mu, dv=lambda*x)`.

A pure-PyTorch implementation (Appendix C, Option S1) is enough for correctness.

```python
# aegdual/ops.py (seeding)
import torch
from .dual import DualTensor

def const_tensor(x: torch.Tensor) -> DualTensor:
    z = torch.zeros_like(x)
    return DualTensor(x, z, z)

def seed_u(x: torch.Tensor) -> DualTensor:
    one = torch.ones_like(x)
    zero = torch.zeros_like(x)
    return DualTensor(x, one, zero)

def seed_v(x: torch.Tensor) -> DualTensor:
    one = torch.ones_like(x)
    zero = torch.zeros_like(x)
    return DualTensor(x, zero, one)

def seed_a(x: torch.Tensor, mu: torch.Tensor, lam: torch.Tensor) -> DualTensor:
    """
    AEG flow seed (Section 3.3):
      val = x
      du  = mu         (broadcastable to x)
      dv  = lam * x
    """
    du = torch.ones_like(x) * mu          # broadcast
    dv = lam * x                          # broadcast
    return DualTensor(x, du, dv)
```

**Engineering note:** If you implement `seed_a` exactly like above, PyTorch autograd automatically yields the VJP rule:

- `bar_x += bar_val + lam * bar_dv`

because `dv = lam * x` is part of the ordinary computation graph.

### D.4 Basic forward/JVP rules as PyTorch ops (reference)

Binary arithmetic:

```python
# aegdual/ops.py (binary)
import torch
from .dual import DualTensor, lift

def add(x: DualTensor, y: DualTensor) -> DualTensor:
    return DualTensor(x.val + y.val, x.du + y.du, x.dv + y.dv)

def sub(x: DualTensor, y: DualTensor) -> DualTensor:
    return DualTensor(x.val - y.val, x.du - y.du, x.dv - y.dv)

def mul(x: DualTensor, y: DualTensor) -> DualTensor:
    # Section 4.2
    return DualTensor(
        x.val * y.val,
        x.du * y.val + x.val * y.du,
        x.dv * y.val + x.val * y.dv,
    )

def div(x: DualTensor, y: DualTensor) -> DualTensor:
    # Section 4.3
    y2 = y.val * y.val
    return DualTensor(
        x.val / y.val,
        (x.du * y.val - x.val * y.du) / y2,
        (x.dv * y.val - x.val * y.dv) / y2,
    )
```

Unary ops (chain rule):

```python
# aegdual/ops.py (unary)
import torch
from .dual import DualTensor

def exp(x: DualTensor) -> DualTensor:
    v = torch.exp(x.val)
    return DualTensor(v, v * x.du, v * x.dv)

def log(x: DualTensor) -> DualTensor:
    return DualTensor(torch.log(x.val), x.du / x.val, x.dv / x.val)

def sin(x: DualTensor) -> DualTensor:
    return DualTensor(torch.sin(x.val), torch.cos(x.val) * x.du, torch.cos(x.val) * x.dv)

def cos(x: DualTensor) -> DualTensor:
    return DualTensor(torch.cos(x.val), -torch.sin(x.val) * x.du, -torch.sin(x.val) * x.dv)

def relu(x: DualTensor) -> DualTensor:
    # Appendix A.4 policy P1: branch on val, mask tangents.
    v = torch.relu(x.val)
    mask = (x.val > 0).to(dtype=x.val.dtype)
    return DualTensor(v, x.du * mask, x.dv * mask)
```

### D.5 Linear primitives: reuse PyTorch kernels (same-kernel-three-channels)

This mirrors Section 6 and Appendix B.4.

```python
# aegdual/nn.py
import torch
import torch.nn.functional as F
from .dual import DualTensor

def linear(x: DualTensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> DualTensor:
    y_val = F.linear(x.val, weight, bias)
    y_du  = F.linear(x.du,  weight, None)  # bias is constant by default
    y_dv  = F.linear(x.dv,  weight, None)
    return DualTensor(y_val, y_du, y_dv)

def conv2d(x: DualTensor, weight: torch.Tensor, bias: torch.Tensor | None = None, **kw) -> DualTensor:
    y_val = F.conv2d(x.val, weight, bias, **kw)
    y_du  = F.conv2d(x.du,  weight, None, **kw)
    y_dv  = F.conv2d(x.dv,  weight, None, **kw)
    return DualTensor(y_val, y_du, y_dv)
```

### D.6 Optional: a fused `seed_a` custom op (with backward + jvp)

This section is a **template** for implementations that want a fused kernel (Appendix C, Option S2).  
Even if the forward is written in Python, the structure below is what you would use if you later replace the body with a C++/CUDA kernel.

#### D.6.1 Broadcast-safe `sum_to_shape` helper

When parameters broadcast (e.g., scalar `mu` or scalar `lambda`), gradients must be **summed** back to the parameter shape.

```python
# aegdual/torchfunc.py (helper)
import torch

def sum_to_shape(x: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    Reduce x by summation so that x.shape == shape, assuming x was produced by
    broadcasting a tensor of `shape` to x.shape.
    """
    if tuple(x.shape) == tuple(shape):
        return x
    if len(shape) == 0:
        return x.sum()

    # Left-pad shape to match rank
    ndim = x.ndim
    target = (1,) * (ndim - len(shape)) + tuple(shape)

    # Sum over broadcasted dims
    for dim in range(ndim):
        if target[dim] == 1 and x.shape[dim] != 1:
            x = x.sum(dim=dim, keepdim=True)

    # Remove left padding dims
    for _ in range(ndim - len(shape)):
        x = x.squeeze(dim=0)

    return x
```

#### D.6.2 `torch.autograd.Function` template (transform-friendly)

Key implementation details:

- If you override `setup_context`, **do not return an input tensor “as-is”** *and* also save it for backward:
  return a view (e.g. `x.view_as(x)`) to satisfy PyTorch’s constraint.
- Save what you need:
  - `save_for_backward(...)` for reverse-mode / `backward()`
  - `save_for_forward(...)` for forward-mode / `jvp()`

```python
# aegdual/torchfunc.py (seed_a fused template)
import torch
from .dual import DualTensor
from .torchfunc import sum_to_shape

class _SeedA(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor, mu: torch.Tensor, lam: torch.Tensor):
        # IMPORTANT: avoid returning x as-is if you save it in setup_context.
        val = x.view_as(x)
        du  = torch.ones_like(x) * mu
        dv  = lam * x
        return val, du, dv

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        x, mu, lam = inputs
        # reverse-mode
        ctx.save_for_backward(x, mu, lam)
        # forward-mode (torch.func.jvp)
        ctx.save_for_forward(x, mu, lam)
        ctx.mu_shape = mu.shape
        ctx.lam_shape = lam.shape

    @staticmethod
    def backward(ctx, g_val, g_du, g_dv):
        x, mu, lam = ctx.saved_tensors

        # VJP of seed_a (Section 8.3 / Appendix B.7):
        g_x = g_val + lam * g_dv

        # Optional parameter gradients (only if you want mu/lambda differentiable):
        g_mu  = sum_to_shape(g_du, ctx.mu_shape)           # du = mu
        g_lam = sum_to_shape(g_dv * x, ctx.lam_shape)      # dv = lam * x

        return g_x, g_mu, g_lam

    @staticmethod
    def jvp(ctx, x_dot, mu_dot, lam_dot):
        # JVP of seed_a (Appendix C.3):
        x, mu, lam = ctx.saved_for_forward

        val_dot = x_dot
        du_dot  = torch.ones_like(x) * mu_dot
        dv_dot  = lam_dot * x + lam * x_dot
        return val_dot, du_dot, dv_dot

def seed_a_fused(x: torch.Tensor, mu: torch.Tensor, lam: torch.Tensor) -> DualTensor:
    val, du, dv = _SeedA.apply(x, mu, lam)
    return DualTensor(val, du, dv)
```

### D.7 Minimal conformance tests (reference)

These tests correspond to Section 12 and Appendix C.6.

```python
# aegdual/testing.py
import torch
from torch.func import jvp
from .ops import seed_a

def test_seed_a_jvp_matches_torch_func():
    torch.manual_seed(0)
    x = torch.randn(4, 5)
    mu = torch.tensor(2.0)
    lam = torch.tensor(0.5)

    dx_u = mu * torch.ones_like(x)
    dx_v = lam * x

    def f(z):
        # Any differentiable function. Keep it simple here.
        return torch.sin(z) + z * z

    # Reference via torch.func.jvp
    _, j_u = jvp(f, (x,), (dx_u,))
    _, j_v = jvp(f, (x,), (dx_v,))

    # AEG Dual result
    X = seed_a(x, mu, lam)         # gives val=x, du=dx_u, dv=dx_v
    Y_du  = (torch.cos(X.val) + 2 * X.val) * X.du
    Y_dv  = (torch.cos(X.val) + 2 * X.val) * X.dv

    torch.testing.assert_close(Y_du, j_u)
    torch.testing.assert_close(Y_dv, j_v)

def test_seed_a_vjp_transpose_rule():
    torch.manual_seed(0)
    x = torch.randn(7, requires_grad=True)
    mu = torch.tensor(2.0)
    lam = torch.tensor(0.5)

    X = seed_a(x, mu, lam)
    # choose arbitrary upstream adjoints for the 3 channels
    g_val = torch.randn_like(X.val)
    g_du  = torch.randn_like(X.du)
    g_dv  = torch.randn_like(X.dv)

    # scalarize: L = <g_val, val> + <g_du, du> + <g_dv, dv>
    L = (X.val * g_val + X.du * g_du + X.dv * g_dv).sum()
    L.backward()

    # expected: grad_x = g_val + lam * g_dv
    torch.testing.assert_close(x.grad, g_val + lam * g_dv)
```

### D.8 Practical usage pattern (reference)

A typical training-time use is:

1. ordinary PyTorch forward for data loss
2. Dual forward to compute AEG regularizers using `(du, dv)`
3. use PyTorch backprop to train parameters

```python
# examples/mlp_reg.py (sketch)
import torch
import torch.nn as nn
import torch.nn.functional as F
from aegdual.ops import seed_a, relu
from aegdual.nn import linear

class TinyMLP(nn.Module):
    def __init__(self, din, dh, dout):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(dh, din) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(dh))
        self.w2 = nn.Parameter(torch.randn(dout, dh) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(dout))

    def forward_dual(self, x_dual):
        h = linear(x_dual, self.w1, self.b1)
        h = relu(h)
        y = linear(h, self.w2, self.b2)
        return y

    def forward(self, x):
        h = F.linear(x, self.w1, self.b1)
        h = F.relu(h)
        y = F.linear(h, self.w2, self.b2)
        return y

model = TinyMLP(din=16, dh=32, dout=10)

x = torch.randn(64, 16)
target = torch.randint(0, 10, (64,))
mu = torch.tensor(1.0)
lam = torch.tensor(0.1)

# normal data loss
logits = model(x)
data_loss = F.cross_entropy(logits, target)

# AEG dual forward for regularizer
X = seed_a(x, mu, lam)
Y = model.forward_dual(X)
reg = (Y.du.square().mean() + Y.dv.square().mean())

loss = data_loss + 0.01 * reg
loss.backward()
```
