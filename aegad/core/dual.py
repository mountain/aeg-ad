from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class DualTensor(torch.Tensor):
    _primal: torch.Tensor
    _du: torch.Tensor
    _dv: torch.Tensor

    @staticmethod
    def __new__(cls, val: torch.Tensor, du: torch.Tensor, dv: torch.Tensor) -> DualTensor:
        if not isinstance(val, torch.Tensor):
            raise TypeError("val must be a torch.Tensor")
        if not isinstance(du, torch.Tensor) or not isinstance(dv, torch.Tensor):
            raise TypeError("du/dv must be torch.Tensor")

        obj: DualTensor = torch.Tensor._make_subclass(cls, val, require_grad=val.requires_grad)
        obj._primal = val
        obj._du = du
        obj._dv = dv
        obj._check_invariants()
        return obj

    def _check_invariants(self) -> None:
        val = self._primal
        for name, t in (("du", self._du), ("dv", self._dv)):
            if t.shape != val.shape:
                raise ValueError(f"{name}.shape must equal val.shape")
            if t.dtype != val.dtype:
                raise ValueError(f"{name}.dtype must equal val.dtype")
            if t.device != val.device:
                raise ValueError(f"{name}.device must equal val.device")

    @property
    def val(self) -> torch.Tensor:
        return self._primal

    @property
    def du(self) -> torch.Tensor:
        return self._du

    @property
    def dv(self) -> torch.Tensor:
        return self._dv

    @classmethod
    def __torch_dispatch__(
        cls,
        func: Any,
        types: Any,
        args: tuple[Any, ...],
        kwargs: dict[Any, Any],
    ) -> Any:
        op = str(func)
        if op.startswith("aten.") and op.split(".")[1].endswith("_"):
            raise RuntimeError(f"in-place op not supported for DualTensor: {op}")

        def unwrap(x: Any) -> tuple[Any, Any, Any, bool]:
            if isinstance(x, DualTensor):
                return x.val, x.du, x.dv, True
            if isinstance(x, torch.Tensor):
                z = torch.zeros_like(x)
                return x, z, z, False
            return x, x, x, False

        def unwrap_tree(x: Any) -> tuple[Any, Any, Any, bool]:
            if isinstance(x, tuple):
                vp, duv, dvv, any_dual = [], [], [], False
                for xi in x:
                    p, u, v, a = unwrap_tree(xi)
                    vp.append(p)
                    duv.append(u)
                    dvv.append(v)
                    any_dual = any_dual or a
                return tuple(vp), tuple(duv), tuple(dvv), any_dual
            if isinstance(x, list):
                vp, duv, dvv, any_dual = [], [], [], False
                for xi in x:
                    p, u, v, a = unwrap_tree(xi)
                    vp.append(p)
                    duv.append(u)
                    dvv.append(v)
                    any_dual = any_dual or a
                return vp, duv, dvv, any_dual
            return unwrap(x)

        def wrap(primal: Any, du: Any, dv: Any) -> Any:
            if isinstance(primal, torch.Tensor):
                return DualTensor(primal, du, dv)
            if isinstance(primal, tuple):
                return tuple(wrap(p, u, v) for p, u, v in zip(primal, du, dv, strict=True))
            if isinstance(primal, list):
                return [wrap(p, u, v) for p, u, v in zip(primal, du, dv, strict=True)]
            return primal

        primals, dus, dvs, any_dual = unwrap_tree(args)
        if not any_dual:
            return func(*args, **kwargs)

        if op in {"aten.size.int", "aten.size", "aten.dim", "aten.numel"}:
            return func(*primals, **kwargs)

        if op in {"aten.view.default", "aten._unsafe_view.default", "aten.reshape.default"}:
            out = func(*primals, **kwargs)
            out_du = func(*dus, **kwargs)
            out_dv = func(*dvs, **kwargs)
            return wrap(out, out_du, out_dv)

        if op in {"aten.flatten.using_ints", "aten.flatten.int"}:
            out = func(*primals, **kwargs)
            out_du = func(*dus, **kwargs)
            out_dv = func(*dvs, **kwargs)
            return wrap(out, out_du, out_dv)

        if op in {
            "aten.permute.default",
            "aten.transpose.int",
            "aten.t.default",
            "aten.contiguous.default",
        }:
            out = func(*primals, **kwargs)
            out_du = func(*dus, **kwargs)
            out_dv = func(*dvs, **kwargs)
            return wrap(out, out_du, out_dv)

        if op in {"aten.clone.default", "aten.detach.default"}:
            out = func(*primals, **kwargs)
            out_du = func(*dus, **kwargs)
            out_dv = func(*dvs, **kwargs)
            return wrap(out, out_du, out_dv)

        if op in {"aten.add.Tensor", "aten.add.Scalar"}:
            x, x_du, x_dv, _ = unwrap(args[0])
            y, y_du, y_dv, _ = unwrap(args[1])
            alpha = kwargs.get("alpha", 1)
            if (
                not isinstance(x, torch.Tensor)
                or not isinstance(x_du, torch.Tensor)
                or not isinstance(x_dv, torch.Tensor)
            ):
                raise RuntimeError("expected tensor input for add")
            if not isinstance(y_du, torch.Tensor) or not isinstance(y_dv, torch.Tensor):
                y_du = torch.zeros_like(x)
                y_dv = torch.zeros_like(x)
            out = torch.add(x, y, **kwargs)
            out_du = x_du + alpha * y_du
            out_dv = x_dv + alpha * y_dv
            return DualTensor(out, out_du, out_dv)

        if op in {"aten.sub.Tensor", "aten.sub.Scalar"}:
            x, x_du, x_dv, _ = unwrap(args[0])
            y, y_du, y_dv, _ = unwrap(args[1])
            alpha = kwargs.get("alpha", 1)
            if (
                not isinstance(x, torch.Tensor)
                or not isinstance(x_du, torch.Tensor)
                or not isinstance(x_dv, torch.Tensor)
            ):
                raise RuntimeError("expected tensor input for sub")
            if not isinstance(y_du, torch.Tensor) or not isinstance(y_dv, torch.Tensor):
                y_du = torch.zeros_like(x)
                y_dv = torch.zeros_like(x)
            out = torch.sub(x, y, **kwargs)
            out_du = x_du - alpha * y_du
            out_dv = x_dv - alpha * y_dv
            return DualTensor(out, out_du, out_dv)

        if op in {"aten.neg.default"}:
            x, x_du, x_dv, _ = unwrap(args[0])
            if not isinstance(x_du, torch.Tensor) or not isinstance(x_dv, torch.Tensor):
                raise RuntimeError("expected tensor input for neg")
            out = torch.neg(x)
            return DualTensor(out, -x_du, -x_dv)

        if op in {"aten.mul.Tensor", "aten.mul.Scalar"}:
            x, x_du, x_dv, _ = unwrap(args[0])
            y, y_du, y_dv, _ = unwrap(args[1])
            if (
                not isinstance(x, torch.Tensor)
                or not isinstance(x_du, torch.Tensor)
                or not isinstance(x_dv, torch.Tensor)
            ):
                raise RuntimeError("expected tensor input for mul")
            if not isinstance(y_du, torch.Tensor) or not isinstance(y_dv, torch.Tensor):
                y_du = torch.zeros_like(x)
                y_dv = torch.zeros_like(x)
            out = torch.mul(x, y)
            out_du = x_du * y + x * y_du
            out_dv = x_dv * y + x * y_dv
            return DualTensor(out, out_du, out_dv)

        if op in {"aten.div.Tensor", "aten.div.Scalar"}:
            x, x_du, x_dv, _ = unwrap(args[0])
            y, y_du, y_dv, _ = unwrap(args[1])
            if (
                not isinstance(x, torch.Tensor)
                or not isinstance(x_du, torch.Tensor)
                or not isinstance(x_dv, torch.Tensor)
            ):
                raise RuntimeError("expected tensor input for div")
            if not isinstance(y_du, torch.Tensor) or not isinstance(y_dv, torch.Tensor):
                y_du = torch.zeros_like(x)
                y_dv = torch.zeros_like(x)
            out = torch.div(x, y, **kwargs)
            y2 = y * y
            out_du = (x_du * y - x * y_du) / y2
            out_dv = (x_dv * y - x * y_dv) / y2
            return DualTensor(out, out_du, out_dv)

        if op in {"aten.exp.default"}:
            x, x_du, x_dv, _ = unwrap(args[0])
            if not isinstance(x_du, torch.Tensor) or not isinstance(x_dv, torch.Tensor):
                raise RuntimeError("expected tensor input for exp")
            out = torch.exp(x)
            return DualTensor(out, out * x_du, out * x_dv)

        if op in {"aten.log.default"}:
            x, x_du, x_dv, _ = unwrap(args[0])
            if not isinstance(x_du, torch.Tensor) or not isinstance(x_dv, torch.Tensor):
                raise RuntimeError("expected tensor input for log")
            out = torch.log(x)
            return DualTensor(out, x_du / x, x_dv / x)

        if op in {"aten.tanh.default"}:
            x, x_du, x_dv, _ = unwrap(args[0])
            if not isinstance(x_du, torch.Tensor) or not isinstance(x_dv, torch.Tensor):
                raise RuntimeError("expected tensor input for tanh")
            out = torch.tanh(x)
            d = 1 - out * out
            return DualTensor(out, d * x_du, d * x_dv)

        if op in {"aten.sigmoid.default"}:
            x, x_du, x_dv, _ = unwrap(args[0])
            if not isinstance(x_du, torch.Tensor) or not isinstance(x_dv, torch.Tensor):
                raise RuntimeError("expected tensor input for sigmoid")
            out = torch.sigmoid(x)
            d = out * (1 - out)
            return DualTensor(out, d * x_du, d * x_dv)

        if op in {"aten.relu.default"}:
            x, x_du, x_dv, _ = unwrap(args[0])
            if not isinstance(x_du, torch.Tensor) or not isinstance(x_dv, torch.Tensor):
                raise RuntimeError("expected tensor input for relu")
            out = F.relu(x)
            mask = (x > 0).to(dtype=x.dtype)
            return DualTensor(out, x_du * mask, x_dv * mask)

        if op in {"aten.threshold.default"}:
            x, x_du, x_dv, _ = unwrap(args[0])
            threshold = args[1]
            if not isinstance(x_du, torch.Tensor) or not isinstance(x_dv, torch.Tensor):
                raise RuntimeError("expected tensor input for threshold")
            out = F.threshold(x, float(threshold), float(args[2]))
            mask = (x > threshold).to(dtype=x.dtype)
            return DualTensor(out, x_du * mask, x_dv * mask)

        if op in {"aten.pow.Tensor_Scalar"}:
            x, x_du, x_dv, _ = unwrap(args[0])
            n = args[1]
            out = torch.pow(x, n)
            if not isinstance(x_du, torch.Tensor) or not isinstance(x_dv, torch.Tensor):
                raise RuntimeError("expected tensor input for pow")
            if not isinstance(n, (int, float)):
                raise NotImplementedError("pow.Tensor_Scalar only supports numeric scalar exponent")
            d = n * torch.pow(x, n - 1)
            return DualTensor(out, d * x_du, d * x_dv)

        if op in {"aten.mm.default", "aten.matmul.default", "aten.bmm.default"}:
            x, x_du, x_dv, _ = unwrap(args[0])
            y, y_du, y_dv, _ = unwrap(args[1])
            if not isinstance(x_du, torch.Tensor) or not isinstance(x_dv, torch.Tensor):
                raise RuntimeError("expected tensor input for matmul")
            if not isinstance(y_du, torch.Tensor) or not isinstance(y_dv, torch.Tensor):
                raise RuntimeError("expected tensor input for matmul")
            out = torch.matmul(x, y)
            out_du = torch.matmul(x_du, y) + torch.matmul(x, y_du)
            out_dv = torch.matmul(x_dv, y) + torch.matmul(x, y_dv)
            return DualTensor(out, out_du, out_dv)

        if op in {"aten.addmm.default"}:
            inp, inp_du, inp_dv, _ = unwrap(args[0])
            mat1, mat1_du, mat1_dv, _ = unwrap(args[1])
            mat2, mat2_du, mat2_dv, _ = unwrap(args[2])
            beta = kwargs.get("beta", 1)
            alpha = kwargs.get("alpha", 1)
            if (
                not isinstance(inp_du, torch.Tensor)
                or not isinstance(inp_dv, torch.Tensor)
                or not isinstance(mat1_du, torch.Tensor)
                or not isinstance(mat1_dv, torch.Tensor)
                or not isinstance(mat2_du, torch.Tensor)
                or not isinstance(mat2_dv, torch.Tensor)
            ):
                raise RuntimeError("expected tensor inputs for addmm")
            out = torch.addmm(inp, mat1, mat2, beta=beta, alpha=alpha)
            out_du = beta * inp_du + alpha * (
                torch.matmul(mat1_du, mat2) + torch.matmul(mat1, mat2_du)
            )
            out_dv = beta * inp_dv + alpha * (
                torch.matmul(mat1_dv, mat2) + torch.matmul(mat1, mat2_dv)
            )
            return DualTensor(out, out_du, out_dv)

        if op in {"aten.linear.default"}:
            x, x_du, x_dv, _ = unwrap(args[0])
            weight = args[1]
            bias = args[2] if len(args) > 2 else None
            if isinstance(weight, DualTensor) or isinstance(bias, DualTensor):
                raise NotImplementedError("Dual weight/bias not supported in phase 1")
            if x_du is None or x_dv is None:
                raise RuntimeError("expected tensor input for linear")
            with torch.enable_grad():
                out = F.linear(x, weight, bias)
                out_du = F.linear(x_du, weight, None)
                out_dv = F.linear(x_dv, weight, None)
            return DualTensor(out, out_du, out_dv)

        if op in {"aten.convolution.default"}:
            x, x_du, x_dv, _ = unwrap(args[0])
            weight = args[1]
            bias = args[2]
            stride = args[3]
            padding = args[4]
            dilation = args[5]
            transposed = args[6]
            output_padding = args[7]
            groups = args[8]
            if transposed:
                raise NotImplementedError("transposed convolution not supported in phase 1")
            if isinstance(weight, DualTensor) or isinstance(bias, DualTensor):
                raise NotImplementedError("Dual weight/bias not supported in phase 1")
            if x_du is None or x_dv is None:
                raise RuntimeError("expected tensor input for convolution")
            if x.dim() != 4:
                raise NotImplementedError("only conv2d is supported in phase 1")
            if not (
                output_padding == 0
                or (
                    isinstance(output_padding, (tuple, list))
                    and all(int(v) == 0 for v in output_padding)
                )
            ):
                raise NotImplementedError("output_padding not supported in phase 1")
            with torch.enable_grad():
                out = F.conv2d(
                    x,
                    weight,
                    bias,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                out_du = F.conv2d(
                    x_du,
                    weight,
                    None,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                out_dv = F.conv2d(
                    x_dv,
                    weight,
                    None,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
            return DualTensor(out, out_du, out_dv)

        if op in {"aten.max_pool2d_with_indices.default"}:
            x, x_du, x_dv, _ = unwrap(args[0])
            kernel_size = args[1]
            stride = args[2] if len(args) > 2 else None
            padding = args[3] if len(args) > 3 else 0
            dilation = args[4] if len(args) > 4 else 1
            ceil_mode = args[5] if len(args) > 5 else False
            y_val, idx = F.max_pool2d(
                x,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
                return_indices=True,
            )
            if x_du is None or x_dv is None:
                raise RuntimeError("expected tensor input for max_pool2d_with_indices")
            flat_idx = idx.reshape(idx.shape[0], idx.shape[1], -1)
            y_du = (
                x_du.reshape(x_du.shape[0], x_du.shape[1], -1)
                .gather(2, flat_idx)
                .reshape_as(y_val)
            )
            y_dv = (
                x_dv.reshape(x_dv.shape[0], x_dv.shape[1], -1)
                .gather(2, flat_idx)
                .reshape_as(y_val)
            )
            return DualTensor(y_val, y_du, y_dv), idx

        if op in {"aten.max_pool2d.default"}:
            x, x_du, x_dv, _ = unwrap(args[0])
            kernel_size = args[1]
            stride = args[2] if len(args) > 2 else None
            padding = args[3] if len(args) > 3 else 0
            dilation = args[4] if len(args) > 4 else 1
            ceil_mode = args[5] if len(args) > 5 else False
            y_val, idx = F.max_pool2d(
                x,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
                return_indices=True,
            )
            if x_du is None or x_dv is None:
                raise RuntimeError("expected tensor input for max_pool2d")
            flat_idx = idx.reshape(idx.shape[0], idx.shape[1], -1)
            y_du = (
                x_du.reshape(x_du.shape[0], x_du.shape[1], -1)
                .gather(2, flat_idx)
                .reshape_as(y_val)
            )
            y_dv = (
                x_dv.reshape(x_dv.shape[0], x_dv.shape[1], -1)
                .gather(2, flat_idx)
                .reshape_as(y_val)
            )
            return DualTensor(y_val, y_du, y_dv)

        raise NotImplementedError(f"DualTensor missing rule for op: {op}")
