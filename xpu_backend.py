"""Pure PyTorch XPU backend for comfy_kitchen + ComfyUI INT8 compute unlock."""

from __future__ import annotations

import logging
from typing import Any

import torch

from comfy_kitchen.backends._activations import apply_input_act as _apply_input_act
from comfy_kitchen.backends._activations import apply_residual as _apply_residual
from comfy_kitchen.backends.eager.quantization import DTYPE_CODE_TO_DTYPE, _round_up
from comfy_kitchen.constraints import ExactDims, FunctionConstraints, ParamConstraint
from comfy_kitchen.registry import registry
from comfy_kitchen.tensor.int8_utils import (
    _build_hadamard,
    _rotate_activation,
    _rotate_weight,
)

logger = logging.getLogger("comfyui_xpu_backend")

# ~32 MB FP32 scratch budget per chunk so large fallback layers don't spike VRAM.
_FP32_CHUNK_ELEMS = 8 << 20
# ~64 MB INT32/FP32 rescale threshold before switching to chunked rescale.
_RESCALE_CHUNK_ELEMS = 16 << 20

_LOW_PREC_DTYPES = frozenset({torch.float16, torch.bfloat16})
_INSTALLED = False


def _prepare_scale(scale: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(scale, torch.Tensor):
        return torch.tensor(scale, device=device, dtype=dtype)
    return scale.to(device=device, dtype=dtype)


def _dequantize_simple_to(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize INT8 tensor directly in target floating dtype to halve temporary VRAM."""
    out = q.to(dtype=dtype)
    out.mul_(_prepare_scale(scale, q.device, dtype))
    return out


def dequantize_int8_simple(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize INT8 tensor to float32 in-place to avoid an extra q.float() temporary."""
    return _dequantize_simple_to(q, scale, torch.float32)


def dequantize_int8_simple_dtype(
    q: torch.Tensor,
    scale: torch.Tensor,
    output_dtype_code: int,
) -> torch.Tensor:
    """Dequantize INT8 tensor directly into float16/bfloat16/float32 without FP32 bloat."""
    target_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    if target_dtype in _LOW_PREC_DTYPES or target_dtype == torch.float32:
        return _dequantize_simple_to(q, scale, target_dtype)
    return _dequantize_simple_to(q, scale, torch.float32).to(target_dtype)


def _dequantize_convrot_chunked_fp32(
    q: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize and un-rotate INT8 ConvRot weights in row chunks capping FP32 scratch."""
    out_f, in_f = q.shape
    if out_f == 0 or in_f == 0:
        return torch.empty((out_f, in_f), device=q.device, dtype=out_dtype)

    h = _build_hadamard(group_size, device=q.device, dtype=torch.float32)
    scale_f32 = _prepare_scale(scale, q.device, torch.float32)
    per_row_scale = scale_f32.dim() >= 2 and scale_f32.shape[0] == out_f

    chunk_rows = max(1, min(out_f, _FP32_CHUNK_ELEMS // in_f))
    if chunk_rows >= out_f:
        w = q.to(dtype=torch.float32)
        w.mul_(scale_f32)
        rot = _rotate_weight(w, h, group_size)
        return rot if out_dtype == torch.float32 else rot.to(out_dtype)

    out = torch.empty((out_f, in_f), device=q.device, dtype=out_dtype)
    for start in range(0, out_f, chunk_rows):
        end = min(start + chunk_rows, out_f)
        w_chunk = q[start:end].to(dtype=torch.float32)
        s_chunk = scale_f32[start:end] if per_row_scale else scale_f32
        w_chunk.mul_(s_chunk)
        rot_chunk = _rotate_weight(w_chunk, h, group_size)
        out[start:end] = rot_chunk if out_dtype == torch.float32 else rot_chunk.to(out_dtype)
    return out


def dequantize_int8_convrot_weight(
    q: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Dequantize INT8 ConvRot weights to float32 with bounded scratch memory."""
    return _dequantize_convrot_chunked_fp32(q, scale, group_size, torch.float32)


def dequantize_int8_convrot_weight_dtype(
    q: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    output_dtype_code: int,
) -> torch.Tensor:
    """Dequantize INT8 ConvRot weights into target dtype (direct low-precision path on XPU)."""
    target_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    if target_dtype in _LOW_PREC_DTYPES:
        w = _dequantize_simple_to(q, scale, target_dtype)
        h = _build_hadamard(group_size, device=q.device, dtype=target_dtype)
        return _rotate_weight(w, h, group_size)
    if target_dtype == torch.float32:
        return _dequantize_convrot_chunked_fp32(q, scale, group_size, torch.float32)
    return _dequantize_convrot_chunked_fp32(q, scale, group_size, torch.float32).to(target_dtype)


def _xpu_int8_mm(a: torch.Tensor, weight_nk: torch.Tensor) -> torch.Tensor:
    """Compute a[M, K] @ weight_nk[N, K].T in int32 using torch._int_mm without copying aligned weights."""
    orig_m, k = a.shape
    orig_n = weight_nk.shape[0]
    if orig_m == 0 or k == 0 or orig_n == 0:
        return torch.empty((orig_m, orig_n), device=a.device, dtype=torch.int32)

    if k % 8 == 0 and orig_n % 8 == 0:
        # Zero-copy transpose view works directly with PyTorch XPU torch._int_mm.
        return torch._int_mm(a, weight_nk.T)

    padded_k = _round_up(k, 8)
    padded_n = _round_up(orig_n, 8)
    a_pad = a
    w_pad = weight_nk
    if padded_k != k:
        a_pad = torch.cat((a_pad, torch.zeros((orig_m, padded_k - k), device=a.device, dtype=a.dtype)), dim=1)
        w_pad = torch.cat((w_pad, torch.zeros((orig_n, padded_k - k), device=w_pad.device, dtype=w_pad.dtype)), dim=1)
    if padded_n != orig_n:
        w_pad = torch.cat((w_pad, torch.zeros((padded_n - orig_n, padded_k), device=w_pad.device, dtype=w_pad.dtype)), dim=0)

    out = torch._int_mm(a_pad, w_pad.T)
    return out[:orig_m, :orig_n]


def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    convrot: bool = False,
    convrot_groupsize: int = 256,
    input_act: str | None = None,
    input_act_weight: torch.Tensor | None = None,
    input_act_eps: float = 0.0,
    residual: torch.Tensor | None = None,
    residual_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hybrid XPU INT8 linear: native torch._int_mm for large weights, low-prec dequant for large M."""
    x = _apply_input_act(x, input_act, input_act_weight, input_act_eps)
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"Input and weight inner dimensions must match, got {x.shape[-1]} and {weight.shape[-1]}"
        )

    n, k = weight.shape
    orig_shape = x.shape
    m = x.numel() // k if k > 0 else 0

    if m == 0:
        out = torch.empty((*orig_shape[:-1], n), device=x.device, dtype=out_dtype)
        if bias is not None:
            out.copy_(bias.to(device=x.device, dtype=out_dtype))
        else:
            out.zero_()
        return _apply_residual(out, residual, residual_scale)

    weight = weight.to(device=x.device)
    if not weight.is_contiguous():
        weight = weight.contiguous()

    weight_scale_f32 = _prepare_scale(weight_scale, x.device, torch.float32).reshape(-1)
    if weight_scale_f32.numel() not in (1, n):
        raise ValueError(
            f"INT8 weight scale must be scalar or per-output-channel, got {tuple(weight_scale.shape)} "
            f"for weight shape {tuple(weight.shape)}"
        )

    if convrot and k % convrot_groupsize != 0:
        raise ValueError(
            f"ConvRot group size {convrot_groupsize} does not divide input features {k}"
        )

    # When tokens M >= K on fp16/bf16 weights, in-place low-precision dequant + XMX F.linear
    # avoids materializing large [M, N] int32/float32 buffers and benchmarks ~1.6x-2.4x faster.
    if out_dtype in _LOW_PREC_DTYPES and m >= k and x.dtype in _LOW_PREC_DTYPES:
        compute_dtype = x.dtype
        scale_view = weight_scale.to(device=x.device, dtype=compute_dtype)
        w_deq = _dequantize_simple_to(weight, scale_view, compute_dtype)
        if convrot:
            h = _build_hadamard(convrot_groupsize, device=x.device, dtype=compute_dtype)
            w_deq = _rotate_weight(w_deq, h, convrot_groupsize)
        bias_cast = bias.to(device=x.device, dtype=compute_dtype) if bias is not None else None
        out = torch.nn.functional.linear(x, w_deq, bias_cast)
        if out.dtype != out_dtype:
            out = out.to(out_dtype)
        return _apply_residual(out, residual, residual_scale)

    x_rot = x
    if convrot:
        h = _build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        x_rot = _rotate_activation(x_rot, h, convrot_groupsize)

    x_2d = x_rot.reshape(-1, k)
    x_scale = (x_2d.abs().amax(dim=-1, keepdim=True).float() * (1.0 / 127.0)).clamp_(min=1e-30)
    x_8 = (x_2d / x_scale.to(x_2d.dtype)).round_().clamp_(-128.0, 127.0).to(torch.int8)

    acc = _xpu_int8_mm(x_8, weight)
    ws_row = weight_scale_f32.reshape(1, -1)

    if m * n <= _RESCALE_CHUNK_ELEMS:
        scaled = acc.to(dtype=torch.float32)
        scaled.mul_(x_scale)
        scaled.mul_(ws_row)
        result = scaled.to(out_dtype)
    else:
        chunk_rows = max(1, min(m, _FP32_CHUNK_ELEMS // n))
        result = torch.empty((m, n), device=acc.device, dtype=out_dtype)
        for start in range(0, m, chunk_rows):
            end = min(start + chunk_rows, m)
            chunk = acc[start:end].to(dtype=torch.float32)
            chunk.mul_(x_scale[start:end])
            chunk.mul_(ws_row)
            result[start:end] = chunk.to(out_dtype)

    if bias is not None:
        result.add_(bias.to(device=result.device, dtype=out_dtype).reshape(1, -1))

    result = result.reshape(*orig_shape[:-1], n)
    return _apply_residual(result, residual, residual_scale)


def _build_xpu_constraints() -> dict[str, FunctionConstraints]:
    xpu_only = frozenset({"xpu"})
    xpu_or_cpu = frozenset({"xpu", "cpu"})
    standard_floats = frozenset({torch.float32, torch.float16, torch.bfloat16})
    int8_set = frozenset({torch.int8})
    int_set = frozenset({int})
    bool_set = frozenset({bool})

    return {
        "dequantize_int8_simple": FunctionConstraints(
            params={
                "q": ParamConstraint(dtypes=int8_set),
                "scale": ParamConstraint(dtypes=standard_floats, devices=xpu_or_cpu),
            },
            default_devices=xpu_only,
        ),
        "dequantize_int8_simple_dtype": FunctionConstraints(
            params={
                "q": ParamConstraint(dtypes=int8_set),
                "scale": ParamConstraint(dtypes=standard_floats, devices=xpu_or_cpu),
                "output_dtype_code": ParamConstraint(dtypes=int_set),
            },
            default_devices=xpu_only,
        ),
        "dequantize_int8_convrot_weight": FunctionConstraints(
            params={
                "q": ParamConstraint(dtypes=int8_set, shape_rules=(ExactDims(2),)),
                "scale": ParamConstraint(dtypes=standard_floats, devices=xpu_or_cpu),
                "group_size": ParamConstraint(dtypes=int_set),
            },
            default_devices=xpu_only,
        ),
        "dequantize_int8_convrot_weight_dtype": FunctionConstraints(
            params={
                "q": ParamConstraint(dtypes=int8_set, shape_rules=(ExactDims(2),)),
                "scale": ParamConstraint(dtypes=standard_floats, devices=xpu_or_cpu),
                "group_size": ParamConstraint(dtypes=int_set),
                "output_dtype_code": ParamConstraint(dtypes=int_set),
            },
            default_devices=xpu_only,
        ),
        "int8_linear": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "weight": ParamConstraint(dtypes=int8_set, shape_rules=(ExactDims(2),)),
                "weight_scale": ParamConstraint(dtypes=standard_floats, devices=xpu_or_cpu),
                "bias": ParamConstraint(dtypes=standard_floats, devices=xpu_or_cpu),
                "convrot": ParamConstraint(dtypes=bool_set),
                "convrot_groupsize": ParamConstraint(dtypes=int_set),
                "residual": ParamConstraint(dtypes=standard_floats),
                "residual_scale": ParamConstraint(dtypes=standard_floats, devices=xpu_or_cpu),
            },
            default_devices=xpu_only,
        ),
    }


def register_xpu_backend() -> bool:
    """Register pure-PyTorch XPU backend in comfy_kitchen.registry if an XPU device is available."""
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        registry.mark_unavailable("xpu", "PyTorch XPU runtime not available")
        return False

    import sys

    registry.register(
        name="xpu",
        module=sys.modules[__name__],
        capabilities=_build_xpu_constraints(),
    )
    priority = [name for name in registry._priority if name != "xpu"]
    insert_at = priority.index("eager") if "eager" in priority else len(priority)
    if "triton" in priority:
        insert_at = min(insert_at, priority.index("triton"))
    priority.insert(insert_at, "xpu")
    registry.set_priority(priority)
    return True


def install_comfy_xpu_hooks() -> dict[str, Any]:
    """Idempotently enable XPU comfy_kitchen ops and ComfyUI int8_tensorwise compute gates."""
    global _INSTALLED

    xpu_registered = register_xpu_backend()
    if not xpu_registered:
        return {"registered": False, "gates_patched": False}

    if _INSTALLED:
        return {"registered": True, "gates_patched": True}

    gates_patched = False
    try:
        import comfy.model_management as mm
        import comfy.ops as comfy_ops
        from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

        orig_supports_int8 = getattr(mm, "_orig_supports_int8_compute", mm.supports_int8_compute)
        mm._orig_supports_int8_compute = orig_supports_int8

        def _xpu_supports_int8_compute(device=None):
            if mm.is_device_cpu(device):
                return False
            if mm.is_intel_xpu() and (device is None or getattr(device, "type", str(device)).startswith("xpu")):
                return True
            return orig_supports_int8(device)

        mm.supports_int8_compute = _xpu_supports_int8_compute

        orig_get_disabled = getattr(comfy_ops, "_orig_get_disabled_quant_formats", comfy_ops.get_disabled_quant_formats)
        comfy_ops._orig_get_disabled_quant_formats = orig_get_disabled

        def _xpu_get_disabled_quant_formats(device=None):
            disabled = orig_get_disabled(device)
            if mm.is_intel_xpu() and not mm.is_device_cpu(device):
                # Re-disable unverified packed INT4/INT6 grouped formats on XPU; unlock int8_tensorwise.
                disabled.discard("int8_tensorwise")
                disabled.add("convrot_w4a4")
                disabled.update(comfy_ops._GROUPED_INT8_FORMATS)
            return disabled

        comfy_ops.get_disabled_quant_formats = _xpu_get_disabled_quant_formats

        orig_fast_mm = getattr(TensorWiseINT8Layout, "_orig_supports_fast_matmul", TensorWiseINT8Layout.supports_fast_matmul)
        TensorWiseINT8Layout._orig_supports_fast_matmul = orig_fast_mm

        @classmethod
        def _xpu_supports_fast_matmul(cls) -> bool:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return True
            return orig_fast_mm()

        TensorWiseINT8Layout.supports_fast_matmul = _xpu_supports_fast_matmul
        gates_patched = True
    except ImportError as exc:
        logger.debug("ComfyUI modules not imported; skipping gate patch: %s", exc)

    _INSTALLED = True
    logger.info(
        "[comfyui-xpu-backend] registered comfy_kitchen 'xpu' backend (priority=%s, gates_patched=%s)",
        registry._priority,
        gates_patched,
    )
    return {"registered": True, "gates_patched": gates_patched}
