"""Verification tests for comfyui-xpu-backend (registry dispatch, accuracy, edge cases, VRAM, speed)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
_CANDIDATES = [
    os.environ.get("COMFY_ROOT"),
    REPO_ROOT.parents[2] if len(REPO_ROOT.parents) > 2 else None,
    Path.cwd(),
]
COMFY_ROOT = next(
    (Path(p) for p in _CANDIDATES if p and (Path(p) / "comfy").is_dir()),
    None,
)
if COMFY_ROOT is not None and str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import comfy.model_management as mm
import comfy.ops as comfy_ops
import comfy_kitchen as ck
from comfy_kitchen.registry import registry
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from xpu_backend import install_comfy_xpu_hooks


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).norm().item() / max(b.float().norm().item(), 1e-12)


def main() -> None:
    assert torch.xpu.is_available(), "Intel XPU device not available"
    dev = torch.device("xpu:0")

    status = install_comfy_xpu_hooks()
    assert status == {"registered": True, "gates_patched": True}, f"Unexpected status: {status}"
    # Idempotency check
    status2 = install_comfy_xpu_hooks()
    assert status2 == {"registered": True, "gates_patched": True}

    # 1. Verify ComfyUI gates
    assert mm.supports_int8_compute() is True
    assert mm.supports_int8_compute(torch.device("cpu")) is False
    disabled = comfy_ops.get_disabled_quant_formats()
    assert "int8_tensorwise" not in disabled, f"int8_tensorwise still disabled: {disabled}"
    assert "convrot_w4a4" in disabled
    assert TensorWiseINT8Layout.supports_fast_matmul() is True

    # 2. Verify registry dispatches XPU to 'xpu' and CPU to 'eager'
    qx = torch.randint(-127, 127, (64, 256), device=dev, dtype=torch.int8)
    sx = torch.rand(64, 1, device=dev, dtype=torch.float32) * 0.02 + 0.001
    qc = qx.cpu()
    sc = sx.cpu()

    assert registry.get_capable_backend("dequantize_int8_convrot_weight_dtype", {"q": qx, "scale": sx, "group_size": 256, "output_dtype_code": 2}) == "xpu"
    assert registry.get_capable_backend("dequantize_int8_convrot_weight_dtype", {"q": qc, "scale": sc, "group_size": 256, "output_dtype_code": 2}) == "eager"

    xx = torch.randn(32, 256, device=dev, dtype=torch.bfloat16)
    assert registry.get_capable_backend("int8_linear", {"x": xx, "weight": qx, "weight_scale": sx, "convrot": True, "convrot_groupsize": 256}) == "xpu"

    # 3. Verify dequantize accuracy & peak memory
    w_big = torch.randn(13824, 5120, device=dev, dtype=torch.bfloat16) * 0.02
    qw, ws = torch.ops.comfy_kitchen.quantize_int8_convrot_weight(w_big, 256)

    with registry.use_backend("eager"):
        ref_fp32 = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight(qw, ws, 256)
        torch.xpu.reset_peak_memory_stats()
        m0 = torch.xpu.memory_allocated()
        ref_bf16 = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(qw, ws, 256, 2)
        torch.xpu.synchronize()
        eager_peak_mb = (torch.xpu.max_memory_allocated() - m0) / (1024**2)

    del ref_bf16
    xpu_fp32 = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight(qw, ws, 256)
    assert rel_err(xpu_fp32, ref_fp32) < 1e-6, "FP32 convrot dequant drifted from eager"
    del xpu_fp32

    torch.xpu.reset_peak_memory_stats()
    m0 = torch.xpu.memory_allocated()
    xpu_bf16 = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(qw, ws, 256, 2)
    torch.xpu.synchronize()
    xpu_peak_mb = (torch.xpu.max_memory_allocated() - m0) / (1024**2)
    assert rel_err(xpu_bf16, ref_fp32) < 0.01, f"BF16 dequant rel_err too high: {rel_err(xpu_bf16, ref_fp32)}"
    assert xpu_peak_mb < eager_peak_mb, f"Expected lower peak VRAM ({xpu_peak_mb:.1f} vs {eager_peak_mb:.1f})"
    del ref_fp32, xpu_bf16

    # Simple dequantize dtype check (fp16)
    with registry.use_backend("eager"):
        ref_s = torch.ops.comfy_kitchen.dequantize_int8_simple_dtype(qw, ws, 1)
    xpu_s = torch.ops.comfy_kitchen.dequantize_int8_simple_dtype(qw, ws, 1)
    assert xpu_s.dtype == torch.float16
    assert rel_err(xpu_s, ref_s) < 0.005
    del ref_s, xpu_s, qw, ws, w_big

    # 4. Verify int8_linear across edge cases and both dispatch regimes
    cases = [
        ("large_weight_int8_mm", 512, 2048, 4096, True, torch.bfloat16, True, True),
        ("large_tokens_dq_linear", 2048, 1024, 1024, True, torch.bfloat16, True, False),
        ("small_M1_convrot", 1, 512, 512, True, torch.float16, True, False),
        ("unaligned_K_N_pad", 17, 130, 77, False, torch.bfloat16, False, False),
        ("empty_batch", 0, 256, 256, True, torch.bfloat16, True, False),
    ]
    for label, M, K, N, convrot, dtype, with_bias, with_res in cases:
        w = torch.randn(N, K, device=dev, dtype=dtype) * 0.02
        if convrot:
            q_w, w_s = torch.ops.comfy_kitchen.quantize_int8_convrot_weight(w, 256)
        else:
            q_w, w_s = torch.ops.comfy_kitchen.quantize_int8_tensorwise(w)
        x = torch.randn(M, K, device=dev, dtype=dtype) * 0.5
        bias = torch.randn(N, device=dev, dtype=dtype) * 0.01 if with_bias else None
        res = torch.randn(M, N, device=dev, dtype=dtype) if with_res else None
        rscale = torch.rand(N, device=dev, dtype=dtype) if with_res else None

        if M == 0:
            actual = ck.int8_linear(x, q_w, w_s, bias, dtype, convrot=convrot, convrot_groupsize=256, residual=res, residual_scale=rscale)
            assert actual.shape == (0, N) and actual.dtype == dtype
            print(f"PASS [{label}] shape=({M},{K})->({M},{N}) dtype={dtype}")
            continue

        with registry.use_backend("eager"):
            expected = ck.int8_linear(x, q_w, w_s, bias, dtype, convrot=convrot, convrot_groupsize=256, residual=res, residual_scale=rscale)
        actual = ck.int8_linear(x, q_w, w_s, bias, dtype, convrot=convrot, convrot_groupsize=256, residual=res, residual_scale=rscale)
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        if M > 0:
            err = rel_err(actual, expected)
            assert err < 0.01, f"{label} rel_err={err}"
        print(f"PASS [{label}] shape=({M},{K})->({M},{N}) dtype={dtype}")

    # 5. Speed benchmark summary (M=1024, K=5120, N=13824 WAN FFN + M=4096, K=3072, N=3072 Qwen)
    for M, K, N in [(1024, 5120, 13824), (4096, 3072, 3072)]:
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        q_w, w_s = torch.ops.comfy_kitchen.quantize_int8_convrot_weight(w, 256)
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.5

        for bname in ("eager", "xpu"):
            def _run():
                with registry.use_backend(bname):
                    if bname == "eager":
                        return torch.nn.functional.linear(x, torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(q_w, w_s, 256, 2))
                    return ck.int8_linear(x, q_w, w_s, None, torch.bfloat16, convrot=True, convrot_groupsize=256)

            _ = _run()
            torch.xpu.synchronize()
            torch.xpu.reset_peak_memory_stats()
            m0 = torch.xpu.memory_allocated()
            t0 = time.perf_counter()
            for _ in range(10):
                _ = _run()
            torch.xpu.synchronize()
            ms = (time.perf_counter() - t0) * 100
            pk = (torch.xpu.max_memory_allocated() - m0) / (1024**2)
            tag = "eager_fallback" if bname == "eager" else "xpu_backend"
            print(f"BENCH M={M} K={K} N={N} | {tag:14s}: {ms:5.2f} ms | peak={pk:6.1f} MB")

    print(f"SUMMARY dequant_13824x5120 peak VRAM: eager={eager_peak_mb:.1f} MB -> xpu={xpu_peak_mb:.1f} MB")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
