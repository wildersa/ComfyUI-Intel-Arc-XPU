"""Idempotent XPU batched eager adapter for comfy_kitchen na3d."""

from __future__ import annotations

import functools
import importlib
import logging
import os
from typing import Any

import torch

log = logging.getLogger("ComfyUI-OmniXPU")

_PATCH_MARKER = "__omnixpu_na3d_eager_original__"
_DEFAULT_KV_STACK_BUDGET = 2**28
_LARGE_VRAM_THRESHOLD_BYTES = 16 * (1024**3)
_LARGE_VRAM_GMAX = 4
_SMALL_VRAM_GMAX = 2
_GMAX_ENV = "OMNIXPU_NA3D_EAGER_GMAX"

_logged_active = False
_logged_oom = False


def _select_xpu_g_max(device: Any) -> int:
    """Choose a conservative tile-stacking factor for XPU eager na3d."""
    raw = os.environ.get(_GMAX_ENV, "").strip()
    if raw:
        try:
            override = int(raw)
        except ValueError:
            pass
        else:
            return override if override > 1 else 1

    try:
        props = torch.xpu.get_device_properties(device)
        total_memory = int(getattr(props, "total_memory", 0))
    except Exception:
        return _SMALL_VRAM_GMAX

    if total_memory <= 0:
        return _SMALL_VRAM_GMAX
    if total_memory >= _LARGE_VRAM_THRESHOLD_BYTES:
        return _LARGE_VRAM_GMAX
    return _SMALL_VRAM_GMAX


def _get_registry_eager_module() -> Any | None:
    """Return the eager backend module registered in comfy_kitchen.registry, if any."""
    for mod_name in ("comfy_kitchen", "comfy_kitchen.registry"):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        reg = getattr(mod, "registry", mod)
        if hasattr(reg, "registry"):
            reg = reg.registry
        backends = getattr(reg, "_backends", None)
        if isinstance(backends, dict):
            return backends.get("eager")
    return None


def _is_oom_error(exc: BaseException) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return "out of memory" in str(exc).lower()


def _run_batched_xpu_na3d(
    eager_na: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: list[int],
    is_causal: list[bool] | None,
    scale: float | None,
    xpu_g_max: int,
) -> torch.Tensor:
    """Execute eager na3d on XPU with bounded tile-group batching."""
    batch, t, h, w, nh, hd = q.shape
    dims = (t, h, w)
    causal = [False, False, False] if is_causal is None else list(is_causal)
    kernels = [
        k_ if c else min(k_, d)
        for k_, c, d in zip(kernel_size, causal, dims, strict=True)
    ]
    if scale is None:
        scale = hd**-0.5
    device = q.device
    if scale != 1.0:
        q = q * scale

    bounds = [
        eager_na._window_bounds(d, k_, c)
        for d, k_, c in zip(dims, kernels, causal, strict=True)
    ]
    tile_t, tile_h, tile_w = eager_na._pick_tiles(
        dims, [min(k_, d) for k_, d in zip(kernels, dims, strict=True)]
    )

    groups: dict[Any, list[tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]]]] = {}
    for t0 in range(0, t, tile_t):
        t1 = min(t0 + tile_t, t)
        rt0, rt1 = bounds[0][0][t0], bounds[0][1][t1 - 1]
        rel_t = (
            tuple(s - rt0 for s in bounds[0][0][t0:t1]),
            tuple(e - rt0 for e in bounds[0][1][t0:t1]),
        )
        for h0 in range(0, h, tile_h):
            h1 = min(h0 + tile_h, h)
            rh0, rh1 = bounds[1][0][h0], bounds[1][1][h1 - 1]
            rel_h = (
                tuple(s - rh0 for s in bounds[1][0][h0:h1]),
                tuple(e - rh0 for e in bounds[1][1][h0:h1]),
            )
            for w0 in range(0, w, tile_w):
                w1 = min(w0 + tile_w, w)
                rw0, rw1 = bounds[2][0][w0], bounds[2][1][w1 - 1]
                rel_w = (
                    tuple(s - rw0 for s in bounds[2][0][w0:w1]),
                    tuple(e - rw0 for e in bounds[2][1][w0:w1]),
                )
                groups.setdefault((rel_t, rel_h, rel_w), []).append(
                    (
                        (slice(t0, t1), slice(h0, h1), slice(w0, w1)),
                        (slice(rt0, rt1), slice(rh0, rh1), slice(rw0, rw1)),
                    )
                )

    out = torch.empty((batch, t, h, w, nh, hd), device=device, dtype=v.dtype)
    kv_budget = int(getattr(eager_na, "NA_KV_STACK_BUDGET", _DEFAULT_KV_STACK_BUDGET))
    functional = getattr(eager_na, "functional", None) or torch.nn.functional

    for rel, tiles in groups.items():
        mask = eager_na._group_mask(rel, q.dtype, device)
        nq, nk = mask.shape[2], mask.shape[3]
        g_max = min(
            xpu_g_max,
            max(1, kv_budget // max(1, batch * nh * nk * hd * 2)),
        )
        qs0, _ = tiles[0]
        tq = qs0[0].stop - qs0[0].start
        th = qs0[1].stop - qs0[1].start
        tw = qs0[2].stop - qs0[2].start
        for c0 in range(0, len(tiles), g_max):
            chunk = tiles[c0 : c0 + g_max]
            g = len(chunk)
            q_s = torch.stack([q[:, qs[0], qs[1], qs[2]] for qs, _ in chunk])
            k_s = torch.stack([k[:, rs[0], rs[1], rs[2]] for _, rs in chunk])
            v_s = torch.stack([v[:, rs[0], rs[1], rs[2]] for _, rs in chunk])
            q_s = q_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nq, hd)
            k_s = k_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nk, hd)
            v_s = v_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nk, hd)
            o = functional.scaled_dot_product_attention(
                q_s, k_s, v_s, attn_mask=mask, scale=1.0
            )
            o = o.view(g, batch, nh, tq, th, tw, hd).permute(0, 1, 3, 4, 5, 2, 6)
            for i, (qs, _) in enumerate(chunk):
                out[:, qs[0], qs[1], qs[2]] = o[i]

    return out


def _make_na3d_wrapper(original, eager_na):
    @functools.wraps(original)
    def wrapped_na3d(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kernel_size: list[int],
        is_causal: list[bool] | None = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        global _logged_active, _logged_oom

        compiler = getattr(torch, "compiler", None)
        is_compiling = bool(
            compiler is not None and getattr(compiler, "is_compiling", lambda: False)()
        )
        if is_compiling or getattr(getattr(q, "device", None), "type", None) != "xpu":
            return original(q, k, v, kernel_size, is_causal, scale)

        xpu_g_max = _select_xpu_g_max(q.device)
        if xpu_g_max <= 1:
            return original(q, k, v, kernel_size, is_causal, scale)

        if not _logged_active:
            _logged_active = True
            log.info(
                "[OmniXPU] na3d_eager: XPU batched eager active (g_max=%d, shape=%s, kernel=%s)",
                xpu_g_max,
                tuple(q.shape),
                list(kernel_size),
            )

        try:
            return _run_batched_xpu_na3d(
                eager_na, q, k, v, kernel_size, is_causal, scale, xpu_g_max
            )
        except RuntimeError as exc:
            if not _is_oom_error(exc):
                raise
            try:
                if hasattr(torch, "xpu") and hasattr(torch.xpu, "empty_cache"):
                    torch.xpu.empty_cache()
            except Exception:
                pass
            if not _logged_oom:
                _logged_oom = True
                log.warning(
                    "[OmniXPU] na3d_eager: XPU batched eager OOM (%s); falling back to g_max=1",
                    exc,
                )
            return original(q, k, v, kernel_size, is_causal, scale)

    setattr(wrapped_na3d, _PATCH_MARKER, original)
    return wrapped_na3d


def apply() -> tuple[bool, str]:
    """Patch comfy_kitchen eager na3d to enable bounded tile-group batching on XPU."""
    xpu = getattr(torch, "xpu", None)
    cuda = getattr(torch, "cuda", None)
    if xpu is None or not xpu.is_available():
        return False, "XPU runtime unavailable"
    if cuda is not None and cuda.is_available():
        return False, "CUDA runtime active; skipping XPU na3d_eager adapter"

    try:
        eager_na = importlib.import_module("comfy_kitchen.backends.eager.na")
        eager_pkg = importlib.import_module("comfy_kitchen.backends.eager")
    except ImportError as exc:
        return False, f"comfy_kitchen eager backend unavailable: {exc}"

    targets = [eager_na, eager_pkg]
    reg_eager = _get_registry_eager_module()
    if reg_eager is not None and reg_eager not in targets:
        targets.append(reg_eager)

    if all(hasattr(getattr(mod, "na3d", None), _PATCH_MARKER) for mod in targets):
        return True, "already patched"

    existing_wrapped = next(
        (
            getattr(mod, "na3d")
            for mod in targets
            if hasattr(getattr(mod, "na3d", None), _PATCH_MARKER)
        ),
        None,
    )
    if existing_wrapped is not None:
        wrapped = existing_wrapped
    else:
        original = getattr(eager_na, "na3d", None) or getattr(eager_pkg, "na3d", None)
        if original is None:
            return False, "comfy_kitchen eager na3d symbol missing"
        wrapped = _make_na3d_wrapper(original, eager_na)

    for mod in targets:
        setattr(mod, "na3d", wrapped)

    return True, ""


__all__ = ["apply"]
