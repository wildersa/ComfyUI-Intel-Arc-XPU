"""Reclaim DynamicVRAM residency at Windows XPU model boundaries."""

from __future__ import annotations

import functools
import logging
import os
import sys
import time
import types
from contextlib import nullcontext

import torch

try:
    from comfy_aimdo import control as _aimdo_control
    from comfy_aimdo import model_vbar as _aimdo_model_vbar
except ImportError:
    _aimdo_control = None
    _aimdo_model_vbar = None

try:
    from ..config import config as _config
except ImportError:
    try:
        from config import config as _config
    except ImportError:
        _config = None


log = logging.getLogger("ComfyUI-OmniXPU")

_PATCH_MARKER = "__omnixpu_dynamic_vram_boundary_original__"
_GET_FREE_MEMORY_PATCH_MARKER = "__omnixpu_dynamic_vram_get_free_memory_original__"
_MP_GET_FREE_MEMORY_PATCH_MARKER = "__omnixpu_dynamic_vram_mp_get_free_memory_original__"
_STEP_PATCH_MARKER = "__omnixpu_dynamic_vram_step_original__"
_FAULT_PATCH_MARKER = "__omnixpu_dynamic_vram_fault_original__"
_TILED_SCALE_PATCH_MARKER = "__omnixpu_dynamic_vram_tiled_scale_original__"
_FAULT_COUNTS = {"ok": 0, "none": 0}
_TRIM_STATE = {}
_OBSERVED_NON_DYNAMIC_PEAKS = {}
_ACTIVE_NON_DYNAMIC_CONTEXT = {}
_MAX_OBSERVED_PEAK_ENTRIES = 64
_MODEL_MANAGEMENT_REF = None
_DEFAULT_TORCH_CACHE_TRIM_INTERVAL_S = 2.0
_DEFAULT_TORCH_CACHE_TRIM_MIN_MIB = 512
_DEFAULT_TORCH_CACHE_TRIM_FREE_MIB = 4096
_WARNED_FALLBACK = False
_MIB = 1024 ** 2
_VBAR_PAGE_BYTES = 32 * _MIB
_DEFAULT_BUDGET_KNEE_MIB = 6144
_DEFAULT_BUDGET_EXCESS_SCALE = 0.35
_DEFAULT_BUDGET_MAX_FRACTION = 0.40
_DEFAULT_NON_DYNAMIC_MEMORY_SCALE = 1.5
_DEFAULT_NON_DYNAMIC_MAX_TOTAL_FRACTION = 0.85
_FIT_TRIM_ENABLED = (
    _config.dynamic_vram_fit_trim
    if _config is not None
    else os.environ.get("OMNIXPU_DYNAMIC_VRAM_FIT_TRIM", "1") != "0"
)


def _fit_trim_enabled():
    env_val = os.environ.get("OMNIXPU_DYNAMIC_VRAM_FIT_TRIM")
    if env_val is not None:
        return os.environ.get("OMNIXPU_ENABLE", "1") != "0" and env_val != "0"
    return bool(_FIT_TRIM_ENABLED)


def _non_dynamic_trim_enabled():
    env_val = os.environ.get("OMNIXPU_DYNAMIC_VRAM_NON_DYNAMIC_TRIM")
    if env_val is not None:
        return os.environ.get("OMNIXPU_ENABLE", "1") != "0" and env_val != "0"
    if _config is not None:
        return bool(getattr(_config, "dynamic_vram_non_dynamic_trim", True))
    return True


def _inter_tile_reclaim_enabled():
    if not _non_dynamic_trim_enabled():
        return False
    env_val = os.environ.get("OMNIXPU_DYNAMIC_VRAM_INTER_TILE_RECLAIM")
    if env_val is not None:
        return os.environ.get("OMNIXPU_ENABLE", "1") != "0" and env_val != "0"
    if _config is not None:
        return bool(getattr(_config, "dynamic_vram_inter_tile_reclaim", True))
    return True


def _get_free_memory_patch_enabled():
    env_val = os.environ.get("OMNIXPU_DYNAMIC_VRAM_GET_FREE_MEMORY")
    if env_val is not None:
        return os.environ.get("OMNIXPU_ENABLE", "1") != "0" and env_val != "0"
    if _config is not None:
        return bool(getattr(_config, "dynamic_vram_get_free_memory", True))
    return True


def _budget_calibration_enabled():
    env_val = os.environ.get("OMNIXPU_DYNAMIC_VRAM_BUDGET_CALIBRATION")
    if env_val is not None:
        return os.environ.get("OMNIXPU_ENABLE", "1") != "0" and env_val != "0"
    if _config is not None:
        return bool(getattr(_config, "dynamic_vram_budget_calibration", True))
    return True


def _telemetry_enabled():
    env_val = os.environ.get("OMNIXPU_DYNAMIC_VRAM_TELEMETRY")
    if env_val is not None:
        return os.environ.get("OMNIXPU_ENABLE", "1") != "0" and env_val != "0"
    if _config is not None:
        return bool(getattr(_config, "dynamic_vram_telemetry", True))
    return True


def _step_trace_enabled():
    env_val = os.environ.get("OMNIXPU_DYNAMIC_VRAM_STEP_TRACE")
    if env_val is not None:
        return os.environ.get("OMNIXPU_ENABLE", "1") != "0" and env_val == "1"
    if _config is not None:
        return bool(getattr(_config, "dynamic_vram_step_trace", False))
    return False


def _torch_cache_trim_enabled():
    env_val = os.environ.get("OMNIXPU_DYNAMIC_VRAM_TORCH_CACHE_TRIM")
    if env_val is not None:
        return (
            os.environ.get("OMNIXPU_ENABLE", "1") != "0"
            and env_val.strip().lower() not in ("", "0", "false", "no", "off")
        )
    if _config is not None:
        return bool(getattr(_config, "dynamic_vram_torch_cache_trim", True))
    return os.environ.get("OMNIXPU_ENABLE", "1") != "0"


def _env_float(name, default, low, high):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return min(high, max(low, float(raw)))
    except ValueError:
        return default


def _env_int(name, default, low):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(low, int(raw))
    except ValueError:
        return default


def _load_argument(args, kwargs, name, position, default):
    if name in kwargs:
        return kwargs[name]
    if len(args) > position:
        return args[position]
    return default


def _is_dynamic_model(model) -> bool:
    checker = getattr(model, "is_dynamic", None)
    return bool(checker()) if callable(checker) else False


def _device_index(device) -> int:
    idx = getattr(device, "index", None)
    if idx is not None:
        return int(idx)
    if getattr(device, "type", None) == "xpu":
        getter = getattr(torch.xpu, "current_device", None)
        if callable(getter):
            try:
                return int(getter())
            except Exception:
                pass
    return 0


def _same_device(dev_a, dev_b) -> bool:
    if dev_a == dev_b:
        return True
    if getattr(dev_a, "type", None) == "xpu" and getattr(dev_b, "type", None) == "xpu":
        return _device_index(dev_a) == _device_index(dev_b)
    return False


def _expanded_models(models):
    pending = list(models)
    expanded = []
    seen = set()
    while pending:
        model = pending.pop(0)
        identity = id(model)
        if identity in seen:
            continue
        seen.add(identity)
        expanded.append(model)
        patches_fn = getattr(model, "model_patches_models", None)
        if callable(patches_fn):
            pending.extend(patches_fn())
    return expanded


def _extra_memory_target(
    model_management, args, kwargs, *, device=None, non_dynamic=False
):
    memory_required = _load_argument(args, kwargs, "memory_required", 0, 0) or 0
    reserved = model_management.extra_reserved_memory()
    inference = model_management.minimum_inference_memory()
    unscaled = max(inference, memory_required + reserved)
    total_vram = (
        _device_total_vram(model_management, device)
        if device is not None
        else 0
    )
    if not non_dynamic or memory_required <= 0:
        return min(unscaled, total_vram) if total_vram > 0 else unscaled

    scale = _env_float(
        "OMNIXPU_DYNAMIC_VRAM_NON_DYNAMIC_SCALE",
        _DEFAULT_NON_DYNAMIC_MEMORY_SCALE,
        1.0,
        2.5,
    )
    scaled = max(inference, int(memory_required * scale) + reserved)
    if total_vram > 0:
        cap = int(total_vram * _DEFAULT_NON_DYNAMIC_MAX_TOTAL_FRACTION)
        scaled = min(scaled, max(unscaled, cap), total_vram)
    return scaled


def _minimum_memory_target(model_management, args, kwargs, *, device=None):
    minimum_required = _load_argument(
        args, kwargs, "minimum_memory_required", 2, None
    )
    if minimum_required is None:
        return _extra_memory_target(model_management, args, kwargs, device=device)
    reserved = model_management.extra_reserved_memory()
    inference = model_management.minimum_inference_memory()
    target = max(inference, minimum_required + reserved)
    if device is not None:
        total_vram = _device_total_vram(model_management, device)
        if total_vram > 0:
            target = min(target, total_vram)
    return target


def _inference_memory_budget(args, kwargs):
    memory_required = _load_argument(args, kwargs, "memory_required", 0, 0)
    minimum_required = _load_argument(
        args, kwargs, "minimum_memory_required", 2, None
    )
    if minimum_required is None:
        return max(0, int(memory_required))
    return max(0, int(minimum_required))


def _aimdo_budget_scope(models, budget):
    if _aimdo_model_vbar is None:
        return nullcontext(), ()
    budget_scope = getattr(_aimdo_model_vbar, "inference_memory_budget", None)
    if budget_scope is None:
        return nullcontext(), ()

    devices = sorted(
        {
            _device_index(model.load_device)
            for model in models
            if _is_dynamic_model(model)
            and getattr(model.load_device, "type", None) == "xpu"
        }
    )
    if not devices or budget <= 0:
        return nullcontext(), ()
    return budget_scope(budget, devices), tuple(devices)


def _reusable_torch_bytes(device) -> int:
    stats_fn = getattr(torch.xpu, "memory_stats", None)
    if callable(stats_fn):
        try:
            stats = stats_fn(device)
            if isinstance(stats, dict):
                return max(
                    0,
                    int(stats.get("reserved_bytes.all.current", 0))
                    - int(stats.get("active_bytes.all.current", 0)),
                )
        except Exception:
            pass
    try:
        rsv = int(torch.xpu.memory_reserved(device))
        alloc = int(torch.xpu.memory_allocated(device))
        return max(0, rsv - alloc)
    except Exception:
        return 0


def _unpatched_get_free_memory(model_management):
    fn = getattr(model_management, "get_free_memory", None)
    while fn is not None and hasattr(fn, _GET_FREE_MEMORY_PATCH_MARKER):
        fn = getattr(fn, _GET_FREE_MEMORY_PATCH_MARKER)
    return fn


def _is_aimdo_enabled(memory_management=None) -> bool:
    if memory_management is not None:
        return bool(getattr(memory_management, "aimdo_enabled", False))
    mod = sys.modules.get("comfy.memory_management")
    if mod is None:
        try:
            import comfy.memory_management as mod
        except Exception:
            mod = None
    if mod is not None:
        return bool(getattr(mod, "aimdo_enabled", False))
    return _aimdo_model_vbar is not None


def _xpu_actual_free_and_torch(
    model_management, device, original_get_free=None, memory_management=None
) -> tuple[int, int]:
    """Return (free_total, free_torch_cached) for an XPU device without calling patched hooks."""
    try:
        free_device, total_device = torch.xpu.mem_get_info(device)
        mem_free_torch = _reusable_torch_bytes(device)
        total_free = max(0, int(free_device) + mem_free_torch)
        if total_device and int(total_device) > 0:
            total_free = min(total_free, int(total_device))
        return total_free, mem_free_torch
    except Exception:
        pass

    mem_free_torch = _reusable_torch_bytes(device)
    unpatched = original_get_free or _unpatched_get_free_memory(model_management)
    base_free = int(unpatched(device)) if callable(unpatched) else 0
    total_vram = _device_total_vram(model_management, device)
    if (
        _is_aimdo_enabled(memory_management)
        and _aimdo_model_vbar is not None
        and hasattr(_aimdo_model_vbar, "vbars_analyze")
    ):
        try:
            dev_idx = _device_index(device)
            aimdo_used = max(0, int(_aimdo_model_vbar.vbars_analyze(dev_idx)))
            free_est = max(0, base_free - aimdo_used)
            if total_vram > 0:
                free_est = min(free_est, total_vram)
            return free_est, mem_free_torch
        except Exception:
            pass
    free_est = max(0, base_free)
    if total_vram > 0:
        free_est = min(free_est, total_vram)
    return free_est, mem_free_torch


def _actual_free_vram(model_management, device) -> int:
    """Return real free XPU VRAM including AIMDO VBAR allocations."""
    if getattr(device, "type", None) != "xpu":
        unpatched = _unpatched_get_free_memory(model_management)
        return int(unpatched(device)) if callable(unpatched) else 0
    total_free, _ = _xpu_actual_free_and_torch(model_management, device)
    return total_free


def _peak_torch_growth(device) -> int:
    try:
        rsv = int(torch.xpu.memory_reserved(device))
        peak = int(torch.xpu.max_memory_reserved(device))
        return max(0, peak - rsv)
    except Exception:
        return 0


def _model_unloaded_bytes(model, device) -> int:
    size_fn = getattr(model, "model_size", None)
    loaded_fn = getattr(model, "loaded_size", None)
    size = max(0, int(size_fn())) if callable(size_fn) else 0
    loaded = max(0, int(loaded_fn())) if callable(loaded_fn) else 0
    curr_dev_fn = getattr(model, "current_loaded_device", None)
    if callable(curr_dev_fn):
        curr_dev = curr_dev_fn()
        if (
            curr_dev is None
            or getattr(curr_dev, "type", None) == "cpu"
            or not _same_device(curr_dev, device)
        ):
            return size
    return max(0, size - loaded)


def _unloaded_model_bytes(models, device) -> tuple[int, int]:
    """Return (dynamic_bytes, non_dynamic_bytes) still unloaded on device."""
    unloaded_dynamic = 0
    unloaded_static = 0
    for model in models:
        if not _same_device(getattr(model, "load_device", None), device):
            continue
        unloaded = _model_unloaded_bytes(model, device)
        if _is_dynamic_model(model):
            unloaded_dynamic += unloaded
        else:
            unloaded_static += unloaded
    return unloaded_dynamic, unloaded_static


def _unloaded_dynamic_bytes(models, device) -> int:
    unloaded_dynamic, _ = _unloaded_model_bytes(models, device)
    return unloaded_dynamic


def _flush_xpu_cache(model_management, device) -> bool:
    """Release dead PyTorch XPU allocator segments for static VRAM consumers."""
    flushed = False
    soft_empty = getattr(model_management, "soft_empty_cache", None)
    if callable(soft_empty):
        soft_empty()
        flushed = True
    else:
        empty_cache = getattr(torch.xpu, "empty_cache", None)
        if callable(empty_cache):
            sync = getattr(torch.xpu, "synchronize", None)
            if callable(sync):
                sync()
            empty_cache()
            flushed = True
    if not flushed:
        return False
    publish = getattr(_aimdo_control, "publish_torch_cached_bytes", None)
    if callable(publish):
        try:
            publish(_device_index(device), _reusable_torch_bytes(device))
        except Exception:
            pass
    return True


def _trim_pass(model_management, device, target, requested, include_requested):
    reclaimed = 0
    trimmed = 0
    for loaded in reversed(model_management.current_loaded_models):
        model = loaded.model
        if (
            model is None
            or not _same_device(loaded.device, device)
            or loaded.is_dead()
            or not _is_dynamic_model(model)
        ):
            continue
        is_requested = id(model) in requested
        if is_requested != include_requested:
            continue

        shortfall = target - _actual_free_vram(model_management, device)
        if shortfall <= 0:
            break
        if model.loaded_size() <= 0:
            continue

        freed = int(model.partially_unload(model.offload_device, shortfall))
        if freed > 0:
            reclaimed += freed
            trimmed += 1
    return reclaimed, trimmed


def _xpu_device(dev_idx: int):
    dev_cls = getattr(torch, "device", None)
    if callable(dev_cls):
        try:
            return dev_cls("xpu", dev_idx)
        except Exception:
            pass
    return types.SimpleNamespace(type="xpu", index=dev_idx)


def _sample_max_memory_reserved(device_or_idx) -> int:
    fn = getattr(torch.xpu, "max_memory_reserved", None)
    if not callable(fn):
        return 0
    try:
        return max(0, int(fn(device_or_idx)))
    except Exception:
        return 0


def _model_class_label(model) -> str:
    inner = getattr(model, "model", None)
    target = inner if inner is not None else model
    return type(target).__name__


def _non_dynamic_peak_key(dev_idx: int, non_dyn_models, memory_required: int):
    label = "+".join(_model_class_label(m) for m in non_dyn_models) or "unknown"
    bucket_mib = int(round(max(0, int(memory_required)) / (64 * _MIB))) * 64
    return (int(dev_idx), label, bucket_mib)


def _record_observed_peak(key, peak_bytes: int) -> int:
    peak_bytes = max(0, int(peak_bytes))
    if peak_bytes <= 0:
        return _OBSERVED_NON_DYNAMIC_PEAKS.get(key, 0)
    prev = _OBSERVED_NON_DYNAMIC_PEAKS.get(key, 0)
    new_val = max(prev, peak_bytes)
    if key in _OBSERVED_NON_DYNAMIC_PEAKS:
        _OBSERVED_NON_DYNAMIC_PEAKS.pop(key, None)
    elif len(_OBSERVED_NON_DYNAMIC_PEAKS) >= _MAX_OBSERVED_PEAK_ENTRIES:
        oldest_key = next(iter(_OBSERVED_NON_DYNAMIC_PEAKS))
        _OBSERVED_NON_DYNAMIC_PEAKS.pop(oldest_key, None)
    _OBSERVED_NON_DYNAMIC_PEAKS[key] = new_val
    return new_val


def _finalize_active_non_dynamic_context(dev_idx: int) -> None:
    ctx = _ACTIVE_NON_DYNAMIC_CONTEXT.pop(dev_idx, None)
    if ctx is None:
        return
    cur_max_rsv = _sample_max_memory_reserved(dev_idx)
    if cur_max_rsv > ctx["pre_max_rsv"]:
        _record_observed_peak(ctx["key"], cur_max_rsv)


def _inter_tile_safe_floor(total_vram: int) -> int:
    if total_vram > 0:
        return max(512 * _MIB, min(3072 * _MIB, int(total_vram * 0.12)))
    return 1024 * _MIB


def _resolve_model_management(active_ctx=None):
    if active_ctx is not None:
        mm = active_ctx.get("model_management")
        if mm is not None:
            return mm
    if _MODEL_MANAGEMENT_REF is not None:
        return _MODEL_MANAGEMENT_REF
    mod = sys.modules.get("comfy.model_management")
    if mod is not None:
        return mod
    try:
        import comfy.model_management as mod

        return mod
    except Exception:
        return None


def _trim_dynamic_boundary(
    model_management,
    models,
    target,
    extra_target=None,
    *,
    load_args=None,
    load_kwargs=None,
):
    global _MODEL_MANAGEMENT_REF
    _MODEL_MANAGEMENT_REF = model_management
    if not models:
        return

    all_dynamic = all(_is_dynamic_model(model) for model in models)
    if not all_dynamic and not _non_dynamic_trim_enabled():
        return

    requested = {id(model) for model in models}
    fit_trim = _fit_trim_enabled()
    seen_xpu_indices = set()
    for model in models:
        device = getattr(model, "load_device", None)
        if getattr(device, "type", None) != "xpu":
            continue
        dev_idx = _device_index(device)
        if dev_idx in seen_xpu_indices:
            continue
        seen_xpu_indices.add(dev_idx)

        _finalize_active_non_dynamic_context(dev_idx)

        device_models = [
            m
            for m in models
            if _same_device(getattr(m, "load_device", None), device)
        ]
        device_has_non_dynamic = any(not _is_dynamic_model(m) for m in device_models)
        total_vram = _device_total_vram(model_management, device)
        base_target = int(target)
        if total_vram > 0:
            base_target = min(base_target, total_vram)
        unscaled_base = int(target)
        if device_has_non_dynamic:
            dev_extra = extra_target
            if load_args is not None:
                dev_extra = _extra_memory_target(
                    model_management,
                    load_args,
                    load_kwargs or {},
                    device=device,
                    non_dynamic=True,
                )
                unscaled_base = max(
                    unscaled_base,
                    _extra_memory_target(
                        model_management,
                        load_args,
                        load_kwargs or {},
                        device=device,
                        non_dynamic=False,
                    ),
                )
            if total_vram > 0:
                unscaled_base = min(unscaled_base, total_vram)
            if dev_extra is not None:
                base_target = max(base_target, int(dev_extra))

            cur_max_rsv = _sample_max_memory_reserved(dev_idx)
            non_dyn_models = [m for m in device_models if not _is_dynamic_model(m)]
            raw_mem_req = int(
                _load_argument(
                    load_args or (), load_kwargs or {}, "memory_required", 0, 0
                )
                or 0
            )
            peak_key = _non_dynamic_peak_key(dev_idx, non_dyn_models, raw_mem_req)
            observed_peak = _OBSERVED_NON_DYNAMIC_PEAKS.get(peak_key, 0)
            if observed_peak > 0:
                reserved = int(model_management.extra_reserved_memory())
                observed_target = int(observed_peak) + reserved
                if total_vram > 0:
                    obs_cap = max(
                        unscaled_base,
                        int(total_vram * _DEFAULT_NON_DYNAMIC_MAX_TOTAL_FRACTION),
                    )
                    observed_target = min(observed_target, obs_cap, total_vram)
                if observed_target > base_target:
                    base_target = observed_target
                    log.info(
                        "[OmniXPU] DynamicVRAM non-dynamic observed peak: "
                        "model=%s key_mib=%d peak_rsv=%.1fMiB -> base_target=%.1fMiB",
                        peak_key[1],
                        peak_key[2],
                        observed_peak / _MIB,
                        base_target / _MIB,
                    )

            _ACTIVE_NON_DYNAMIC_CONTEXT[dev_idx] = {
                "key": peak_key,
                "pre_max_rsv": cur_max_rsv,
                "model_ids": requested,
                "device": device,
                "model_management": model_management,
            }

        unloaded_dynamic, unloaded_static = _unloaded_model_bytes(device_models, device)
        # Mirror ComfyUI load_models_gpu (1.1x unloaded static weights + extra_mem).
        static_budget = (unloaded_static * 11 + 9) // 10 if unloaded_static > 0 else 0
        inactive_target = base_target + static_budget
        if device_has_non_dynamic and total_vram > 0:
            non_dyn_cap = max(
                min(unscaled_base + static_budget, total_vram),
                int(total_vram * _DEFAULT_NON_DYNAMIC_MAX_TOTAL_FRACTION),
            )
            inactive_target = min(inactive_target, non_dyn_cap)
        if fit_trim and unloaded_dynamic > 0:
            inactive_target = max(
                inactive_target,
                base_target + static_budget + unloaded_dynamic + _peak_torch_growth(device),
            )
        active_target = base_target + static_budget
        if total_vram > 0:
            inactive_target = min(inactive_target, total_vram)
            active_target = min(active_target, total_vram)

        free_before = _actual_free_vram(model_management, device)
        reclaimed = 0
        inactive_bytes = 0
        inactive_models = 0
        active_bytes = 0
        active_models = 0

        if free_before < inactive_target:
            inactive_bytes, inactive_models = _trim_pass(
                model_management, device, inactive_target, requested, False
            )
            active_bytes, active_models = _trim_pass(
                model_management, device, active_target, requested, True
            )
            reclaimed = inactive_bytes + active_bytes

        cache_flushed = False
        if device_has_non_dynamic and (
            reclaimed > 0
            or _reusable_torch_bytes(device) > max(64 * _MIB, free_before * 0.25)
        ):
            cache_flushed = _flush_xpu_cache(model_management, device)

        if free_before >= inactive_target and not cache_flushed:
            continue

        free_after = _actual_free_vram(model_management, device)
        extra_parts = []
        if unloaded_static > 0:
            extra_parts.append(f"static_unloaded={unloaded_static / _MIB:.1f}MiB")
        if cache_flushed:
            extra_parts.append("cache_flushed=1")
        extra_text = (" " + " ".join(extra_parts)) if extra_parts else ""

        if inactive_models > 0 or active_models > 0 or cache_flushed:
            log.info(
                "[OmniXPU] DynamicVRAM boundary trim: device=%s target=%.1fMiB "
                "free=%.1f->%.1fMiB reclaimed=%.1fMiB models=%d inactive/%d active%s",
                device,
                inactive_target / _MIB,
                free_before / _MIB,
                free_after / _MIB,
                reclaimed / _MIB,
                inactive_models,
                active_models,
                extra_text,
            )
        else:
            log.debug(
                "[OmniXPU] DynamicVRAM boundary trim: device=%s target=%.1fMiB "
                "free=%.1f->%.1fMiB reclaimed=0.0MiB models=0 inactive/0 active%s",
                device,
                inactive_target / _MIB,
                free_before / _MIB,
                free_after / _MIB,
                extra_text,
            )


def _is_flash_attention_active(model_management):
    backend = os.environ.get("OMNI_ATTN_BACKEND", "").strip().lower()
    if backend in ("cute", "auto"):
        return True
    checker = getattr(model_management, "pytorch_attention_flash_attention", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return False
    return False


def _device_total_vram(model_management, device):
    try:
        _, total = torch.xpu.mem_get_info(device)
        if total and total > 0:
            return int(total)
    except Exception:
        pass
    getter = getattr(model_management, "get_total_memory", None)
    if callable(getter):
        try:
            return max(0, int(getter(device)))
        except Exception:
            return 0
    return 0


def _with_minimum_memory_required(args, kwargs, minimum_value):
    new_args = list(args)
    new_kwargs = dict(kwargs)
    if "minimum_memory_required" in new_kwargs:
        new_kwargs["minimum_memory_required"] = minimum_value
    elif len(new_args) > 2:
        new_args[2] = minimum_value
    else:
        new_kwargs["minimum_memory_required"] = minimum_value
    return tuple(new_args), new_kwargs


def _calibration_knee_and_cap(model_management, xpu_devices, max_fraction):
    raw_knee_env = os.environ.get("OMNIXPU_AIMDO_BUDGET_KNEE_MIB")
    knee_bytes = (
        _env_int("OMNIXPU_AIMDO_BUDGET_KNEE_MIB", _DEFAULT_BUDGET_KNEE_MIB, 512)
        * _MIB
    )
    totals = [
        t
        for t in (_device_total_vram(model_management, dev) for dev in xpu_devices)
        if t > 0
    ]
    if not totals:
        return knee_bytes, None
    min_total = min(totals)
    cap = int(min_total * max_fraction)
    if raw_knee_env is None:
        effective_knee = min(knee_bytes, int(min_total * 0.25), cap)
    else:
        effective_knee = min(knee_bytes, cap)
    return effective_knee, cap


def _calibrate_inference_budget(model_management, expanded, args, kwargs):
    raw_min_arg = _load_argument(args, kwargs, "minimum_memory_required", 2, None)
    raw_budget = _inference_memory_budget(args, kwargs)
    if (
        not _budget_calibration_enabled()
        or raw_min_arg is None
        or isinstance(raw_min_arg, bool)
        or not expanded
        or not all(_is_dynamic_model(model) for model in expanded)
        or not _is_flash_attention_active(model_management)
    ):
        return args, kwargs, raw_budget, raw_budget, False

    xpu_devices = [
        model.load_device
        for model in expanded
        if getattr(model.load_device, "type", None) == "xpu"
    ]
    if not xpu_devices:
        return args, kwargs, raw_budget, raw_budget, False

    max_fraction = _env_float(
        "OMNIXPU_AIMDO_BUDGET_MAX_FRACTION", _DEFAULT_BUDGET_MAX_FRACTION, 0.1, 1.0
    )
    effective_knee, cap = _calibration_knee_and_cap(
        model_management, xpu_devices, max_fraction
    )
    if raw_budget <= effective_knee:
        return args, kwargs, raw_budget, raw_budget, False

    excess_scale = _env_float(
        "OMNIXPU_AIMDO_BUDGET_EXCESS_SCALE", _DEFAULT_BUDGET_EXCESS_SCALE, 0.0, 1.0
    )
    calibrated = effective_knee + int((raw_budget - effective_knee) * excess_scale)
    if cap is not None:
        calibrated = min(calibrated, cap)

    calibrated = max(effective_knee, min(raw_budget, calibrated))
    if calibrated >= raw_budget:
        return args, kwargs, raw_budget, raw_budget, False

    cal_args, cal_kwargs = _with_minimum_memory_required(args, kwargs, calibrated)
    return cal_args, cal_kwargs, raw_budget, calibrated, True


def _xpu_vram_telemetry(device):
    stats = {}
    try:
        stats["alloc"] = int(torch.xpu.memory_allocated(device))
        stats["max_alloc"] = int(torch.xpu.max_memory_allocated(device))
        stats["rsv"] = int(torch.xpu.memory_reserved(device))
        stats["max_rsv"] = int(torch.xpu.max_memory_reserved(device))
        free_dev, total_dev = torch.xpu.mem_get_info(device)
        stats["free"] = int(free_dev)
        stats["total"] = int(total_dev)
    except Exception:
        return None
    if _aimdo_model_vbar is not None and hasattr(_aimdo_model_vbar, "vbars_analyze"):
        try:
            dev_idx = _device_index(device)
            stats["vbar"] = int(_aimdo_model_vbar.vbars_analyze(dev_idx))
        except Exception:
            stats["vbar"] = -1
    return stats


def _log_boundary_telemetry(expanded, raw_budget, calibrated_budget):
    if not _telemetry_enabled() or calibrated_budget < 200 * _MIB:
        return
    seen_idx = set()
    for model in expanded:
        device = model.load_device
        if getattr(device, "type", None) != "xpu":
            continue
        dev_idx = _device_index(device)
        if dev_idx in seen_idx:
            continue
        seen_idx.add(dev_idx)
        t = _xpu_vram_telemetry(device)
        if not t:
            continue
        vbar_text = f"{t['vbar'] / _MIB:.1f}MiB" if t.get("vbar", -1) >= 0 else "n/a"
        log.info(
            "[OmniXPU] DynamicVRAM telemetry: device=%s raw_min=%.1fMiB "
            "cal_min=%.1fMiB torch_alloc=%.1fMiB torch_max_alloc=%.1fMiB "
            "torch_rsv=%.1fMiB torch_max_rsv=%.1fMiB vbar=%s free=%.1f/%.1fMiB",
            device,
            raw_budget / _MIB,
            calibrated_budget / _MIB,
            t["alloc"] / _MIB,
            t["max_alloc"] / _MIB,
            t["rsv"] / _MIB,
            t["max_rsv"] / _MIB,
            vbar_text,
            t["free"] / _MIB,
            t["total"] / _MIB,
        )


def _prepare_boundary(model_management, models, args, kwargs):
    global _WARNED_FALLBACK
    try:
        expanded = _expanded_models(models)
        cal_args, cal_kwargs, raw_budget, inference_budget, was_calibrated = (
            _calibrate_inference_budget(model_management, expanded, args, kwargs)
        )
        xpu_devs = [
            m.load_device
            for m in expanded
            if getattr(getattr(m, "load_device", None), "type", None) == "xpu"
        ]
        if was_calibrated:
            max_fraction = _env_float(
                "OMNIXPU_AIMDO_BUDGET_MAX_FRACTION",
                _DEFAULT_BUDGET_MAX_FRACTION,
                0.1,
                1.0,
            )
            effective_knee, _ = _calibration_knee_and_cap(
                model_management, xpu_devs, max_fraction
            )
            scale = _env_float(
                "OMNIXPU_AIMDO_BUDGET_EXCESS_SCALE", _DEFAULT_BUDGET_EXCESS_SCALE, 0.0, 1.0
            )
            log.info(
                "[OmniXPU] DynamicVRAM budget calibrated (flash): "
                "raw=%.1fMiB -> %.1fMiB (knee=%dMiB scale=%.2f)",
                raw_budget / _MIB,
                inference_budget / _MIB,
                effective_knee // _MIB,
                scale,
            )
        _log_boundary_telemetry(expanded, raw_budget, inference_budget)
        single_xpu_dev = (
            xpu_devs[0]
            if xpu_devs and len({_device_index(d) for d in xpu_devs}) == 1
            else None
        )
        target = _minimum_memory_target(
            model_management, cal_args, cal_kwargs, device=single_xpu_dev
        )
        _trim_dynamic_boundary(
            model_management,
            expanded,
            target,
            load_args=cal_args,
            load_kwargs=cal_kwargs,
        )
        budget_scope, budget_devices = _aimdo_budget_scope(
            expanded, inference_budget
        )
        if budget_devices:
            log.info(
                "[OmniXPU] AIMDO inference budget: devices=%s "
                "minimum=%.1fMiB",
                ",".join(map(str, budget_devices)),
                inference_budget / (1024 ** 2),
            )
        return budget_scope, cal_args, cal_kwargs
    except Exception as exc:
        if not _WARNED_FALLBACK:
            _WARNED_FALLBACK = True
            log.warning(
                "[OmniXPU] DynamicVRAM boundary trim conflict (%s); "
                "falling back to stock ComfyUI loader",
                exc,
            )
        return nullcontext(), args, kwargs


def _vbar_for_patcher(patcher):
    model = getattr(patcher, "model", None)
    vbars = getattr(model, "dynamic_vbars", None)
    device = getattr(patcher, "load_device", None)
    if not isinstance(vbars, dict) or device is None:
        return None
    return vbars.get(device)


def _fault_device_index(alloc):
    if isinstance(alloc, tuple) and alloc:
        vbar = alloc[0]
        dev = getattr(vbar, "device", 0)
        try:
            return int(dev)
        except (TypeError, ValueError):
            return 0
    return 0


def _trim_state_for_device(dev_idx: int) -> dict:
    return _TRIM_STATE.setdefault(
        dev_idx,
        {"last_attempt": 0.0, "last_unmap": 0, "calls": 0, "freed": 0},
    )


def _maybe_trim_torch_cache(alloc):
    if not _torch_cache_trim_enabled():
        return 0
    dev_idx = _fault_device_index(alloc)
    state = _trim_state_for_device(dev_idx)
    now = time.monotonic()
    interval = _env_float(
        "OMNIXPU_TORCH_CACHE_TRIM_INTERVAL_S",
        _DEFAULT_TORCH_CACHE_TRIM_INTERVAL_S,
        0.2,
        60.0,
    )
    if (now - state["last_attempt"]) < interval:
        return 0
    state["last_attempt"] = now

    try:
        rsv = int(torch.xpu.memory_reserved(dev_idx))
        alloc_b = int(torch.xpu.memory_allocated(dev_idx))
    except Exception:
        return 0

    reusable = max(0, rsv - alloc_b)
    min_reusable = _env_int(
        "OMNIXPU_TORCH_CACHE_TRIM_MIN_MIB", _DEFAULT_TORCH_CACHE_TRIM_MIN_MIB, 64
    ) * _MIB
    if reusable < min_reusable:
        return 0

    pressure = False
    cur_vmm = _xpu_vmm_snapshot()
    if cur_vmm is not None:
        unmap_bytes = int(cur_vmm.get("unmap_bytes", 0))
        if unmap_bytes > state["last_unmap"]:
            pressure = True
        state["last_unmap"] = unmap_bytes

    if not pressure:
        env_free_mib = os.environ.get("OMNIXPU_TORCH_CACHE_TRIM_FREE_MIB")
        configured_free_bytes = (
            _env_int(
                "OMNIXPU_TORCH_CACHE_TRIM_FREE_MIB",
                _DEFAULT_TORCH_CACHE_TRIM_FREE_MIB,
                256,
            )
            * _MIB
        )
        try:
            free_bytes, total_vram = torch.xpu.mem_get_info(dev_idx)
            free_bytes = int(free_bytes)
            total_vram = int(total_vram)
        except Exception:
            free_bytes = configured_free_bytes + 1
            total_vram = 0

        free_threshold = configured_free_bytes
        if total_vram > 0:
            if env_free_mib is not None:
                free_threshold = min(configured_free_bytes, total_vram)
            else:
                free_threshold = min(
                    total_vram,
                    max(
                        256 * _MIB,
                        min(configured_free_bytes, int(total_vram * 0.15)),
                    ),
                )
        pressure = free_bytes <= free_threshold

    if not pressure:
        return 0

    try:
        torch.xpu.empty_cache()
        after_rsv = int(torch.xpu.memory_reserved(dev_idx))
        after_alloc = int(torch.xpu.memory_allocated(dev_idx))
        publish = getattr(_aimdo_control, "publish_torch_cached_bytes", None)
        if callable(publish):
            publish(dev_idx, max(0, after_rsv - after_alloc))
    except Exception:
        return 0
    finally:
        state["last_attempt"] = time.monotonic()

    freed = max(0, rsv - after_rsv)
    state["calls"] += 1
    state["freed"] += freed
    return freed


def _ensure_fault_tracer():
    if _aimdo_model_vbar is None:
        return False
    original = getattr(_aimdo_model_vbar, "vbar_fault", None)
    if original is None:
        return False
    if hasattr(original, _FAULT_PATCH_MARKER):
        return True

    @functools.wraps(original)
    def traced_fault(alloc):
        _maybe_trim_torch_cache(alloc)
        res = original(alloc)
        _FAULT_COUNTS["none" if res is None else "ok"] += 1
        return res

    setattr(traced_fault, _FAULT_PATCH_MARKER, original)
    _aimdo_model_vbar.vbar_fault = traced_fault
    return True


def _xpu_vmm_snapshot():
    fn = getattr(_aimdo_control, "get_xpu_vmm_stats", None) if _aimdo_control is not None else None
    if not callable(fn):
        return None
    try:
        return fn() or None
    except Exception:
        return None


def _xpu_ur_hook_snapshot():
    fn = getattr(_aimdo_control, "get_xpu_ur_hook_stats", None) if _aimdo_control is not None else None
    if not callable(fn):
        return None
    try:
        return fn() or None
    except Exception:
        return None


def _vbar_metrics(vbar):
    if vbar is None:
        return None
    try:
        loaded = int(vbar.loaded_size())
        wm_pages = int(vbar.get_watermark())
        nr_pages = int(vbar.get_nr_pages())
        max_size = int(getattr(vbar, "max_size", 0) or 0)
        return loaded, wm_pages, nr_pages, max_size
    except Exception:
        return None


def _wrap_step_callback(patcher, callback):
    if patcher is None or not callable(getattr(patcher, "is_dynamic", None)) or not patcher.is_dynamic():
        return callback
    device = getattr(patcher, "load_device", None)
    if getattr(device, "type", None) != "xpu":
        return callback
    if not _step_trace_enabled():
        return callback

    try:
        model_bytes = int(patcher.model_size())
    except Exception:
        model_bytes = 0
    if model_bytes < 1024 * _MIB:
        return callback

    vbar = _vbar_for_patcher(patcher)
    if vbar is None:
        return callback

    dev_idx = _device_index(device)
    trim_state = _trim_state_for_device(dev_idx)
    fault_traced = _ensure_fault_tracer()
    t_last = [time.perf_counter()]
    prev_faults = [dict(_FAULT_COUNTS)]
    prev_trim = {"calls": trim_state["calls"], "freed": trim_state["freed"]}
    prev_vmm = [_xpu_vmm_snapshot()]
    prev_hook = [_xpu_ur_hook_snapshot()]

    def _emit_step(tag, dt):
        t = _xpu_vram_telemetry(device)
        if not t:
            return
        metrics = _vbar_metrics(vbar)
        if metrics is not None:
            loaded, wm_pages, nr_pages, max_size = metrics
            cap_desc = f"/{max_size / _MIB:.0f}MiB" if max_size > 0 else ""
            vbar_desc = f"{loaded / _MIB:.1f}{cap_desc}(wm={wm_pages}/{nr_pages}p)"
        else:
            vbar_desc = "n/a"

        extra_parts = []
        if fault_traced:
            cur_f = dict(_FAULT_COUNTS)
            df_ok = cur_f["ok"] - prev_faults[0]["ok"]
            df_none = cur_f["none"] - prev_faults[0]["none"]
            prev_faults[0] = cur_f
            extra_parts.append(f"fault=ok:{df_ok}/none:{df_none}")

        cur_vmm = _xpu_vmm_snapshot()
        if cur_vmm is not None and prev_vmm[0] is not None:
            pv = prev_vmm[0]
            map_mib = (cur_vmm.get("map_bytes", 0) - pv.get("map_bytes", 0)) / _MIB
            unmap_mib = (cur_vmm.get("unmap_bytes", 0) - pv.get("unmap_bytes", 0)) / _MIB
            rel_calls = cur_vmm.get("physical_release_calls", 0) - pv.get("physical_release_calls", 0)
            fb_calls = (
                cur_vmm.get("small_vbar_copy_fallback_calls", 0)
                - pv.get("small_vbar_copy_fallback_calls", 0)
            )
            extra_parts.append(
                f"vmm=+{map_mib:.0f}/-{unmap_mib:.0f}MiB(rel={rel_calls},fb={fb_calls})"
            )
        prev_vmm[0] = cur_vmm

        cur_hook = _xpu_ur_hook_snapshot()
        if cur_hook is not None and prev_hook[0] is not None:
            ph = prev_hook[0]
            syn_oom = cur_hook.get("synthetic_oom_calls", 0) - ph.get("synthetic_oom_calls", 0)
            rt_oom = cur_hook.get("runtime_oom_calls", 0) - ph.get("runtime_oom_calls", 0)
            retry_ev = cur_hook.get("retry_eviction_calls", 0) - ph.get("retry_eviction_calls", 0)
            dir_p = cur_hook.get("direct_pressure_calls", 0) - ph.get("direct_pressure_calls", 0)
            dir_mib = (cur_hook.get("direct_pressure_bytes", 0) - ph.get("direct_pressure_bytes", 0)) / _MIB
            skip = cur_hook.get("cache_lever_skipped_calls", 0) - ph.get("cache_lever_skipped_calls", 0)
            extra_parts.append(
                f"hook=dir:{dir_p}({dir_mib:.0f}MiB),syn_oom:{syn_oom},rt_oom:{rt_oom},retry:{retry_ev},skip:{skip}"
            )
        prev_hook[0] = cur_hook
        if trim_state["calls"] > prev_trim["calls"] or _torch_cache_trim_enabled():
            tc = trim_state["calls"] - prev_trim["calls"]
            tf_mib = (trim_state["freed"] - prev_trim["freed"]) / _MIB
            prev_trim["calls"] = trim_state["calls"]
            prev_trim["freed"] = trim_state["freed"]
            extra_parts.append(f"trim={tc}({tf_mib:.0f}MiB)")
        extra = (" " + " ".join(extra_parts)) if extra_parts else ""

        log.info(
            "[OmniXPU] DynamicVRAM %s dt=%.2fs vbar=%s "
            "torch_alloc=%.1fMiB torch_max_alloc=%.1fMiB "
            "torch_rsv=%.1fMiB torch_max_rsv=%.1fMiB free=%.1fMiB%s",
            tag,
            dt,
            vbar_desc,
            t["alloc"] / _MIB,
            t["max_alloc"] / _MIB,
            t["rsv"] / _MIB,
            t["max_rsv"] / _MIB,
            t["free"] / _MIB,
            extra,
        )

    _emit_step("step=init", 0.0)

    def wrapped(step, x0, x, total_steps):
        result = callback(step, x0, x, total_steps) if callback is not None else None
        now = time.perf_counter()
        dt = now - t_last[0]
        t_last[0] = now
        _emit_step(f"step={int(step) + 1}/{total_steps}", dt)
        return result

    return wrapped


def _patch_cfg_guider_steps():
    try:
        import comfy.samplers
    except Exception:
        return False

    cfg_guider = getattr(comfy.samplers, "CFGGuider", None)
    original = getattr(cfg_guider, "inner_sample", None)
    if original is None or hasattr(original, _STEP_PATCH_MARKER):
        return False

    @functools.wraps(original)
    def stepped(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
        wrapped_cb = _wrap_step_callback(getattr(self, "model_patcher", None), callback)
        return original(
            self,
            noise,
            latent_image,
            device,
            sampler,
            sigmas,
            denoise_mask,
            wrapped_cb,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )

    setattr(stepped, _STEP_PATCH_MARKER, original)
    cfg_guider.inner_sample = stepped
    return True


def _patch_get_free_memory(model_management, memory_management=None):
    original = getattr(model_management, "get_free_memory", None)
    if original is None or hasattr(original, _GET_FREE_MEMORY_PATCH_MARKER):
        return False

    @functools.wraps(original)
    def xpu_get_free_memory(dev=None, torch_free_too=False):
        if not _get_free_memory_patch_enabled():
            return (
                original(dev, torch_free_too=True)
                if torch_free_too
                else original(dev)
            )
        target_dev = dev
        if target_dev is None:
            getter = getattr(model_management, "get_torch_device", None)
            if callable(getter):
                target_dev = getter()
        if getattr(target_dev, "type", None) != "xpu":
            return (
                original(dev, torch_free_too=True)
                if torch_free_too
                else original(dev)
            )
        total_free, torch_free = _xpu_actual_free_and_torch(
            model_management,
            target_dev,
            original_get_free=original,
            memory_management=memory_management,
        )
        total_free = max(0, total_free)
        torch_free = max(0, torch_free)
        total_vram = _device_total_vram(model_management, target_dev)
        if total_vram > 0:
            total_free = min(total_free, total_vram)
            torch_free = min(torch_free, total_free)
        if torch_free_too:
            return (total_free, torch_free)
        return total_free

    setattr(xpu_get_free_memory, _GET_FREE_MEMORY_PATCH_MARKER, original)
    model_management.get_free_memory = xpu_get_free_memory
    return True


def _patch_model_patcher_get_free_memory(
    model_patcher_cls, model_management, memory_management=None
):
    original = getattr(model_patcher_cls, "get_free_memory", None)
    if original is None or hasattr(original, _MP_GET_FREE_MEMORY_PATCH_MARKER):
        return False

    @functools.wraps(original)
    def xpu_mp_get_free_memory(self, device):
        if not _get_free_memory_patch_enabled():
            return original(self, device)

        dev_type = getattr(device, "type", None)
        if dev_type == "cpu":
            return int(model_management.get_free_memory(device))
        if dev_type != "xpu":
            return original(self, device)

        aimdo_mem = 0
        aimdo_enabled = bool(getattr(memory_management, "aimdo_enabled", False))
        if (
            aimdo_enabled
            and _aimdo_model_vbar is not None
            and hasattr(_aimdo_model_vbar, "vbars_analyze")
        ):
            try:
                aimdo_mem = max(
                    0, int(_aimdo_model_vbar.vbars_analyze(_device_index(device)))
                )
            except Exception:
                aimdo_mem = 0
        result = max(0, int(model_management.get_free_memory(device)) + aimdo_mem)
        total_vram = _device_total_vram(model_management, device)
        if total_vram > 0:
            result = min(result, total_vram)
        return result

    setattr(xpu_mp_get_free_memory, _MP_GET_FREE_MEMORY_PATCH_MARKER, original)
    model_patcher_cls.get_free_memory = xpu_mp_get_free_memory
    return True


def _after_tile_step(f_args):
    try:
        active_ctx = None
        if _ACTIVE_NON_DYNAMIC_CONTEXT:
            if (
                f_args
                and getattr(getattr(f_args[0], "device", None), "type", None) == "xpu"
            ):
                active_ctx = _ACTIVE_NON_DYNAMIC_CONTEXT.get(
                    _device_index(f_args[0].device)
                )
            if active_ctx is None:
                active_ctx = next(iter(_ACTIVE_NON_DYNAMIC_CONTEXT.values()), None)

        if active_ctx is not None:
            device = active_ctx["device"]
            dev_idx = _device_index(device)
        else:
            dev_idx = (
                _device_index(f_args[0].device)
                if f_args
                and getattr(getattr(f_args[0], "device", None), "type", None) == "xpu"
                else 0
            )
            device = _xpu_device(dev_idx)

        cur_rsv = int(torch.xpu.memory_reserved(dev_idx))
        cur_alloc = int(torch.xpu.memory_allocated(dev_idx))
        cur_max_rsv = _sample_max_memory_reserved(dev_idx)

        if active_ctx is not None:
            peak_obs = cur_rsv
            if cur_max_rsv > active_ctx["pre_max_rsv"]:
                peak_obs = max(peak_obs, cur_max_rsv)
            _record_observed_peak(active_ctx["key"], peak_obs)

        free_before, total_vram = torch.xpu.mem_get_info(dev_idx)
        free_before = int(free_before)
        total_vram = int(total_vram)
        mm = _resolve_model_management(active_ctx)
        if total_vram <= 0 and mm is not None:
            total_vram = _device_total_vram(mm, device)

        safe_floor = _inter_tile_safe_floor(total_vram)
        if free_before >= safe_floor:
            return

        cache_flushed = False
        reusable = max(0, cur_rsv - cur_alloc)
        if reusable >= 256 * _MIB:
            empty_cache = getattr(torch.xpu, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
                cache_flushed = True
                after_rsv = int(torch.xpu.memory_reserved(dev_idx))
                after_alloc = int(torch.xpu.memory_allocated(dev_idx))
                after_reusable = max(0, after_rsv - after_alloc)
                publish = (
                    getattr(_aimdo_control, "publish_torch_cached_bytes", None)
                    if _aimdo_control is not None
                    else None
                )
                if callable(publish):
                    try:
                        publish(dev_idx, after_reusable)
                    except Exception:
                        pass

        free_bytes = int(torch.xpu.mem_get_info(dev_idx)[0])
        vbar_reclaimed = 0
        if (
            free_bytes < safe_floor
            and mm is not None
            and hasattr(mm, "current_loaded_models")
        ):
            active_model_ids = (
                active_ctx["model_ids"] if active_ctx is not None else set()
            )
            trim_target = safe_floor + _reusable_torch_bytes(device)
            if total_vram > 0:
                trim_target = min(trim_target, total_vram)
            vbar_reclaimed, _ = _trim_pass(
                mm, device, trim_target, active_model_ids, include_requested=False
            )

        free_after = int(torch.xpu.mem_get_info(dev_idx)[0])
        if cache_flushed or vbar_reclaimed > 0:
            log.info(
                "[OmniXPU] DynamicVRAM inter-tile reclaim: device=xpu:%d "
                "free=%.1f->%.1fMiB floor=%.1fMiB vbar_reclaimed=%.1fMiB cache_flushed=%d",
                dev_idx,
                free_before / _MIB,
                free_after / _MIB,
                safe_floor / _MIB,
                vbar_reclaimed / _MIB,
                1 if cache_flushed else 0,
            )
        else:
            log.debug(
                "[OmniXPU] DynamicVRAM inter-tile reclaim: device=xpu:%d "
                "free=%.1f->%.1fMiB floor=%.1fMiB vbar_reclaimed=0.0MiB cache_flushed=0",
                dev_idx,
                free_before / _MIB,
                free_after / _MIB,
                safe_floor / _MIB,
            )
    except Exception as exc:
        log.debug("[OmniXPU] DynamicVRAM inter-tile check skipped: %s", exc)


def _patch_tiled_scale_multidim(utils_mod=None):
    if utils_mod is None:
        try:
            import comfy.utils as utils_mod
        except Exception:
            return False

    original = getattr(utils_mod, "tiled_scale_multidim", None)
    if original is None or hasattr(original, _TILED_SCALE_PATCH_MARKER):
        return False

    @functools.wraps(original)
    def xpu_tiled_scale_multidim(samples, function, *args, **kwargs):
        compiler = getattr(torch, "compiler", None)
        is_compiling_fn = getattr(compiler, "is_compiling", None)
        is_compiling = bool(is_compiling_fn()) if callable(is_compiling_fn) else False
        xpu_avail_fn = getattr(getattr(torch, "xpu", None), "is_available", None)
        xpu_avail = bool(xpu_avail_fn()) if callable(xpu_avail_fn) else False
        if not xpu_avail or not _inter_tile_reclaim_enabled() or is_compiling:
            return original(samples, function, *args, **kwargs)

        def guarded_function(*f_args, **f_kwargs):
            result = function(*f_args, **f_kwargs)
            _after_tile_step(f_args)
            return result

        return original(samples, guarded_function, *args, **kwargs)

    setattr(xpu_tiled_scale_multidim, _TILED_SCALE_PATCH_MARKER, original)
    utils_mod.tiled_scale_multidim = xpu_tiled_scale_multidim
    return True


def _patch_model_loader(model_management):
    global _MODEL_MANAGEMENT_REF
    _MODEL_MANAGEMENT_REF = model_management
    original = model_management.load_models_gpu
    if hasattr(original, _PATCH_MARKER):
        return

    @functools.wraps(original)
    def boundary_trimmed(models, *args, **kwargs):
        budget_scope, cal_args, cal_kwargs = _prepare_boundary(
            model_management, models, args, kwargs
        )
        with budget_scope:
            return original(models, *cal_args, **cal_kwargs)

    setattr(boundary_trimmed, _PATCH_MARKER, original)
    model_management.load_models_gpu = boundary_trimmed


def apply():
    if sys.platform != "win32":
        return False, "Windows only"

    import comfy.model_management

    required = (
        "current_loaded_models",
        "extra_reserved_memory",
        "get_free_memory",
        "load_models_gpu",
        "minimum_inference_memory",
    )
    missing = [name for name in required if not hasattr(comfy.model_management, name)]
    if missing:
        return False, "missing ComfyUI hooks: " + ", ".join(missing)

    _patch_model_loader(comfy.model_management)
    if _get_free_memory_patch_enabled():
        try:
            import comfy.memory_management as comfy_memory_management
        except Exception:
            comfy_memory_management = None
        _patch_get_free_memory(
            comfy.model_management, memory_management=comfy_memory_management
        )
        try:
            import comfy.model_patcher as comfy_model_patcher

            mp_cls = getattr(comfy_model_patcher, "ModelPatcher", None)
            if mp_cls is not None and hasattr(mp_cls, "get_free_memory"):
                _patch_model_patcher_get_free_memory(
                    mp_cls, comfy.model_management, comfy_memory_management
                )
        except Exception as exc:
            log.debug(
                "[OmniXPU] ModelPatcher.get_free_memory patch skipped: %s", exc
            )
    _patch_tiled_scale_multidim()
    _patch_cfg_guider_steps()
    if _torch_cache_trim_enabled() or _step_trace_enabled():
        _ensure_fault_tracer()
    return True, "Windows XPU DynamicVRAM boundary reclaim enabled"


__all__ = ["apply"]
