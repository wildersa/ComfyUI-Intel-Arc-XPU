"""Reclaim DynamicVRAM residency at Windows XPU model boundaries."""

from __future__ import annotations

import functools
import logging
import os
import sys
import time
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
_STEP_PATCH_MARKER = "__omnixpu_dynamic_vram_step_original__"
_FAULT_PATCH_MARKER = "__omnixpu_dynamic_vram_fault_original__"
_FAULT_COUNTS = {"ok": 0, "none": 0}
_TRIM_STATE = {"last_attempt": 0.0, "last_unmap": 0, "calls": 0, "freed": 0}
_DEFAULT_TORCH_CACHE_TRIM_INTERVAL_S = 2.0
_DEFAULT_TORCH_CACHE_TRIM_MIN_MIB = 512
_DEFAULT_TORCH_CACHE_TRIM_FREE_MIB = 4096
_WARNED_FALLBACK = False
_MIB = 1024 ** 2
_VBAR_PAGE_BYTES = 32 * _MIB
_DEFAULT_BUDGET_KNEE_MIB = 6144
_DEFAULT_BUDGET_EXCESS_SCALE = 0.35
_DEFAULT_BUDGET_MAX_FRACTION = 0.40
_FIT_TRIM_ENABLED = (
    _config.dynamic_vram_fit_trim
    if _config is not None
    else os.environ.get("OMNIXPU_DYNAMIC_VRAM_FIT_TRIM", "1") != "0"
)


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
        return os.environ.get("OMNIXPU_ENABLE", "1") != "0" and env_val == "1"
    if _config is not None:
        return bool(getattr(_config, "dynamic_vram_torch_cache_trim", False))
    return False


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
        pending.extend(model.model_patches_models())
    return expanded


def _minimum_memory_target(model_management, args, kwargs):
    memory_required = _load_argument(args, kwargs, "memory_required", 0, 0)
    minimum_required = _load_argument(
        args, kwargs, "minimum_memory_required", 2, None
    )
    reserved = model_management.extra_reserved_memory()
    inference = model_management.minimum_inference_memory()
    if minimum_required is None:
        return max(inference, memory_required + reserved)
    return max(inference, minimum_required + reserved)


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
            model.load_device.index
            for model in models
            if model.is_dynamic()
            and getattr(model.load_device, "type", None) == "xpu"
            and model.load_device.index is not None
        }
    )
    if not devices or budget <= 0:
        return nullcontext(), ()
    return budget_scope(budget, devices), tuple(devices)


def _actual_free_vram(model_management, device) -> int:
    """Return real free XPU VRAM including AIMDO VBAR allocations."""
    base_free = int(model_management.get_free_memory(device))
    if getattr(device, "type", None) != "xpu":
        return base_free
    try:
        free_device, _ = torch.xpu.mem_get_info(device)
        stats = torch.xpu.memory_stats(device)
        reusable_torch = max(
            0,
            int(stats.get("reserved_bytes.all.current", 0))
            - int(stats.get("active_bytes.all.current", 0)),
        )
        return int(free_device) + reusable_torch
    except Exception:
        pass
    if _aimdo_model_vbar is not None and hasattr(_aimdo_model_vbar, "vbars_analyze"):
        try:
            dev_idx = 0 if device.index is None else int(device.index)
            aimdo_used = int(_aimdo_model_vbar.vbars_analyze(dev_idx))
            return max(0, base_free - aimdo_used)
        except Exception:
            pass
    return base_free


def _peak_torch_growth(device) -> int:
    try:
        rsv = int(torch.xpu.memory_reserved(device))
        peak = int(torch.xpu.max_memory_reserved(device))
        return max(0, peak - rsv)
    except Exception:
        return 0


def _unloaded_dynamic_bytes(models, device) -> int:
    total = 0
    for model in models:
        if not model.is_dynamic() or model.load_device != device:
            continue
        size = max(0, int(model.model_size()))
        loaded = max(0, int(model.loaded_size()))
        if size > loaded:
            total += size - loaded
    return total


def _trim_pass(model_management, device, target, requested, include_requested):
    reclaimed = 0
    trimmed = 0
    for loaded in reversed(model_management.current_loaded_models):
        model = loaded.model
        if loaded.device != device or loaded.is_dead() or not model.is_dynamic():
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


def _trim_dynamic_boundary(model_management, models, target):
    if not models or not all(model.is_dynamic() for model in models):
        return

    requested = {id(model) for model in models}
    devices = {model.load_device for model in models}
    for device in devices:
        if getattr(device, "type", None) != "xpu":
            continue

        free_before = _actual_free_vram(model_management, device)
        inactive_target = int(target)
        if _FIT_TRIM_ENABLED:
            unloaded_req = _unloaded_dynamic_bytes(models, device)
            if unloaded_req > 0:
                inactive_target = max(
                    inactive_target,
                    int(target) + unloaded_req + _peak_torch_growth(device),
                )
        if free_before >= inactive_target:
            continue

        inactive_bytes, inactive_models = _trim_pass(
            model_management, device, inactive_target, requested, False
        )
        active_bytes, active_models = _trim_pass(
            model_management, device, int(target), requested, True
        )
        free_after = _actual_free_vram(model_management, device)
        log.info(
            "[OmniXPU] DynamicVRAM boundary trim: device=%s target=%.1fMiB "
            "free=%.1f->%.1fMiB reclaimed=%.1fMiB models=%d inactive/%d active",
            device,
            inactive_target / (1024 ** 2),
            free_before / (1024 ** 2),
            free_after / (1024 ** 2),
            (inactive_bytes + active_bytes) / (1024 ** 2),
            inactive_models,
            active_models,
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


def _calibrate_inference_budget(model_management, expanded, args, kwargs):
    raw_min_arg = _load_argument(args, kwargs, "minimum_memory_required", 2, None)
    raw_budget = _inference_memory_budget(args, kwargs)
    if (
        not _budget_calibration_enabled()
        or raw_min_arg is None
        or isinstance(raw_min_arg, bool)
        or not expanded
        or not all(model.is_dynamic() for model in expanded)
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

    knee_bytes = _env_int("OMNIXPU_AIMDO_BUDGET_KNEE_MIB", _DEFAULT_BUDGET_KNEE_MIB, 512) * _MIB
    if raw_budget <= knee_bytes:
        return args, kwargs, raw_budget, raw_budget, False

    excess_scale = _env_float(
        "OMNIXPU_AIMDO_BUDGET_EXCESS_SCALE", _DEFAULT_BUDGET_EXCESS_SCALE, 0.0, 1.0
    )
    max_fraction = _env_float(
        "OMNIXPU_AIMDO_BUDGET_MAX_FRACTION", _DEFAULT_BUDGET_MAX_FRACTION, 0.1, 1.0
    )
    calibrated = knee_bytes + int((raw_budget - knee_bytes) * excess_scale)

    totals = [
        t for t in (_device_total_vram(model_management, dev) for dev in xpu_devices) if t > 0
    ]
    if totals:
        cap = max(knee_bytes, int(min(totals) * max_fraction))
        calibrated = min(calibrated, cap)

    calibrated = max(knee_bytes, min(raw_budget, calibrated))
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
            dev_idx = 0 if device.index is None else int(device.index)
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
        dev_idx = 0 if device.index is None else int(device.index)
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
        if was_calibrated:
            knee = _env_int("OMNIXPU_AIMDO_BUDGET_KNEE_MIB", _DEFAULT_BUDGET_KNEE_MIB, 512)
            scale = _env_float(
                "OMNIXPU_AIMDO_BUDGET_EXCESS_SCALE", _DEFAULT_BUDGET_EXCESS_SCALE, 0.0, 1.0
            )
            log.info(
                "[OmniXPU] DynamicVRAM budget calibrated (flash): "
                "raw=%.1fMiB -> %.1fMiB (knee=%dMiB scale=%.2f)",
                raw_budget / _MIB,
                inference_budget / _MIB,
                knee,
                scale,
            )
        _log_boundary_telemetry(expanded, raw_budget, inference_budget)
        target = _minimum_memory_target(model_management, cal_args, cal_kwargs)
        _trim_dynamic_boundary(model_management, expanded, target)
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


def _maybe_trim_torch_cache(alloc):
    if not _torch_cache_trim_enabled():
        return 0
    now = time.monotonic()
    interval = _env_float(
        "OMNIXPU_TORCH_CACHE_TRIM_INTERVAL_S",
        _DEFAULT_TORCH_CACHE_TRIM_INTERVAL_S,
        0.2,
        60.0,
    )
    if (now - _TRIM_STATE["last_attempt"]) < interval:
        return 0
    _TRIM_STATE["last_attempt"] = now

    dev_idx = _fault_device_index(alloc)
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
        if unmap_bytes > _TRIM_STATE["last_unmap"]:
            pressure = True
        _TRIM_STATE["last_unmap"] = unmap_bytes

    if not pressure:
        free_threshold = _env_int(
            "OMNIXPU_TORCH_CACHE_TRIM_FREE_MIB", _DEFAULT_TORCH_CACHE_TRIM_FREE_MIB, 256
        ) * _MIB
        try:
            free_bytes = int(torch.xpu.mem_get_info(dev_idx)[0])
        except Exception:
            free_bytes = free_threshold + 1
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
        _TRIM_STATE["last_attempt"] = time.monotonic()

    freed = max(0, rsv - after_rsv)
    _TRIM_STATE["calls"] += 1
    _TRIM_STATE["freed"] += freed
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

    fault_traced = _ensure_fault_tracer()
    t_last = [time.perf_counter()]
    prev_faults = [dict(_FAULT_COUNTS)]
    prev_trim = {"calls": _TRIM_STATE["calls"], "freed": _TRIM_STATE["freed"]}
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
        if _TRIM_STATE["calls"] > prev_trim["calls"] or _torch_cache_trim_enabled():
            tc = _TRIM_STATE["calls"] - prev_trim["calls"]
            tf_mib = (_TRIM_STATE["freed"] - prev_trim["freed"]) / _MIB
            prev_trim["calls"] = _TRIM_STATE["calls"]
            prev_trim["freed"] = _TRIM_STATE["freed"]
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


def _patch_model_loader(model_management):
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
    _patch_cfg_guider_steps()
    if _torch_cache_trim_enabled() or _step_trace_enabled():
        _ensure_fault_tracer()
    return True, "Windows XPU DynamicVRAM boundary reclaim enabled"


__all__ = ["apply"]
