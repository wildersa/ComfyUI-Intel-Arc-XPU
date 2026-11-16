"""Reclaim DynamicVRAM residency at Windows XPU model boundaries."""

from __future__ import annotations

import functools
import logging
import os
import sys
from contextlib import nullcontext

import torch

try:
    from comfy_aimdo import model_vbar as _aimdo_model_vbar
except ImportError:
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
_WARNED_FALLBACK = False
_FIT_TRIM_ENABLED = (
    _config.dynamic_vram_fit_trim
    if _config is not None
    else os.environ.get("OMNIXPU_DYNAMIC_VRAM_FIT_TRIM", "1") != "0"
)


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


def _prepare_boundary(model_management, models, args, kwargs):
    global _WARNED_FALLBACK
    try:
        expanded = _expanded_models(models)
        target = _minimum_memory_target(model_management, args, kwargs)
        inference_budget = _inference_memory_budget(args, kwargs)
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
        return budget_scope
    except Exception as exc:
        if not _WARNED_FALLBACK:
            _WARNED_FALLBACK = True
            log.warning(
                "[OmniXPU] DynamicVRAM boundary trim conflict (%s); "
                "falling back to stock ComfyUI loader",
                exc,
            )
        return nullcontext()


def _patch_model_loader(model_management):
    original = model_management.load_models_gpu
    if hasattr(original, _PATCH_MARKER):
        return

    @functools.wraps(original)
    def boundary_trimmed(models, *args, **kwargs):
        budget_scope = _prepare_boundary(
            model_management, models, args, kwargs
        )
        with budget_scope:
            return original(models, *args, **kwargs)

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
    return True, "Windows XPU DynamicVRAM boundary reclaim enabled"


__all__ = ["apply"]
