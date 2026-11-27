"""Stdlib-only workflow VRAM protection and transition guard tests (no GPU or torch required)."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]

MIB = 1024 ** 2
GIB = 1024 ** 3
VBAR_PAGE_BYTES = 32 * MIB

# Representative Intel Arc SKUs: 12 GiB (B580), 16 GiB (B50), 24 GiB (B60), 32 GiB (B70)
SKUS_GIB = (12, 16, 24, 32)
RESERVES_GIB = (0.0, 1.0, 2.0, 4.0)

_PKG_PREFIX = "omnixpu_workflow_ci_pkg"
_LOADED_MODULES: dict[str, types.ModuleType] = {}


def _make_fake_torch(*, xpu_available: bool = True, cuda_available: bool = False, total_vram: int = 32 * GIB):
    torch_mod = types.ModuleType("torch")
    torch_mod.__version__ = "2.13.0+xpu"
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.device = lambda dev_type="cpu", index=None: _FakeDevice(dev_type, index)
    torch_mod.xpu = types.SimpleNamespace(
        is_available=lambda: xpu_available,
        current_device=lambda: 0,
        memory_allocated=lambda dev=0: 0,
        memory_reserved=lambda dev=0: 0,
        max_memory_allocated=lambda dev=0: 0,
        max_memory_reserved=lambda dev=0: 0,
        memory_stats=lambda dev=0: {
            "reserved_bytes.all.current": 0,
            "active_bytes.all.current": 0,
        },
        empty_cache=lambda: None,
        synchronize=lambda: None,
        mem_get_info=lambda dev=0: (total_vram, total_vram),
    )
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    return torch_mod


def _load_repo_module(name: str, rel_path: str, *, fake_torch=None):
    full_name = f"{_PKG_PREFIX}.{name}"
    if full_name in _LOADED_MODULES:
        return _LOADED_MODULES[full_name]

    added_keys = []
    if _PKG_PREFIX not in sys.modules:
        pkg = types.ModuleType(_PKG_PREFIX)
        pkg.__path__ = [str(REPO_ROOT)]
        sys.modules[_PKG_PREFIX] = pkg
        added_keys.append(_PKG_PREFIX)

    adapters_pkg = f"{_PKG_PREFIX}.adapters"
    if adapters_pkg not in sys.modules:
        apkg = types.ModuleType(adapters_pkg)
        apkg.__path__ = [str(REPO_ROOT / "adapters")]
        sys.modules[adapters_pkg] = apkg
        added_keys.append(adapters_pkg)

    cfg_name = f"{_PKG_PREFIX}.config"
    if cfg_name not in sys.modules:
        cfg_spec = importlib.util.spec_from_file_location(cfg_name, REPO_ROOT / "config.py")
        cfg_mod = importlib.util.module_from_spec(cfg_spec)
        sys.modules[cfg_name] = cfg_mod
        added_keys.append(cfg_name)
        cfg_spec.loader.exec_module(cfg_mod)

    spec = importlib.util.spec_from_file_location(full_name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    added_keys.append(full_name)

    torch_stub = fake_torch if fake_torch is not None else _make_fake_torch()
    try:
        with mock.patch.dict(sys.modules, {"torch": torch_stub}):
            spec.loader.exec_module(mod)
    except Exception:
        for key in added_keys:
            sys.modules.pop(key, None)
        raise

    _LOADED_MODULES[full_name] = mod
    return mod


def tearDownModule():
    _LOADED_MODULES.clear()
    for key in list(sys.modules):
        if key == _PKG_PREFIX or key.startswith(f"{_PKG_PREFIX}."):
            sys.modules.pop(key, None)


class _FakeDevice:
    def __init__(self, dev_type: str = "xpu", index: int | None = 0):
        self.type = dev_type
        self.index = index

    def __eq__(self, other):
        return (
            isinstance(other, _FakeDevice)
            and self.type == other.type
            and self.index == other.index
        )

    def __hash__(self):
        return hash((self.type, self.index))

    def __repr__(self):
        return f"{self.type}:{self.index}" if self.index is not None else self.type


class _FakeModel:
    def __init__(
        self,
        *,
        name: str = "model",
        dynamic: bool = True,
        device=None,
        size: int = 0,
        loaded: int = 0,
        patches=None,
        attachments=None,
        partially_unload_fn=None,
    ):
        self.name = name
        self._dynamic = dynamic
        self.load_device = device or _FakeDevice("xpu", 0)
        self.offload_device = _FakeDevice("cpu", None)
        self._size = int(size)
        self._loaded = int(loaded)
        self.patches = patches or {}
        self.attachments = attachments if attachments is not None else {}
        self._partially_unload_fn = partially_unload_fn
        self._current_device = self.load_device if loaded > 0 else self.offload_device

    def is_dynamic(self):
        return self._dynamic

    def model_patches_models(self):
        return []

    def model_size(self):
        return self._size

    def loaded_size(self):
        return self._loaded

    def current_loaded_device(self):
        return self._current_device

    def partially_unload(self, offload_device, shortfall):
        if self._partially_unload_fn is not None:
            return self._partially_unload_fn(self, shortfall)
        freed = min(self._loaded, max(0, int(shortfall)))
        self._loaded -= freed
        if self._loaded == 0:
            self._current_device = offload_device
        return freed


class _FakeLoaded:
    def __init__(self, model):
        self.model = model
        self.device = model.load_device

    def is_dead(self):
        return False


def _make_mm(*, total_vram: int, reserve_vram: int, flash: bool = True, free_vram: int | None = None):
    mm = types.SimpleNamespace()
    mm.pytorch_attention_flash_attention = lambda: flash
    mm.get_total_memory = lambda _dev=None: int(total_vram)
    mm.extra_reserved_memory = lambda: int(reserve_vram)
    mm.minimum_inference_memory = lambda: int(0.8 * GIB) + int(reserve_vram)
    mm.get_free_memory = lambda _dev=None, torch_free_too=False: (
        (int(total_vram if free_vram is None else free_vram), 0)
        if torch_free_too
        else int(total_vram if free_vram is None else free_vram)
    )
    mm.soft_empty_cache = lambda: None
    mm.current_loaded_models = []
    mm.get_torch_device = lambda: _FakeDevice("xpu", 0)
    return mm


class _WorkflowTransitionSimulator:
    """Simulates multi-stage ComfyUI workflow transitions under Windows XPU AIMDO semantics."""

    def __init__(
        self,
        total_gib: int,
        reserve_gib: float,
        *,
        dynamic_vram_mod,
        lora_memory_mod,
        enable_boundary_trim: bool = True,
        enable_calibration: bool = True,
        aimdo_enabled: bool = True,
    ):
        self.total_vram = int(total_gib * GIB)
        self.reserve_vram = int(reserve_gib * GIB)
        self.dynamic_vram = dynamic_vram_mod
        self.lora_memory = lora_memory_mod
        self.enable_boundary_trim = enable_boundary_trim
        self.enable_calibration = enable_calibration
        self.aimdo_enabled = aimdo_enabled
        self.device = _FakeDevice("xpu", 0)

        self.models: dict[str, _FakeModel] = {}
        self.vbar_logical: dict[str, int] = {}
        self.vbar_physical: dict[str, int] = {}
        self.vbar_deferred_evicted: dict[str, int] = {}
        self.static_alloc: dict[str, int] = {}
        self.loaded_order: list[str] = []
        self.peak_torch_growth = 0
        self.events: list[dict] = []

    @staticmethod
    def _align_up_page(nbytes: int) -> int:
        if nbytes <= 0:
            return 0
        return ((nbytes + VBAR_PAGE_BYTES - 1) // VBAR_PAGE_BYTES) * VBAR_PAGE_BYTES

    @staticmethod
    def _align_down_page(nbytes: int) -> int:
        if nbytes <= 0:
            return 0
        return (nbytes // VBAR_PAGE_BYTES) * VBAR_PAGE_BYTES

    def total_vbar_physical(self) -> int:
        return sum(self.vbar_physical.values())

    def total_static_alloc(self) -> int:
        return sum(self.static_alloc.values())

    def physical_used(self) -> int:
        return self.total_vbar_physical() + self.total_static_alloc()

    def physical_free(self) -> int:
        return max(0, self.total_vram - self.physical_used())

    def _explicit_free_vbar(self, name: str, shortfall: int) -> int:
        """Explicit partially_unload reclaims physical 32 MiB VBAR pages immediately."""
        avail = self.vbar_physical.get(name, 0)
        if avail <= 0 or shortfall <= 0:
            return 0
        pages_needed = min(
            avail // VBAR_PAGE_BYTES,
            (shortfall + VBAR_PAGE_BYTES - 1) // VBAR_PAGE_BYTES,
        )
        freed_phys = pages_needed * VBAR_PAGE_BYTES
        self.vbar_physical[name] -= freed_phys

        def_ev = self.vbar_deferred_evicted.get(name, 0)
        take_def = min(def_ev, freed_phys)
        self.vbar_deferred_evicted[name] = def_ev - take_def
        take_log = min(self.vbar_logical.get(name, 0), freed_phys - take_def)
        self.vbar_logical[name] = max(0, self.vbar_logical.get(name, 0) - take_log)
        self.models[name]._loaded = self.vbar_logical[name]
        if self.vbar_logical[name] == 0:
            self.models[name]._current_device = self.models[name].offload_device
        return freed_phys

    def _flush_deferred_on_vbar_fault(self) -> int:
        """Next vbar_fault() commits deferred physical VBAR reclaim."""
        reclaimed = 0
        for name, def_bytes in list(self.vbar_deferred_evicted.items()):
            if def_bytes > 0:
                take = min(self.vbar_physical.get(name, 0), def_bytes)
                self.vbar_physical[name] -= take
                self.vbar_deferred_evicted[name] = 0
                reclaimed += take
        return reclaimed

    def _evict_vbar_deferred(self, bytes_needed: int, *, exclude: str | None = None) -> int:
        """Evicts logical VBAR pages while deferring physical reclaim until next vbar_fault()."""
        evicted = 0
        for name in list(self.loaded_order):
            if name == exclude:
                continue
            model = self.models.get(name)
            if model is None or not model.is_dynamic():
                continue
            avail_log = self.vbar_logical.get(name, 0)
            if avail_log <= 0:
                continue
            take = min(avail_log, self._align_up_page(bytes_needed - evicted))
            self.vbar_logical[name] -= take
            self.vbar_deferred_evicted[name] = self.vbar_deferred_evicted.get(name, 0) + take
            model._loaded = self.vbar_logical[name]
            evicted += take
            if evicted >= bytes_needed:
                break
        return evicted

    def _vbar_fault(self, name: str, needed_bytes: int) -> bool:
        self._flush_deferred_on_vbar_fault()
        if self.physical_free() < needed_bytes:
            shortfall = needed_bytes - self.physical_free()
            self._evict_vbar_deferred(shortfall, exclude=name)
            self._flush_deferred_on_vbar_fault()
        if self.physical_free() < needed_bytes:
            return False
        aligned = self._align_up_page(needed_bytes)
        if self.physical_free() < aligned:
            aligned = self._align_down_page(self.physical_free())
        self.vbar_logical[name] = self.vbar_logical.get(name, 0) + aligned
        self.vbar_physical[name] = self.vbar_physical.get(name, 0) + aligned
        self.models[name]._loaded = self.vbar_logical[name]
        self.models[name]._current_device = self.device
        return True

    def _measure_patched_free_memory(self) -> tuple[int, int]:
        """Exercise patched get_free_memory and ModelPatcher.get_free_memory at the current state."""
        static_alloc = self.total_static_alloc()
        vbar_phys = self.total_vbar_physical()
        hw_free = max(0, self.total_vram - static_alloc - vbar_phys)

        fake_xpu = types.SimpleNamespace(
            current_device=lambda: 0,
            mem_get_info=lambda _d=0: (hw_free, self.total_vram),
            memory_stats=lambda _d=0: {
                "reserved_bytes.all.current": static_alloc,
                "active_bytes.all.current": static_alloc,
            },
        )
        fake_vbar_mod = types.SimpleNamespace(vbars_analyze=lambda _idx: vbar_phys)
        mm_probe = _make_mm(total_vram=self.total_vram, reserve_vram=self.reserve_vram)
        mm_probe.get_free_memory = lambda _d=None, torch_free_too=False: (
            (self.total_vram - static_alloc, 0)
            if torch_free_too
            else (self.total_vram - static_alloc)
        )
        mp_cls = type(
            "_ProbeModelPatcher",
            (),
            {
                "get_free_memory": lambda self_mp, dev: (
                    mm_probe.get_free_memory(dev) + fake_vbar_mod.vbars_analyze(0)
                )
            },
        )
        mem_mgr = types.SimpleNamespace(aimdo_enabled=self.aimdo_enabled)

        with (
            mock.patch.object(self.dynamic_vram.torch, "xpu", fake_xpu, create=True),
            mock.patch.object(self.dynamic_vram, "_aimdo_model_vbar", fake_vbar_mod),
            mock.patch.dict(os.environ, {"OMNIXPU_DYNAMIC_VRAM_GET_FREE_MEMORY": "1"}, clear=False),
        ):
            self.dynamic_vram._patch_get_free_memory(mm_probe, memory_management=mem_mgr)
            self.dynamic_vram._patch_model_patcher_get_free_memory(mp_cls, mm_probe, mem_mgr)
            free_mem = int(mm_probe.get_free_memory(self.device))
            mp_free = int(mp_cls().get_free_memory(self.device))
        return free_mem, mp_free

    def step(
        self,
        name: str,
        *,
        dynamic: bool,
        model_size: int,
        working_vram: int,
        minimum_vram: int | None = None,
        lora_staging: int = 0,
        lora_layer_peak: int = 0,
        active_vbar_need: int | None = None,
    ) -> dict:
        if name not in self.models:
            self.models[name] = _FakeModel(
                name=name,
                dynamic=dynamic,
                device=self.device,
                size=model_size,
                loaded=0,
                partially_unload_fn=lambda m, sf, n=name: self._explicit_free_vbar(n, sf),
            )
            self.vbar_logical[name] = 0
            self.vbar_physical[name] = 0
            self.vbar_deferred_evicted[name] = 0
            self.static_alloc[name] = 0
        else:
            self.models[name]._size = model_size
            self.models[name]._dynamic = dynamic

        model = self.models[name]
        if name in self.loaded_order:
            self.loaded_order.remove(name)
        self.loaded_order.append(name)

        mm = _make_mm(total_vram=self.total_vram, reserve_vram=self.reserve_vram, flash=True)
        mm.current_loaded_models = [
            _FakeLoaded(self.models[n])
            for n in self.loaded_order
            if (self.vbar_physical.get(n, 0) > 0 or self.static_alloc.get(n, 0) > 0)
        ]

        # 1. LoRA budget contribution
        req_arg = working_vram
        min_arg = minimum_vram
        if lora_staging > 0:
            min_lora = self.lora_memory._minimum_lora_budget(lora_staging, lora_layer_peak, dynamic)
            req_arg += lora_staging
            if min_arg is not None:
                min_arg += min_lora

        load_kwargs = {"memory_required": req_arg}
        if min_arg is not None:
            load_kwargs["minimum_memory_required"] = min_arg

        # 2. DynamicVRAM boundary calibration + trim
        cal_args, cal_kwargs = (), dict(load_kwargs)
        seen_target = None
        if self.enable_boundary_trim:
            env_cal = "1" if self.enable_calibration else "0"
            with (
                mock.patch.dict(os.environ, {"OMNIXPU_DYNAMIC_VRAM_BUDGET_CALIBRATION": env_cal}, clear=False),
                mock.patch.object(self.dynamic_vram, "_device_total_vram", return_value=self.total_vram),
                mock.patch.object(self.dynamic_vram, "_actual_free_vram", side_effect=lambda *_: self.physical_free()),
                mock.patch.object(self.dynamic_vram, "_reusable_torch_bytes", return_value=0),
                mock.patch.object(self.dynamic_vram, "_peak_torch_growth", return_value=self.peak_torch_growth),
            ):
                expanded = [model]
                cal_args, cal_kwargs, _, _, _ = self.dynamic_vram._calibrate_inference_budget(
                    mm, expanded, (), load_kwargs
                )
                seen_target = self.dynamic_vram._minimum_memory_target(
                    mm, cal_args, cal_kwargs, device=self.device
                )
                self.dynamic_vram._trim_dynamic_boundary(
                    mm, expanded, seen_target, load_args=cal_args, load_kwargs=cal_kwargs
                )

        # 3. Loader free_memory passes
        unloaded_static = 0 if (dynamic or self.static_alloc[name] > 0) else model_size
        extra_mem = max(mm.minimum_inference_memory(), cal_kwargs["memory_required"] + self.reserve_vram)
        free_req = int(unloaded_static * 1.1) + extra_mem
        for old_name in list(self.loaded_order):
            if old_name == name:
                continue
            old_m = self.models[old_name]
            if old_m.is_dynamic() and dynamic:
                continue
            need = free_req - self.physical_free()
            if need <= 0:
                break
            if old_m.is_dynamic():
                self._explicit_free_vbar(old_name, need)
            else:
                self.static_alloc[old_name] = 0
                old_m._loaded = 0
                old_m._current_device = old_m.offload_device

        eff_min = cal_kwargs.get("minimum_memory_required")
        min_mem_req = (
            extra_mem
            if eff_min is None
            else max(mm.minimum_inference_memory(), eff_min + self.reserve_vram)
        )
        if self.physical_free() < min_mem_req:
            for old_name in list(self.loaded_order):
                if old_name == name:
                    continue
                old_m = self.models[old_name]
                if old_m.is_dynamic() and dynamic:
                    continue
                need = min_mem_req - self.physical_free()
                if need <= 0:
                    break
                if old_m.is_dynamic():
                    self._explicit_free_vbar(old_name, need)
                else:
                    self.static_alloc[old_name] = 0
                    old_m._loaded = 0
                    old_m._current_device = old_m.offload_device

        # 4. Execution phase
        oom = False
        oom_reason = ""
        if dynamic:
            self._flush_deferred_on_vbar_fault()
            target_vbar = model_size if active_vbar_need is None else active_vbar_need
            already_res = self.vbar_logical[name]
            to_fault = max(0, target_vbar - already_res)
            max_faultable = max(0, self.physical_free() - working_vram)
            faulted = min(to_fault, max_faultable)
            if self.physical_free() < working_vram:
                self._evict_vbar_deferred(working_vram - self.physical_free(), exclude=name)
                self._flush_deferred_on_vbar_fault()
            if self.physical_free() < working_vram:
                oom = True
                oom_reason = (
                    f"dynamic working_vram={working_vram / GIB:.2f}GiB > "
                    f"free={self.physical_free() / GIB:.2f}GiB"
                )
            elif faulted > 0 and not self._vbar_fault(name, faulted):
                oom = True
                oom_reason = "vbar_fault OOM"
            warm_hit_bytes = already_res
        else:
            warm_hit_bytes = self.static_alloc[name]
            weights_needed = max(0, model_size - self.static_alloc[name])
            total_needed = weights_needed + working_vram
            if self.physical_free() < total_needed:
                self._evict_vbar_deferred(total_needed - self.physical_free(), exclude=name)
                if self.physical_free() < total_needed:
                    oom = True
                    oom_reason = (
                        f"static_oom need={total_needed / GIB:.2f}GiB "
                        f"phys_free={self.physical_free() / GIB:.2f}GiB "
                        f"deferred_vbar={sum(self.vbar_deferred_evicted.values()) / GIB:.2f}GiB"
                    )
            if not oom:
                self.static_alloc[name] += weights_needed
                model._loaded = self.static_alloc[name]
                model._current_device = self.device

        free_mem, mp_free_mem = self._measure_patched_free_memory()
        ev = {
            "name": name,
            "dynamic": dynamic,
            "oom": oom,
            "oom_reason": oom_reason,
            "target_bytes": seen_target,
            "phys_used_bytes": self.physical_used(),
            "phys_used_gib": self.physical_used() / GIB,
            "phys_free_gib": self.physical_free() / GIB,
            "vbar_phys_gib": self.total_vbar_physical() / GIB,
            "deferred_gib": sum(self.vbar_deferred_evicted.values()) / GIB,
            "warm_hit_gib": warm_hit_bytes / GIB,
            "get_free_memory": free_mem,
            "mp_get_free_memory": mp_free_mem,
            "total_vram": self.total_vram,
        }
        self.events.append(ev)
        return ev


class TestWorkflowChainProtection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_torch = _make_fake_torch(xpu_available=True, cuda_available=False)
        cls.dyn = _load_repo_module(
            "adapters.dynamic_vram", "adapters/dynamic_vram.py", fake_torch=fake_torch
        )
        cls.lora = _load_repo_module(
            "adapters.lora_memory", "adapters/lora_memory.py", fake_torch=fake_torch
        )

    def _new_sim(self, total_gib: int, reserve_gib: float = 1.0, **kwargs) -> _WorkflowTransitionSimulator:
        return _WorkflowTransitionSimulator(
            total_gib,
            reserve_gib,
            dynamic_vram_mod=self.dyn,
            lora_memory_mod=self.lora,
            **kwargs,
        )

    def _assert_step_invariants(self, events: list[dict], *, context: str):
        for idx, ev in enumerate(events):
            self.assertFalse(ev["oom"], f"{context} step={idx} ({ev['name']}) OOM: {ev['oom_reason']}")
            self.assertLessEqual(
                ev["phys_used_bytes"],
                ev["total_vram"],
                f"{context} step={idx} ({ev['name']}) exceeded total_vram",
            )
            if ev["target_bytes"] is not None:
                self.assertLessEqual(
                    ev["target_bytes"],
                    ev["total_vram"],
                    f"{context} step={idx} ({ev['name']}) target exceeded total_vram",
                )
            self.assertGreaterEqual(ev["get_free_memory"], 0, f"{context} step={idx} free < 0")
            self.assertLessEqual(
                ev["get_free_memory"],
                ev["mp_get_free_memory"],
                f"{context} step={idx} get_free_memory > ModelPatcher.get_free_memory",
            )
            self.assertLessEqual(
                ev["mp_get_free_memory"],
                ev["total_vram"],
                f"{context} step={idx} ModelPatcher.get_free_memory > total_vram",
            )

    def _run_chain_1(self, sim: _WorkflowTransitionSimulator) -> list[dict]:
        """TE (Dyn) -> DiT (Dyn) -> LatentUpsampler (Dyn) -> DiT (Dyn) -> AudioVAE (Static) -> VideoVAE (Static) -> Warm DiT (Dyn)."""
        return [
            sim.step("TE", dynamic=True, model_size=int(14.6 * GIB), working_vram=int(2.0 * GIB), minimum_vram=int(2.0 * GIB)),
            sim.step("DiT", dynamic=True, model_size=int(20.5 * GIB), working_vram=int(6.0 * GIB), minimum_vram=int(6.0 * GIB)),
            sim.step("LatentUpsampler", dynamic=True, model_size=int(0.95 * GIB), working_vram=int(1.0 * GIB), minimum_vram=int(1.0 * GIB)),
            sim.step("DiT", dynamic=True, model_size=int(20.5 * GIB), working_vram=int(6.0 * GIB), minimum_vram=int(6.0 * GIB)),
            sim.step("AudioVAE", dynamic=False, model_size=int(0.69 * GIB), working_vram=int(0.5 * GIB)),
            sim.step("VideoVAE", dynamic=False, model_size=int(1.4 * GIB), working_vram=int(3.4 * GIB)),
            sim.step("DiT", dynamic=True, model_size=int(20.5 * GIB), working_vram=int(6.0 * GIB), minimum_vram=int(6.0 * GIB)),
        ]

    def test_chain_1_audio_video_dit_upscaler_static_vaes_and_deferred_reclaim(self):
        for total_gib in SKUS_GIB:
            sim = self._new_sim(total_gib, 1.0, enable_boundary_trim=True)
            events = self._run_chain_1(sim)
            self._assert_step_invariants(events, context=f"chain1-{total_gib}G")
            # Before static VAEs execute, deferred VBAR bytes must never strand static allocations
            self.assertEqual(events[4]["deferred_gib"], 0.0)
            self.assertEqual(events[5]["deferred_gib"], 0.0)
            if total_gib >= 24:
                self.assertGreater(events[3]["warm_hit_gib"], 0.0)
            if total_gib == 32:
                self.assertGreater(events[3]["warm_hit_gib"], 10.0)

        # Verify deferred physical reclaim behavior without explicit reclaim vs vbar_fault flush
        sim_deferred = self._new_sim(16, 1.0, enable_boundary_trim=False)
        sim_deferred.step("DiT", dynamic=True, model_size=int(14.0 * GIB), working_vram=int(2.0 * GIB))
        free_before = sim_deferred.physical_free()
        evicted_logical = sim_deferred._evict_vbar_deferred(4 * GIB)
        self.assertEqual(evicted_logical, 4 * GIB)
        self.assertEqual(sim_deferred.physical_free(), free_before)
        self.assertGreater(sum(sim_deferred.vbar_deferred_evicted.values()), 0)
        sim_deferred._flush_deferred_on_vbar_fault()
        self.assertEqual(sim_deferred.physical_free(), free_before + 4 * GIB)

    def test_chain_2_dynamic_dit_lora_and_dynamic_vaes_warm_reuse(self):
        # High-memory DiT (16 GiB working budget) + 1.87 GiB LoRA on 24/32 GiB SKUs
        for total_gib in (24, 32):
            sim = self._new_sim(total_gib, 1.0, enable_boundary_trim=True)
            events = [
                sim.step("VideoVAE", dynamic=True, model_size=int(2.7 * GIB), working_vram=int(2.0 * GIB), minimum_vram=int(2.0 * GIB)),
                sim.step("TE", dynamic=True, model_size=int(15.0 * GIB), working_vram=int(2.0 * GIB), minimum_vram=int(2.0 * GIB)),
                sim.step(
                    "DiT",
                    dynamic=True,
                    model_size=int(20.0 * GIB),
                    working_vram=int(16.0 * GIB),
                    minimum_vram=int(16.0 * GIB),
                    lora_staging=int(1.87 * GIB),
                    lora_layer_peak=16 * MIB,
                ),
                sim.step("AudioVAE", dynamic=True, model_size=int(0.8 * GIB), working_vram=int(0.8 * GIB), minimum_vram=int(0.8 * GIB)),
                sim.step("VideoVAE", dynamic=True, model_size=int(2.7 * GIB), working_vram=int(2.0 * GIB), minimum_vram=int(2.0 * GIB)),
                sim.step("TE", dynamic=True, model_size=int(15.0 * GIB), working_vram=int(2.0 * GIB), minimum_vram=int(2.0 * GIB)),
            ]
            self._assert_step_invariants(events, context=f"chain2-hi-{total_gib}G")

        # SKU-scaled dynamic DiT + LoRA chain across all 12/16/24/32 GiB SKUs
        for total_gib in SKUS_GIB:
            sim = self._new_sim(total_gib, 1.0, enable_boundary_trim=True)
            events = [
                sim.step("VideoVAE", dynamic=True, model_size=int(2.7 * GIB), working_vram=int(1.5 * GIB), minimum_vram=int(1.5 * GIB)),
                sim.step("TE", dynamic=True, model_size=int(10.0 * GIB), working_vram=int(2.0 * GIB), minimum_vram=int(2.0 * GIB)),
                sim.step(
                    "DiT",
                    dynamic=True,
                    model_size=int(14.0 * GIB),
                    working_vram=int(6.0 * GIB),
                    minimum_vram=int(6.0 * GIB),
                    lora_staging=int(1.87 * GIB),
                    lora_layer_peak=16 * MIB,
                ),
                sim.step("AudioVAE", dynamic=True, model_size=int(0.8 * GIB), working_vram=int(0.8 * GIB), minimum_vram=int(0.8 * GIB)),
                sim.step("VideoVAE", dynamic=True, model_size=int(2.7 * GIB), working_vram=int(1.5 * GIB), minimum_vram=int(1.5 * GIB)),
                sim.step("TE", dynamic=True, model_size=int(10.0 * GIB), working_vram=int(2.0 * GIB), minimum_vram=int(2.0 * GIB)),
            ]
            self._assert_step_invariants(events, context=f"chain2-all-{total_gib}G")
            if total_gib >= 24:
                self.assertGreater(events[4]["warm_hit_gib"], 0.0)
                self.assertGreater(events[5]["warm_hit_gib"], 0.0)

        # Verify LoRA peak minimum budget + Flash calibration interaction across all SKUs
        full_lora = int(1.87 * GIB)
        min_lora = self.lora._minimum_lora_budget(full_lora, 16 * MIB, all_dynamic=True)
        self.assertEqual(min_lora, 128 * MIB)
        model = _FakeModel(name="DiT", dynamic=True, size=20 * GIB)
        raw_min = 16 * GIB
        _, lora_kwargs, _, _, _, eff_min = self.lora._add_budget_to_load_args(
            (), {"memory_required": 20 * GIB, "minimum_memory_required": raw_min}, full_lora, min_budget=min_lora
        )
        self.assertEqual(eff_min, raw_min + 128 * MIB)

        for total_gib, expected_delta in ((12, 0), (16, 0), (24, int(128 * MIB * 0.35)), (32, int(128 * MIB * 0.35))):
            total_b = total_gib * GIB
            mm = _make_mm(total_vram=total_b, reserve_vram=1 * GIB, flash=True)
            with (
                mock.patch.dict(os.environ, {"OMNIXPU_DYNAMIC_VRAM_BUDGET_CALIBRATION": "1"}, clear=False),
                mock.patch.object(self.dyn, "_device_total_vram", return_value=total_b),
            ):
                os.environ.pop("OMNIXPU_AIMDO_BUDGET_KNEE_MIB", None)
                _, _, _, cal_base, _ = self.dyn._calibrate_inference_budget(
                    mm, [model], (), {"memory_required": 20 * GIB, "minimum_memory_required": raw_min}
                )
                _, _, _, cal_lora, _ = self.dyn._calibrate_inference_budget(
                    mm, [model], (), lora_kwargs
                )
            self.assertAlmostEqual(cal_lora - cal_base, expected_delta, delta=2)
            self.assertLessEqual(cal_lora, int(total_b * 0.40))

    def test_chain_3_dual_model_high_low_swap_and_static_vae(self):
        for total_gib in SKUS_GIB:
            sim = self._new_sim(total_gib, 1.0, enable_boundary_trim=True)
            events = [
                sim.step("HighModel", dynamic=True, model_size=int(14.0 * GIB), working_vram=int(5.0 * GIB), minimum_vram=int(5.0 * GIB)),
                sim.step("LowModel", dynamic=True, model_size=int(14.0 * GIB), working_vram=int(5.0 * GIB), minimum_vram=int(5.0 * GIB)),
                sim.step("StaticVAE", dynamic=False, model_size=int(1.5 * GIB), working_vram=int(4.0 * GIB)),
                sim.step("HighModel", dynamic=True, model_size=int(14.0 * GIB), working_vram=int(5.0 * GIB), minimum_vram=int(5.0 * GIB)),
            ]
            self._assert_step_invariants(events, context=f"chain3-{total_gib}G")
            if total_gib in (12, 16):
                # Tighter SKUs must evict HighModel cleanly when StaticVAE loads
                self.assertLess(sim.vbar_physical["HighModel"], int(14.0 * GIB))
            if total_gib == 32:
                # 32 GiB retains partial warm residency across LowModel / StaticVAE transitions
                self.assertGreater(events[2]["phys_used_gib"], 0.0)
                self.assertGreater(events[3]["warm_hit_gib"], 0.0)

    def test_chain_4_consecutive_static_models_high_reserve_protection(self):
        for total_gib in SKUS_GIB:
            for reserve_gib in RESERVES_GIB:
                sim = self._new_sim(total_gib, reserve_gib, enable_boundary_trim=True)
                events = [
                    sim.step("DynDiT", dynamic=True, model_size=int(22.0 * GIB), working_vram=int(4.0 * GIB), minimum_vram=int(4.0 * GIB)),
                    sim.step("StaticA", dynamic=False, model_size=int(1.0 * GIB), working_vram=int(1.5 * GIB)),
                    sim.step("StaticB", dynamic=False, model_size=int(2.5 * GIB), working_vram=int(5.0 * GIB)),
                    sim.step("DynDiT", dynamic=True, model_size=int(22.0 * GIB), working_vram=int(4.0 * GIB), minimum_vram=int(4.0 * GIB)),
                ]
                self._assert_step_invariants(
                    events, context=f"chain4-{total_gib}G-rsv{reserve_gib}G"
                )


class TestTransitionGuardInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_torch = _make_fake_torch(xpu_available=True, cuda_available=False)
        cls.dyn = _load_repo_module(
            "adapters.dynamic_vram", "adapters/dynamic_vram.py", fake_torch=fake_torch
        )

    def test_calibration_monotonicity_knee_and_cap_across_skus(self):
        xpu0 = _FakeDevice("xpu", 0)
        model = _FakeModel(dynamic=True, device=xpu0, size=10 * GIB)
        expected_knees = {12: 3 * GIB, 16: 4 * GIB, 24: 6 * GIB, 32: 6 * GIB}
        sweep_mib = sorted(
            set(range(1024, 20 * 1024 + 1, 512))
            | {3071, 3072, 3073, 4095, 4096, 4097, 6143, 6144, 6145}
        )

        for total_gib in SKUS_GIB:
            total_b = total_gib * GIB
            cap = int(total_b * 0.40)
            exp_knee = expected_knees[total_gib]
            mm = _make_mm(total_vram=total_b, reserve_vram=1 * GIB, flash=True)
            prev = -1
            with (
                mock.patch.dict(os.environ, {"OMNIXPU_DYNAMIC_VRAM_BUDGET_CALIBRATION": "1"}, clear=False),
                mock.patch.object(self.dyn, "_device_total_vram", return_value=total_b),
            ):
                os.environ.pop("OMNIXPU_AIMDO_BUDGET_KNEE_MIB", None)
                for mib in sweep_mib:
                    raw = mib * MIB
                    _, _, seen_raw, cal, changed = self.dyn._calibrate_inference_budget(
                        mm, [model], (), {"memory_required": total_b, "minimum_memory_required": raw}
                    )
                    self.assertEqual(seen_raw, raw)
                    self.assertLessEqual(cal, raw)
                    self.assertGreaterEqual(cal, prev, f"non-monotonic on {total_gib}G at {mib}MiB")
                    if raw <= exp_knee:
                        self.assertFalse(changed)
                        self.assertEqual(cal, raw)
                    else:
                        self.assertLessEqual(cal, cap)
                    prev = cal

    def test_get_free_memory_bounds_and_aimdo_disabled_fallback(self):
        xpu0 = _FakeDevice("xpu", 0)
        for total_gib in SKUS_GIB:
            total_b = total_gib * GIB
            hw_free = int(total_b * 0.50)
            torch_active = int(total_b * 0.15)
            # Deliberately over-report vbar_phys above remaining headroom to verify clamp
            vbar_reported = int(total_b * 0.70)

            fake_xpu = types.SimpleNamespace(
                current_device=lambda: 0,
                mem_get_info=lambda _d=0, f=hw_free, t=total_b: (f, t),
                memory_stats=lambda _d=0, a=torch_active: {
                    "reserved_bytes.all.current": a,
                    "active_bytes.all.current": a,
                },
            )
            fake_vbar = types.SimpleNamespace(vbars_analyze=lambda _idx, v=vbar_reported: v)
            mm = _make_mm(total_vram=total_b, reserve_vram=1 * GIB)
            mm.get_free_memory = lambda _d=None, torch_free_too=False, t=total_b, a=torch_active: (
                (t - a, 0) if torch_free_too else (t - a)
            )
            mp_cls = type(
                "_ClampMP",
                (),
                {"get_free_memory": lambda self_mp, dev, m=mm, v=fake_vbar: m.get_free_memory(dev) + v.vbars_analyze(0)},
            )
            mem_mgr = types.SimpleNamespace(aimdo_enabled=True)

            with (
                mock.patch.object(self.dyn.torch, "xpu", fake_xpu, create=True),
                mock.patch.object(self.dyn, "_aimdo_model_vbar", fake_vbar),
                mock.patch.dict(os.environ, {"OMNIXPU_DYNAMIC_VRAM_GET_FREE_MEMORY": "1"}, clear=False),
            ):
                self.dyn._patch_get_free_memory(mm, memory_management=mem_mgr)
                self.dyn._patch_model_patcher_get_free_memory(mp_cls, mm, mem_mgr)
                free_total = mm.get_free_memory(xpu0)
                mp_free = mp_cls().get_free_memory(xpu0)
                self.assertEqual(free_total, hw_free)
                self.assertEqual(mp_free, total_b)
                self.assertGreaterEqual(free_total, 0)
                self.assertLessEqual(free_total, mp_free)
                self.assertLessEqual(mp_free, total_b)

                # When aimdo_enabled=False, ModelPatcher.get_free_memory adds zero VBAR
                mem_mgr.aimdo_enabled = False
                self.assertEqual(mp_cls().get_free_memory(xpu0), free_total)

                # Fallback when mem_get_info fails also skips VBAR subtraction if aimdo_enabled=False
                fake_xpu_err = types.SimpleNamespace(
                    mem_get_info=mock.Mock(side_effect=RuntimeError("driver unavailable")),
                    memory_stats=lambda _d=0: {"reserved_bytes.all.current": 0, "active_bytes.all.current": 0},
                )
                with (
                    mock.patch.object(self.dyn.torch, "xpu", fake_xpu_err, create=True),
                    mock.patch.dict(sys.modules, {"comfy.memory_management": mem_mgr}),
                ):
                    free_fb, _ = self.dyn._xpu_actual_free_and_torch(
                        mm, xpu0, original_get_free=lambda _d: total_b - torch_active
                    )
                self.assertEqual(free_fb, total_b - torch_active)

    def test_windows_apply_hooks_and_clean_sys_modules_restoration(self):
        fake_torch = _make_fake_torch(xpu_available=True, cuda_available=False)
        dyn = _load_repo_module("adapters.dynamic_vram_apply_probe", "adapters/dynamic_vram.py", fake_torch=fake_torch)
        lora = _load_repo_module("adapters.lora_memory_apply_probe", "adapters/lora_memory.py", fake_torch=fake_torch)

        comfy_pkg = types.ModuleType("comfy")
        comfy_mm = types.ModuleType("comfy.model_management")
        comfy_mm.current_loaded_models = []
        comfy_mm.extra_reserved_memory = lambda: 1 * GIB
        comfy_mm.get_free_memory = lambda dev=None: 16 * GIB
        comfy_mm.load_models_gpu = lambda models, *a, **kw: "ok"
        comfy_mm.minimum_inference_memory = lambda: int(1.8 * GIB)
        comfy_mm.get_torch_device = lambda: _FakeDevice("xpu", 0)
        comfy_mm.soft_empty_cache = lambda: None

        comfy_mem = types.ModuleType("comfy.memory_management")
        comfy_mem.aimdo_enabled = True

        class _FakeMP:
            def get_free_memory(self, dev):
                return comfy_mm.get_free_memory(dev)

        comfy_mp = types.ModuleType("comfy.model_patcher")
        comfy_mp.ModelPatcher = _FakeMP

        comfy_sd = types.ModuleType("comfy.sd")
        comfy_sd.load_lora_for_models = lambda model, clip, lora_data, strength_model, strength_clip: (model, clip)
        comfy_ops = types.ModuleType("comfy.ops")

        comfy_pkg.model_management = comfy_mm
        comfy_pkg.memory_management = comfy_mem
        comfy_pkg.model_patcher = comfy_mp
        comfy_pkg.sd = comfy_sd
        comfy_pkg.ops = comfy_ops

        fake_aimdo = types.ModuleType("comfy_aimdo")
        fake_aimdo_vbar = types.ModuleType("comfy_aimdo.model_vbar")
        fake_aimdo_vbar.vbars_analyze = lambda _idx: 0
        fake_aimdo_vbar.vbar_fault = lambda _alloc: None
        fake_aimdo.model_vbar = fake_aimdo_vbar

        before_keys = set(sys.modules.keys())
        stub_modules = {
            "torch": fake_torch,
            "comfy": comfy_pkg,
            "comfy.model_management": comfy_mm,
            "comfy.memory_management": comfy_mem,
            "comfy.model_patcher": comfy_mp,
            "comfy.sd": comfy_sd,
            "comfy.ops": comfy_ops,
            "comfy_aimdo": fake_aimdo,
            "comfy_aimdo.model_vbar": fake_aimdo_vbar,
        }
        with (
            mock.patch.object(sys, "platform", "win32"),
            mock.patch.dict(sys.modules, stub_modules),
            mock.patch.object(dyn, "_aimdo_model_vbar", fake_aimdo_vbar),
        ):
            ok_dyn, _ = dyn.apply()
            ok_lora, _ = lora.apply()
            self.assertTrue(ok_dyn)
            self.assertTrue(ok_lora)
            self.assertTrue(hasattr(comfy_mm.load_models_gpu, dyn._PATCH_MARKER) or hasattr(comfy_mm.load_models_gpu, lora._PATCH_MARKER))
            self.assertTrue(hasattr(comfy_mm.get_free_memory, dyn._GET_FREE_MEMORY_PATCH_MARKER))
            self.assertTrue(hasattr(_FakeMP.get_free_memory, dyn._MP_GET_FREE_MEMORY_PATCH_MARKER))

        self.assertEqual(set(sys.modules.keys()), before_keys)


if __name__ == "__main__":
    unittest.main()
