"""Unit and environment-safety tests for ComfyUI-Intel-Arc-XPU (no GPU required)."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_mock_torch(*, xpu_available: bool = True, cuda_available: bool = False):
    torch_mod = types.ModuleType("torch")
    torch_mod.__version__ = "2.13.0+xpu"
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.xpu = types.SimpleNamespace(
        is_available=lambda: xpu_available,
        memory_allocated=lambda dev=0: 0,
        memory_reserved=lambda dev=0: 0,
        max_memory_allocated=lambda dev=0: 0,
        max_memory_reserved=lambda dev=0: 0,
        empty_cache=lambda: None,
        synchronize=lambda: None,
        mem_get_info=lambda dev=0: (24 * 1024**3, 32 * 1024**3),
    )
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    return torch_mod


def _load_repo_module(name: str, rel_path: str, *, fake_torch=None):
    pkg_name = "omnixpu_ci_pkg"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(REPO_ROOT)]
        sys.modules[pkg_name] = pkg
    adapters_pkg = f"{pkg_name}.adapters"
    if adapters_pkg not in sys.modules:
        apkg = types.ModuleType(adapters_pkg)
        apkg.__path__ = [str(REPO_ROOT / "adapters")]
        sys.modules[adapters_pkg] = apkg

    cfg_name = f"{pkg_name}.config"
    if cfg_name not in sys.modules:
        cfg_spec = importlib.util.spec_from_file_location(cfg_name, REPO_ROOT / "config.py")
        cfg_mod = importlib.util.module_from_spec(cfg_spec)
        sys.modules[cfg_name] = cfg_mod
        cfg_spec.loader.exec_module(cfg_mod)

    full_name = f"{pkg_name}.{name}"
    spec = importlib.util.spec_from_file_location(full_name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    if fake_torch is not None:
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            spec.loader.exec_module(mod)
    else:
        spec.loader.exec_module(mod)
    return mod


class ExtensionContractTests(unittest.TestCase):
    def test_config_defaults_and_opt_out(self):
        cfg_mod = _load_repo_module("config_probe", "config.py")
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = cfg_mod.Config()
            self.assertTrue(cfg.dynamic_vram_torch_cache_trim)
            self.assertTrue(cfg.dynamic_vram_budget_calibration)
            self.assertTrue(cfg.lora_min_budget_peak)
            self.assertTrue(cfg.custom_nodes_xpu)
            self.assertFalse(cfg.dynamic_vram_step_trace)
        with mock.patch.dict(os.environ, {"OMNIXPU_DYNAMIC_VRAM_TORCH_CACHE_TRIM": "0"}, clear=True):
            cfg_off = cfg_mod.Config()
            self.assertFalse(cfg_off.dynamic_vram_torch_cache_trim)

    def test_custom_nodes_xpu_skips_cleanly_when_optional_packs_absent(self):
        fake_torch = _make_mock_torch(xpu_available=True, cuda_available=False)
        mod = _load_repo_module("adapters.custom_nodes_xpu", "adapters/custom_nodes_xpu.py", fake_torch=fake_torch)
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            applied = mod.patch_loaded_custom_nodes()
            self.assertEqual(applied, [])
            ok, msg = mod.apply()
            self.assertTrue(ok)
            self.assertEqual(msg, "")

    def test_custom_nodes_xpu_disabled_when_xpu_unavailable_or_cuda_present(self):
        no_xpu = _make_mock_torch(xpu_available=False, cuda_available=False)
        mod = _load_repo_module("adapters.custom_nodes_xpu_noxpu", "adapters/custom_nodes_xpu.py", fake_torch=no_xpu)
        with mock.patch.dict(sys.modules, {"torch": no_xpu}):
            self.assertEqual(mod.patch_loaded_custom_nodes(), [])
            ok, _ = mod.apply()
            self.assertFalse(ok)

        both = _make_mock_torch(xpu_available=True, cuda_available=True)
        with mock.patch.dict(sys.modules, {"torch": both}):
            self.assertEqual(mod.patch_loaded_custom_nodes(), [])

    def test_dynamic_vram_calibration_and_safety(self):
        fake_torch = _make_mock_torch(xpu_available=True, cuda_available=False)
        dyn = _load_repo_module("adapters.dynamic_vram", "adapters/dynamic_vram.py", fake_torch=fake_torch)
        # Small models below 6 GiB knee stay untouched (calibrated == False, raw == cal)
        _, _, raw_min, cal_min, calibrated = dyn._calibrate_inference_budget(
            types.SimpleNamespace(), [], (), {"minimum_memory_required": 2 * 1024**3}
        )
        self.assertFalse(calibrated)
        self.assertEqual(raw_min, 2 * 1024**3)
        self.assertEqual(cal_min, raw_min)
        # When Flash/CUTE attention is off, skip calibration; when cute, detect it
        mm = types.SimpleNamespace(pytorch_attention_flash_attention=lambda: False)
        with mock.patch.dict(os.environ, {"OMNI_ATTN_BACKEND": "torch"}, clear=False):
            self.assertFalse(dyn._is_flash_attention_active(mm))
        with mock.patch.dict(os.environ, {"OMNI_ATTN_BACKEND": "cute"}, clear=False):
            self.assertTrue(dyn._is_flash_attention_active(mm))

    def test_docs_and_installer_reserve_vram_and_sol_attn_consistent(self):
        for rel in ("__init__.py", "install.py", "install.ps1", "README.md", "WORKFLOW_COMPATIBILITY.md", "pyproject.toml"):
            txt = (REPO_ROOT / rel).read_text(encoding="utf-8")
            self.assertNotIn("--reserve-vram 4", txt, f"stale --reserve-vram 4 found in {rel}")
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        compat = (REPO_ROOT / "WORKFLOW_COMPATIBILITY.md").read_text(encoding="utf-8")
        self.assertNotIn("CUDA-only `sol-attn`", readme)
        self.assertNotIn("(`sol-attn`", compat)

    def test_installer_environment_contract(self):
        inst = _load_repo_module("install_mod", "install.py")
        ok, _ = inst.check_environment(plat="win32", py_ver=(3, 13), ptr_bytes=8, torch_ver="2.13.0+xpu")
        self.assertTrue(ok)
        ok_bad, msg = inst.check_environment(plat="win32", py_ver=(3, 12), ptr_bytes=8, torch_ver="2.13.0+xpu")
        self.assertFalse(ok_bad)
        self.assertIn("CONFLICT:", msg)
        hashes = inst.resolve_expected_hashes(REPO_ROOT)
        self.assertIn("comfy_aimdo_xpu_runtime-0.6.0-cp39-abi3-win_amd64.whl", hashes)


if __name__ == "__main__":
    unittest.main()
