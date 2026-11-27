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
    fixes_pkg = f"{pkg_name}.fixes"
    if fixes_pkg not in sys.modules:
        fpkg = types.ModuleType(fixes_pkg)
        fpkg.__path__ = [str(REPO_ROOT / "fixes")]
        sys.modules[fixes_pkg] = fpkg

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
            self.assertTrue(cfg.dynamic_vram_non_dynamic_trim)
            self.assertTrue(cfg.dynamic_vram_get_free_memory)
            self.assertTrue(cfg.lora_min_budget_peak)
            self.assertTrue(cfg.custom_nodes_xpu)
            self.assertTrue(cfg.te_long_sdpa_bf16)
            self.assertTrue(cfg.h3_cute_q_chunk)
            self.assertFalse(cfg.dynamic_vram_step_trace)
        with mock.patch.dict(os.environ, {"OMNIXPU_DYNAMIC_VRAM_TORCH_CACHE_TRIM": "0"}, clear=True):
            cfg_off = cfg_mod.Config()
            self.assertFalse(cfg_off.dynamic_vram_torch_cache_trim)
        with mock.patch.dict(os.environ, {"OMNIXPU_DYNAMIC_VRAM_NON_DYNAMIC_TRIM": "0"}, clear=True):
            cfg_nondyn_off = cfg_mod.Config()
            self.assertFalse(cfg_nondyn_off.dynamic_vram_non_dynamic_trim)
        with mock.patch.dict(os.environ, {"OMNIXPU_DYNAMIC_VRAM_GET_FREE_MEMORY": "0"}, clear=True):
            cfg_free_off = cfg_mod.Config()
            self.assertFalse(cfg_free_off.dynamic_vram_get_free_memory)
        with mock.patch.dict(os.environ, {"OMNIXPU_TE_LONG_SDPA_BF16": "0"}, clear=True):
            cfg_te_off = cfg_mod.Config()
            self.assertFalse(cfg_te_off.te_long_sdpa_bf16)
        with mock.patch.dict(os.environ, {"OMNIXPU_H3_CUTE_Q_CHUNK": "0"}, clear=True):
            cfg_h3_off = cfg_mod.Config()
            self.assertFalse(cfg_h3_off.h3_cute_q_chunk)

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
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("MANDATORY:", pyproject)
        self.assertIn("https://github.com/wildersa/ComfyUI-Intel-Arc-XPU", pyproject)
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

    def test_te_long_sdpa_bf16_routes_only_long_xpu_float32_and_preserves_gqa(self):
        import contextlib

        class _FakeTensor:
            def __init__(self, shape, dtype, device_type="xpu"):
                self.shape = shape
                self.dtype = dtype
                self.device = types.SimpleNamespace(type=device_type)
                self.ndim = len(shape)

            def is_floating_point(self):
                return self.dtype in ("float32", "bfloat16")

            def to(self, dtype):
                return _FakeTensor(self.shape, dtype, self.device.type)

            def unsqueeze(self, dim):
                s = list(self.shape)
                s.insert(dim, 1)
                return _FakeTensor(tuple(s), self.dtype, self.device.type)

            def transpose(self, d0, d1):
                s = list(self.shape)
                s[d0], s[d1] = s[d1], s[d0]
                return _FakeTensor(tuple(s), self.dtype, self.device.type)

            def reshape(self, *shape):
                if -1 in shape:
                    total = 1
                    for dim in self.shape:
                        total *= dim
                    known = 1
                    for dim in shape:
                        if dim != -1:
                            known *= dim
                    shape = tuple(total // known if dim == -1 else dim for dim in shape)
                return _FakeTensor(shape, self.dtype, self.device.type)

        sdpa_calls = []
        base_calls = []
        kernel_contexts = []

        @contextlib.contextmanager
        def _sdpa_kernel(backends):
            kernel_contexts.append(list(backends))
            yield

        def _fake_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
            sdpa_calls.append((q, k, v, attn_mask, dropout_p, is_causal, kw))
            return _FakeTensor(q.shape, q.dtype, q.device.type)

        fake_torch = _make_mock_torch(xpu_available=True, cuda_available=False)
        fake_torch.float32 = "float32"
        fake_torch.bool = "bool"
        fake_torch.xpu.is_bf16_supported = lambda: True
        fake_torch.compiler = types.SimpleNamespace(is_compiling=lambda: False)
        fake_torch.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
        fake_torch.nn = types.SimpleNamespace(
            attention=types.SimpleNamespace(
                SDPBackend=types.SimpleNamespace(OVERRIDEABLE="OVERRIDEABLE"),
                sdpa_kernel=_sdpa_kernel,
            ),
            functional=types.SimpleNamespace(scaled_dot_product_attention=_fake_sdpa),
        )

        def _wrap_attn(fn):
            return fn

        def _base_attention_pytorch(q, k, v, heads, mask=None, **kwargs):
            base_calls.append((q, k, v, heads, mask, kwargs))
            return q

        fake_attn_mod = types.SimpleNamespace(
            attention_pytorch=_base_attention_pytorch,
            wrap_attn=_wrap_attn,
            SDP_BATCH_LIMIT=512,
        )
        fake_llama = types.SimpleNamespace(
            optimized_attention_for_device=lambda device, mask=False, small_input=False: _base_attention_pytorch
        )

        _load_repo_module("adapters.errors", "adapters/errors.py", fake_torch=fake_torch)
        mod = _load_repo_module("fixes.te_long_sdpa_bf16", "fixes/te_long_sdpa_bf16.py", fake_torch=fake_torch)
        with mock.patch.object(mod, "torch", fake_torch):
            self.assertTrue(
                mod.patch_text_encoder_module(
                    fake_llama, fake_attn_mod, _sdpa_kernel, "OVERRIDEABLE", 1024
                )
            )
            # Idempotent re-patch returns True without double-wrapping.
            wrapped_once = fake_llama.optimized_attention_for_device
            self.assertTrue(
                mod.patch_text_encoder_module(
                    fake_llama, fake_attn_mod, _sdpa_kernel, "OVERRIDEABLE", 1024
                )
            )
            self.assertIs(fake_llama.optimized_attention_for_device, wrapped_once)

            attn_fn = fake_llama.optimized_attention_for_device(
                types.SimpleNamespace(type="xpu"), mask=True, small_input=True
            )

            # 1. Short sequence (< 1024) stays on stock float32 path.
            q_short = _FakeTensor((1, 64, 512, 128), "float32", "xpu")
            k_short = _FakeTensor((1, 8, 512, 128), "float32", "xpu")
            v_short = _FakeTensor((1, 8, 512, 128), "float32", "xpu")
            attn_fn(q_short, k_short, v_short, 64, skip_reshape=True, enable_gqa=True)
            self.assertEqual(len(base_calls), 1)
            self.assertEqual(len(sdpa_calls), 0)

            # 2. Non-XPU stays on stock path even when >= 1024.
            q_cpu = _FakeTensor((1, 64, 1024, 128), "float32", "cpu")
            k_cpu = _FakeTensor((1, 8, 1024, 128), "float32", "cpu")
            v_cpu = _FakeTensor((1, 8, 1024, 128), "float32", "cpu")
            attn_fn(q_cpu, k_cpu, v_cpu, 64, skip_reshape=True, enable_gqa=True)
            self.assertEqual(len(base_calls), 2)
            self.assertEqual(len(sdpa_calls), 0)

            # 3. Long XPU float32 uses bf16 OVERRIDEABLE without expanding KV and returns float32.
            q_long = _FakeTensor((1, 64, 2048, 128), "float32", "xpu")
            k_long = _FakeTensor((1, 8, 2048, 128), "float32", "xpu")
            v_long = _FakeTensor((1, 8, 2048, 128), "float32", "xpu")
            mask_long = _FakeTensor((2048, 2048), "float32", "xpu")
            out = attn_fn(
                q_long, k_long, v_long, 64, mask=mask_long,
                skip_reshape=True, skip_output_reshape=False, enable_gqa=True
            )
            self.assertEqual(len(sdpa_calls), 1)
            sq, sk, sv, sm, _, _, skw = sdpa_calls[0]
            self.assertEqual(sq.dtype, "bfloat16")
            self.assertEqual(sk.dtype, "bfloat16")
            self.assertEqual(sk.shape[1], 8)  # native GQA kept (not expanded to 64)
            self.assertEqual(sm.dtype, "bfloat16")
            self.assertEqual(sm.shape, (1, 1, 2048, 2048))
            self.assertEqual(skw, {"enable_gqa": True})
            self.assertEqual(kernel_contexts, [["OVERRIDEABLE"]])
            self.assertEqual(out.dtype, "float32")
            self.assertEqual(out.shape, (1, 2048, 64 * 128))

            # 4. Non-fatal error falls back cleanly; fatal OOM re-raises.
            def _non_fatal(*a, **kw):
                raise RuntimeError("unsupported kernel contract")

            fake_torch.nn.functional.scaled_dot_product_attention = _non_fatal
            attn_fn(q_long, k_long, v_long, 64, skip_reshape=True, enable_gqa=True)
            self.assertEqual(len(base_calls), 3)

            def _fatal_oom(*a, **kw):
                raise fake_torch.OutOfMemoryError("XPU out of memory")

            fake_torch.nn.functional.scaled_dot_product_attention = _fatal_oom
            with self.assertRaises(fake_torch.OutOfMemoryError):
                attn_fn(q_long, k_long, v_long, 64, skip_reshape=True, enable_gqa=True)

    def test_h3_cute_q_chunk_slices_only_above_threshold_and_preallocates(self):
        class _ChunkTensor:
            def __init__(self, shape, dtype="bfloat16", device="xpu:0"):
                self.shape = tuple(shape)
                self.dtype = dtype
                self.device = device
                self.copied_slices = []

            def __getitem__(self, item):
                if isinstance(item, tuple) and len(item) == 4:
                    sl = item[2]
                    return _ChunkTensor(
                        (self.shape[0], self.shape[1], sl.stop - sl.start, self.shape[3]),
                        self.dtype,
                        self.device,
                    )
                if isinstance(item, tuple) and len(item) == 3:
                    sl = item[1]
                    target = self
                    return types.SimpleNamespace(
                        copy_=lambda src: target.copied_slices.append((sl.start, sl.stop, src.shape))
                    )
                raise AssertionError(f"unexpected slice: {item}")

            def transpose(self, d0, d1):
                s = list(self.shape)
                s[d0], s[d1] = s[d1], s[d0]
                return _ChunkTensor(tuple(s), self.dtype, self.device)

            def reshape(self, *shape):
                if -1 in shape:
                    total = 1
                    for dim in self.shape:
                        total *= dim
                    known = 1
                    for dim in shape:
                        if dim != -1:
                            known *= dim
                    resolved = tuple((total // known) if dim == -1 else dim for dim in shape)
                    return _ChunkTensor(resolved, self.dtype, self.device)
                return _ChunkTensor(shape, self.dtype, self.device)

        sync_calls = []
        sdp_calls = []
        fake_torch = _make_mock_torch(xpu_available=True, cuda_available=False)
        fake_torch.xpu.synchronize = lambda dev=0: sync_calls.append(dev)
        fake_torch.compiler = types.SimpleNamespace(is_compiling=lambda: False)
        fake_torch.empty = lambda shape, device=None, dtype=None: _ChunkTensor(shape, dtype, device)

        attn_mod = _load_repo_module(
            "adapters.attention_qchunk", "adapters/attention.py", fake_torch=fake_torch
        )
        attn_mod._h3_q_chunk_log_count = 0
        attn_mod._backend_sdp = types.SimpleNamespace(
            sdp_bhld_d128=lambda q, k, v: (
                sdp_calls.append((q.shape, k.shape, v.shape))
                or _ChunkTensor(q.shape, q.dtype, q.device)
            )
        )

        # Below default min_seq (44000): runs single unchunked dispatch without sync.
        q_short = _ChunkTensor((1, 56, 43000, 128))
        out_short = attn_mod._run_minimax_h3_cute_d128(
            q_short, q_short, q_short, b=1, heads=56, q_len=43000, kv_len=43000, dim_head=128, skip_output_reshape=False
        )
        self.assertEqual(len(sdp_calls), 1)
        self.assertEqual(sdp_calls[0][0], (1, 56, 43000, 128))
        self.assertEqual(out_short.shape, (1, 43000, 56 * 128))
        self.assertEqual(sync_calls, [])

        # Above default min_seq (44000) with default chunk_size (16384): chunks Q and syncs.
        sdp_calls.clear()
        q_mid = _ChunkTensor((1, 56, 45000, 128))
        with mock.patch.dict(
            os.environ,
            {
                "OMNIXPU_H3_CUTE_Q_CHUNK": "1",
                "OMNIXPU_H3_CUTE_Q_CHUNK_SYNC": "1",
            },
            clear=False,
        ):
            out_mid = attn_mod._run_minimax_h3_cute_d128(
                q_mid, q_mid, q_mid, b=1, heads=56, q_len=45000, kv_len=45000, dim_head=128, skip_output_reshape=False
            )
        self.assertEqual([c[0][2] for c in sdp_calls], [16384, 16384, 12232])
        self.assertEqual(len(sync_calls), 3)
        self.assertEqual(out_mid.shape, (1, 45000, 56 * 128))

        # Explicit overrides also chunk Q along seq and write directly into pre-allocated output.
        sdp_calls.clear()
        sync_calls.clear()
        q_long = _ChunkTensor((1, 56, 50000, 128))
        with mock.patch.dict(
            os.environ,
            {
                "OMNIXPU_H3_CUTE_Q_CHUNK": "1",
                "OMNIXPU_H3_CUTE_Q_CHUNK_MIN_SEQ": "48000",
                "OMNIXPU_H3_CUTE_Q_CHUNK_SIZE": "20480",
                "OMNIXPU_H3_CUTE_Q_CHUNK_SYNC": "1",
            },
            clear=False,
        ):
            out_long = attn_mod._run_minimax_h3_cute_d128(
                q_long, q_long, q_long, b=1, heads=56, q_len=50000, kv_len=50000, dim_head=128, skip_output_reshape=False
            )
        self.assertEqual([c[0][2] for c in sdp_calls], [20480, 20480, 9040])
        self.assertTrue(all(c[1] == (1, 56, 50000, 128) for c in sdp_calls))
        self.assertEqual(out_long.copied_slices, [(0, 20480, (1, 20480, 7168)), (20480, 40960, (1, 20480, 7168)), (40960, 50000, (1, 9040, 7168))])
        self.assertEqual(len(sync_calls), 3)

        # Opt-out env disables Q-chunking cleanly.
        sdp_calls.clear()
        sync_calls.clear()
        with mock.patch.dict(os.environ, {"OMNIXPU_H3_CUTE_Q_CHUNK": "0"}, clear=False):
            attn_mod._run_minimax_h3_cute_d128(
                q_long, q_long, q_long, b=1, heads=56, q_len=50000, kv_len=50000, dim_head=128, skip_output_reshape=False
            )
        self.assertEqual(len(sdp_calls), 1)
        self.assertEqual(sync_calls, [])

    def test_h3_rms_modulation_collapses_contiguous_tensor_runs(self):
        class _FakeIntVec:
            def __init__(self, values):
                self._values = list(values)

            def tolist(self):
                return list(self._values)

            def numel(self):
                return len(self._values)

        class _FakeRowTensor:
            def __init__(self, values):
                self._values = list(values)
                self.ndim = 1
                self.dtype = "int64"
                self.requires_grad = False
                self._version = 0

            def numel(self):
                return len(self._values)

        def _fake_unique_consecutive(tensor, return_counts=False):
            vals, counts = [], []
            for v in tensor._values:
                if not vals or vals[-1] != v:
                    vals.append(v)
                    counts.append(1)
                else:
                    counts[-1] += 1
            return _FakeIntVec(vals), _FakeIntVec(counts)

        fake_torch = _make_mock_torch(xpu_available=True, cuda_available=False)
        fake_torch.Tensor = _FakeRowTensor
        fake_torch.int64 = "int64"
        fake_torch.int32 = "int32"
        fake_torch.int16 = "int16"
        fake_torch.int8 = "int8"
        fake_torch.uint8 = "uint8"
        fake_torch.unique_consecutive = _fake_unique_consecutive

        mod_mod = _load_repo_module(
            "adapters.h3_rms_mod_collapse", "adapters/h3_rms_modulation.py", fake_torch=fake_torch
        )
        row_tensor = _FakeRowTensor([3, 3, 3, 6, 6, 9, 9, 9, 9])
        segments = [(0, 10, 1), (10, 19, row_tensor)]
        collapsed = mod_mod._collapse_mod_segments(segments)
        self.assertEqual(collapsed, [(0, 10, 1), (10, 13, 3), (13, 15, 6), (15, 19, 9)])
        self.assertEqual(mod_mod._segment_reason(collapsed, 19, 12), "")
        # Second call reuses the weakref cache for the same tensor object.
        self.assertIsNotNone(mod_mod._collapse_mod_segments(segments))

        # Inference tensors disallow _version access and must not raise RuntimeError.
        class _FakeInferenceRowTensor:
            def __init__(self, values):
                self._values = list(values)
                self.ndim = 1
                self.dtype = "int64"
                self.requires_grad = False

            def numel(self):
                return len(self._values)

            def is_inference(self):
                return True

            @property
            def _version(self):
                raise RuntimeError("Inference tensors do not track version counter.")

        fake_torch.Tensor = (_FakeRowTensor, _FakeInferenceRowTensor)
        inf_tensor = _FakeInferenceRowTensor([4, 4, 7, 7])
        inf_segments = [(0, 4, inf_tensor)]
        self.assertEqual(mod_mod._collapse_mod_segments(inf_segments), [(0, 2, 4), (2, 4, 7)])
        self.assertEqual(mod_mod._collapse_mod_segments(inf_segments), [(0, 2, 4), (2, 4, 7)])

    def test_qsv_video_encode_adapter_patches_and_falls_back_cleanly(self):
        import enum
        import types

        class _VideoCodec(enum.Enum):
            AUTO = "auto"
            H264 = "h264"
            AV1 = "av1"
            PRORES = "prores"

        orig_calls = []

        class _VideoFromComponents:
            def save_to(self, path, *args, **kwargs):
                orig_calls.append(
                    ("components", fake_vt.VIDEO_ENCODERS[_VideoCodec.H264], kwargs.get("bit_depth"))
                )
                if fake_vt.VIDEO_ENCODERS[_VideoCodec.H264] == "h264_qsv" and kwargs.get("force_fail"):
                    raise RuntimeError("simulated qsv error")
                return path

        class _VideoFromFile:
            def save_to(self, path, *args, **kwargs):
                orig_calls.append(
                    ("file", fake_vt.VIDEO_ENCODERS[_VideoCodec.H264], kwargs.get("bit_depth"))
                )
                return path

        fake_vt = types.ModuleType("comfy_api.latest._input_impl.video_types")
        fake_vt.VIDEO_ENCODERS = {
            _VideoCodec.H264: "h264",
            _VideoCodec.AV1: "libsvtav1",
            _VideoCodec.PRORES: "prores_ks",
        }
        fake_vt.video_pixel_format = lambda codec, bit_depth, chroma_subsampling=None, has_alpha=False: "yuv420p"
        fake_vt.video_encoder_options = lambda codec, crf, preset=None, pix_fmt=None: {
            "crf": str(18 if crf is None else crf)
        }
        fake_vt.VideoFromComponents = _VideoFromComponents
        fake_vt.VideoFromFile = _VideoFromFile

        fake_util = types.ModuleType("comfy_api.latest._util")
        fake_util.VideoCodec = _VideoCodec

        fake_ca = types.ModuleType("comfy_api")
        fake_ca_latest = types.ModuleType("comfy_api.latest")
        fake_ca_impl = types.ModuleType("comfy_api.latest._input_impl")

        ve = _load_repo_module("adapters.video_encode_test", "adapters/video_encode.py")
        ve._PATCHED = False

        with mock.patch.dict(
            sys.modules,
            {
                "comfy_api": fake_ca,
                "comfy_api.latest": fake_ca_latest,
                "comfy_api.latest._util": fake_util,
                "comfy_api.latest._input_impl": fake_ca_impl,
                "comfy_api.latest._input_impl.video_types": fake_vt,
            },
        ), mock.patch.object(ve, "_probe_h264_qsv", return_value=(True, "")):
            ok, reason = ve.apply()
            self.assertTrue(ok, reason)
            self.assertEqual(fake_vt.VIDEO_ENCODERS[_VideoCodec.H264], "h264_qsv")
            self.assertEqual(fake_vt.video_pixel_format(_VideoCodec.H264, 8, "420", False), "nv12")
            self.assertEqual(fake_vt.video_pixel_format(_VideoCodec.H264, 10, "420", False), "yuv420p")
            self.assertEqual(
                fake_vt.video_encoder_options(_VideoCodec.H264, 20.0, preset=None, pix_fmt="nv12"),
                {"preset": "medium", "global_quality": "20"},
            )

            comp = _VideoFromComponents()
            comp.save_to("a.mp4", codec=_VideoCodec.AUTO, bit_depth=8)
            comp.save_to("b.mp4", codec=_VideoCodec.H264, bit_depth=10)
            comp.save_to("c.mp4", codec=_VideoCodec.H264, bit_depth=8, force_fail=True)
            _VideoFromFile().save_to("d.mp4", codec=_VideoCodec.AUTO)

        self.assertEqual(
            orig_calls,
            [
                ("components", "h264_qsv", 8),
                ("components", "h264", 10),
                ("components", "h264_qsv", 8),
                ("components", "h264", 8),
                ("file", "h264_qsv", None),
            ],
        )
        ok_again, _ = ve.apply()
        self.assertTrue(ok_again)

    @staticmethod
    def _make_boundary_harness(*, free_bytes, total_bytes=32 * 1024**3, reserved_bytes=1024 * 1024**2):
        mib = 1024**2

        class _FakeDevice:
            def __init__(self, dev_type="xpu", index=0):
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

        state = {
            "free": free_bytes,
            "reusable_torch": 0,
            "soft_empty_calls": 0,
            "load_calls": [],
        }

        class _FakeModel:
            def __init__(
                self,
                name,
                *,
                dynamic,
                size,
                loaded=0,
                load_device=None,
                current_device=None,
                max_freeable=None,
            ):
                self.name = name
                self._dynamic = dynamic
                self._size = size
                self._loaded = loaded
                self.load_device = load_device or _FakeDevice("xpu", 0)
                self.offload_device = _FakeDevice("cpu", None)
                self._current_device = current_device
                self._max_freeable = max_freeable
                self.unload_calls = []

            def is_dynamic(self):
                return self._dynamic

            def model_size(self):
                return self._size

            def loaded_size(self):
                return self._loaded

            def current_loaded_device(self):
                if self._current_device is not None:
                    return self._current_device
                return self.load_device if self._loaded > 0 else self.offload_device

            def model_patches_models(self):
                return []

            def partially_unload(self, offload_device, memory_to_free):
                if not self._dynamic:
                    raise AssertionError(f"partially_unload called on non-dynamic {self.name}")
                self.unload_calls.append(memory_to_free)
                cap = self._loaded if self._max_freeable is None else min(self._loaded, self._max_freeable)
                freed = max(0, min(cap, int(memory_to_free)))
                self._loaded -= freed
                state["free"] += freed
                return freed

        class _FakeLoadedModel:
            def __init__(self, model):
                self.model = model
                self.device = model.load_device

            def is_dead(self):
                return False

        fake_torch = _make_mock_torch(xpu_available=True, cuda_available=False)
        fake_torch.xpu.current_device = lambda: 0
        fake_torch.xpu.mem_get_info = lambda dev=0: (state["free"], total_bytes)
        fake_torch.xpu.memory_stats = lambda dev=0: {
            "reserved_bytes.all.current": state["reusable_torch"],
            "active_bytes.all.current": 0,
        }
        fake_torch.xpu.memory_reserved = lambda dev=0: state["reusable_torch"]
        fake_torch.xpu.max_memory_reserved = lambda dev=0: state["reusable_torch"]

        def _soft_empty_cache():
            state["soft_empty_calls"] += 1
            state["free"] += state["reusable_torch"]
            state["reusable_torch"] = 0

        mm = types.SimpleNamespace(
            current_loaded_models=[],
            extra_reserved_memory=lambda: reserved_bytes,
            minimum_inference_memory=lambda: int(0.8 * 1024**3) + reserved_bytes,
            # Simulate stock ComfyUI XPU get_free_memory ignoring AIMDO VBARs.
            get_free_memory=lambda dev=None: total_bytes - 800 * mib,
            get_total_memory=lambda dev=None: total_bytes,
            soft_empty_cache=_soft_empty_cache,
            pytorch_attention_flash_attention=lambda: False,
        )

        def _stock_load_models_gpu(models, memory_required=0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False):
            state["load_calls"].append(
                {
                    "models": list(models),
                    "memory_required": memory_required,
                    "minimum_memory_required": minimum_memory_required,
                    "force_full_load": force_full_load,
                    "free_at_load": state["free"] + state["reusable_torch"],
                }
            )

        mm.load_models_gpu = _stock_load_models_gpu
        dyn = _load_repo_module("adapters.dynamic_vram_boundary_test", "adapters/dynamic_vram.py", fake_torch=fake_torch)
        dyn._patch_model_loader(mm)
        return dyn, mm, state, _FakeDevice, _FakeModel, _FakeLoadedModel

    def test_dynamic_vram_non_dynamic_trims_inactive_vbar_and_flushes_cache(self):
        mib = 1024**2
        free_before = 4400 * mib
        dyn, mm, state, dev_cls, model_cls, loaded_cls = self._make_boundary_harness(
            free_bytes=free_before, reserved_bytes=1024 * mib
        )
        ltxav = model_cls("LTXAV", dynamic=True, size=25696 * mib, loaded=25696 * mib)
        mm.current_loaded_models = [loaded_cls(ltxav)]

        # Also exercise indexless torch.device("xpu") matching xpu:0 in current_loaded_models.
        vae = model_cls(
            "CausalDiffusionVAE",
            dynamic=False,
            size=1404 * mib,
            loaded=0,
            load_device=dev_cls("xpu", None),
        )
        expected_min_shortfall = int(3400 * mib * 1.5) + 1024 * mib + int(1.1 * 1404 * mib) - free_before
        mm.load_models_gpu([vae], memory_required=3400 * mib, force_full_load=True)

        self.assertEqual(vae.unload_calls, [])
        self.assertEqual(len(ltxav.unload_calls), 1)
        self.assertGreaterEqual(ltxav.unload_calls[0], expected_min_shortfall)
        self.assertEqual(state["soft_empty_calls"], 1)
        self.assertGreaterEqual(
            state["load_calls"][0]["free_at_load"],
            int(3400 * mib * 1.5) + 1024 * mib + int(1.1 * 1404 * mib),
        )

        # Custom scale ("1.0") and 0.85 * total_vram cap + memory_required=0 invariant.
        ltxav.unload_calls.clear()
        state["free"] = free_before
        state["soft_empty_calls"] = 0
        with mock.patch.dict(os.environ, {"OMNIXPU_DYNAMIC_VRAM_NON_DYNAMIC_SCALE": "1.0"}, clear=False):
            mm.load_models_gpu([vae], memory_required=3400 * mib, force_full_load=True)
        self.assertEqual(
            ltxav.unload_calls,
            [3400 * mib + 1024 * mib + ((1404 * mib * 11 + 9) // 10) - free_before],
        )

        # Scaled target is capped safely at 0.85 * total_vram when total_vram is known.
        capped_target = dyn._extra_memory_target(
            mm, (22000 * mib,), {}, device=dev_cls("xpu", 0), non_dynamic=True
        )
        self.assertEqual(capped_target, int(32 * 1024**3 * 0.85))

        # Small memory_required=0 loads keep minimum_inference_memory + 1.1*static.
        zero_req_target = dyn._extra_memory_target(
            mm, (0,), {}, device=dev_cls("xpu", 0), non_dynamic=True
        )
        self.assertEqual(zero_req_target, mm.minimum_inference_memory())

        # Opt-out env disables non-dynamic trimming cleanly.
        ltxav.unload_calls.clear()
        state["free"] = free_before
        state["soft_empty_calls"] = 0
        with mock.patch.dict(os.environ, {"OMNIXPU_DYNAMIC_VRAM_NON_DYNAMIC_TRIM": "0"}, clear=False):
            mm.load_models_gpu([vae], memory_required=3400 * mib, force_full_load=True)
        self.assertEqual(ltxav.unload_calls, [])
        self.assertEqual(state["soft_empty_calls"], 0)

    def test_dynamic_vram_already_loaded_non_dynamic_avoids_double_count(self):
        mib = 1024**2
        # 6200 MiB is above scaled target (1.5*3400 + 1024 = 6124 MiB) but below + 1.1*1404 (7669 MiB).
        dyn, mm, state, dev_cls, model_cls, loaded_cls = self._make_boundary_harness(
            free_bytes=6200 * mib, reserved_bytes=1024 * mib
        )
        ltxav = model_cls("LTXAV", dynamic=True, size=25696 * mib, loaded=25696 * mib)
        mm.current_loaded_models = [loaded_cls(ltxav)]
        vae_loaded = model_cls(
            "CausalDiffusionVAE",
            dynamic=False,
            size=1404 * mib,
            loaded=1404 * mib,
        )
        mm.load_models_gpu([vae_loaded], memory_required=3400 * mib, force_full_load=True)
        self.assertEqual(ltxav.unload_calls, [])
        self.assertEqual(state["soft_empty_calls"], 0)

        # When free_before drops below 6124 MiB, trim only the target shortfall (not +1.1*1404).
        state["free"] = 5500 * mib
        mm.load_models_gpu([vae_loaded], memory_required=3400 * mib, force_full_load=True)
        self.assertEqual(ltxav.unload_calls, [624 * mib])
        self.assertEqual(state["soft_empty_calls"], 1)

        # Stale loaded_size on CPU or None current_loaded_device must still count full model_size.
        stale_cpu = model_cls(
            "StaleCPU",
            dynamic=False,
            size=1404 * mib,
            loaded=1404 * mib,
            current_device=dev_cls("cpu", None),
        )
        self.assertEqual(dyn._model_unloaded_bytes(stale_cpu, dev_cls("xpu", 0)), 1404 * mib)
        stale_none = model_cls(
            "StaleNone",
            dynamic=False,
            size=1404 * mib,
            loaded=1404 * mib,
        )
        stale_none.current_loaded_device = lambda: None
        self.assertEqual(dyn._model_unloaded_bytes(stale_none, dev_cls("xpu", 0)), 1404 * mib)

    def test_dynamic_vram_mixed_dynamic_and_non_dynamic_models(self):
        mib = 1024**2
        free_before = 4400 * mib
        dyn, mm, state, _, model_cls, loaded_cls = self._make_boundary_harness(
            free_bytes=free_before, reserved_bytes=1024 * mib
        )
        inactive_dyn = model_cls(
            "InactiveDyn", dynamic=True, size=1000 * mib, loaded=1000 * mib, max_freeable=600 * mib
        )
        active_dyn = model_cls(
            "ActiveDyn", dynamic=True, size=20000 * mib, loaded=20000 * mib
        )
        vae = model_cls("CausalDiffusionVAE", dynamic=False, size=1404 * mib, loaded=0)
        # current_loaded_models is newest-first; reversed() trims inactive_dyn first.
        mm.current_loaded_models = [loaded_cls(active_dyn), loaded_cls(inactive_dyn)]

        mm.load_models_gpu([vae, active_dyn], memory_required=3400 * mib, force_full_load=True)
        static_budget = (1404 * mib * 11 + 9) // 10
        scaled_req = int(3400 * mib * 1.5)
        expected_total_trim = (scaled_req + 1024 * mib + static_budget) - free_before
        self.assertEqual(vae.unload_calls, [])
        self.assertEqual(inactive_dyn.unload_calls, [expected_total_trim])
        self.assertEqual(len(active_dyn.unload_calls), 1)
        self.assertEqual(inactive_dyn.unload_calls[0] - 600 * mib, active_dyn.unload_calls[0])
        self.assertEqual(state["soft_empty_calls"], 1)
        self.assertGreaterEqual(
            state["load_calls"][0]["free_at_load"],
            scaled_req + 1024 * mib + int(1.1 * 1404 * mib),
        )

        # Mixed with an unloaded dynamic model also accounts for both static (1.1x) and dynamic weights.
        _, mm2, _, _, model_cls2, loaded_cls2 = self._make_boundary_harness(
            free_bytes=free_before, reserved_bytes=1024 * mib
        )
        inactive_big = model_cls2("InactiveBig", dynamic=True, size=25696 * mib, loaded=25696 * mib)
        new_dyn = model_cls2("NewDyn", dynamic=True, size=3000 * mib, loaded=0)
        vae2 = model_cls2("CausalDiffusionVAE", dynamic=False, size=1404 * mib, loaded=0)
        mm2.current_loaded_models = [loaded_cls2(inactive_big)]
        mm2.load_models_gpu([vae2, new_dyn], memory_required=3400 * mib)
        self.assertEqual(
            inactive_big.unload_calls,
            [(scaled_req + 1024 * mib + static_budget + 3000 * mib) - free_before],
        )

    def test_dynamic_vram_get_free_memory_and_model_patcher_patches(self):
        mib = 1024**2
        gib = 1024**3
        total_bytes = 32 * gib
        vbar_bytes = 25696 * mib
        free_driver = 4400 * mib
        reusable_torch = 1000 * mib
        stock_calls = []
        vbar_queries = []

        class _Dev:
            def __init__(self, dev_type, index=None):
                self.type = dev_type
                self.index = index

        xpu0 = _Dev("xpu", 0)
        xpu1 = _Dev("xpu", 1)
        cpu_dev = _Dev("cpu", None)
        cuda0 = _Dev("cuda", 0)

        fake_torch = _make_mock_torch(xpu_available=True, cuda_available=False)
        fake_torch.xpu.current_device = lambda: 0
        fake_torch.xpu.mem_get_info = lambda dev=0: (free_driver, total_bytes)
        fake_torch.xpu.memory_stats = lambda dev=0: {
            "reserved_bytes.all.current": 1800 * mib,
            "active_bytes.all.current": 800 * mib,
        }

        def _stock_get_free(dev=None, torch_free_too=False):
            stock_calls.append((dev, torch_free_too))
            if getattr(dev, "type", None) == "cpu":
                val = 48 * gib
                return (val, val) if torch_free_too else val
            if getattr(dev, "type", None) == "cuda":
                val = 12 * gib
                return (val, 512 * mib) if torch_free_too else val
            # Stock XPU bug: ignores VBAR allocations (~31.1 GiB reported free).
            val = total_bytes - 800 * mib
            return (val, reusable_torch) if torch_free_too else val

        mm = types.SimpleNamespace(
            get_free_memory=_stock_get_free,
            get_torch_device=lambda: xpu0,
        )
        mem_mgmt = types.SimpleNamespace(aimdo_enabled=True)

        class _FakeModelPatcher:
            def get_free_memory(self, device):
                # Stock ModelPatcher fallback for non-XPU (e.g. CUDA)
                return int(mm.get_free_memory(device)) + 999 * mib

        dyn = _load_repo_module(
            "adapters.dynamic_vram_free_test", "adapters/dynamic_vram.py", fake_torch=fake_torch
        )
        fake_vbar = types.SimpleNamespace(
            vbars_analyze=lambda dev=None: (
                vbar_queries.append(dev) or (vbar_bytes if dev in (0, 1) else 0)
            )
        )
        with mock.patch.object(dyn, "_aimdo_model_vbar", fake_vbar):
            self.assertTrue(dyn._patch_get_free_memory(mm))
            self.assertFalse(dyn._patch_get_free_memory(mm))  # idempotent
            self.assertTrue(dyn._patch_model_patcher_get_free_memory(_FakeModelPatcher, mm, mem_mgmt))
            self.assertFalse(dyn._patch_model_patcher_get_free_memory(_FakeModelPatcher, mm, mem_mgmt))

            # 1. get_free_memory(xpu) and get_free_memory(None) return real free + reusable_torch.
            self.assertEqual(mm.get_free_memory(xpu0), free_driver + reusable_torch)
            self.assertEqual(mm.get_free_memory(), free_driver + reusable_torch)
            self.assertEqual(
                mm.get_free_memory(xpu0, torch_free_too=True),
                (free_driver + reusable_torch, reusable_torch),
            )
            self.assertEqual(stock_calls, [])

            # Non-XPU devices delegate to stock get_free_memory unchanged.
            self.assertEqual(mm.get_free_memory(cpu_dev), 48 * gib)
            self.assertEqual(mm.get_free_memory(cuda0, torch_free_too=True), (12 * gib, 512 * mib))
            self.assertEqual(len(stock_calls), 2)

            # 2. ModelPatcher.get_free_memory(xpu) passes explicit device index (no 56.6 GiB double count).
            mp = _FakeModelPatcher()
            self.assertEqual(mp.get_free_memory(xpu1), free_driver + reusable_torch + vbar_bytes)
            self.assertEqual(vbar_queries, [1])
            # CPU device never triggers vbars_analyze(None).
            self.assertEqual(mp.get_free_memory(cpu_dev), 48 * gib)
            self.assertEqual(vbar_queries, [1])
            # CUDA delegates to original ModelPatcher.get_free_memory.
            self.assertEqual(mp.get_free_memory(cuda0), 12 * gib + 999 * mib)

            # 3. Fallback when mem_get_info raises: calls saved original once, subtracts vbar once, no recursion.
            stock_calls.clear()
            vbar_queries.clear()
            with mock.patch.object(
                fake_torch.xpu, "mem_get_info", side_effect=RuntimeError("driver unavailable")
            ):
                expected_fallback = (total_bytes - 800 * mib) - vbar_bytes
                self.assertEqual(mm.get_free_memory(xpu0), expected_fallback)
                self.assertEqual(dyn._actual_free_vram(mm, xpu0), expected_fallback)
                self.assertEqual(stock_calls, [(xpu0, False), (xpu0, False)])
                self.assertEqual(vbar_queries, [0, 0])

            # 4. Opt-out env restores stock behavior.
            stock_calls.clear()
            with mock.patch.dict(os.environ, {"OMNIXPU_DYNAMIC_VRAM_GET_FREE_MEMORY": "0"}, clear=False):
                self.assertEqual(mm.get_free_memory(xpu0), total_bytes - 800 * mib)
                self.assertEqual(mp.get_free_memory(xpu0), total_bytes - 800 * mib + 999 * mib)

    def test_dynamic_vram_dynamic_only_path_regression(self):
        mib = 1024**2
        dyn, mm, state, _, model_cls, loaded_cls = self._make_boundary_harness(
            free_bytes=4000 * mib, reserved_bytes=1024 * mib
        )
        inactive_dyn = model_cls("InactiveDyn", dynamic=True, size=10000 * mib, loaded=10000 * mib)
        active_dyn = model_cls("ActiveDyn", dynamic=True, size=22000 * mib, loaded=20000 * mib)
        mm.current_loaded_models = [loaded_cls(active_dyn), loaded_cls(inactive_dyn)]

        with mock.patch.dict(os.environ, {"OMNI_ATTN_BACKEND": "torch"}, clear=False):
            mm.load_models_gpu(
                [active_dyn],
                memory_required=10000 * mib,
                minimum_memory_required=2000 * mib,
            )
        # Dynamic-only inactive target uses minimum_memory_required (2000) + reserve (1024) + unloaded (2000) = 5024 MiB.
        self.assertEqual(inactive_dyn.unload_calls, [1024 * mib])
        self.assertEqual(active_dyn.unload_calls, [])
        self.assertEqual(state["soft_empty_calls"], 0)

        # When no inactive model can free more and free < target (3024 MiB), active dynamic model trims to target.
        _, mm2, state2, _, model_cls2, loaded_cls2 = self._make_boundary_harness(
            free_bytes=2500 * mib, reserved_bytes=1024 * mib
        )
        active_only = model_cls2("ActiveOnly", dynamic=True, size=22000 * mib, loaded=22000 * mib)
        mm2.current_loaded_models = [loaded_cls2(active_only)]
        with mock.patch.dict(os.environ, {"OMNI_ATTN_BACKEND": "torch"}, clear=False):
            mm2.load_models_gpu(
                [active_only],
                memory_required=10000 * mib,
                minimum_memory_required=2000 * mib,
            )
        self.assertEqual(active_only.unload_calls, [524 * mib])
        self.assertEqual(state2["soft_empty_calls"], 0)

    def test_multi_sku_calibration_knee_cap_and_monotonicity(self):
        gib = 1024**3
        mib = 1024**2
        fake_torch = _make_mock_torch(xpu_available=True, cuda_available=False)
        dyn = _load_repo_module("adapters.dynamic_vram_sku_cal", "adapters/dynamic_vram.py", fake_torch=fake_torch)
        xpu0 = types.SimpleNamespace(type="xpu", index=0)
        model = types.SimpleNamespace(load_device=xpu0, is_dynamic=lambda: True, model_patches_models=lambda: [])
        mm = types.SimpleNamespace(pytorch_attention_flash_attention=lambda: True)

        expected_knees = {8: 2 * gib, 12: 3 * gib, 16: 4 * gib, 24: 6 * gib, 32: 6 * gib}
        for total_gib, exp_knee in expected_knees.items():
            total_b = total_gib * gib
            cap = int(total_b * 0.40)
            with (
                mock.patch.dict(os.environ, {"OMNIXPU_DYNAMIC_VRAM_BUDGET_CALIBRATION": "1"}, clear=False),
                mock.patch.object(dyn, "_device_total_vram", return_value=total_b),
            ):
                os.environ.pop("OMNIXPU_AIMDO_BUDGET_KNEE_MIB", None)
                prev = -1
                for raw_mib in (512, 1024, 2048, 3072, 4096, 5120, 6144, 8192, 12288, 16384, 20480):
                    raw = raw_mib * mib
                    _, _, _, cal, changed = dyn._calibrate_inference_budget(
                        mm, [model], (), {"memory_required": total_b, "minimum_memory_required": raw}
                    )
                    self.assertLessEqual(cal, raw)
                    self.assertLessEqual(cal, max(raw, cap) if raw <= exp_knee else cap)
                    self.assertGreaterEqual(cal, prev)
                    if raw <= exp_knee:
                        self.assertFalse(changed)
                        self.assertEqual(cal, raw)
                    prev = cal
                # Large minimum_memory_required is always capped at 40% of total VRAM.
                _, _, _, cal_max, changed_max = dyn._calibrate_inference_budget(
                    mm, [model], (), {"memory_required": total_b, "minimum_memory_required": total_b}
                )
                self.assertTrue(changed_max)
                self.assertEqual(cal_max, cap)

            # Explicit knee override still respects max_fraction cap.
            with (
                mock.patch.dict(
                    os.environ,
                    {"OMNIXPU_DYNAMIC_VRAM_BUDGET_CALIBRATION": "1", "OMNIXPU_AIMDO_BUDGET_KNEE_MIB": "6144"},
                    clear=False,
                ),
                mock.patch.object(dyn, "_device_total_vram", return_value=total_b),
            ):
                _, _, _, cal_ovr, _ = dyn._calibrate_inference_budget(
                    mm, [model], (), {"memory_required": total_b, "minimum_memory_required": 10 * gib}
                )
                self.assertLessEqual(cal_ovr, cap)

    def test_torch_cache_trim_flag_per_device_state_and_proportional_threshold(self):
        gib = 1024**3
        mib = 1024**2
        fake_torch = _make_mock_torch(xpu_available=True, cuda_available=False)
        dyn = _load_repo_module("adapters.dynamic_vram_trim_sku", "adapters/dynamic_vram.py", fake_torch=fake_torch)

        # Config alignment: defaults to True even if _config is None; parses truthy/falsy env values.
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(dyn, "_config", None):
            self.assertTrue(dyn._torch_cache_trim_enabled())
        for truthy in ("1", "true", "TRUE", "yes", "on"):
            with mock.patch.dict(os.environ, {"OMNIXPU_DYNAMIC_VRAM_TORCH_CACHE_TRIM": truthy}, clear=True):
                self.assertTrue(dyn._torch_cache_trim_enabled(), truthy)
        for falsy in ("", "0", "false", "no", "off"):
            with mock.patch.dict(os.environ, {"OMNIXPU_DYNAMIC_VRAM_TORCH_CACHE_TRIM": falsy}, clear=True):
                self.assertFalse(dyn._torch_cache_trim_enabled(), falsy)

        # Proportional 15% free threshold capped at 4096 MiB on 32 GiB; per-device cooldown isolation.
        dyn._TRIM_STATE.clear()
        empty_calls = []
        mem_state = {
            0: {"rsv": 5 * gib, "alloc": 1 * gib, "free": 2 * gib, "total": 8 * gib},
            1: {"rsv": 5 * gib, "alloc": 1 * gib, "free": 1500 * mib, "total": 12 * gib},
        }
        fake_xpu = types.SimpleNamespace(
            memory_reserved=lambda dev: mem_state[dev]["rsv"],
            memory_allocated=lambda dev: mem_state[dev]["alloc"],
            mem_get_info=lambda dev: (mem_state[dev]["free"], mem_state[dev]["total"]),
            empty_cache=lambda: empty_calls.append(True),
        )
        alloc0 = (types.SimpleNamespace(device=0), 0, 64 * mib)
        alloc1 = (types.SimpleNamespace(device=1), 0, 64 * mib)
        clock = iter([100.0, 100.5, 100.6])
        with (
            mock.patch.dict(os.environ, {"OMNIXPU_DYNAMIC_VRAM_TORCH_CACHE_TRIM": "1"}, clear=False),
            mock.patch.object(dyn.torch, "xpu", fake_xpu, create=True),
            mock.patch.object(dyn, "_aimdo_control", None),
            mock.patch.object(dyn, "_xpu_vmm_snapshot", return_value=None),
            mock.patch.object(dyn.time, "monotonic", side_effect=clock),
        ):
            os.environ.pop("OMNIXPU_TORCH_CACHE_TRIM_FREE_MIB", None)
            # Dev 0 (8 GiB, threshold=1228.8 MiB): free=2 GiB > 1228.8 MiB -> no trim.
            self.assertEqual(dyn._maybe_trim_torch_cache(alloc0), 0)
            # Dev 1 (12 GiB, threshold=1843.2 MiB): independent cooldown from dev 0, free=1500 MiB -> trims!
            mem_state[1]["rsv_after"] = 1 * gib
            fake_xpu.empty_cache = lambda: (empty_calls.append(True), mem_state[1].__setitem__("rsv", 1 * gib))
            self.assertEqual(dyn._maybe_trim_torch_cache(alloc1), 4 * gib)
        self.assertEqual(len(empty_calls), 1)
        self.assertIn(0, dyn._TRIM_STATE)
        self.assertIn(1, dyn._TRIM_STATE)

    def test_total_vram_clamps_and_aimdo_disabled_fallback(self):
        gib = 1024**3
        mib = 1024**2
        total_b = 12 * gib
        dyn, mm, state, dev_cls, model_cls, loaded_cls = self._make_boundary_harness(
            free_bytes=6 * gib, total_bytes=total_b, reserved_bytes=4 * gib
        )
        xpu0 = dev_cls("xpu", 0)
        # _minimum_memory_target clamps to total_vram when minimum_memory_required + reserve > total_vram.
        clamped_target = dyn._minimum_memory_target(mm, (), {"minimum_memory_required": 10 * gib}, device=xpu0)
        self.assertEqual(clamped_target, total_b)

        # Boundary trim clamps inactive_target <= total_vram (prevents >100% VRAM target on small SKUs).
        ltx = model_cls("LTX", dynamic=True, size=10 * gib, loaded=6 * gib)
        vae = model_cls("VAE", dynamic=False, size=2 * gib, loaded=0)
        mm.current_loaded_models = [loaded_cls(ltx)]
        mm.load_models_gpu([vae], memory_required=5 * gib, force_full_load=True)
        # Unscaled needed = 5G + 4G + ceil(1.1 * 2G) <= 12G -> leaves ~0.8G resident (does not overshoot 12G).
        expected_shortfall = 5 * gib + 4 * gib + ((2 * gib * 11 + 9) // 10) - 6 * gib
        self.assertGreater(ltx.loaded_size(), 0)
        self.assertEqual(ltx.unload_calls, [expected_shortfall])

        # ModelPatcher.get_free_memory clamps at total_vram even if vbars_analyze over-reports.
        fake_torch = _make_mock_torch(xpu_available=True, cuda_available=False)
        fake_torch.xpu.mem_get_info = lambda dev=0: (6 * gib, total_b)
        fake_vbar = types.SimpleNamespace(vbars_analyze=lambda dev=0: 8 * gib)
        mm2 = types.SimpleNamespace(get_free_memory=lambda dev=None, torch_free_too=False: total_b, get_torch_device=lambda: xpu0, get_total_memory=lambda dev=None: total_b)
        mp_cls = type("_MP", (), {"get_free_memory": lambda self, dev: mm2.get_free_memory(dev)})
        mem_mgmt = types.SimpleNamespace(aimdo_enabled=True)
        dyn2 = _load_repo_module("adapters.dynamic_vram_clamp_test", "adapters/dynamic_vram.py", fake_torch=fake_torch)
        with mock.patch.object(dyn2, "_aimdo_model_vbar", fake_vbar):
            dyn2._patch_get_free_memory(mm2, memory_management=mem_mgmt)
            dyn2._patch_model_patcher_get_free_memory(mp_cls, mm2, mem_mgmt)
            self.assertEqual(mp_cls().get_free_memory(xpu0), total_b)

            # Fallback skips vbars_analyze subtraction when comfy.memory_management.aimdo_enabled is False.
            fake_torch.xpu.mem_get_info = mock.Mock(side_effect=RuntimeError("no mem_get_info"))
            with mock.patch.dict(sys.modules, {"comfy.memory_management": types.SimpleNamespace(aimdo_enabled=False)}):
                free_fb, _ = dyn2._xpu_actual_free_and_torch(
                    mm2, xpu0, original_get_free=lambda _d: 9 * gib
                )
            self.assertEqual(free_fb, 9 * gib)

    def test_lora_memory_passes_model_load_device_to_xpu_snapshot(self):
        fake_torch = _make_mock_torch(xpu_available=True, cuda_available=False)
        lora_mod = _load_repo_module("adapters.lora_memory_dev_test", "adapters/lora_memory.py", fake_torch=fake_torch)
        seen_devices = []
        xpu1 = types.SimpleNamespace(type="xpu", index=1)
        patcher = types.SimpleNamespace(
            load_device=xpu1,
            is_dynamic=lambda: True,
            model_patches_models=lambda: [],
            attachments={
                lora_mod._ATTACHMENT_KEY: {
                    "version": lora_mod._ATTACHMENT_VERSION,
                    "entries": ((1, 64 * 1024**2, 64 * 1024**2),),
                    "peak": 16 * 1024**2,
                }
            },
        )
        mm = types.SimpleNamespace(load_models_gpu=lambda models, *a, **kw: "ok")
        with mock.patch.object(
            lora_mod,
            "_xpu_snapshot",
            side_effect=lambda dev=None: (seen_devices.append(dev) or {"allocated": 0, "reserved": 0, "free": 0, "total": 0, "errors": {}}),
        ):
            lora_mod._patch_model_loader(mm)
            self.assertEqual(mm.load_models_gpu([patcher], memory_required=1024**3), "ok")
        self.assertEqual(seen_devices, [xpu1, xpu1])


if __name__ == "__main__":
    unittest.main()
