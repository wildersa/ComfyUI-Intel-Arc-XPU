# Changelog

All notable changes to **ComfyUI-Intel-Arc-XPU** are documented in this file.

## [0.2.9] - 2026-11-27

Faster and more stable LTX-2.5 video generation on Intel Arc (XPU): speeds up LTX-2.5 video decoding (~4s faster on 1280x704 5s clips) and improves automatic VRAM cleanup between decode tiles to prevent stalls across 8 GB to 32 GB Intel Arc GPUs. Also added LTX-2.5 and updated MiniMax H3 10s benchmarks to the README. Updating from v0.2.x only requires updating this node and restarting ComfyUI.

### Changed
- Republishes `v0.2.8` on Comfy Registry as `v0.2.9` with concise plain-text Updates notes (Comfy Registry does not allow reusing a deleted version number).

## [0.2.8] - 2026-11-27

Faster and more stable LTX-2.5 video generation on Intel Arc (XPU): speeds up LTX-2.5 video decoding (~4s faster on 1280x704 5s clips) and improves automatic VRAM cleanup between decode tiles to prevent stalls across 8 GB to 32 GB Intel Arc GPUs. Also added LTX-2.5 and updated MiniMax H3 10s benchmarks to the README. Updating from v0.2.x only requires updating this node and restarting ComfyUI.

### Added
- **Batched XPU eager `na3d` adapter (`adapters/na3d_eager.py`, `config.py`, `patches/__init__.py`):** batches neighborhood-3D attention groups on Intel XPU instead of falling back to `g_max=1` (`g_max=4` on `>=16 GB` GPUs, `g_max=2` on `<16 GB`, controlled via `OMNIXPU_NA3D_EAGER=1` and `OMNIXPU_NA3D_EAGER_GMAX`), with automatic OOM fallback to stock `na3d` and exact numerical parity (`max_err=0.0`). Saves `~4.1s` on LTX-2.5 `1280×704` `CausalDiffusionVAE` decode (`69.04s → 67.26s` warm).
- **Observed non-dynamic peak tracking & inter-tile VRAM reclaim (`adapters/dynamic_vram.py`, `config.py`):** tracks observed `torch.xpu.memory_reserved` peaks for non-dynamic models (`_OBSERVED_NON_DYNAMIC_PEAKS`, 64-entry LRU without calling `reset_peak_memory_stats`) and adds an inter-tile safety valve on `comfy.utils.tiled_scale_multidim` (`OMNIXPU_DYNAMIC_VRAM_INTER_TILE_RECLAIM=1`, multi-SKU safety floor `max(512 MiB, min(3072 MiB, 12% * total_vram))`) to flush PyTorch cache and reclaim inactive DynamicVRAM VBAR cleanly outside C/UR allocator locks during tiled VAE decodes.
- **LTX-2.5 T2V & updated MiniMax H3 10s benchmarks (`README.md`, `benchmarks/README.md`):** added verified Windows 11 Intel Arc Pro B70 (`32 GB`) timings for `LTX-2.5 T2V` (`22B` distilled: `1280×704` `107.99s` cold / `67.26s` warm, short `26.79s` warm, high-res `175.63s` warm) and updated `MiniMax H3 Ref2VA` 10s (`342.62s` with `sol-attn`).
- **Multi-SKU workflow VRAM protection test suite (`tests/test_workflow_vram_protection.py`):** added 7 stdlib-only end-to-end transition tests across `8 / 12 / 16 / 24 / 32 GiB` VRAM profiles.

### Changed
- Expanded CPU-only CI test suite to **30 stdlib-only tests** (`tests/test_extension_contracts.py` + `tests/test_workflow_vram_protection.py`) covering `na3d_eager`, observed non-dynamic peaks, inter-tile VRAM reclaim, and multi-SKU workflow transitions.
- Automated Comfy Registry **Updates** (`COMFY_NODE_CHANGELOG`) extraction from `CHANGELOG.md` in `.github/workflows/publish.yml`.
- Added the experimental hardware-validation scope notice and direct upstream build links (`intel/llm-scaler`, `xiangyuT/comfy-kitchen-xpu`, `xiangyuT/comfy-aimdo-xpu`) to `README.md`.

## [0.2.7] - 2026-11-27

### Added
- **Non-dynamic transition VBAR trim & XPU `get_free_memory` accounting (`adapters/dynamic_vram.py`, `config.py`):** reclaims inactive DynamicVRAM residency explicitly before loading non-dynamic models (e.g. LTX-2.5 `CausalDiffusionVAE`), preventing Windows deferred VBAR reclaim from causing WDDM spillover, and fixes XPU `get_free_memory` / `ModelPatcher.get_free_memory` VBAR accounting (`OMNIXPU_DYNAMIC_VRAM_NON_DYNAMIC_TRIM=1`, `OMNIXPU_DYNAMIC_VRAM_GET_FREE_MEMORY=1`).
- **Intel Quick Sync `h264_qsv` SaveVideo adapter (`adapters/video_encode.py`, `config.py`):** routes eligible 8-bit 4:2:0 H.264 `SaveVideo` encodes to `h264_qsv` with probe validation and automatic CPU `libx264` fallback (`OMNIXPU_QSV_VIDEO_ENCODE=1`).
- **Long float32 text-encoder SDPA BF16 route (`fixes/te_long_sdpa_bf16.py`, `config.py`):** keeps long float32 Llama/Qwen3VL text-encoder SDPA on XPU `SDPBackend.OVERRIDEABLE` (`bfloat16`) to avoid quadratic `MATH` fallback (`OMNIXPU_TE_LONG_SDPA_BF16=1`).
- **MiniMax H3 CUTE D128 Q-chunking & segment collapse (`adapters/attention.py`, `adapters/h3_rms_modulation.py`):** chunks long `heads=56` CUTE D128 attention queries to stay well below Windows WDDM TDR limits (`OMNIXPU_H3_CUTE_Q_CHUNK=1`) and collapses contiguous 1D token-index tensors in segmented RMS modulation.

### Changed
- **Multi-SKU DynamicVRAM safety (`8 / 12 B580 / 16 B50 / 24 B60 / 32 B70 GiB`):** scaled the default budget calibration knee and reactive `torch_cache_trim` free threshold proportionally for smaller VRAM cards, isolated per-device trim state, enforced `[0, total_vram]` upper/lower clamps across memory targets and `get_free_memory`, gated fallback VBAR subtraction on `aimdo_enabled`, and passed `model.load_device` to `_xpu_snapshot` in `adapters/lora_memory.py`.
- Updated `pyproject.toml` to `0.2.7` and clarified that updating from `v0.2.x` only requires updating the node and restarting ComfyUI (no wheel reinstall needed).
- Expanded `tests/test_extension_contracts.py` to 19 CPU-only contract tests covering multi-SKU math, non-dynamic transitions, QSV encode fallback, TE long SDPA, and H3 Q-chunking.

## [0.2.6] - 2026-11-22

### Changed
- Updated `[project].description` in `pyproject.toml` so the Comfy Registry listing keeps the **`MANDATORY:`** warning up front (pointing users to the README, Python `3.13` + `torch==2.13.0+xpu`, native XPU wheel installation, and `--enable-dynamic-vram --reserve-vram 1` / `2`).
- Added a contract assertion in `tests/test_extension_contracts.py` ensuring `pyproject.toml` retains the mandatory setup notice.

## [0.2.5] - 2026-11-22

### Added
- Added standalone CPU-only extension contract tests (`tests/test_extension_contracts.py`) and GitHub Actions workflow (`.github/workflows/test.yml`).

### Changed
- Aligned Windows launch flag guidance across `__init__.py`, `install.py`, `install.ps1`, and `WORKFLOW_COMPATIBILITY.md` to recommend `--enable-dynamic-vram --reserve-vram 1` (compute-only GPU) or `--reserve-vram 2` (display GPU).
- Resolved contradictory documentation around `sol-attn` / `ComfyUI-Ref2VA-VSA` across `README.md`, `WORKFLOW_COMPATIBILITY.md`, and `benchmarks/README.md`, clarifying that `BlockSparseAttention` is automatically adapted to XPU when installed and cleanly skipped when absent.
- Documented clearly in `benchmarks/README.md` that the bundled `comfy_aimdo_xpu_runtime-0.6.0` wheel predates upstream [`xiangyuT/comfy-aimdo-xpu#37`](https://github.com/xiangyuT/comfy-aimdo-xpu/pull/37).

## [0.2.4] - 2026-11-22

### Added
- **DynamicVRAM inference budget calibration & LoRA peak budget (`adapters/dynamic_vram.py`, `config.py`):** avoids over-evicting models during large Flash/CUTE attention steps and accounts for LoRA patch peak memory.
- **Custom node XPU adapters (`adapters/custom_nodes_xpu.py`):** opportunistic runtime XPU adaptations (with clean skip when absent and hook on `nodes.init_external_custom_nodes`) for `ComfyUI-Ref2VA-VSA` (`sol_attn`), `ComfyUI_InstantID`, `ComfyUI_Fill-ChatterBox`, `comfyui-deno-custom-nodes`, `h3_character_swap`, and `ComfyUI-MiniMaxH3-Context-Loop`.
- **10s / 243-frame MiniMax H3 Ref2VA benchmark entries** in `README.md` and `benchmarks/README.md`.

### Changed
- Enabled `OMNIXPU_DYNAMIC_VRAM_TORCH_CACHE_TRIM=1` by default (`0` disables) for reactive rate-limited `torch.xpu.empty_cache()` during `vbar_fault` pressure.

## [0.2.3] - 2026-11-15 to 2026-11-19

### Added
- Standalone setup notice UI (`js/omnixpu_notice.js`) + `/omnixpu/setup-status` endpoint when installed via Comfy Registry before native `.whl` setup is run.
- [`WORKFLOW_COMPATIBILITY.md`](./WORKFLOW_COMPATIBILITY.md) covering `int8_convrot` vs. `fp8_scaled`, `ModelAttentionBackend`, and `ComfyUI-XPU-Advisor` integration.
- Verified Stock ComfyUI (OOTB) vs. Extension benchmark tables and `Cold` vs. `Warm` legend in `README.md` and `benchmarks/README.md`.

### Changed
- Bundled prebuilt wheels inside `wheels/`, removed `exec()` patches for Comfy Registry compliance, updated frontend fetch calls to direct `api.fetchApi`, and clarified standalone/Git URL/Registry installation options plus `intel/llm-scaler` build paths.

## [0.2.2] - 2026-11-15

### Changed
- Updated Comfy Registry package description in `pyproject.toml` with repository benchmark notes.

## [0.2.1] - 2026-11-15

### Added
- Comfy Registry metadata (`pyproject.toml`), `.comfyignore`, and GitHub Actions publish workflow (`.github/workflows/publish.yml`).
- Improved XPU DynamicVRAM free-memory trim and CUTE D128 CFG attention routing (`adapters/attention.py`, `adapters/dynamic_vram.py`), plus `Z-Image Turbo` and `Qwen Image 2.1 T2I` benchmark samples.

### Changed
- Clarified Windows prebuilt binaries vs. pure-Python provider routing and updated package classifiers for frontend compatibility.

## [0.2.0] - 2026-11-14

### Added
- Integrated full `ComfyUI-OmniXPU` model adapters, fixes, and `OmniXPU Status` diagnostics node (`adapters/`, `fixes/`, `patches/`, `nodes/`).
- Automated Windows installers (`install.ps1`, `install.py`) with `SHA256SUMS.txt` verification and `comfy_aimdo_xpu_runtime` / `comfy_kitchen_xpu_runtime` / `omni_xpu_kernel` provider bootstrap (`runtime_bootstrap.py`).
- Reference timings, troubleshooting guide, and sample video output in `benchmarks/`.

## [0.1.0] - 2026-11-13

### Added
- Initial release of **ComfyUI-Intel-Arc-XPU** with Windows Intel Arc XPU runtime setup, DynamicVRAM guide, and observational benchmark documentation.
