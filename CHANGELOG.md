# Changelog

All notable changes to **ComfyUI-Intel-Arc-XPU** are documented in this file.

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
