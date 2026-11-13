# ComfyUI-Intel-Arc-XPU

Experimental optimizations for running **ComfyUI on Windows with Intel Arc GPUs (`xpu`)**, combining two complementary improvements:

1. **Custom Node with a pure PyTorch `comfy_kitchen` `'xpu'` backend** — accelerates quantized `INT8` operations (`int8_tensorwise` / `convrot`) and drastically reduces peak VRAM usage during dequantization without relying on CUDA/Triton.
2. **Windows build of `comfy-aimdo-xpu` (Release `v0.1.0`)** — enables **DynamicVRAM** support on Windows for Intel Arc (`xpu`) through Level Zero + Microsoft Detours hooks on top of `torch==2.13.0+xpu` (`sycl9.dll`).

---

> [!WARNING]
> **IMPORTANT NOTICE / DISCLAIMER (READ BEFORE USING)**
>
> **We have not performed enough testing yet — this setup has only been validated on my machine ("works on my machine").**
>
> - **Tested hardware & OS:** Windows 11, **Intel Arc Pro B70 GPU (`xpu`)**, **Python 3.13 (`cp313`)**, **PyTorch `2.13.0+xpu`** + **`torchvision 0.28.0+xpu`** (`intel-sycl-rt==2026.0.0` / `sycl9.dll`), **ComfyUI `v0.45.3`**.
> - **Tests completed successfully:**
>   - `test_xpu_backend.py` (`ALL TESTS PASSED`, including an INT8 microbenchmark and peak VRAM reduction);
>   - Real-world `VNCCS_PoseStudio_QI2.1` workflow (`qwen_image_2.1_int8_convrot.safetensors`, `1024x1024`, `steps=25`, LoRA enabled), with a **~19%** sampler speedup (`37s` -> `31s`, `1.51–1.52 s/it` -> `1.20 s/it` steady-state loop / `1.25 s/it` average);
>   - Actual ComfyUI Desktop startup with the log message `DynamicVRAM support detected and enabled`;
>   - Windows/VBAR hook smoke and lifecycle tests (`scripts\test-windows-xpu-hook.cmd`, `run_windows_ur_hook_smoke.py`, `run_xpu_vbar_resident_growth.py`, `run_xpu_vbar_retirement_safety.py` in default and async modes for 100 cycles, `run_xpu_vbar_consumer_lifecycle.py`).
> - **Limitations and pending tests:**
>   - Two synthetic memory-policy scripts from the upstream repository (`vbar-pageout-xpu.py` and `repro_xpu_platform_memory_policy.py`) failed their assertions (`AssertionError: []` and `AssertionError: [1]`);
>   - The upstream `pytest` suite (`tests/test_xpu_backend.py`) has not been run in this `.venv`;
>   - Heavy video workflows (such as `WAN_Animate2`) have not yet been covered by controlled same-baseline A/B benchmarks ([`benchmarks/direct/`](./benchmarks/direct/README.md)), though an initial observational run completed cleanly ([`benchmarks/observational/`](./benchmarks/observational/2026-11-13-qwen-and-wan-animate2-analysis.md)).
>
> *Summary: **Experimental / early release.** Tested only on Windows 11 + Intel Arc Pro B70 + `torch==2.13.0+xpu` + ComfyUI `v0.45.3`. Works on my machine, but has not yet been broadly tested across Intel Arc GPU models or long multi-model workflows.*

---

## Why We Built This (The Short Story)

Running large quantized models (like `qwen_image_2.1_int8_convrot` and `wan_animate_2_int8_convrot`) on **Windows + Intel Arc (`xpu`)** ran straight into two bottlenecks at once:

1. **Slow `eager` fallback & temporary `FP32` memory spikes:** Because ComfyUI's `comfy_kitchen` only had CUDA/Triton accelerated backends, Intel Arc fell back to PyTorch `eager`. In that path, `int8_tensorwise` fast compute was disabled and dequantization promoted large weight tensors into temporary `float32` buffers, slowing steps down and eating hundreds of megabytes of extra VRAM per layer.
2. **All-or-nothing VRAM loading, `XPU out of memory`, and Copy/Compute thrashing:** Without Windows DynamicVRAM (`comfy-aimdo-xpu`), heavy workflows like `WAN_Animate2` (`~15.5 GiB` diffusion model + VAE + CLIP + text encoder) tried to load completely into VRAM (`full load: True`) and repeatedly crashed with `XPU out of memory` at step `0/6` — or stalled as WDDM shuffled blocks back and forth, visible in Windows Task Manager as **Copy** bursts constantly interrupting **Compute**.

To fix both walls without rewriting ComfyUI:

- First, we wrote a pure-PyTorch `'xpu'` backend (`xpu_backend.py`) that dequantizes directly into `bf16`/`fp16`, chunks ConvRot rotations to keep temporary scratch memory bounded, and unlocks `int8_tensorwise` compute (`torch._int_mm` / low-precision XMX `F.linear`) — immediately yielding a ~19% speedup on `VNCCS_PoseStudio_QI2.1`.
- Next, we compiled `comfy-aimdo-xpu` from source for Windows (fixing a missing `dxcore.lib` link against Windows SDK `10.0.26100.0`, upgrading PyTorch XPU to `2.13.0+xpu` to match `sycl9.dll`, and launching with `--enable-dynamic-vram --reserve-vram 4`) so large models could stage virtual pages instead of crashing on load.
- With both pieces active together, `WAN_Animate2` went from crashing at step `0/6` to finishing cleanly in `370.91s` with steady GPU **Compute** utilization in Task Manager.

---

## What We Did and How It Works

### Improvement 1 — Pure PyTorch `'xpu'` Backend (`xpu_backend.py` + `__init__.py`)

By default, `comfy_kitchen` prioritizes CUDA/Triton backends and falls back to `eager` on Intel Arc GPUs (`xpu`). With the `eager` fallback, several dequantization operations temporarily promote weights to `float32` in VRAM, and ComfyUI disables fast `int8_tensorwise` paths on XPU.

The custom node in this repository automatically registers an `'xpu'` backend when ComfyUI starts (`__init__.py` -> `install_comfy_xpu_hooks()`):

- **Priority registration in `comfy_kitchen.registry`:** inserts the `'xpu'` backend immediately before `triton`/`eager`.
- **Dequantization without FP32 memory spikes (`_dequantize_simple_to`, `dequantize_int8_simple`, `dequantize_int8_simple_dtype`):** converts quantized weights directly to the target `dtype` (`bfloat16` or `float16`), halving temporary peak VRAM usage compared with `eager`.
- **Chunked ConvRot un-rotation (`dequantize_int8_convrot_weight`, `dequantize_int8_convrot_weight_dtype`):** processes the FP32 stage in blocks limited by `_FP32_CHUNK_ELEMS = 8 << 20` (~32 MB of scratch memory) before converting back to `bf16`/`fp16`.
- **Hybrid `int8_linear` matrix multiplication:**
  - When `M >= K` (token-dominated layers) and input/output are in `fp16`/`bf16`, uses low-precision dequantization + `torch.nn.functional.linear` on Intel Arc's XMX engines;
  - When `M < K`, uses `torch._int_mm(a, weight_nk.T)` without an extra transpose copy (with automatic padding to multiples of 8 when necessary) and performs FP32 rescaling in chunks above `_RESCALE_CHUNK_ELEMS = 16 << 20`.
- **Enabling native quantization in ComfyUI (`install_comfy_xpu_hooks`):** applies idempotent patches to:
  - `comfy.model_management.supports_int8_compute`;
  - `comfy.ops.get_disabled_quant_formats` (enables `int8_tensorwise` on XPU while keeping `convrot_w4a4` and `_GROUPED_INT8_FORMATS` disabled);
  - `TensorWiseINT8Layout.supports_fast_matmul`.

#### Measured Results (Improvement 1 Only)

| Test | Baseline (`eager` / `torch==2.12.1+xpu`) | With `'xpu'` backend | Improvement |
| --- | --- | --- | --- |
| `test_xpu_backend.py` (`M=1024, K=5120, N=13824`) | `5.69 ms` / peak `513.0 MB` | `1.58 ms` / peak `123.0 MB` | **~3.6x** faster, **-76%** peak VRAM |
| Peak VRAM during dequantization (`test_xpu_backend.py`) | `540.0 MB` | `272.0 MB` | **~50%** less temporary VRAM |
| `VNCCS_PoseStudio_QI2.1` workflow (`1024x1024`, `25` steps, LoRA) | `1.51–1.52 s/it` (`37s` sampler) | `1.20 s/it` steady state (`1.25 s/it` average, `31s` sampler) | **~19%** faster |

---

### Improvement 2 — Prebuilt Windows `comfy-aimdo-xpu` (`DynamicVRAM`)

Upstream `comfy-aimdo` focuses DynamicVRAM support on CUDA, and there was no official prebuilt Windows wheel for Intel Arc (`xpu`). Rather than shipping build instructions, we compiled the upstream [`xiangyuT/comfy-aimdo-xpu`](https://github.com/xiangyuT/comfy-aimdo-xpu) project (`intel/llm-scaler` reference, commit `3bc6afbcb92e10a2d4737e79d170190ed787bf89`) for Windows ourselves — building Microsoft Detours and `oneapi-src/level-zero` alongside it and adding the missing `dxcore.lib` link needed on Windows SDK `10.0.26100.0` so `aimdo_xpu.dll` compiles cleanly — and uploaded the ready-to-install `.whl` to **Release `v0.1.0`**:

- **Prebuilt Wheel (Release `v0.1.0`):** `comfy_aimdo-0.6.0+xpu.g3bc6afb.dirty-cp39-abi3-win_amd64.whl` (`157,964` bytes)
- **Wheel SHA-256:** `68dde78cae9655a2ced31f2595e43b7e0026341ac6d6e7172226a5377820aaae`
- **Bundled `aimdo_xpu.dll` SHA-256:** `B1850393799D9F1D3E70A47754564737173F0ADEF4A699CA3B41F27E5ABA1CA3`

> [!IMPORTANT]
> **Mandatory requirement: `torch==2.13.0+xpu` (`sycl9.dll`)**
>
> The prebuilt `aimdo_xpu.dll` links against Intel oneAPI 2026.0's **`sycl9.dll`** runtime (`intel-sycl-rt==2026.0.0`).
> If you are still on `torch==2.12.1+xpu` (`sycl8.dll`), the XPU extension will fail to load until you upgrade PyTorch XPU to `2.13.0+xpu`.

Additionally, ComfyUI's `main.py` only enables DynamicVRAM automatically on NVIDIA/ROCm GPUs (`dynamic_vram_supported()`), so on Intel Arc you **must explicitly pass** `--enable-dynamic-vram` at startup (we also recommend `--reserve-vram 4`).

---

## How to Use (Adapted `.py` Backend + Prebuilt `.whl`)

### 1. Install the Adapted `.py` Custom Node (`__init__.py` + `xpu_backend.py`)

From your ComfyUI directory (`<ComfyUI_Root>`), clone this repository into `custom_nodes` so ComfyUI automatically loads our `'xpu'` backend and quantization hooks at startup:

```powershell
cd "<ComfyUI_Root>\custom_nodes"
git clone https://github.com/wildersa/ComfyUI-Intel-Arc-XPU.git
```

### 2. Upgrade ComfyUI's `.venv` to `torch==2.13.0+xpu`

Back up your current environment before changing packages:

```powershell
"<ComfyUI_Root>\.venv\Scripts\python.exe" -m pip freeze > venv_backup_freeze.txt
"<ComfyUI_Root>\.venv\Scripts\python.exe" -m pip install --upgrade torch==2.13.0+xpu torchvision==0.28.0+xpu --index-url https://download.pytorch.org/whl/xpu
```

*(Adjust `torchaudio` or other packages if your environment requires specific versions.)*

### 3. Install the `comfy-aimdo-xpu` Wheel from Release `v0.1.0`

Download `comfy_aimdo-0.6.0+xpu.g3bc6afb.dirty-cp39-abi3-win_amd64.whl` from the [Releases](https://github.com/wildersa/ComfyUI-Intel-Arc-XPU/releases) page and install it in ComfyUI's `.venv`:

```powershell
"<ComfyUI_Root>\.venv\Scripts\python.exe" -m pip install --force-reinstall --no-deps "<Download_Path>\comfy_aimdo-0.6.0+xpu.g3bc6afb.dirty-cp39-abi3-win_amd64.whl"
"<ComfyUI_Root>\.venv\Scripts\python.exe" -m pip check
```

### 4. Configure ComfyUI Startup Flags

Add the following flags to ComfyUI's launch arguments (or to `launchArgs` in ComfyUI Desktop):

```text
--enable-dynamic-vram --reserve-vram 4
```

When starting ComfyUI, check the log (`comfyui.log`) for:

- `DynamicVRAM support detected and enabled`
- `Native ops: int8_tensorwise` / `'xpu'` backend registration in `comfy_kitchen`

### 5. Run the `'xpu'` Backend Self-Test (Optional)

```powershell
"<ComfyUI_Root>\.venv\Scripts\python.exe" "<ComfyUI_Root>\custom_nodes\ComfyUI-Intel-Arc-XPU\test_xpu_backend.py"
```

---

## Repository Structure

- [`__init__.py`](./__init__.py) — ComfyUI custom node entry point; calls `install_comfy_xpu_hooks()`.
- [`xpu_backend.py`](./xpu_backend.py) — Implementation of the `'xpu'` backend for `comfy_kitchen` + ComfyUI hooks.
- [`test_xpu_backend.py`](./test_xpu_backend.py) — Correctness unit tests and latency/VRAM microbenchmark.
- [`benchmarks/`](./benchmarks/) — Dedicated benchmarks folder separating direct A/B benchmarks ([`direct/`](./benchmarks/direct/README.md), pending) from non-controlled observational analyses ([`observational/`](./benchmarks/observational/)).
- [`LICENSE`](./LICENSE) — **GNU General Public License v3.0 (GPL-3.0)**.
- **Windows binary (`.whl`):** available exclusively under [Releases (`v0.1.0`)](https://github.com/wildersa/ComfyUI-Intel-Arc-XPU/releases) to keep the `custom_nodes` clone clean.

---

## Credits and License

- Distributed under the **GPL-3.0** license ([`LICENSE`](./LICENSE)), in compliance with `comfy-aimdo` / `comfy-aimdo-xpu`.
- Credits to the upstream projects:
  - [`xiangyuT/comfy-aimdo-xpu`](https://github.com/xiangyuT/comfy-aimdo-xpu) (commit `3bc6afbcb92e10a2d4737e79d170190ed787bf89`)
  - [`intel/llm-scaler`](https://github.com/intel/llm-scaler)
  - [`Comfy-Org/comfy-aimdo`](https://github.com/Comfy-Org/comfy-aimdo)
  - [`oneapi-src/level-zero`](https://github.com/oneapi-src/level-zero)
  - [`microsoft/Detours`](https://github.com/microsoft/Detours)
