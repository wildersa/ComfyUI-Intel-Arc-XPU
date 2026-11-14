# ComfyUI-Intel-Arc-XPU

Native **Intel Arc Battlemage (`bmg` / Arc Pro B70)** acceleration for **ComfyUI on Windows**, built from Intel's official [`intel/llm-scaler`](https://github.com/intel/llm-scaler) (`omni`) stack and packaged for one-step installation via **ComfyUI Manager** or **`install.ps1`**.

---

## Why This Repository Exists

Out of the box, upstream ComfyUI can run on Intel `xpu` via PyTorch, **but it is not natively wired to unlock the full hardware power of Intel Arc Battlemage GPUs (such as the Intel Arc Pro B70)**:

- Generic PyTorch XPU fallback misses Intel's native SYCL, ESIMD, CUTLASS-SYCL (`CUTE`), and oneDNN quantized kernels.
- Official `comfy-kitchen` does not ship an XPU backend out of the box for native INT8 / ConvRot / FP8 / W4A8 / SVDQuant / RoPE / AdaLN acceleration.
- Official `comfy-aimdo` DynamicVRAM and host RAM weight staging do not work natively on Windows XPU without a Windows Level Zero + Microsoft Detours runtime provider.
- Model-specific bottlenecks (attention layouts, segmented RMS modulation, prefix KV caching, fused INT8 FFN, large-video preprocessing) remain unadapted.

**Intel and upstream contributors already solved these gaps** in [`intel/llm-scaler`](https://github.com/intel/llm-scaler) (`omni 0.2.0-b3`), [`xiangyuT/comfy-kitchen-xpu`](https://github.com/xiangyuT/comfy-kitchen-xpu), and [`xiangyuT/comfy-aimdo-xpu`](https://github.com/xiangyuT/comfy-aimdo-xpu). However, upstream `llm-scaler` primarily targets Linux Docker containers (`intel/llm-scaler-omni`) or requires a complex Windows source build (Intel oneAPI compiler toolchain, Visual Studio Build Tools, Windows SDK, CMake/Ninja, Microsoft Detours, Level Zero, and manual `.whl` + custom-node wiring).

**What this repository does differently:**
- Follows Intel's official `llm-scaler` (`omni 0.2.0-b3`) build and integration instructions to compile the native stack for **Windows 11 x64 (`cp313`, `torch==2.13.0+xpu`, Battlemage `bmg`)**.
- **Tested directly on Windows** with an **Intel Arc Pro B70** on **ComfyUI `v0.45.3`** (ComfyUI Desktop & `.venv`).
- Distributes the prebuilt Windows wheels via [GitHub Releases (`v0.2.0`)](https://github.com/wildersa/ComfyUI-Intel-Arc-XPU/releases/tag/v0.2.0) alongside automated installers (`install.py` & `install.ps1`) — **no C++/SYCL compiler required, and zero core ComfyUI files modified.**

---

## Stock ComfyUI (Windows XPU) vs. `ComfyUI-Intel-Arc-XPU`

| Capability | Stock ComfyUI on Windows (`xpu`) | With `ComfyUI-Intel-Arc-XPU` (`llm-scaler` Stack) |
| --- | --- | --- |
| **Compute Kernels** | PyTorch generic fallback | Native Intel `omni_xpu_kernel` (CUTLASS-SYCL FMHA, ESIMD SDP, oneDNN 3.11.2 GEMM) |
| **`comfy-kitchen` Backend** | Eager / fallback only | Native XPU provider (`comfy-kitchen-xpu-runtime`): INT8, ConvRot, FP8 QDQ, W4A8, SVDQuant, AdaLN, RoPE |
| **Windows `DynamicVRAM` (`comfy-aimdo`)** | Not enabled for Windows XPU | Enabled via `comfy-aimdo-xpu-runtime` (Level Zero + Microsoft Detours + host RAM weight staging) |
| **Model-Specific Adapters** | None | `ComfyUI-OmniXPU` adapters for Qwen-Image 2.1, MiniMax H3, SeedVR2, Lumina/Z-Image INT8 FFN, LoRA memory budgets |
| **Core ComfyUI Files** | Clean | **Untouched** (runs purely as a `custom_nodes` extension + `prestartup_script.py` + private providers) |
| **Windows Setup** | Manual if compiling `llm-scaler` | Automated via **ComfyUI Manager** (`Install via Git URL`) or interactive **`install.ps1`** |

---

## Architecture: What's Included (`v0.2.0`)

This package uses Intel's three-layer **OmniXPU** architecture so official ComfyUI packages (`comfy-kitchen==0.2.41` and `comfy-aimdo==0.6.0`) remain completely intact:

1. **`ComfyUI-OmniXPU` Layer (`prestartup_script.py`, `runtime_bootstrap.py`, `adapters/`, `nodes/`)**
   - Bootstraps the XPU runtime providers before PyTorch/ComfyUI initialize and sets the XPU caching-allocator fraction (`0.99` default on Windows).
   - Adapts ComfyUI call sites without patching core repo files: attention routing, `LayerNorm`/`RMSNorm`, MiniMax H3 segmented RMS modulation, FP8 model bridge, fused Lumina/Z-Image INT8 FFN, Qwen Image 2.1 cache & QKV, SeedVR2 capacity/GroupNorm, DynamicVRAM boundary trim, host RAM weight staging, LoRA memory caching, and large-video preprocessing.
   - Adds the **`OmniXPU Status`** diagnostic node inside ComfyUI.
2. **`omni_xpu_kernel` (`0.2.0b3+torch213.bmg`)**
   - Native Windows AOT-compiled (`cp313-win_amd64`) Intel XPU kernels for Battlemage (`bmg` / `intel_gpu_bmg_g31`), including CUTLASS-SYCL `CUTE` FMHA, hand-written ESIMD flash attention, and oneDNN quantized GEMMs.
3. **Co-Installable XPU Runtime Providers**
   - `comfy-kitchen-xpu-runtime==0.2.41` — registers the native XPU ops backend into official `comfy-kitchen`.
   - `comfy-aimdo-xpu-runtime==0.6.0` — hooks Windows DynamicVRAM / WDDM memory management for XPU via Level Zero and Microsoft Detours.

> [!NOTE]
> **Tested Windows Environment**
>
> - **OS / Architecture:** Windows 11 x64 (`win_amd64`)
> - **GPU:** Intel Arc Battlemage (`bmg`) — tested on **Intel Arc Pro B70** (`32 GB` VRAM)
> - **Python:** `3.13` (`cp313`)
> - **PyTorch:** `torch==2.13.0+xpu` (`sycl9.dll` / `intel-sycl-rt==2026.0.0`)
> - **ComfyUI:** `v0.45.3` (`comfy-aimdo==0.6.0`, `comfy-kitchen==0.2.41`)

---

## Installation

### Prerequisite: `torch==2.13.0+xpu` on Python 3.13

Make sure your ComfyUI Python 3.13 environment has `torch==2.13.0+xpu` installed:

```powershell
"<ComfyUI_Python>\python.exe" -m pip install --upgrade torch==2.13.0+xpu torchvision==0.28.0+xpu --index-url https://download.pytorch.org/whl/xpu
```

---

### Method 1: ComfyUI Manager (*Install via Git URL*)

1. Open **ComfyUI Manager** → **Install via Git URL**.
2. Paste:
   ```text
   https://github.com/wildersa/ComfyUI-Intel-Arc-XPU
   ```
3. ComfyUI Manager automatically runs [`install.py`](./install.py), which:
   - verifies Windows x64, Python `3.13`, and `torch==2.13.0+xpu` (exits cleanly without altering non-XPU environments);
   - skips reinstalling packages that already match the exact target versions (avoiding Windows DLL lock issues during restarts);
   - downloads the 3 prebuilt wheels from Release `v0.2.0` (or uses local `wheels\*.whl` if present), verifies SHA-256 checksums against [`SHA256SUMS.txt`](./SHA256SUMS.txt), and installs with `--no-deps`.
4. Restart ComfyUI / Comfy Desktop.

---

### Method 2: Clone / Download ZIP + `.\install.ps1` (Interactive PowerShell Installer)

Recommended for **ComfyUI Desktop** or **Windows Portable** setups if you want automatic install discovery, running-process checks, conflict cleanup, and post-install kernel verification:

```powershell
git clone https://github.com/wildersa/ComfyUI-Intel-Arc-XPU.git
cd ComfyUI-Intel-Arc-XPU
.\install.ps1
```

*(If you downloaded a `.zip` instead of cloning, extract it and run `.\install.ps1` inside the extracted folder. Supports `-ComfyPath "<path>"`, `-DryRun`, and `-Yes`.)*

`install.ps1` will:
1. Auto-detect your ComfyUI installation (`%APPDATA%\Comfy Desktop\installations.json`, `config.json`/`settings.json`, or common portable paths) or prompt for the path.
2. Check whether Comfy Desktop / ComfyUI `python.exe` is running and ask you to close it before modifying `.venv` packages.
3. Validate Windows x64, Python `3.13`, and `torch==2.13.0+xpu`.
4. Detect older conflicting XPU custom nodes (`custom_nodes\comfyui-xpu-backend` or duplicate `ComfyUI-OmniXPU`) and offer to disable/remove them after confirmation.
5. Print a full action summary and ask `Y/N` before downloading/verifying wheels, installing packages, copying the node into `custom_nodes\ComfyUI-Intel-Arc-XPU`, verifying `omni_xpu_kernel.cute.is_available()`, and optionally setting `OMNI_ATTN_BACKEND=cute` in your Windows User environment variables.

---

## Important Windows Configuration Notes

1. **Attention Backend (`OMNI_ATTN_BACKEND`):**
   On Windows, `ComfyUI-OmniXPU` defaults `OMNI_ATTN_BACKEND` to `torch` (upstream PyTorch SDPA) for maximum stability across models.
   - To opt into Intel's CUTLASS-SYCL FMHA (`omni_xpu_kernel.cute`, `fp32` accumulation) — e.g. for fast attention on Qwen-Image and compatible `head_dim=128` models — set `OMNI_ATTN_BACKEND=cute`:
     ```powershell
     [Environment]::SetEnvironmentVariable("OMNI_ATTN_BACKEND", "cute", "User")
     ```
   - `install.ps1` offers to configure this automatically.
   - If a specific workflow hits an experimental route issue under `cute`, switch back to `torch` (or remove the variable) and restart ComfyUI:
     ```powershell
     [Environment]::SetEnvironmentVariable("OMNI_ATTN_BACKEND", "torch", "User")
     ```
2. **Enable DynamicVRAM (`--enable-dynamic-vram`):**
   The `comfy-aimdo-xpu-runtime` provider activates when DynamicVRAM is enabled in ComfyUI launch arguments (or Comfy Desktop `launchArgs`):
   ```text
   --enable-dynamic-vram --reserve-vram 4
   ```
3. **Avoid Conflicting XPU Nodes:**
   Do not keep `custom_nodes\comfyui-xpu-backend` or a duplicate `custom_nodes\ComfyUI-OmniXPU` folder enabled alongside `ComfyUI-Intel-Arc-XPU`, as they will conflict over `comfy_kitchen` and ComfyUI adapters.
4. **Verify Inside ComfyUI:**
   Drop the **`OmniXPU Status`** node into any workflow (or inspect `comfyui.log`) to verify that `omni_xpu_kernel`, `comfy_kitchen` (`xpu`), `comfy_aimdo` (`DynamicVRAM`), and the adapters loaded cleanly.

### Useful Environment Variables

| Variable | Default (Windows) | Purpose |
| --- | --- | --- |
| `OMNI_ATTN_BACKEND` | `torch` | Attention policy: `torch` (PyTorch SDPA), `cute` (CUTLASS-SYCL FMHA), `auto`, or `esimd`. |
| `OMNIXPU_PROVIDER_BOOTSTRAP` | `auto` | Controls provider loading (`auto`, `off`, or `required`). |
| `OMNIXPU_XPU_MEMORY_FRACTION` | `0.99` | Caching-allocator fraction applied during prestartup (`0` / `off` disables). |
| `OMNIXPU_ENABLE` | `1` | Set to `0` to disable all OmniXPU patches and providers. |
| `OMNIXPU_DEBUG` / `OMNIXPU_DEBUG_VERBOSE` | `0` | Set to `1` for kernel or dispatch tracing. |

---

## Log Diagnostics & Troubleshooting

If something feels slow, fails with OOM, or falls back unexpectedly, scan your `<ComfyUI>\user\comfyui.log` in PowerShell to inspect what actually loaded and ran:

```powershell
Select-String -Path "<ComfyUI>\user\comfyui.log" -Pattern "OmniXPU|comfy_kitchen backend xpu|comfy_kitchen backend triton|DynamicVRAM|aimdo|Native ops|full load:|CUTE|OMNI_ATTN_BACKEND|XPU out of memory|comfy kitchen attention"
```

### Healthy Startup Signatures

You should see lines like these in `comfyui.log` when the full Intel stack is active:

```text
[OmniXPU] omni_xpu_kernel 0.2.0b3+torch213.bmg - available: sdp, norm, rotary, linear_fp8, int8, layout
Found comfy_kitchen backend xpu: {'available': True, 'disabled': False, ...}
aimdo_setup_hooks: native Torch XPU allocator retained; arbitrating Unified Runtime USM allocations
DynamicVRAM support detected and enabled
[OmniXPU] XPU RAM weight staging enabled: budget ... MiB
```

### Common Log Findings & Fixes

| What appears in `comfyui.log` | Meaning | What to do |
| --- | --- | --- |
| `XPU out of memory` + `full load: True` (and **no** `DynamicVRAM support detected and enabled`) | ComfyUI tried to load the whole model into VRAM without DynamicVRAM staging | Add `--enable-dynamic-vram --reserve-vram 4` to your launch arguments and ensure `comfy-aimdo-xpu-runtime` is installed without conflicts. |
| `Found comfy_kitchen backend xpu: {'available': False, ...}` or `int8_convrot` model missing `Native ops: int8_tensorwise` | Native XPU provider did not activate (wrong `.venv`, missing wheels, or conflicting `custom_nodes` patch) | Re-verify `omni_xpu_kernel` and `comfy-kitchen-xpu-runtime` in ComfyUI's Python and remove older `comfyui-xpu-backend` / duplicate `ComfyUI-OmniXPU` folders. |
| `experimental capability-driven BMG D128 CUTE route enabled` | `OMNI_ATTN_BACKEND=cute` routed an experimental attention contract (e.g. MiniMax H3) through CUTE | If the workflow shows visual artifacts or instability, test `[Environment]::SetEnvironmentVariable("OMNI_ATTN_BACKEND", "torch", "User")` and restart ComfyUI to isolate. |
| `Attention backend 'comfy kitchen attention' is unavailable; using PyTorch attention.` | Workflow imported from a CUDA setup requests NVIDIA-only `comfy kitchen attention` | Switch `ModelAttentionBackend` to `"pytorch attention"` (and bypass `BlockSparseAttention` if it uses CUDA-only `sol-attn`). |
| `Found comfy_kitchen backend triton: {'available': False, 'disabled': True, 'unavailable_reason': 'Triton is unavailable by default on Windows...'}` | **Expected on Windows (false alarm)** — Triton is disabled by default on Windows | Ignore it; XPU acceleration uses the **`xpu`** backend, not `triton`. |

---

## Release `v0.2.0` Wheel Assets & Checksums

| File | Package / Version | SHA-256 |
| --- | --- | --- |
| `omni_xpu_kernel-0.2.0b3+torch213.bmg-cp313-cp313-win_amd64.whl` | `omni-xpu-kernel==0.2.0b3+torch213.bmg` | `a5f3d152bce545c517a2e8f1db958c3c14d67238e7467cfa89bbbb859cb9fd25` |
| `comfy_kitchen_xpu_runtime-0.2.41-py3-none-any.whl` | `comfy-kitchen-xpu-runtime==0.2.41` | `638756e43526542b625d5a81717bc0721f503e53240416479b42922cde52c605` |
| `comfy_aimdo_xpu_runtime-0.6.0-cp39-abi3-win_amd64.whl` | `comfy-aimdo-xpu-runtime==0.6.0` | `6f962d5bdc1f6be61e9ee1d4654b846ff3edea31e4c4fc4640ded6a405756981` |

*(Also recorded in [`SHA256SUMS.txt`](./SHA256SUMS.txt).)*

---

## Benchmarks & Telemetry

Reference timings recorded on **Windows 11 (`Intel Arc Pro B70`, `torch==2.13.0+xpu`)**:

| Workflow | Config | Baseline / Prior | With `OmniXPU` + `DynamicVRAM` |
| --- | --- | --- | --- |
| **`QwenImage21`** | `25` steps, `int8_convrot` | `61.35s` | **`24.44s`–`28.77s`** warm |
| **`WAN_Animate2`** | `2 × 6` steps, `int8_convrot` | `XPU OOM` at step `0/6` | **`370.91s`** (0 OOMs) |
| **`MiniMax H3 Ref2VA`** | `8` steps turbo, `960×544`, `8s` + audio | — *(reference)* | **`281.92s`–`297.45s`** · [sample MP4](./benchmarks/assets/minimax_h3_ref2va_8s_b70.mp4) |

*(Observational session timings; full breakdown and notes in [`benchmarks/README.md`](./benchmarks/README.md).)*

---

## Credits & Licenses

- **Custom Node & Installer Scripts:** [`LICENSE`](./LICENSE) / [`LICENSE-APACHE-2.0.txt`](./LICENSE-APACHE-2.0.txt) (Apache-2.0)
- **`comfy-aimdo-xpu-runtime` Binary Wheel:** [`LICENSE-GPL-3.0.txt`](./LICENSE-GPL-3.0.txt) (GPL-3.0)
- **Full Attributions & Component Notices:** [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md)
  - [`intel/llm-scaler`](https://github.com/intel/llm-scaler) (`ComfyUI-OmniXPU`, `omni_xpu_kernel`) — Apache-2.0
  - [`xiangyuT/comfy-kitchen-xpu`](https://github.com/xiangyuT/comfy-kitchen-xpu) — Apache-2.0
  - [`xiangyuT/comfy-aimdo-xpu`](https://github.com/xiangyuT/comfy-aimdo-xpu) & [`Comfy-Org/comfy-aimdo`](https://github.com/Comfy-Org/comfy-aimdo) — GPL-3.0
  - `oneDNN`, `Strata`, [`oneapi-src/level-zero`](https://github.com/oneapi-src/level-zero), [`microsoft/Detours`](https://github.com/microsoft/Detours)
