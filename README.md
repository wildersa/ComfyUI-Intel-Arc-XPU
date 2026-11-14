# ComfyUI-Intel-Arc-XPU

Native **Intel Arc Battlemage (`bmg` / Arc Pro B70)** acceleration for **ComfyUI on Windows**, built from Intel's official [`intel/llm-scaler`](https://github.com/intel/llm-scaler) (`omni 0.2.0-b3`) stack and packaged for one-step installation via **ComfyUI Manager** or **`install.ps1`**.

---

## Why This Exists

Out of the box, ComfyUI runs on Intel `xpu` through generic PyTorch fallback, leaving most of Battlemage's native hardware stack unused: no Intel SYCL/CUTE/oneDNN kernels, no `comfy-kitchen` XPU backend, and no Windows `DynamicVRAM` support in `comfy-aimdo`.

Intel and upstream contributors solved those gaps in [`intel/llm-scaler`](https://github.com/intel/llm-scaler) (`omni`), [`xiangyuT/comfy-kitchen-xpu`](https://github.com/xiangyuT/comfy-kitchen-xpu), and [`xiangyuT/comfy-aimdo-xpu`](https://github.com/xiangyuT/comfy-aimdo-xpu) — but upstream targets Linux Docker images or manual Windows C++/SYCL builds (oneAPI, Visual Studio, Detours, Level Zero). **This repository packages, compiles, and tests that official stack directly on Windows 11 (`Intel Arc Pro B70`)** with prebuilt wheels ([Release `v0.2.0`](https://github.com/wildersa/ComfyUI-Intel-Arc-XPU/releases/tag/v0.2.0)) and automated installers — **no compiler toolchain needed, and zero core ComfyUI files modified.**

| Capability | Stock ComfyUI (Windows `xpu`) | With `ComfyUI-Intel-Arc-XPU` (`llm-scaler` Stack) |
| --- | --- | --- |
| **Compute Kernels** | PyTorch generic fallback | Native `omni_xpu_kernel==0.2.0b3+torch213.bmg` (CUTLASS-SYCL `CUTE` FMHA, ESIMD SDP, oneDNN GEMM) |
| **`comfy-kitchen` Backend** | Eager / fallback only | Native XPU provider (`comfy-kitchen-xpu-runtime==0.2.41`): INT8, ConvRot, FP8 QDQ, W4A8, SVDQuant, AdaLN, RoPE |
| **Windows `DynamicVRAM`** | Unavailable for Windows XPU | Enabled via `comfy-aimdo-xpu-runtime==0.6.0` (Level Zero + Microsoft Detours + host RAM weight staging) |
| **Model & Node Integration** | Unadapted | `ComfyUI-OmniXPU` (`prestartup_script.py`, Qwen-Image 2.1, MiniMax H3, SeedVR2, INT8 FFN) + **`OmniXPU Status`** node |
| **Install & Core Files** | Clean | **Core untouched**; installs via **ComfyUI Manager** (`install.py`) or **`install.ps1`** |

> [!NOTE]
> **Tested Environment:** Windows 11 x64 (`win_amd64`) · **Intel Arc Pro B70** (`32 GB`, `bmg`) · Python `3.13` (`cp313`) · `torch==2.13.0+xpu` · ComfyUI `v0.45.3` (`comfy-aimdo==0.6.0`, `comfy-kitchen==0.2.41`).

---

## Installation

**Prerequisite:** Ensure your ComfyUI Python `3.13` environment already has `torch==2.13.0+xpu`:

```powershell
"<ComfyUI_Python>\python.exe" -m pip install --upgrade torch==2.13.0+xpu torchvision==0.28.0+xpu --index-url https://download.pytorch.org/whl/xpu
```

### Method 1: ComfyUI Manager (*Install via Git URL*)
1. In **ComfyUI Manager → Install via Git URL**, paste `https://github.com/wildersa/ComfyUI-Intel-Arc-XPU`.
2. [`install.py`](./install.py) validates Windows `cp313` + `torch==2.13.0+xpu`, downloads and SHA-256-verifies the 3 prebuilt wheels from Release `v0.2.0` (`--no-deps`), and skips already-matching packages.
3. Restart ComfyUI / Comfy Desktop.

### Method 2: Clone / ZIP + `.\install.ps1` *(Recommended for Comfy Desktop / Portable)*
```powershell
git clone https://github.com/wildersa/ComfyUI-Intel-Arc-XPU.git
cd ComfyUI-Intel-Arc-XPU
.\install.ps1
```
`install.ps1` auto-detects your ComfyUI install, checks for running `python.exe` locks, flags conflicting older XPU nodes, installs and verifies the wheels + custom node after confirmation, and optionally sets `OMNI_ATTN_BACKEND=cute`. *(Supports `-ComfyPath`, `-DryRun`, and `-Yes`.)*

---

## Windows Configuration

1. **Enable DynamicVRAM:** Add these ComfyUI launch flags (or Comfy Desktop `launchArgs`):
   ```text
   --enable-dynamic-vram --reserve-vram 4
   ```
2. **Attention Backend (`OMNI_ATTN_BACKEND`):** Defaults to `torch` (PyTorch SDPA) on Windows for stability. To opt into CUTLASS-SYCL FMHA (`omni_xpu_kernel.cute`) — or switch back to `torch` to isolate experimental routes — set it in User environment variables and restart ComfyUI:
   ```powershell
   [Environment]::SetEnvironmentVariable("OMNI_ATTN_BACKEND", "cute", "User")   # enable CUTE FMHA
   [Environment]::SetEnvironmentVariable("OMNI_ATTN_BACKEND", "torch", "User")  # back to Windows default
   ```
3. **Verify:** Add **`OmniXPU Status`** to any workflow (or check `comfyui.log` below). Remove older `comfyui-xpu-backend` or duplicate `ComfyUI-OmniXPU` folders so patches do not conflict.

<details>
<summary>Advanced Environment Variables & Release v0.2.0 Wheels</summary>

| Variable | Default (Windows) | Purpose |
| --- | --- | --- |
| `OMNI_ATTN_BACKEND` | `torch` | `torch` (PyTorch SDPA), `cute` (CUTLASS-SYCL FMHA), `auto`, or `esimd`. |
| `OMNIXPU_PROVIDER_BOOTSTRAP` | `auto` | Provider loading: `auto`, `off`, or `required`. |
| `OMNIXPU_XPU_MEMORY_FRACTION` | `0.99` | Caching-allocator fraction applied on prestartup (`0` / `off` disables). |
| `OMNIXPU_ENABLE` | `1` | Set to `0` to disable all OmniXPU patches and providers. |
| `OMNIXPU_DEBUG` / `OMNIXPU_DEBUG_VERBOSE` | `0` | Set to `1` for kernel or dispatch tracing. |

Release `v0.2.0` prebuilt assets (`omni_xpu_kernel-0.2.0b3+torch213.bmg`, `comfy_kitchen_xpu_runtime-0.2.41`, `comfy_aimdo_xpu_runtime-0.6.0`) and their hashes are listed in [`SHA256SUMS.txt`](./SHA256SUMS.txt) and [Releases (`v0.2.0`)](https://github.com/wildersa/ComfyUI-Intel-Arc-XPU/releases/tag/v0.2.0).

</details>

---

## Log Diagnostics & Troubleshooting

Scan `<ComfyUI>\user\comfyui.log` in PowerShell to verify what loaded or spot common issues:

```powershell
Select-String -Path "<ComfyUI>\user\comfyui.log" -Pattern "OmniXPU|comfy_kitchen backend xpu|comfy_kitchen backend triton|DynamicVRAM|aimdo|Native ops|full load:|CUTE|OMNI_ATTN_BACKEND|XPU out of memory|comfy kitchen attention"
```

**Expected when healthy:**
```text
[OmniXPU] omni_xpu_kernel 0.2.0b3+torch213.bmg - available: sdp, norm, rotary, linear_fp8, int8, layout
Found comfy_kitchen backend xpu: {'available': True, 'disabled': False, ...}
DynamicVRAM support detected and enabled
[OmniXPU] XPU RAM weight staging enabled: budget ... MiB
```

| Log Signature | Meaning / Fix |
| --- | --- |
| `XPU out of memory` + `full load: True` *(without `DynamicVRAM support detected and enabled`)* | Add `--enable-dynamic-vram --reserve-vram 4` and verify `comfy-aimdo-xpu-runtime` is active. |
| `Found comfy_kitchen backend xpu: {'available': False...}` or missing `Native ops: int8_tensorwise` | Verify `omni_xpu_kernel` + `comfy-kitchen-xpu-runtime` in the active `.venv` and remove conflicting XPU `custom_nodes`. |
| `experimental capability-driven BMG D128 CUTE route enabled` | `OMNI_ATTN_BACKEND=cute` used an experimental attention route; switch to `torch` and restart to isolate artifacts. |
| `Attention backend 'comfy kitchen attention' is unavailable; using PyTorch attention.` | CUDA-origin workflow; switch `ModelAttentionBackend` to `"pytorch attention"` (and bypass CUDA-only `sol-attn` sparse nodes). |
| `Found comfy_kitchen backend triton: {'available': False, 'disabled': True...}` | **Expected on Windows** (not an error); XPU uses the `xpu` backend, not `triton`. |

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

- **Custom Node & Installer Scripts:** [`LICENSE`](./LICENSE) / [`LICENSE-APACHE-2.0.txt`](./LICENSE-APACHE-2.0.txt) (Apache-2.0) · **`comfy-aimdo-xpu-runtime`:** [`LICENSE-GPL-3.0.txt`](./LICENSE-GPL-3.0.txt) (GPL-3.0)
- **Full Attributions:** [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md) — [`intel/llm-scaler`](https://github.com/intel/llm-scaler), [`xiangyuT/comfy-kitchen-xpu`](https://github.com/xiangyuT/comfy-kitchen-xpu), [`xiangyuT/comfy-aimdo-xpu`](https://github.com/xiangyuT/comfy-aimdo-xpu), [`Comfy-Org/comfy-aimdo`](https://github.com/Comfy-Org/comfy-aimdo), `oneDNN`, `Strata`, [`oneapi-src/level-zero`](https://github.com/oneapi-src/level-zero), [`microsoft/Detours`](https://github.com/microsoft/Detours).
