# ComfyUI-Intel-Arc-XPU

**The problem:** Out of the box, ComfyUI on Windows does **not** ship native **Intel Arc (`xpu`)** compute kernels, an XPU backend for `comfy-kitchen`, Windows `DynamicVRAM` (`comfy-aimdo` XPU overlay), or Intel XPU model adapters. Even with PyTorch XPU installed, stock ComfyUI falls back to generic PyTorch paths, slow weight emulation, and static VRAM allocation — causing multi-minute stalls or `XPU out of memory` on large image and video models.

**What this extension solves:** `ComfyUI-Intel-Arc-XPU` brings native **Intel Arc Battlemage (`bmg` / Arc Pro B70)** acceleration to **ComfyUI on Windows** as a drop-in custom node extension (**zero core ComfyUI files modified**):
- Bundled **Windows XPU native kernels** (`omni_xpu_kernel`: SDP/attention, norms, rotary, INT8, FP8, layout) and `comfy-kitchen` XPU backend (`INT8`, `ConvRot`, `W4A8`, `FP8`).
- **Windows `DynamicVRAM`** (`comfy-aimdo-xpu-runtime`) with host RAM weight staging so large models (`WAN`, `MiniMax H3`, `FLUX`, `Qwen`) run without OOM.
- **Model adapters & `CUTE` FMHA attention routing** from `ComfyUI-OmniXPU`, plus diagnostics (`OmniXPU Status`) and clean fallbacks if ComfyUI updates.

> [!NOTE]
> **Your workflow also matters:** even with XPU and native kernels installed, a workflow must use **Intel XPU-compatible nodes, attention routes, and weights** (rather than CUDA-only custom nodes or emulated `fp8` defaults) to actually execute on these kernels. See **[Section 5 — Workflow Compatibility](#5-workflow-compatibility-using-the-xpu-kernels)** and [`WORKFLOW_COMPATIBILITY.md`](./WORKFLOW_COMPATIBILITY.md).

---

## 1. Prerequisite (Required)

Your Windows ComfyUI environment **must** already have **Windows 11 x64**, **Python `3.13`**, and **`torch==2.13.0+xpu`** installed:

```powershell
"<ComfyUI_Python>\python.exe" -m pip install --upgrade torch==2.13.0+xpu torchvision==0.28.0+xpu --index-url https://download.pytorch.org/whl/xpu
```

> [!IMPORTANT]
> If Python or PyTorch does not match **`Python 3.13` + `torch==2.13.0+xpu`** (or if Intel XPU is unavailable), the extension prints a **`CONFLICT:`** message and disables itself cleanly instead of crashing ComfyUI. If an upstream ComfyUI update changes an internal hook, conflicting patches fall back to stock ComfyUI behavior.

---

## 2. Installation

Choose **one** of the options below (you do **not** need to run multiple options).

### Option A — PowerShell All-in-One Install *(Recommended)*

Clone anywhere and run `.\install.ps1`:

```powershell
git clone https://github.com/wildersa/ComfyUI-Intel-Arc-XPU.git
cd ComfyUI-Intel-Arc-XPU
.\install.ps1
```

`install.ps1` automatically searches standard ComfyUI locations (Comfy Desktop, `%USERPROFILE%\ComfyUI`, `C/D/E:\ComfyUI*`), asks only if multiple installs exist or none are found (or pass `-ComfyPath "<path>"`), copies the extension into `custom_nodes\ComfyUI-Intel-Arc-XPU`, and installs the verified `.whl` binaries into your ComfyUI `.venv`.

### Option B — ComfyUI Manager → Install via Git URL

1. Set `allow_git_url_install = True` in `<ComfyUI>\user\__manager\config.ini`.
2. In **ComfyUI Manager**, click **Install via Git URL** and paste:
   ```text
   https://github.com/wildersa/ComfyUI-Intel-Arc-XPU.git
   ```
3. Restart ComfyUI — the repository includes the bundled `.whl` files and ComfyUI Manager runs `install.py` automatically.

### Option C — ComfyUI Manager Registry *(+ PowerShell binary step)*

Per **Comfy Registry security standards**, Registry packages exclude prebuilt `.whl` binaries and automated `pip` installers. Installing **`ComfyUI Intel Arc XPU (OmniXPU Windows)`** (`comfyui-intel-arc-xpu`) via the Manager Registry installs the Python adapters and **`OmniXPU Status`** diagnostic node; finish setup by running `.\install.ps1` (Option A above) so the native `.whl` runtimes are installed into your ComfyUI environment.

- Remove any older `comfyui-xpu-backend` folder in `custom_nodes/` if present so patches do not conflict.

---

## 3. Required / Recommended Flags & Environment Setup

After installing, configure these two Windows settings for full performance:

### 1. ComfyUI Launch Flags *(Explicitly Recommended)*

Add these launch arguments to your ComfyUI startup command or **Comfy Desktop → Settings → Launch Arguments (`launchArgs`)**:

```text
--enable-dynamic-vram --reserve-vram 4
```

**What works with vs. without `--enable-dynamic-vram --reserve-vram 4`:**
- **Without `--enable-dynamic-vram`:** Compute acceleration still works — native Intel XPU kernels (`omni_xpu_kernel`), `comfy-kitchen`'s `xpu` backend (INT8, ConvRot, FP8, W4A8), and model adapters still accelerate workflows that fit inside VRAM.
- **With `--enable-dynamic-vram --reserve-vram 4`:** Unlocks **Windows `DynamicVRAM`** (`comfy-aimdo` XPU overlay) and host RAM weight staging, preventing `XPU out of memory` on large/video workflows (`WAN_Animate2`, `MiniMax H3`).

### 2. Attention Backend (`OMNI_ATTN_BACKEND`)

Windows defaults to `torch` (PyTorch SDPA) for stability. To enable CUTLASS-SYCL `CUTE` FMHA attention on Intel Arc BMG (or revert to `torch` when isolating experimental routes):

```powershell
[Environment]::SetEnvironmentVariable("OMNI_ATTN_BACKEND", "cute", "User")   # enable CUTE FMHA
[Environment]::SetEnvironmentVariable("OMNI_ATTN_BACKEND", "torch", "User")  # revert to Windows default
```

Restart ComfyUI / Comfy Desktop after changing environment variables or `launchArgs`.

---

## 4. Verify Installation

Add the **`OmniXPU Status`** node to any workflow, or check `<ComfyUI>\user\comfyui.log` in PowerShell:

```powershell
Select-String -Path "<ComfyUI>\user\comfyui.log" -Pattern "OmniXPU|comfy_kitchen backend xpu|DynamicVRAM|aimdo|Native ops|CUTE|OMNI_ATTN_BACKEND|CONFLICT|XPU out of memory"
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
| `[OmniXPU] CONFLICT: ... requires ... torch==2.13.0+xpu` | Incompatible Python/PyTorch build; install `torch==2.13.0+xpu` on Python `3.13`. |
| `Native Windows XPU runtime binaries (omni_xpu_kernel) are not installed` | Installed via Registry without running the extra binary setup step (`.\install.ps1` / `install.py`). |
| `XPU out of memory` *(without `DynamicVRAM support detected and enabled`)* | Add `--enable-dynamic-vram --reserve-vram 4` to your ComfyUI launch flags. |
| `experimental capability-driven BMG D128 CUTE route enabled` | `OMNI_ATTN_BACKEND=cute` hit an experimental attention route; switch `OMNI_ATTN_BACKEND` back to `torch` to isolate artifacts. |
| `Attention backend 'comfy kitchen attention' is unavailable; using PyTorch attention.` | CUDA-origin workflow; switch `ModelAttentionBackend` to `"pytorch attention"` (and bypass CUDA-only `sol-attn` sparse nodes). |
| `Found comfy_kitchen backend triton: {'available': False, 'disabled': True...}` | **Expected on Windows** (not an error); XPU uses the `xpu` backend, not `triton`. |

---

## 5. Workflow Compatibility (Using the XPU Kernels)

Having `torch==2.13.0+xpu`, `omni_xpu_kernel`, and `DynamicVRAM` installed is only half the setup — **your workflow also needs to hit native XPU paths** instead of emulated formats or CUDA-only custom nodes:

- **Weights (`int8_convrot` vs. `fp8_scaled`):** Prefer `int8_convrot` / `int8_tensorwise` checkpoints over NVIDIA-oriented `fp8_scaled` weights when available (FP8 often falls back to FP16 emulation on Intel Arc BMG).
- **Attention & CUDA nodes:** In workflows imported from NVIDIA setups, switch `ModelAttentionBackend` from `comfy kitchen attention` to **`pytorch attention`** and bypass CUDA-only nodes (`sol-attn`, `SageAttention`, TensorRT/Triton-only packs).
- **Live diagnostics ([`ComfyUI-XPU-Advisor`](https://github.com/wildersa/ComfyUI-XPU-Advisor)):** Install our companion advisor extension to get live UI alerts whenever a workflow loads emulated FP8 weights, triggers Shared GPU Memory spills, or hits CUDA-only fallbacks.

See **[`WORKFLOW_COMPATIBILITY.md`](./WORKFLOW_COMPATIBILITY.md)** for the full workflow checklist and verification steps.

---

## 6. Benchmarks & Telemetry

Measured on **Windows 11 (`Intel Arc Pro B70`, `32 GB`, `torch==2.13.0+xpu`)** comparing **Stock ComfyUI (OOTB)** (out-of-the-box, no extension, `DynamicVRAM` off) against **`ComfyUI-Intel-Arc-XPU`** (`--enable-dynamic-vram`):

| Model | Config | Stock ComfyUI (OOTB) | With Extension (`torch`) | With Extension (`cute`) |
| --- | --- | --- | --- | --- |
| **`MOSS-Image`** | `int8_convrot` + `W4A8`, `25` steps | `375.28s` cold · `50.75s` warm | `103.45s` cold · `20.01s` warm | **`29.53s`** cold (`12.7×`) · **`18.27s`** warm (`2.78×`) |
| **`MiniMax H3 Ref2VA`** | `int8_convrot` + `NVFP4`, `20` steps | `399.91s` cold · `1582.00s` warm | `209.29s` cold · `179.84s` warm | **`186.08s`** cold (`2.15×`) · **`160.25s`** warm (`9.87×`) |
| **`Qwen Image 2.1 T2I`** | `int8_convrot`, `1024×1024`, `25` steps | — *(not tested)* | `22.48s` cold · `12.58s` warm | **`21.51s`** cold · **`11.38s`** warm |
| **`Z-Image Turbo`** | `bf16`, `1024×1024`, `8` steps | — *(not tested)* | — | **`33.41s`** cold · **`5.19s`–`5.27s`** warm |
| **`WAN 2.2 T2V 14B` (4-step LightX2V)** | `fp8_scaled` + `lightx2v` LoRA, `4` steps (`2+2`), `cfg=1.0` | — *(not tested)* | `104.37s` cold | **`93.09s`** cold · **`82.62s`** warm |
| **`WAN 2.2 T2V 14B` (Standard 20-step)** | `fp8_scaled`, `20` steps (`10+10`), `cfg=3.5` *(template default, LoRA bypassed)* | — *(not tested)* | `777.00s` | **`685.00s`–`696.00s`** |

<sub>**`cold`** = first run (loading models from disk + XPU/kernel warmup) · **`warm` (hot)** = subsequent runs with models/caches already loaded. Full tables and sample outputs in [`benchmarks/README.md`](./benchmarks/README.md).</sub>

---

## 7. Licenses, Upstream Sources & Advanced Variables

- **Prefer Not to Use Prebuilt Binaries?** Follow Intel's official `omni` stack and custom-node instructions directly at **<https://github.com/intel/llm-scaler/tree/main/omni>** and **<https://github.com/intel/llm-scaler/tree/main/omni/ComfyUI-OmniXPU>** to build/install from Intel's sources.
- **Licenses:** Apache License 2.0 ([`LICENSE`](./LICENSE)) and GNU General Public License v3.0 ([`LICENSE-GPL-3.0.txt`](./LICENSE-GPL-3.0.txt) for `comfy-aimdo-xpu-runtime`).
- **Attributions & Checksums:** [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md) (`intel/llm-scaler`, `xiangyuT/comfy-kitchen-xpu`, `xiangyuT/comfy-aimdo-xpu`, `Comfy-Org/comfy-aimdo`, `oneDNN`, `Strata`, `Level Zero`, `Detours`) · [`SHA256SUMS.txt`](./SHA256SUMS.txt).

<details>
<summary>Advanced Environment Variables &amp; Manual CLI Verification</summary>

| Variable | Default (Windows) | Purpose |
| --- | --- | --- |
| `OMNI_ATTN_BACKEND` | `torch` | `torch` (PyTorch SDPA), `cute` (CUTLASS-SYCL FMHA), `auto`, or `esimd`. |
| `OMNIXPU_PROVIDER_BOOTSTRAP` | `auto` | Provider routing mode: `auto`, `off`, or `required`. |
| `OMNIXPU_XPU_MEMORY_FRACTION` | `0.99` | Caching-allocator fraction applied on prestartup (`0` / `off` disables). |
| `OMNIXPU_ENABLE` | `1` | Set to `0` to disable all OmniXPU providers and patches. |
| `OMNIXPU_DEBUG` / `OMNIXPU_DEBUG_VERBOSE` | `0` | Set to `1` for kernel or dispatch tracing. |

To inspect or rerun bundled wheel alignment from CLI (`--no-deps` + SHA-256 verified against `SHA256SUMS.txt`):

```powershell
"<ComfyUI_Python>\python.exe" install.py --dry-run
"<ComfyUI_Python>\python.exe" install.py
```

</details>
