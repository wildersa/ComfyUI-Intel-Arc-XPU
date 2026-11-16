# ComfyUI-Intel-Arc-XPU

Native **Intel Arc Battlemage (`bmg` / Arc Pro B70)** acceleration for **ComfyUI on Windows**, combining `ComfyUI-OmniXPU` model adapters and diagnostics with Windows XPU compute kernels, `comfy-kitchen` XPU backend, and `DynamicVRAM` support — zero core ComfyUI files modified.

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

### Option A — ComfyUI Manager / Comfy Registry *(+ 1 extra binary step on Windows)*

Per **Comfy Registry security standards**, Registry packages exclude prebuilt `.whl` binaries and automated `pip` installers. Installing **`ComfyUI Intel Arc XPU (OmniXPU Windows)`** (`comfyui-intel-arc-xpu`) via the Manager Registry installs the Python adapters, conflict checks, and the **`OmniXPU Status`** diagnostic node — then finish setup with this **one extra PowerShell step**:

```powershell
git clone https://github.com/wildersa/ComfyUI-Intel-Arc-XPU.git
cd ComfyUI-Intel-Arc-XPU
.\install.ps1
```

*(If you already have the extension folder via Registry, running `.\install.ps1` updates/installs the XPU runtime packages in your active ComfyUI `.venv` cleanly.)*

### Option B — Direct GitHub Install *(All-in-One via Git Clone or Git URL)*

If installing directly from GitHub (`git clone` into `ComfyUI\custom_nodes\` or **ComfyUI Manager → Install via Git URL**, which requires `allow_git_url_install = True` in `user\__manager\config.ini`), the bundled Windows XPU runtimes are included in the repository and installed automatically by `install.py` on restart.

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

## 5. Benchmarks & Telemetry

Reference timings recorded on **Windows 11 (`Intel Arc Pro B70`, `32 GB`, `torch==2.13.0+xpu`)**:

| Workflow | Config | Stock / Prior | With `ComfyUI-Intel-Arc-XPU` + `DynamicVRAM` |
| --- | --- | --- | --- |
| **`Qwen Image 2.1 T2I`** | `25` steps, `int8_convrot`, `1024×1024` | `12.58s` warm (`torch`) | **`11.65s`–`11.69s`** warm (`21.76s`–`23.41s` cold) |
| **`WAN_Animate2`** | `2 × 6` steps, `int8_convrot` | `XPU OOM` at step `0/6` | **`370.91s`** (`0` OOMs) |
| **`MiniMax H3 Ref2VA`** | `8` steps turbo, `960×544`, `8s` + audio | — *(reference)* | **`281.92s`–`297.45s`** · [sample MP4](./benchmarks/assets/minimax_h3_ref2va_8s_b70.mp4) |
| **`Z-Image Turbo`** | `8` steps `bf16`, `1024×1024`, `cute` | — *(reference)* | **`33.41s`** cold · **`5.18s`–`5.27s`** warm · [samples](./benchmarks/README.md#sample-outputs) |

*(Observational session timings; full methodology and logs in [`benchmarks/README.md`](./benchmarks/README.md).)*

---

## 6. Licenses, Upstream Sources & Advanced Variables

- **Prefer Not to Use Prebuilt Binaries?** Follow Intel's official `omni` stack and custom-node instructions directly at **<https://github.com/intel/llm-scaler/tree/main/omni>** and **<https://github.com/intel/llm-scaler/tree/main/omni/ComfyUI-OmniXPU>** to build/install from Intel's sources. Self-built compilation setups are out of scope for this repository.
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
