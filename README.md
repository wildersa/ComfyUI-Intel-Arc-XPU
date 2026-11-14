# ComfyUI-Intel-Arc-XPU (`v0.2.0`)

Windows-ready **Intel Arc (`xpu` / Battlemage BMG)** runtime integration for **ComfyUI**, packaging upstream [`ComfyUI-OmniXPU`](https://github.com/intel/llm-scaler) together with automated installers (`install.py` & `install.ps1`) and prebuilt Windows `cp313` wheels distributed via [GitHub Releases (`v0.2.0`)](https://github.com/wildersa/ComfyUI-Intel-Arc-XPU/releases/tag/v0.2.0).

---

## What's Included in `v0.2.0`

Instead of replacing core ComfyUI files or maintaining a standalone pure-PyTorch prototype, `v0.2.0` uses the three-layer **OmniXPU** runtime stack:

1. **`ComfyUI-OmniXPU` (Custom Node + `prestartup_script.py`)** — activates co-installable XPU runtime providers before PyTorch loads, configures XPU caching-allocator fraction (`0.99` default on Windows), adapts ComfyUI call sites (attention, norms, FP8/INT8, Qwen Image 2.1 cache, SeedVR2 capacity, DynamicVRAM boundary trim & RAM staging), and provides the **`OmniXPU Status`** diagnostic node.
2. **`omni_xpu_kernel` (`0.2.0b3+torch213.bmg`)** — native Intel XPU SYCL/ESIMD/CUTLASS-SYCL kernels + oneDNN quantized GEMM for Battlemage (`bmg`) on Windows.
3. **Co-installable XPU Runtime Providers** — private provider packages that leave official `comfy-kitchen==0.2.41` and `comfy-aimdo==0.6.0` intact:
   - `comfy-kitchen-xpu-runtime==0.2.41`
   - `comfy-aimdo-xpu-runtime==0.6.0` (enables Windows DynamicVRAM for XPU via Level Zero + Microsoft Detours)

> [!NOTE]
> **Target Environment & Validation Scope**
>
> - **OS / Architecture:** Windows 11 x64 (`win_amd64`)
> - **GPU Target:** Intel Arc Battlemage (`bmg`) — tested on **Intel Arc Pro B70**
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
   - skips reinstalling packages that already match the exact target versions (preventing Windows DLL lock issues on restarts);
   - downloads the 3 prebuilt wheels from Release `v0.2.0` (or uses local `wheels\*.whl` if present), verifies SHA-256 against [`SHA256SUMS.txt`](./SHA256SUMS.txt), and installs with `--no-deps`.
4. Restart ComfyUI / Comfy Desktop.

---

### Method 2: Clone / Download ZIP + `.\install.ps1` (Interactive PowerShell Installer)

Recommended for **ComfyUI Desktop** or **Windows Portable** installations when you want automatic detection, running-process checks, conflict cleanup, and verification:

```powershell
git clone https://github.com/wildersa/ComfyUI-Intel-Arc-XPU.git
cd ComfyUI-Intel-Arc-XPU
.\install.ps1
```

*(If you downloaded the repository as a `.zip`, extract it and run `.\install.ps1` inside the extracted folder. You can also pass `-ComfyPath "<path>"`, `-DryRun`, or `-Yes`.)*

`install.ps1` will:
1. Auto-detect your ComfyUI installation (`%APPDATA%\Comfy Desktop\installations.json`, `config.json`/`settings.json`, or common portable paths) or prompt for the path.
2. Check whether Comfy Desktop / ComfyUI `python.exe` is running and ask you to close it before modifying `.venv` packages.
3. Validate Windows x64, Python `3.13`, and `torch==2.13.0+xpu`.
4. Detect conflicting `custom_nodes\comfyui-xpu-backend` if present and offer to disable/remove it after confirmation.
5. Print a full summary and ask `Y/N` before downloading/verifying wheels, installing packages, copying the custom node into `custom_nodes\ComfyUI-Intel-Arc-XPU`, running an import + `omni_xpu_kernel.cute.is_available()` check, and optionally setting `OMNI_ATTN_BACKEND=cute` in your User environment variables.

---

## Important Windows Configuration Notes

1. **Enable CUTE Attention on Windows (`OMNI_ATTN_BACKEND=cute`):**
   On Windows, `ComfyUI-OmniXPU` defaults attention to PyTorch SDPA (`torch`) unless `OMNI_ATTN_BACKEND=cute` is set in the environment before ComfyUI starts.
   - `install.ps1` offers to set `OMNI_ATTN_BACKEND=cute` in your Windows User environment variables automatically.
   - To set it manually in PowerShell:
     ```powershell
     [Environment]::SetEnvironmentVariable("OMNI_ATTN_BACKEND", "cute", "User")
     ```
2. **Enable DynamicVRAM (`--enable-dynamic-vram`):**
   The `comfy-aimdo-xpu-runtime` provider requires DynamicVRAM to be explicitly enabled on XPU. Add the following to your ComfyUI launch arguments (or Comfy Desktop `launchArgs`):
   ```text
   --enable-dynamic-vram --reserve-vram 4
   ```
3. **Do NOT coexist with `comfyui-xpu-backend`:**
   Remove or disable `custom_nodes\comfyui-xpu-backend` (or any standalone `ComfyUI-OmniXPU` duplicate) to avoid conflicting `comfy_kitchen` / ComfyUI monkey-patches.
4. **Verify inside ComfyUI:**
   Add the **`OmniXPU Status`** node to a workflow (or check `comfyui.log`) to confirm `omni_xpu_kernel`, `comfy_kitchen.xpu`, `comfy_aimdo.xpu`, and patch statuses.

### Useful Environment Variables

| Variable | Default (Windows) | Purpose |
| --- | --- | --- |
| `OMNI_ATTN_BACKEND` | `torch` | Set to `cute` to enable CUTLASS-SYCL FMHA (`omni_xpu_kernel.cute`); options: `auto`, `cute`, `esimd`, `torch`. |
| `OMNIXPU_PROVIDER_BOOTSTRAP` | `auto` | `auto`, `off`, or `required` for `comfy-kitchen-xpu-runtime` and `comfy-aimdo-xpu-runtime`. |
| `OMNIXPU_XPU_MEMORY_FRACTION` | `0.99` | Caching-allocator fraction applied during prestartup (`0` / `off` disables). |
| `OMNIXPU_ENABLE` | `1` | Set to `0` to disable all OmniXPU patches and providers. |
| `OMNIXPU_DEBUG` / `OMNIXPU_DEBUG_VERBOSE` | `0` | Set to `1` for kernel or dispatch tracing. |

---

## Release `v0.2.0` Wheel Assets & Checksums

| File | Package / Version | SHA-256 |
| --- | --- | --- |
| `omni_xpu_kernel-0.2.0b3+torch213.bmg-cp313-cp313-win_amd64.whl` | `omni-xpu-kernel==0.2.0b3+torch213.bmg` | `a5f3d152bce545c517a2e8f1db958c3c14d67238e7467cfa89bbbb859cb9fd25` |
| `comfy_kitchen_xpu_runtime-0.2.41-py3-none-any.whl` | `comfy-kitchen-xpu-runtime==0.2.41` | `638756e43526542b625d5a81717bc0721f503e53240416479b42922cde52c605` |
| `comfy_aimdo_xpu_runtime-0.6.0-cp39-abi3-win_amd64.whl` | `comfy-aimdo-xpu-runtime==0.6.0` | `6f962d5bdc1f6be61e9ee1d4654b846ff3edea31e4c4fc4640ded6a405756981` |

*(Also recorded in [`SHA256SUMS.txt`](./SHA256SUMS.txt).)*

---

## Legacy `v0.1.0` & Benchmarks

- **Legacy `v0.1.0` Prototype:** The original standalone pure-PyTorch `xpu_backend.py` prototype and `comfy_aimdo-0.6.0+xpu` wheel remain preserved on the [`v0.1.0`](https://github.com/wildersa/ComfyUI-Intel-Arc-XPU/releases/tag/v0.1.0) git tag and GitHub Release.
- **Benchmarks:** Historical benchmark notes and observational logs remain in [`benchmarks/`](./benchmarks/).

---

## Credits & Licenses

- **Custom Node & Installer Scripts:** [`LICENSE`](./LICENSE) / [`LICENSE-APACHE-2.0.txt`](./LICENSE-APACHE-2.0.txt) (Apache-2.0)
- **`comfy-aimdo-xpu-runtime` Binary Wheel:** [`LICENSE-GPL-3.0.txt`](./LICENSE-GPL-3.0.txt) (GPL-3.0)
- **Full Attributions & Embedded Component Notices:** [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md)
  - [`intel/llm-scaler`](https://github.com/intel/llm-scaler) (`ComfyUI-OmniXPU`, `omni_xpu_kernel`) — Apache-2.0
  - [`xiangyuT/comfy-kitchen-xpu`](https://github.com/xiangyuT/comfy-kitchen-xpu) — Apache-2.0
  - [`xiangyuT/comfy-aimdo-xpu`](https://github.com/xiangyuT/comfy-aimdo-xpu) & [`Comfy-Org/comfy-aimdo`](https://github.com/Comfy-Org/comfy-aimdo) — GPL-3.0
  - `oneDNN`, `Strata`, [`oneapi-src/level-zero`](https://github.com/oneapi-src/level-zero), [`microsoft/Detours`](https://github.com/microsoft/Detours)
