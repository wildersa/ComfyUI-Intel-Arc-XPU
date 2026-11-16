# Packaging, Bundled Wheels & Upstream Build Sources

> [!NOTE]
> Most users do **not** need this document. Installing via **Git URL / `git clone`** (`install.py` or `.\install.ps1`) installs the bundled binaries from [`wheels/`](../wheels) automatically without any compiler toolchain. (Comfy Registry packages exclude `.whl` files and `install.py` per Registry policy, so Registry users run `.\install.ps1` separately.)

---

## 1. Why Bundled Prebuilt Binaries Are Included

Stock ComfyUI on Windows runs Intel Arc (`xpu`) through generic PyTorch fallback without native Intel SYCL/CUTE/oneDNN kernels, `comfy-kitchen`'s XPU backend, or Windows `DynamicVRAM` support in `comfy-aimdo`. Because upstream `intel/llm-scaler` targets Linux Docker and building those components from source on Windows requires full C++/SYCL toolchains, `ComfyUI-Intel-Arc-XPU` ships the prebuilt Windows wheels directly inside [`wheels/`](../wheels):

| Distribution | Version | Packaging | Role |
| --- | --- | --- | --- |
| `omni-xpu-kernel` | `0.2.0b3+torch213.bmg` | Compiled C++/SYCL (`cp313-win_amd64`) | CUTLASS-SYCL `CUTE` FMHA, ESIMD SDP, oneDNN GEMM, INT8/FP8/ConvRot/RoPE/norm ops |
| `comfy-kitchen-xpu-runtime` | `0.2.41` | Pure Python (`py3-none-any`) | Registers `comfy-kitchen`'s `xpu` backend and routes quantized/ops calls to `omni_xpu_kernel` |
| `comfy-aimdo-xpu-runtime` | `0.6.0` | Compiled C/C++ (`cp39-abi3-win_amd64`) | `aimdo_xpu.dll` + Windows XPU `DynamicVRAM` allocator/offloading overlay |

---

## 2. How `install.py` Installs Them

When ComfyUI Manager installs the extension (or when you run `python install.py` manually):

1. **Environment Gate:** [`install.py`](../install.py) checks for **Windows x64**, **Python `3.13` (`cp313`)**, and **`torch==2.13.0+xpu`**. On mismatch, it prints a `CONFLICT` message with exact instructions and exits without modifying packages.
2. **Official Package Alignment:** Ensures `comfy-aimdo==0.6.0` and `comfy-kitchen==0.2.41` are present (`--no-deps`).
3. **Offline Bundled Wheel Install:** Resolves the three `.whl` files directly from [`wheels/`](../wheels), verifies each file against [`SHA256SUMS.txt`](../SHA256SUMS.txt), and installs them (`--no-deps`). *(If `wheels/` is manually emptied, `install.py` falls back to downloading verified wheels from GitHub Release `v0.2.0`.)*

### Dry Run / Manual CLI Check

```powershell
"<ComfyUI_Python>\python.exe" install.py --dry-run
"<ComfyUI_Python>\python.exe" install.py
```

---

## 3. Manual / Comfy Desktop Helper (`install.ps1`)

For users who cloned/downloaded the repository outside ComfyUI Manager or want interactive setup for Comfy Desktop:

```powershell
git clone https://github.com/wildersa/ComfyUI-Intel-Arc-XPU.git
cd ComfyUI-Intel-Arc-XPU
.\install.ps1
```

`install.ps1` auto-detects your ComfyUI install, checks for running `python.exe` file locks, warns on conflicting older XPU nodes (such as `comfyui-xpu-backend`), installs the packages, and optionally sets `OMNI_ATTN_BACKEND=cute`. *(Supports `-ComfyPath`, `-DryRun`, and `-Yes`.)*

---

## 4. Upstream Repositories & Windows Build Toolchains

Exact upstream revisions and toolchains used to build the bundled components (see [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for full notices and licenses):

- **`intel/llm-scaler`** (`1101fd71e10757e90bb6aae0eddbe1aea6317bb3`): `ComfyUI-OmniXPU` adapters + `omni_xpu_kernel` compiled with Intel oneAPI DPC++/C++, `sycl-tla`, `oneDNN`, and `Strata`.
- **`xiangyuT/comfy-kitchen-xpu`** (`f27f58782c51b6ab314b4cc0f6e65c8a6bafa277`, `0.2.41`): XPU backend provider packaged as `comfy_kitchen_xpu_runtime`.
- **`xiangyuT/comfy-aimdo-xpu`** (`3bc6afbcb92e10a2d4737e79d170190ed787bf89`, `0.6.0`, based on `Comfy-Org/comfy-aimdo`): Windows `aimdo_xpu.dll` compiled with MSVC using `oneapi-src/level-zero` and `microsoft/Detours`.

### Prefer Not to Use Prebuilt Binaries?

If you prefer not to use the precompiled `.whl`/`.dll` files packaged in this repository, follow Intel's official repository and upstream stack instructions directly at **<https://github.com/intel/llm-scaler>** (which references the corresponding upstream XPU components) and build/install the stack directly from Intel's sources. Self-built / custom compilation setups are out of scope for this repository.

---

## 5. Advanced Environment Variables

| Variable | Default (Windows) | Purpose |
| --- | --- | --- |
| `OMNI_ATTN_BACKEND` | `torch` | `torch` (PyTorch SDPA), `cute` (CUTLASS-SYCL FMHA), `auto`, or `esimd`. |
| `OMNIXPU_PROVIDER_BOOTSTRAP` | `auto` | Provider routing mode: `auto`, `off`, or `required`. |
| `OMNIXPU_XPU_MEMORY_FRACTION` | `0.99` | Caching-allocator fraction applied on prestartup (`0` / `off` disables). |
| `OMNIXPU_ENABLE` | `1` | Set to `0` to disable all OmniXPU providers and patches. |
| `OMNIXPU_DEBUG` / `OMNIXPU_DEBUG_VERBOSE` | `0` | Set to `1` for kernel or dispatch tracing. |
