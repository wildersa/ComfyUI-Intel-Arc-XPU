# Third-Party Notices and Licenses

This repository packages Windows-ready Intel Arc XPU (`xpu`) integration for ComfyUI, combining upstream `ComfyUI-OmniXPU` with prebuilt Windows wheels distributed via GitHub Release `v0.2.0`.

- Repository custom-node code and installer scripts: **Apache License 2.0** ([`LICENSE`](./LICENSE) / [`LICENSE-APACHE-2.0.txt`](./LICENSE-APACHE-2.0.txt))
- `comfy-aimdo-xpu-runtime` binary wheel (Release `v0.2.0` asset): **GNU General Public License v3.0 (GPL-3.0)** ([`LICENSE-GPL-3.0.txt`](./LICENSE-GPL-3.0.txt))

---

## 1. `intel/llm-scaler` (`ComfyUI-OmniXPU` & `omni_xpu_kernel`)

- **Upstream Repository:** <https://github.com/intel/llm-scaler>
- **Source Revision:** `1101fd71e10757e90bb6aae0eddbe1aea6317bb3`
- **License:** Apache License, Version 2.0 ([`LICENSE-APACHE-2.0.txt`](./LICENSE-APACHE-2.0.txt))

### Bundled Notices inside `omni_xpu_kernel-0.2.0b3+torch213.bmg-cp313-cp313-win_amd64.whl`

1. **Strata (MIT License)** — `omni_xpu_kernel/licenses/Strata-MIT.txt`
   - Copyright (c) 2026 Niko1221 and the Strata contributors
   - Licensed under the MIT License.

2. **oneAPI Deep Neural Network Library (oneDNN)** — `omni_xpu_kernel/.libs/onednn/LICENSE` & `THIRD-PARTY-PROGRAMS`
   - Licensed under the Apache License, Version 2.0.
   - Includes third-party components documented in `omni_xpu_kernel/.libs/onednn/THIRD-PARTY-PROGRAMS`:
     - **XByak** (`third_party/xbyak/`) — Copyright (c) 2007 MITSUNARI Shigeo (BSD-3-Clause)
     - **Googletest** (`third_party/gtest/`) — Copyright 2005-2015 Google Inc. (BSD-3-Clause)
     - **Instrumentation and Tracing Technology API** (`third_party/ittnotify/`) — Copyright (c) 2005-2014 Intel Corporation (BSD-3-Clause)
     - **CMake** (`cmake/FindOpenCL.cmake`, `FindBLAS.cmake`, `FindACL.cmake`) — Copyright 2000-2020 Kitware, Inc. and Contributors (BSD-3-Clause)
     - **spdlog** (`third_party/spdlog/`) & **fmt** — Copyright (c) 2016 Gabi Melman (MIT License)
     - **Xbyak_aarch64** (`third_party/xbyak_aarch64/`) — Copyright 2019-2020 FUJITSU LIMITED (Apache-2.0)
     - **Boost C++ Libraries** (`src/common/primitive_hashing.hpp`) — Copyright 2005-2014 Daniel James (Boost Software License 1.0)
     - **Intel(R) Graphics Compute Runtime for oneAPI Level Zero and OpenCL(TM)** — Copyright (c) 2018-2021 Intel Corporation et al. (MIT License)
     - **Sphinx** (`doc/sphinx/conf.py`) — Copyright (c) 2007-2021 Sphinx team (BSD-2-Clause)
     - **Intel(R) Metrics Discovery API** (`third_party/mdapi/`) — Copyright (c) 2019 Intel Corporation (MIT License)
     - **Xbyak_riscv** (`third_party/xbyak_riscv/`) — Copyright (c) 2007 MITSUNARI Shigeo
     - **Khronos OpenCL Headers** (`third_party/opencl/`) — Copyright (c) 2008 The Khronos Group Inc. (Apache-2.0)

---

## 2. `xiangyuT/comfy-kitchen-xpu` (`comfy_kitchen_xpu_runtime`)

- **Upstream Repository:** <https://github.com/xiangyuT/comfy-kitchen-xpu>
- **Source Revision:** `f27f58782c51b6ab314b4cc0f6e65c8a6bafa277` (`0.2.41`)
- **License:** Apache License, Version 2.0, Copyright (c) 2025 Comfy Org. All rights reserved.
- **Third-Party Notices (from upstream `NOTICE`):**
  - **PyTorch** — Copyright (c) 2016-present, Meta Platforms, Inc. and PyTorch contributors (BSD-3-Clause)
  - **nanobind** — Copyright (c) 2022 Wenzel Jakob (BSD-3-Clause)
  - **FlashAttention** — Copyright (c) 2022, FlashAttention contributors (BSD-3-Clause)
  - **PyTorch AO (`torchao`)** — Copyright (c) Meta Platforms, Inc. and affiliates (BSD-3-Clause)
  - **SageAttention** — Copyright (c) 2024 by SageAttention team (Apache-2.0)
  - *(Note: NVIDIA CUDA components from upstream `comfy-kitchen` are excluded from the XPU runtime provider wheel.)*

---

## 3. `xiangyuT/comfy-aimdo-xpu` (`comfy_aimdo_xpu_runtime`)

- **Upstream Repository:** <https://github.com/xiangyuT/comfy-aimdo-xpu> (based on [`Comfy-Org/comfy-aimdo`](https://github.com/Comfy-Org/comfy-aimdo))
- **Source Revision:** `3bc6afbcb92e10a2d4737e79d170190ed787bf89` (`0.6.0`)
- **License:** GNU General Public License v3.0 ([`LICENSE-GPL-3.0.txt`](./LICENSE-GPL-3.0.txt))
- **Windows Native Dependencies Used in Build:**
  - [`oneapi-src/level-zero`](https://github.com/oneapi-src/level-zero) (MIT License)
  - [`microsoft/Detours`](https://github.com/microsoft/Detours) (MIT License)
