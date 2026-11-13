# Observational Analysis: `QwenImage21` & `WAN_Animate2` Sessions (2026-11-13)

> [!IMPORTANT]
> **Non-Direct / Observational Data Only**
>
> This document summarizes real usage logs rather than a controlled A/B benchmark. Runs span different PyTorch versions (`2.12.1+xpu` vs. `2.13.0+xpu`), warm vs. cold caches, and cases where the baseline crashed with Out-Of-Memory (OOM) before completing. Treat timing comparisons here as **directional observations**, not controlled benchmarks.

## Test Environment Profile

- **OS:** Windows 11 (64-bit)
- **GPU Class:** Intel Arc Pro B70 (~31.16 GiB / ~31,906 MB VRAM reported)
- **ComfyUI:** `v0.45.3` (Python `3.13.12`)
- **Stack Active in Target Session:**
  - `torch==2.13.0+xpu`
  - `comfyui-xpu-backend` (`'xpu'` backend priority ahead of `triton` and `eager`, `gates_patched=True`)
  - `comfy-aimdo` `0.6.0+xpu.g3bc6afb.dirty` with `--enable-dynamic-vram --reserve-vram 4`

---

## 1. `WAN_Animate2` Video Workflow: OOM Without DynamicVRAM vs. Clean Completion With Both Improvements

### Before DynamicVRAM (3 earlier attempts on `torch==2.12.1+xpu`)

Without DynamicVRAM, ComfyUI attempted `full load: True` for the `15,880.54 MB` (`~15.5 GiB`) `wan_animate_2_int8_convrot` diffusion model alongside VAE/buffers, leaving too little headroom on the ~31.16 GiB GPU and failing with **`XPU out of memory` at step `0/6`** in every logged attempt:

| Attempt | Configuration | Model Load Mode | Failure Point | OOM Allocation & Free VRAM Reported |
| --- | --- | --- | --- | --- |
| 1 | `torch==2.12.1+xpu`, no `'xpu'` backend, no DynamicVRAM | `full load: True` (`15,880.54 MB`) | Step `0/6` (~`00:30` in) | Tried `270.00 MiB` (`2.90 GiB` free; failed inside `eager` `dequantize_int8_convrot_weight_dtype`) |
| 2 | `torch==2.12.1+xpu`, `'xpu'` backend (`int8_tensorwise`), no DynamicVRAM | `full load: True` (`15,880.54 MB`) | Step `0/6` | Tried `882.00 MiB` (`4.50 GiB` free; failed in `gelu` activation inside `int8_linear`) |
| 3 | `torch==2.12.1+xpu`, `'xpu'` backend (`int8_tensorwise`), no DynamicVRAM | `full load: True` (`15,880.54 MB`) | Step `0/6` (~`00:33` in) | Tried `684.00 MiB` (`2.87 GiB` free; failed in `eager` `apply_rope1`) |

### After Both Improvements (`torch==2.13.0+xpu` + `'xpu'` backend + DynamicVRAM)

With `DynamicVRAM support detected and enabled` and `comfy-aimdo` XPU USM/UR hooking active, the exact same model set loaded via DynamicVRAM staging and completed cleanly with **zero OOM or GPU errors**:

- **Staged DynamicVRAM Footprints:**
  - `CLIPVisionModelProjection` (`clip_vision_h`): `1,205 MB Staged` (`132` weights / `330 KB` force pre-loaded)
  - `WanTEModel` (`umt5_xxl_fp8_e4m3fn_scaled`): `6,419 MB Staged` (`73` weights / `488 KB` force pre-loaded)
  - `WanVAE` (`Wan2_1_VAE_bf16`): `241 MB Staged` (`60` weights / `61 KB` force pre-loaded)
  - `WAN_Animate2` (`wan_animate_2_int8_convrot`): `15,878 MB Staged`, `1,059` patches attached (`202` weights / `2,008 KB` force pre-loaded)
- **Quantization Mode:** `Native ops: int8_tensorwise`, `model weight dtype torch.float16, manual cast: torch.float16`
- **Execution Timings:**
  - **Sampler Pass 1 (6 steps):** `01:14` model init, `31.06s–31.27s/it` steady loop, `03:06` sampler total (`31.15s/it` avg)
  - **Sampler Pass 2 (6 steps):** `00:22` model init, `15.32s–15.51s/it` steady loop, `01:32` sampler total (`15.42s/it` avg)
  - **Total Prompt Execution:** **`370.91 seconds` (~6m 11s)**

---

## 2. `QwenImage21` Image Workflow (`25` Steps): No-DynamicVRAM vs. DynamicVRAM (`torch==2.13.0+xpu`)

Three runs on `torch==2.13.0+xpu` with `comfyui-xpu-backend` active (`qwen_image_2.1_int8_convrot`, `Native ops: int8_tensorwise`, `192` patches):

| Run | DynamicVRAM | State | Sampler Steady Speed | Sampler Total | Prompt Total |
| --- | --- | --- | --- | --- | --- |
| Prior session (`comfy-aimdo 0.6.0`) | Off (`full load: True`) | Cold start | `1.21s/it` (init `3.00s`, `2.98s`, `2.16s`) | `00:33` (`1.35s/it` avg) | `61.35s` |
| Target session Run 1 | On (`6,920 MB` / `8,916 MB` / `644 MB Staged`) | Cold start | `1.19–1.22s/it` (init `5.59s`) | `00:27` (`1.12s/it` avg) | `53.41s` |
| Target session Run 2 | On (staged cache warm) | Warm | `1.19–1.22s/it` | `00:33` (`1.35s/it` avg) | `40.69s` |

*Note: Only one prior `61.35s` run without DynamicVRAM on `torch==2.13.0+xpu` was logged in that session, and patch/warmup conditions differed between cold and warm runs.*

---

## 3. Fallback & Op Dispatch Audit

Checking the dispatch logs for both runs in the target session confirms **no unexpected fallback occurred on the main `INT8` models**:

- **Active Native Path (`int8_tensorwise`):** `qwen_image_2.1_int8_convrot` and `wan_animate_2_int8_convrot` both reported `Native ops: int8_tensorwise`, routing `INT8` linear/dequant operations through `xpu_backend.py` (`priority=['ascend', 'cuda', 'xpu', 'triton', 'eager']`).
- **Expected Emulation / Pass-Throughs (Not Failures):**
  1. **`umt5_xxl_fp8_e4m3fn_scaled` (`WanTEModel`):** uses `float8_e4m3fn` weights, which remain listed under `emulated ops` (`MixedPrecisionOps` / `eager`) because `comfyui-xpu-backend` implements `INT8`, not `FP8`. Text encoding runs once per prompt before the diffusion sampler loop.
  2. **Non-INT8 ops in `comfy_kitchen` (e.g., `apply_rope1`):** continue dispatching to `'eager'` (native PyTorch XPU), as `'xpu'` intentionally overrides only `INT8` kernels.
- **No GPU / Memory Errors:** Zero OOMs or backend exceptions occurred during inference (the only errors in the session log were benign browser/websocket `WinError 10054` `ConnectionResetError` callbacks prior to execution).

---

## 4. Qualitative GPU Telemetry (Windows Task Manager)

During the `WAN_Animate2` run with DynamicVRAM + `'xpu'` backend enabled, GPU **Compute** utilization remained steady rather than repeatedly pausing and alternating with high **Copy** queue spikes.

Mechanistically, this matches the architectural difference visible in the logs:

1. Without DynamicVRAM, approaching the ~31 GiB ceiling causes whole-model pressure, synchronous layer/WDDM paging, and heavy **Copy** bursts interrupting **Compute** (and in `WAN_Animate2`, OOM crashes).
2. With `comfy-aimdo-xpu` managing staged virtual residency within `--reserve-vram 4` and `xpu_backend.py` eliminating temporary `FP32` dequantization spikes, the GPU **Compute** pipeline stays fed without constant memory thrashing.
