# Observational Reference Timings (`OmniXPU` Windows `v0.2.0` Stack)

> [!IMPORTANT]
> **Observational Reference Data Only**
>
> These figures come from real Windows 11 sessions on an **Intel Arc Pro B70 (`~31.2 GiB` VRAM)** rather than strictly controlled single-variable A/B runs. Warm vs. cold cache state, attention backend (`torch` vs. `cute`), KV cache reuse, and model precision variants (`int8_convrot` vs. `fp8_scaled`) differed across runs and are noted explicitly below.

## Test System Profile

- **OS:** Windows 11 x64
- **GPU:** Intel Arc Pro B70 (`xpu:0`, `31,906 MB` VRAM reported)
- **ComfyUI:** `v0.45.3` (Python `3.13.12`)
- **PyTorch:** `torch==2.13.0+xpu`
- **Stack:** `ComfyUI-OmniXPU` + `omni_xpu_kernel==0.2.0b3+torch213.bmg` + `comfy-kitchen-xpu-runtime==0.2.41` + `comfy-aimdo-xpu-runtime==0.6.0` (`--enable-dynamic-vram --reserve-vram 4`)

---

## 1. `QwenImage21` (`qwen_image_2.1_int8_convrot`, 25 steps)

Same `QwenImage21` pose workflow tracked across setup changes on `torch==2.13.0+xpu`:

| Setup / Session State | `DynamicVRAM` | `OMNI_ATTN_BACKEND` | Total Prompt Time | Notes |
| --- | --- | --- | --- | --- |
| Early `torch==2.13.0+xpu` baseline | Off (`full load: True`) | — | `61.35s` | Cold start (`1.21s/it`, sampler `33s`) |
| `OmniXPU` (`0.2.0b3`) | On (`6,920 MB` DiT / `8,916 MB` TE staged) | `cute` | `42.22s` (cold) → `26.17s` / `26.27s` (warm) | Native `int8_tensorwise` + `CUTE_BHLD_D128` |
| `OmniXPU` (`0.2.0b3`) | On (`6,920 MB` DiT / `8,916 MB` TE staged) | `torch` | `43.67s` (cold) → `28.77s` (warm) | `25/25` steps in `18s`–`21s` (`1.15–1.34 it/s`) |
| `OmniXPU` (`0.2.0b3`) + `KVCacheModel` | On (`6,920 MB` DiT / `8,916 MB` TE + `172 MB` KV cache staged) | `torch` | **`24.44s`** | `25/25` steps in `10s` (`2.22–2.37 it/s`) |

---

## 2. `WAN_Animate2` (`wan_animate_2_int8_convrot`, 2 × 6 steps)

Video workflow comparing no-DynamicVRAM (`torch==2.12.1+xpu`) vs. DynamicVRAM enabled (`--enable-dynamic-vram --reserve-vram 4`):

| Setup | `DynamicVRAM` | Outcome | Total Prompt Time / Breakdown |
| --- | --- | --- | --- |
| Before DynamicVRAM (3 attempts) | Off (`full load: True`, `15,880.54 MB`) | **Failed (`XPU out of memory` at step `0/6`)** | Crashed trying `270–882 MiB` allocations |
| With DynamicVRAM (`comfy-aimdo` XPU hook) | On (`15,878 MB` DiT + `6,419 MB` TE staged) | **Completed (0 OOMs)** | **`370.91s`** (`31.06–31.27s/it` pass 1, `15.32–15.51s/it` pass 2) |

*(Full failure & recovery telemetry recorded in [`2026-11-13-qwen-and-wan-animate2-analysis.md`](./2026-11-13-qwen-and-wan-animate2-analysis.md).)*

---

## 3. `MiniMax H3 Ref2VA` (8s / 192 Frames @ 24 fps + Audio, 8 Steps)

Reference timings for `MiniMaxH3LongVideo` (`960×544`, `res_multistep`, `simple`, `tiled_decode=True`, `qwen3vl_32b_minimax_h3_nvfp4_awq` + `minimax_h3_ref2v_turbo_8step_v1.1_768p`):

| Diffusion Model Weight | `OMNI_ATTN_BACKEND` | Total Prompt Time | Stage Breakdown |
| --- | --- | --- | --- |
| `minimax_h3_ref2va_pruned_int8_convrot` (`19,995 MB` staged) | `cute` | `305.14s` | Conditioning `18.3s`, sampling `245.0s` (`29.45–29.57s/it`), decode `38.8s`, save `2.3s` |
| `minimax_h3_ref2va_pruned_fp8_scaled` (`~19.5 GiB`, 5 runs) | `torch` | **`281.92s` – `297.45s`** | Fastest warm runs: `281.92s` & `282.27s` (conditioning `12.9–13.0s`, sampling `243.1–243.4s` @ `29.00–29.11s/it`, decode `22.2–22.3s`, save `1.9s`) |

*(Note: those two groups used different quantized weight files (`int8_convrot` vs. `fp8_scaled`), so they are recorded as reference baselines rather than an `OMNI_ATTN_BACKEND` A/B comparison.)*
