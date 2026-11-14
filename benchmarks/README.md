# Benchmarks & Reference Timings

Observational timings recorded on **Windows 11 (`Intel Arc Pro B70`, `xpu:0`, ComfyUI `v0.45.3`, `torch==2.13.0+xpu`)** using `ComfyUI-OmniXPU`, `omni_xpu_kernel==0.2.0b3+torch213.bmg`, `comfy-kitchen-xpu-runtime==0.2.41`, and `comfy-aimdo-xpu-runtime==0.6.0` (`--enable-dynamic-vram --reserve-vram 4`).

## Reference Runs

| Workload | Settings | Baseline / Prior Run | `OmniXPU` + `DynamicVRAM` |
| --- | --- | --- | --- |
| **`QwenImage21`** | `qwen_image_2.1_int8_convrot`, `25` steps | `61.35s` (no DynamicVRAM) | `42.22s`–`43.67s` cold · `26.17s`–`28.77s` warm · **`24.44s`** with `KVCacheModel` (`2.22–2.37 it/s`) |
| **`WAN_Animate2`** | `wan_animate_2_int8_convrot`, `2 × 6` steps | `XPU out of memory` at step `0/6` (3 attempts) | **`370.91s`** (`31.1s/it` pass 1, `15.4s/it` pass 2, **0 OOMs**) |
| **`MiniMax H3 Ref2VA`** | `8-step v1.1` Turbo LoRA, `960×544`, `8s` (`192` frames + audio), `res_multistep` | — *(reference only)* | **`281.92s`–`297.45s`** (`fp8_scaled`, `torch`) · `305.14s` (`int8_convrot`, `cute`) |

## Sample Output

- **[`assets/minimax_h3_ref2va_8s_b70.mp4`](./assets/minimax_h3_ref2va_8s_b70.mp4)** (`6.34 MB`, `960×544`, `24 fps`, `8.0s` video + AAC audio, **0.00 ms** AV drift; `293.36s` total: `14.4s` conditioning, `252.5s` sampling, `22.3s` decode, `2.3s` save).
  - Embeds the ComfyUI workflow (with empty `api_key` and no local paths).

## Notes

- These numbers come from real session logs rather than controlled single-variable A/B tests (`WAN_Animate2` baseline crashed on `torch==2.12.1+xpu` before DynamicVRAM; `MiniMax H3` reference runs used different diffusion weights for `torch` vs. `cute`).
- No host-specific paths, LUIDs, usernames, or credentials are stored in this directory.
