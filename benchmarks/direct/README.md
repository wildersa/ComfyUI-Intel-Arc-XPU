# Direct A/B Benchmarks (Pending)

Controlled same-baseline benchmarks comparing **`torch==2.13.0+xpu` without improvements** against:

1. **Improvement 1 only:** `comfyui-xpu-backend` (`'xpu'` `comfy_kitchen` backend)
2. **Improvements 1 + 2:** `comfyui-xpu-backend` + `comfy-aimdo-xpu` (`--enable-dynamic-vram --reserve-vram 4`)

have **not been executed yet**.

## Planned Test Protocol

To qualify for this folder, runs should hold constant:

- Identical PyTorch XPU runtime (`torch==2.13.0+xpu`) and GPU driver
- Identical workflow JSON, input assets, seed, resolution, frame count, and step count
- Cold-start vs. warm-run separation (second execution after warmup)
- Recorded metrics: total prompt execution time, sampler `s/it` (init step + steady loop), peak VRAM usage, and host RAM/swap behavior
