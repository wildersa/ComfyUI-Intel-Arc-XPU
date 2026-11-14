# Direct A/B Benchmarks (Pending)

Controlled single-variable benchmarks comparing stock PyTorch XPU execution (`torch==2.13.0+xpu`, without OmniXPU runtime providers) against the full **ComfyUI-Intel-Arc-XPU** (`ComfyUI-OmniXPU` + `omni_xpu_kernel` + `comfy-kitchen-xpu-runtime` + `comfy-aimdo-xpu-runtime` with `--enable-dynamic-vram --reserve-vram 4`) stack have **not been executed yet**.

For real-world Windows reference timings and OOM comparisons recorded from session logs, see [`../observational/`](../observational/).

## Planned Test Protocol

To qualify for this folder, runs should hold constant:

- Identical PyTorch XPU runtime (`torch==2.13.0+xpu`) and GPU driver
- Identical workflow JSON, input assets, seed, resolution, frame count, and step count
- Cold-start vs. warm-run separation (second execution after warmup)
- Recorded metrics: total prompt execution time, sampler `s/it` (init step + steady loop), peak VRAM usage, and host RAM/swap behavior
