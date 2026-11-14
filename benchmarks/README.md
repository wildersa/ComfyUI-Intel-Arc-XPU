# Benchmarks & Performance Notes

This directory tracks performance measurements and runtime analyses for **ComfyUI-Intel-Arc-XPU** on Windows (`Intel Arc Pro B70`), separated into two categories so controlled A/B data is never mixed with real-world reference runs:

- **[`observational/`](./observational/)** — Real-world session timings, OOM avoidance records, and reference baselines observed on Windows 11 (`torch==2.13.0+xpu`):
  - [`2026-11-14-omnixpu-reference-timings.md`](./observational/2026-11-14-omnixpu-reference-timings.md) — Reference timings for `QwenImage21`, `WAN_Animate2`, and `MiniMax H3 Ref2VA` using the `v0.2.0` (`OmniXPU` + `omni_xpu_kernel`) stack.
  - [`2026-11-13-qwen-and-wan-animate2-analysis.md`](./observational/2026-11-13-qwen-and-wan-animate2-analysis.md) — Earlier OOM-vs-DynamicVRAM telemetry and `QwenImage21` / `WAN_Animate2` dispatch audit.
- **[`direct/`](./direct/README.md)** — Controlled, single-variable A/B benchmarks on identical workflow parameters. **Status: Pending (not yet executed).**

> [!NOTE]
> All files in `benchmarks/` avoid host-specific paths, hardware LUIDs, usernames, and unrelated local system details.
