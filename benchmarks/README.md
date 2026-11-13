# Benchmarks & Performance Notes

This directory tracks performance measurements and runtime analyses for **ComfyUI-Intel-Arc-XPU**, separated into two categories so controlled A/B data is never mixed with informal or cross-version observations:

- **[`direct/`](./direct/README.md)** — Controlled, same-PyTorch (`torch==2.13.0+xpu`) A/B benchmarks (baseline vs. `comfyui-xpu-backend` vs. full stack with DynamicVRAM). **Status: Pending (not yet executed).**
- **[`observational/`](./observational/)** — Non-controlled real-world session logs, OOM boundary analyses, and qualitative GPU telemetry observed across setups and PyTorch versions.

> [!NOTE]
> All files in `benchmarks/` avoid host-specific paths, hardware LUIDs, usernames, and unrelated local system details.
