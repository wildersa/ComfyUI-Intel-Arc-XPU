# Workflow Compatibility & Kernel Utilization on Intel Arc XPU

Installing **PyTorch XPU**, **`ComfyUI-Intel-Arc-XPU`**, **`omni_xpu_kernel`**, and **`DynamicVRAM`** enables native hardware acceleration — **but your workflow still determines whether those XPU kernels actually run.**

Many workflows shared online are built for NVIDIA CUDA setups. When loaded unchanged, they often fall back to slow weight emulation, CPU/RAM spilling, or unsupported CUDA-only nodes even when every XPU binary is healthy.

---

## 1. Use XPU-Friendly Weight Formats (`int8_convrot` vs. `fp8_scaled`)

- **Prefer `int8_convrot` / `int8_tensorwise` (and `bf16` / `fp16` / `W4A8`) when available:** Intel Arc Battlemage (`bmg`) accelerates INT8 matrix math natively via XPU hardware (`omni_xpu_kernel` and `comfy-kitchen`'s `xpu` backend).
- **Watch out for `fp8_scaled` weights:** NVIDIA workflows default to `fp8_scaled` checkpoints because Ada/Hopper GPUs have native FP8 tensor cores. On Intel Arc BMG, FP8 diffusion weights commonly fall back to dequantization + FP16 emulation, running significantly slower and using extra VRAM compared to an `int8_convrot` variant of the same model.
- **Balance Text Encoder size against total VRAM:** In large video workflows, a quantized Text Encoder (such as `nvfp4_awq` when supported) can keep total working set inside VRAM, whereas forcing a larger `int8` Text Encoder may overflow into host RAM / Windows Shared GPU Memory.

---

## 2. Avoid CUDA-Only Attention & Custom Nodes

- **`ModelAttentionBackend` (`comfy kitchen attention` -> `pytorch attention`):**
  - Workflows saved on CUDA often select `comfy kitchen attention`, which targets CUDA/Triton attention kernels.
  - Switch `ModelAttentionBackend` to **`pytorch attention`** so ComfyUI routes through PyTorch XPU SDPA or `OMNI_ATTN_BACKEND=cute` (CUTLASS-SYCL FMHA on Battlemage) when supported.
- **Bypass CUDA-specific attention / custom nodes:**
  - Nodes or patches targeting CUDA-only libraries (`SageAttention`, `FlashAttention`, `Triton`-only kernels, `RTXVideoSuperResolution`, or NVIDIA TensorRT/VFX packs) cannot run natively on Windows XPU. Bypass or replace them with ComfyUI core equivalents.
  - Optional third-party XPU-adapted nodes (such as `ComfyUI-Ref2VA-VSA` `BlockSparseAttention` / `sol-attn`, `InstantID`, `ChatterBox`, and `DenoTiled`) are automatically routed to XPU/`comfy-kitchen` when installed, and safely skipped if you do not use them.
- **Check `device` dropdowns in third-party nodes:**
  - Avoid hardcoded `"cuda"` strings inside custom nodes when `"default"` / `"auto"` or ComfyUI native loaders (`comfy.model_management`) are available.

---

## 3. Keep Memory Inside VRAM (`DynamicVRAM` & Sampling Cost)

- **Always launch with `--enable-dynamic-vram --reserve-vram 1` (compute-only GPU) or `--reserve-vram 2` (display GPU)** on Windows so WDDM retains headroom below physical capacity for large image and video graphs.
- **Avoid `CFG > 1.0` when a distilled `CFG = 1.0` LoRA fits your quality target:** Running classifier-free guidance (`cfg > 1.0`) processes positive and negative batches together (`batch=2`), roughly doubling attention and compute cost per step.
- **Keep prompt-enhancer LLMs outside the ComfyUI loop when latency matters:** Running large chat/VLM token-generation loops inside ComfyUI (`TextGenerate`) adds Python per-token overhead and JIT warmup per prompt length.

---

## 4. Verify Kernel Usage & Catch Bottlenecks Live

### Recommended: Install [`ComfyUI-XPU-Advisor`](https://github.com/wildersa/ComfyUI-XPU-Advisor)

**[ComfyUI-XPU-Advisor](https://github.com/wildersa/ComfyUI-XPU-Advisor)** is a lightweight, zero-binary companion extension (pure Python/JS) built specifically to catch workflow and runtime bottlenecks on Intel Arc (`xpu`) live inside the ComfyUI UI:

- Flags when a workflow loads **`fp8`** diffusion weights instead of faster `int8_convrot` alternatives.
- Detects **`comfy kitchen attention`** fallback warnings and reminds you to switch to **`pytorch attention`**.
- Alerts on **Windows Shared GPU Memory spikes**, disabled **`DynamicVRAM`**, tight VRAM headroom, or missing `--reserve-vram` headroom before or during a run.
- Displays live GPU/VRAM/Shared Memory metrics and actionable alerts directly in the ComfyUI top bar.

### Manual Log & Dispatch Check

1. Add the **`OmniXPU Status`** node to verify active `omni_xpu_kernel` capabilities and `OMNI_ATTN_BACKEND`.
2. For deep kernel/adapter tracing during a run, set `OMNIXPU_DEBUG=1` temporarily and inspect `<ComfyUI>\user\comfyui.log`.
