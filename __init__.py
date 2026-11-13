"""ComfyUI custom node entrypoint for the pure-PyTorch XPU comfy_kitchen backend."""

from __future__ import annotations

import logging

from .xpu_backend import install_comfy_xpu_hooks

logger = logging.getLogger("comfyui_xpu_backend")

try:
    _status = install_comfy_xpu_hooks()
    if _status.get("registered"):
        print(f"[comfyui-xpu-backend] XPU backend active ({_status})")
except Exception as exc:  # pragma: no cover - never block ComfyUI boot on unexpected runtime issue
    logger.warning("[comfyui-xpu-backend] Failed to initialize XPU backend: %s", exc)

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}
