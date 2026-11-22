import logging
import importlib.util
import os
import sys

log = logging.getLogger("ComfyUI-OmniXPU")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./js"

_SETUP_NOTICE = {"status": "ok", "severity": "info", "summary": "", "detail": ""}


def _register_setup_route():
    try:
        from aiohttp import web
        import server

        prompt_server = getattr(server, "PromptServer", None)
        instance = getattr(prompt_server, "instance", None)
        routes = getattr(instance, "routes", None)
        if routes is None:
            return

        @routes.get("/omnixpu/setup-status")
        async def omnixpu_setup_status(request):
            return web.json_response(_SETUP_NOTICE)
    except Exception:
        # Standalone imports/tests run without a live PromptServer instance.
        pass


def _is_dynamic_vram_enabled() -> bool:
    cli_args = sys.modules.get("comfy.cli_args")
    args = getattr(cli_args, "args", None)
    return bool(getattr(args, "enable_dynamic_vram", False)) if args is not None else False

# Resolve package directory (works regardless of how we're loaded)
_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG = "ComfyUI-OmniXPU"
# ComfyUI registers directory custom nodes under their full path. Our named
# child modules need this package alias for imports from the plugin root.
sys.modules.setdefault(_PKG, sys.modules[__name__])


def _load(rel_path, mod_name):
    """Load a .py file as a named module."""
    fpath = os.path.join(_DIR, *rel_path.split("/"))
    if os.path.isdir(fpath):
        fpath = os.path.join(fpath, "__init__.py")
    elif not fpath.endswith(".py"):
        fpath += ".py"
    search = [os.path.dirname(fpath)] if os.path.basename(fpath) == "__init__.py" else None
    spec = importlib.util.spec_from_file_location(mod_name, fpath, submodule_search_locations=search)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    import torch

    torch_ver = str(getattr(torch, "__version__", ""))
    xpu_ok = hasattr(torch, "xpu") and torch.xpu.is_available()
    py_ok = sys.platform != "win32" or sys.version_info[:2] == (3, 13)
    torch_ok = torch_ver.startswith("2.13.0+xpu")

    if not (py_ok and xpu_ok and torch_ok):
        _SETUP_NOTICE = {
            "status": "conflict",
            "severity": "error",
            "summary": "ComfyUI-Intel-Arc-XPU: Environment Conflict",
            "detail": (
                f"Requires Windows x64, Python 3.13, and torch==2.13.0+xpu with Intel XPU "
                f"(detected Python {sys.version_info[0]}.{sys.version_info[1]}, torch={torch_ver!r}, "
                f"xpu={xpu_ok}). Extension disabled; ComfyUI continues with stock behavior."
            ),
        }
        log.error(
            "[OmniXPU] CONFLICT: ComfyUI-Intel-Arc-XPU requires Windows x64, Python 3.13, "
            "and torch==2.13.0+xpu with Intel XPU available (detected Python %d.%d, "
            "torch=%r, xpu_available=%s). Skipping patches; ComfyUI will continue with stock behavior.",
            sys.version_info[0],
            sys.version_info[1],
            torch_ver,
            xpu_ok,
        )
    else:
        # Load probe and run capability check
        probe = _load("probe.py", f"{_PKG}.probe")
        probe.probe()
        if probe.version is None:
            _SETUP_NOTICE = {
                "status": "missing_binaries",
                "severity": "warn",
                "summary": "ComfyUI-Intel-Arc-XPU: Extra Setup Step Required",
                "detail": (
                    "Native Windows XPU binaries (omni_xpu_kernel) are missing from this Python "
                    "environment. Finish setup from https://github.com/wildersa/ComfyUI-Intel-Arc-XPU "
                    "(run install.ps1 or install.py) and launch with --enable-dynamic-vram --reserve-vram 1 (or 2 with display)."
                ),
            }
            log.warning(
                "[OmniXPU] Native Windows XPU runtime binaries (omni_xpu_kernel) are not installed "
                "in this Python environment yet. If you installed via Comfy Registry, finish the extra "
                "binary setup step from https://github.com/wildersa/ComfyUI-Intel-Arc-XPU (install.ps1 / "
                "install.py on Python 3.13 + torch==2.13.0+xpu) and launch ComfyUI with "
                "--enable-dynamic-vram --reserve-vram 1 (or 2 with display)."
            )
        elif not _is_dynamic_vram_enabled():
            _SETUP_NOTICE = {
                "status": "missing_dynamic_vram",
                "severity": "info",
                "summary": "ComfyUI-Intel-Arc-XPU: DynamicVRAM Flag Recommended",
                "detail": (
                    "XPU compute kernels are active, but --enable-dynamic-vram is not set. "
                    "Add --enable-dynamic-vram --reserve-vram 1 (or 2 with display) to prevent XPU OOM on heavy/video workflows."
                ),
            }

        # Load config and patches
        _load("config.py", f"{_PKG}.config")
        patches = _load("patches", f"{_PKG}.patches")

        config_mod = sys.modules[f"{_PKG}.config"]
        patches.apply_all_patches(config_mod.config)

        # Load diagnostics node
        _load("nodes/__init__.py", f"{_PKG}.nodes")
        diag = _load("nodes/diagnostics.py", f"{_PKG}.nodes.diagnostics")
        NODE_CLASS_MAPPINGS["OmniXPUStatus"] = diag.OmniXPUStatus
        NODE_DISPLAY_NAME_MAPPINGS["OmniXPUStatus"] = "OmniXPU Status"

except Exception as e:
    import traceback
    log.error("[OmniXPU] Initialization failed: %s\n%s", e, traceback.format_exc())

_register_setup_route()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
