"""Adapt third-party custom node packs that assume CUDA on Intel Arc XPU.

Keeps third-party pack directories unmodified so updates never discard XPU fixes
and preserves stock behavior whenever CUDA is available or XPU is absent.
"""

from __future__ import annotations

import contextvars
import functools
import importlib
import inspect
import logging
import sys

import torch


log = logging.getLogger("ComfyUI-OmniXPU")

_MARKER = "_omnixpu_custom_nodes_xpu_patched"
_HOOK_MARKER = "_omnixpu_custom_nodes_xpu_hooked"
_CHATTERBOX_ALLOW_XPU: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_omnixpu_chatterbox_allow_xpu", default=False
)


def _xpu_only() -> bool:
    return (
        hasattr(torch, "xpu")
        and torch.xpu.is_available()
        and not torch.cuda.is_available()
    )


def _node_module(node_name: str, module_suffix: str = ""):
    nodes = sys.modules.get("nodes")
    if nodes is not None:
        cls = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get(node_name)
        if cls is not None:
            mod = sys.modules.get(getattr(cls, "__module__", ""))
            if mod is not None and (not module_suffix or getattr(mod, "__name__", "").endswith(module_suffix)):
                return mod
    if module_suffix:
        for mod_name, mod in list(sys.modules.items()):
            if mod is not None and (mod_name == module_suffix or mod_name.endswith("." + module_suffix)):
                return mod
    return None


def _patch_ref2va_vsa() -> bool:
    mod = _node_module("Ref2VAVSAGatePatch")
    if mod is None:
        return False
    patch_cls = getattr(mod, "FastH3VSAPatch", None)
    orig_attn = getattr(mod, "_vsa_attention", None)
    if patch_cls is None or orig_attn is None:
        return False
    if hasattr(patch_cls.patch, _MARKER):
        return True

    try:
        import comfy_kitchen as ck
        backends = ck.list_backends()
    except Exception:
        return False

    if backends.get("cuda", {}).get("available", False):
        return False
    xpu_caps = backends.get("xpu", {}).get("capabilities", [])
    eager_caps = backends.get("eager", {}).get("capabilities", [])
    if backends.get("xpu", {}).get("available", False) and "sol_attn" in xpu_caps:
        sol_backend = "xpu"
    elif backends.get("eager", {}).get("available", False) and "sol_attn" in eager_caps:
        sol_backend = "eager"
    else:
        return False

    def _vsa_attention_xpu(attn, x, rope_freqs, transformer_options, state):
        if x.device.type != "xpu":
            return orig_attn(attn, x, rope_freqs, transformer_options, state)

        import comfy.model_management
        import comfy.quant_ops
        import comfy_kitchen as ck_inner

        runtime = transformer_options.get(mod._RUNTIME_KEY)
        if runtime is None:
            raise RuntimeError("FastH3 VSA runtime metadata is missing; diffusion wrapper did not run")
        if not hasattr(attn, "to_gate_compress"):
            raise RuntimeError(
                "FastH3 VSA checkpoint gate is missing (attn.to_gate_compress). "
                "Load the FastVideo VSA checkpoint or attach gate patches first."
            )

        s = int(x.shape[0])
        inner = int(attn.heads * attn.head_dim)
        geo = state.geometry(runtime["prefix_segments"], runtime["video_grid"], x.device)
        if geo.total_rows != s:
            raise RuntimeError(f"FastH3 VSA geometry has {geo.total_rows} rows but attention got {s}")

        x_t = x.new_zeros((geo.padded_rows, x.shape[-1]))
        x_t[geo.untile_index] = x
        sp = geo.padded_rows
        q, k, v = attn.qkv_proj(x_t).split(inner, dim=-1)
        v = v.view(1, sp, attn.heads, attn.head_dim).clone()

        if rope_freqs is not None:
            rope_t = rope_freqs.new_zeros((rope_freqs.shape[0], sp, *rope_freqs.shape[2:]))
            rope_t[:, geo.untile_index] = rope_freqs
            q = q.view(1, sp, attn.heads, attn.head_dim)
            k = k.view(1, sp, attn.heads, attn.head_dim)
            qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
            rot = rope_t.shape[-3] * 2
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_t, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
            )
        else:
            q = attn.q_norm(q.view(sp, attn.heads, attn.head_dim)).unsqueeze(0)
            k = attn.k_norm(k.view(sp, attn.heads, attn.head_dim)).unsqueeze(0)

        gate_t = attn.to_gate_compress(x_t).view(1, sp, attn.heads, attn.head_dim)
        del x_t

        if q.dtype != torch.bfloat16 or int(attn.head_dim) != 128:
            raise RuntimeError(
                "FastH3 VSA XPU path requires BF16 with head_dim=128; "
                f"got device={q.device}, dtype={q.dtype}, head_dim={attn.head_dim}"
            )

        keep_ratio = max(0.0, min(1.0, 1.0 - state.sparsity))
        with ck_inner.use_backend(sol_backend):
            out_t = ck_inner.sol_attn(
                q,
                k,
                v,
                scale=attn.head_dim ** -0.5,
                sink_blocks=[0, geo.num_prefix_tiles],
                sink_q=[0, geo.num_prefix_tiles],
                topk_ratio=keep_ratio,
                tail=False,
                block_len=geo.block_len,
                coarse_gate=gate_t,
            )

        del q, k, v, gate_t
        out = out_t[:, geo.untile_index]
        del out_t
        out = out.reshape(s, inner)
        state.calls += 1
        return attn.out_proj(out)

    orig_patch = patch_cls.patch

    @functools.wraps(orig_patch)
    def _patch_xpu(self, model, sparsity=0.90):
        sparsity = float(sparsity)
        if not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity must be in [0,1)")

        patched = model.clone()
        options = patched.model_options
        if options.get(mod._PATCH_MARKER):
            raise RuntimeError("FastH3 VSA patch is already installed on this MODEL")

        transformer = options.setdefault("transformer_options", {})
        if transformer.get("optimized_attention_override") is not None:
            raise RuntimeError(
                "FastH3 VSA cannot stack with another optimized_attention override."
            )
        existing = transformer.get("patches_replace", {}).get("dit", {})
        if existing:
            raise RuntimeError(
                "FastH3 VSA requires an unmodified H3 DiT block chain; another dit patch_replace is already installed."
            )

        dm = getattr(getattr(patched, "model", None), "diffusion_model", None)
        if dm is None or type(dm).__name__ != "MiniMaxH3Model":
            raise ValueError("Ref2VA VSA requires a native ComfyUI MiniMaxH3Model")
        blocks = list(getattr(dm, "blocks", ()))
        if len(blocks) != 50:
            raise RuntimeError(f"FastH3 VSA expected 50 H3 DiT blocks, found {len(blocks)}")

        object_patches = set(getattr(patched, "object_patches", {}))
        missing = [
            i for i, b in enumerate(blocks)
            if not hasattr(b.attn, "to_gate_compress")
            and f"diffusion_model.blocks.{i}.attn.to_gate_compress" not in object_patches
        ]
        if missing:
            raise RuntimeError("FastH3 VSA compression gate is absent from blocks " + str(missing[:8]))

        state = mod._VSAState(sparsity=sparsity)
        patched.add_wrapper_with_key(
            importlib.import_module("comfy.patcher_extension").WrappersMP.DIFFUSION_MODEL,
            mod._WRAPPER_KEY,
            mod._make_diffusion_wrapper(state, dm),
        )
        for i, block in enumerate(blocks):
            patched.set_model_patch_replace(mod._make_block_replace(block, state), "dit", "double_block", i)

        backend_label = "xpu" if sol_backend == "xpu" else "xpu_eager"
        backend_note = "comfy_kitchen xpu" if sol_backend == "xpu" else "comfy_kitchen eager"
        options[mod._PATCH_MARKER] = {"version": 2, "sparsity": sparsity, "tile_size": 64, "backend": backend_label}
        mod.log.info(
            "FastH3 VSA XPU adapter installed: 50 blocks, sparsity=%.2f, tile=64 (%s)",
            sparsity,
            backend_note,
        )
        return (patched,)

    setattr(_patch_xpu, _MARKER, orig_patch)
    setattr(_vsa_attention_xpu, _MARKER, orig_attn)
    mod._vsa_attention = _vsa_attention_xpu
    patch_cls.patch = _patch_xpu
    return True


def _patch_instantid() -> bool:
    mod = _node_module("InstantIDFaceAnalysis")
    if mod is None:
        return False
    cls = getattr(mod, "InstantIDFaceAnalysis", None)
    if cls is None or hasattr(cls.load_insight_face, _MARKER):
        return cls is not None

    try:
        import onnxruntime as ort
        providers = set(ort.get_available_providers())
    except Exception:
        return False

    if "DmlExecutionProvider" not in providers or "CUDAExecutionProvider" in providers:
        return False

    orig = cls.load_insight_face

    @functools.wraps(orig)
    def _load_xpu(self, provider):
        if provider in ("CUDA", "DML", "DirectML"):
            model = mod.FaceAnalysis(
                name="antelopev2",
                root=mod.INSIGHTFACE_DIR,
                providers=["DmlExecutionProvider", "CPUExecutionProvider"],
            )
            model.prepare(ctx_id=0, det_size=(640, 640))
            return (model,)
        return orig(self, provider)

    setattr(_load_xpu, _MARKER, orig)
    cls.load_insight_face = _load_xpu
    return True


def _patch_chatterbox() -> bool:
    mod = _node_module("FL_ChatterboxTTS")
    dialog_mod = _node_module("FL_ChatterboxDialogTTS")
    if mod is None and dialog_mod is None:
        return False

    patched_any = False

    if mod is not None:
        if not hasattr(mod._torch_load_with_default_map_location, _MARKER):
            orig_load = mod._torch_load_with_default_map_location

            @functools.wraps(orig_load)
            def _load_map_xpu(load_func, *args, **kwargs):
                if "map_location" not in kwargs and _xpu_only():
                    kwargs["map_location"] = torch.device("xpu")
                return orig_load(load_func, *args, **kwargs)

            setattr(_load_map_xpu, _MARKER, orig_load)
            mod._torch_load_with_default_map_location = _load_map_xpu
            patched_any = True

        if not hasattr(mod.clear_cached_model, _MARKER):
            orig_clear = mod.clear_cached_model

            @functools.wraps(orig_clear)
            def _clear_xpu(model_type=None):
                orig_clear(model_type)
                if _xpu_only():
                    torch.xpu.empty_cache()

            setattr(_clear_xpu, _MARKER, orig_clear)
            mod.clear_cached_model = _clear_xpu
            patched_any = True

        for loader_name in ("load_turbo_tts_model", "load_tts_model", "load_multilingual_model", "load_vc_model"):
            orig_loader = getattr(mod, loader_name, None)
            if orig_loader is None or hasattr(orig_loader, _MARKER):
                continue

            @functools.wraps(orig_loader)
            def _loader_xpu(device, _orig=orig_loader):
                if device == "cpu" and _CHATTERBOX_ALLOW_XPU.get() and _xpu_only():
                    device = "xpu"
                return _orig(device)

            setattr(_loader_xpu, _MARKER, orig_loader)
            setattr(mod, loader_name, _loader_xpu)
            patched_any = True

        for cls_name, fn_name in (
            ("FL_ChatterboxTTSNode", "generate_speech"),
            ("FL_ChatterboxTurboTTSNode", "generate_turbo_speech"),
            ("FL_ChatterboxMultilingualTTSNode", "generate_multilingual_speech"),
            ("FL_ChatterboxVCNode", "generate_vc"),
        ):
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue
            orig_fn = getattr(cls, fn_name, None)
            if orig_fn is None or hasattr(orig_fn, _MARKER):
                continue

            @functools.wraps(orig_fn)
            def _wrapped(self, *args, _orig=orig_fn, **kwargs):
                bound = inspect.signature(_orig).bind(self, *args, **kwargs)
                bound.apply_defaults()
                use_cpu = bool(bound.arguments.get("use_cpu", False))
                seed = bound.arguments.get("seed")
                if seed is not None and _xpu_only():
                    torch.xpu.manual_seed(seed)
                    torch.xpu.manual_seed_all(seed)
                tok = _CHATTERBOX_ALLOW_XPU.set((not use_cpu) and _xpu_only())
                try:
                    return _orig(self, *args, **kwargs)
                finally:
                    _CHATTERBOX_ALLOW_XPU.reset(tok)

            setattr(_wrapped, _MARKER, orig_fn)
            setattr(cls, fn_name, _wrapped)
            patched_any = True

    if dialog_mod is not None:
        tts_cls = getattr(dialog_mod, "ChatterboxTTS", None)
        node_cls = getattr(dialog_mod, "FL_ChatterboxDialogTTSNode", None)
        if tts_cls is not None and not hasattr(tts_cls.from_pretrained, _MARKER):
            orig_fp = tts_cls.from_pretrained.__func__ if isinstance(inspect.getattr_static(tts_cls, "from_pretrained"), classmethod) else tts_cls.from_pretrained

            @classmethod
            def _from_pretrained_xpu(cls, device="cpu", *args, _orig=orig_fp, **kwargs):
                if device == "cpu" and _CHATTERBOX_ALLOW_XPU.get() and _xpu_only():
                    device = "xpu"
                return _orig(cls, device=device, *args, **kwargs)

            setattr(_from_pretrained_xpu, _MARKER, orig_fp)
            tts_cls.from_pretrained = _from_pretrained_xpu
            patched_any = True

        if node_cls is not None and not hasattr(node_cls.generate_dialog, _MARKER):
            orig_gd = node_cls.generate_dialog

            @functools.wraps(orig_gd)
            def _dialog_xpu(self, *args, _orig=orig_gd, **kwargs):
                bound = inspect.signature(_orig).bind(self, *args, **kwargs)
                bound.apply_defaults()
                use_cpu = bool(bound.arguments.get("use_cpu", False))
                seed = bound.arguments.get("seed")
                if seed is not None and _xpu_only():
                    torch.xpu.manual_seed(seed)
                    torch.xpu.manual_seed_all(seed)
                tok = _CHATTERBOX_ALLOW_XPU.set((not use_cpu) and _xpu_only())
                try:
                    return _orig(self, *args, **kwargs)
                finally:
                    _CHATTERBOX_ALLOW_XPU.reset(tok)

            setattr(_dialog_xpu, _MARKER, orig_gd)
            node_cls.generate_dialog = _dialog_xpu
            patched_any = True

    return patched_any


def _patch_deno_tiled() -> bool:
    mod = _node_module("DenoMiniMaxH3TiledSampler", "deno_minimax_h3_tiled_sampler")
    if mod is None:
        return False
    orig_reset = getattr(mod, "peak_reset", None)
    orig_stats = getattr(mod, "peak_stats", None)
    if orig_reset is None or orig_stats is None:
        return False
    if hasattr(orig_reset, _MARKER):
        return True

    @functools.wraps(orig_reset)
    def _reset_xpu():
        if _xpu_only():
            torch.xpu.reset_peak_memory_stats()
            return
        return orig_reset()

    @functools.wraps(orig_stats)
    def _stats_xpu():
        if _xpu_only():
            return {
                "torch_peak_allocated_gib": round(torch.xpu.max_memory_allocated() / 2**30, 3),
                "torch_peak_reserved_gib": round(torch.xpu.max_memory_reserved() / 2**30, 3),
            }
        return orig_stats()

    setattr(_reset_xpu, _MARKER, orig_reset)
    setattr(_stats_xpu, _MARKER, orig_stats)
    mod.peak_reset = _reset_xpu
    mod.peak_stats = _stats_xpu
    return True


def _patch_h3_character_swap() -> bool:
    ctrl_mod = _node_module("H3CharacterSwapController")
    if ctrl_mod is not None:
        pkg_prefix = ctrl_mod.__name__.rsplit(".", 2)[0]
        sampling_mod = sys.modules.get(f"{pkg_prefix}.sampling")
    else:
        sampling_mod = _node_module("", "h3_character_swap.sampling") or next(
            (m for k, m in list(sys.modules.items()) if m is not None and k.endswith(".sampling") and hasattr(m, "stage_handoff")),
            None,
        )
    if sampling_mod is None:
        return False
    orig = getattr(sampling_mod, "stage_handoff", None)
    if orig is None or hasattr(orig, _MARKER):
        return orig is not None

    @functools.wraps(orig)
    def _stage_handoff_xpu(label):
        if not _xpu_only():
            return orig(label)
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
        logging.info(
            "H3_STAGE_HANDOFF %s: inactive models moved to RAM; XPU allocated %.1f MiB",
            label,
            torch.xpu.memory_allocated() / 2**20,
        )

    setattr(_stage_handoff_xpu, _MARKER, orig)
    sampling_mod.stage_handoff = _stage_handoff_xpu
    return True


def _patch_context_loop_selflift() -> bool:
    node_mod = _node_module("MiniMaxH3ChainLoopStart")
    if node_mod is None:
        return False
    pkg_prefix = node_mod.__name__.rsplit(".", 1)[0]
    patched = False

    try:
        mem_mod = sys.modules.get(f"{pkg_prefix}.selflift_runtime.memory") or importlib.import_module(
            f"{pkg_prefix}.selflift_runtime.memory"
        )
    except Exception:
        mem_mod = None
    if mem_mod is not None and hasattr(mem_mod, "release_stage_models"):
        orig_release = mem_mod.release_stage_models
        if hasattr(orig_release, _MARKER):
            patched = True
        else:
            @functools.wraps(orig_release)
            def _release_xpu(patchers, *, keep_models=(), stage):
                if _xpu_only():
                    torch.xpu.synchronize()
                return orig_release(patchers, keep_models=keep_models, stage=stage)

            setattr(_release_xpu, _MARKER, orig_release)
            mem_mod.release_stage_models = _release_xpu
            patched = True

    try:
        diag_mod = sys.modules.get(f"{pkg_prefix}.selflift_runtime.diagnostics") or importlib.import_module(
            f"{pkg_prefix}.selflift_runtime.diagnostics"
        )
    except Exception:
        diag_mod = None
    if diag_mod is not None and hasattr(diag_mod, "log_memory"):
        orig_log = diag_mod.log_memory
        if hasattr(orig_log, _MARKER):
            patched = True
        else:
            @functools.wraps(orig_log)
            def _log_xpu(label, device, *, force=False):
                dev = torch.device(device)
                if dev.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_initialized():
                    import os
                    if not force and os.environ.get("SELFLIFT_MEMORY_LOG", "0") != "1":
                        return
                    free, total = torch.xpu.mem_get_info(dev)
                    logging.info(
                        "[SelfLift memory] %s device=%s; allocated=%.2f MiB; reserved=%.2f MiB; device_used=%.2f MiB; device_free=%.2f MiB",
                        label,
                        dev,
                        torch.xpu.memory_allocated(dev) / 2**20,
                        torch.xpu.memory_reserved(dev) / 2**20,
                        (total - free) / 2**20,
                        free / 2**20,
                    )
                    return
                return orig_log(label, device, force=force)

            setattr(_log_xpu, _MARKER, orig_log)
            diag_mod.log_memory = _log_xpu
            patched = True

    return patched


def patch_loaded_custom_nodes() -> list[str]:
    if not _xpu_only():
        return []
    applied = []
    for name, fn in (
        ("Ref2VA-VSA", _patch_ref2va_vsa),
        ("InstantID-DirectML", _patch_instantid),
        ("Fill-ChatterBox-XPU", _patch_chatterbox),
        ("DenoTiled-XPU", _patch_deno_tiled),
        ("H3CharacterSwap-XPU", _patch_h3_character_swap),
        ("H3ContextLoop-XPU", _patch_context_loop_selflift),
    ):
        try:
            if fn():
                applied.append(name)
        except Exception as exc:
            log.warning("[OmniXPU] third-party adapter %s skipped: %s", name, exc)
    if applied:
        log.info("[OmniXPU] third-party custom node XPU adapters active: %s", ", ".join(applied))
    return applied


def apply():
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        return False, "XPU runtime unavailable"

    patch_loaded_custom_nodes()

    # Custom nodes load in filesystem order; hook init_external_custom_nodes so
    # packs loaded after ComfyUI-Intel-Arc-XPU are adapted right after import.
    nodes = sys.modules.get("nodes")
    if nodes is not None and hasattr(nodes, "init_external_custom_nodes"):
        orig_init = nodes.init_external_custom_nodes
        if not hasattr(orig_init, _HOOK_MARKER):
            @functools.wraps(orig_init)
            async def _wrapped_init(*args, **kwargs):
                try:
                    return await orig_init(*args, **kwargs)
                finally:
                    patch_loaded_custom_nodes()

            setattr(_wrapped_init, _HOOK_MARKER, orig_init)
            nodes.init_external_custom_nodes = _wrapped_init

    return True, ""


__all__ = ["apply", "patch_loaded_custom_nodes"]
