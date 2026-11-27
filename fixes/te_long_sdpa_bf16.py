"""Keep long float32 Llama/Qwen3VL text-encoder SDPA on XPU bfloat16 OVERRIDEABLE.

ComfyUI's ``SDClipModel`` runs text encoders in ``torch.float32`` while
``Llama2_`` / ``Qwen35VisionModel`` request ``small_input=True``.  On PyTorch
XPU, ``SDPBackend.OVERRIDEABLE`` (oneDNN tiled SDPA) only supports
``bfloat16``/``float16`` and falls back to quadratic ``MATH`` for ``float32``.
Casting only the attention inputs inside SDPA keeps residuals, norms,
connectors, T5, and short sequences in ``float32`` while avoiding ``O(S^2)``
allocations and unnecessary GQA KV expansion.
"""

from __future__ import annotations

import functools
import logging
import os

import torch

from ..adapters.errors import is_fatal_accelerator_error

log = logging.getLogger("ComfyUI-OmniXPU")

_MARKER = "_omnixpu_te_long_sdpa_bf16_original"
_MIN_SEQ_ENV = "OMNIXPU_TE_LONG_SDPA_MIN_SEQ"
_DEFAULT_MIN_SEQ = 1024
_LOGGED_CALLS = 0
_FALLBACK_LOGGED = False


def _min_seq_len() -> int:
    raw = os.environ.get(_MIN_SEQ_ENV, str(_DEFAULT_MIN_SEQ)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_MIN_SEQ


def _xpu_overrideable_bf16_supported():
    """Return ``(sdpa_kernel, overrideable_backend)`` when XPU BF16 SDPA is ready."""
    xpu = getattr(torch, "xpu", None)
    cuda = getattr(torch, "cuda", None)
    if xpu is None or not xpu.is_available():
        return None, "XPU is unavailable"
    if cuda is not None and cuda.is_available():
        return None, "CUDA is present"
    is_bf16 = getattr(xpu, "is_bf16_supported", None)
    if callable(is_bf16) and not is_bf16():
        return None, "XPU device does not report bfloat16 support"

    attn_ns = getattr(torch.nn, "attention", None)
    sdp_backend = getattr(attn_ns, "SDPBackend", None)
    sdpa_kernel = getattr(attn_ns, "sdpa_kernel", None)
    overrideable = getattr(sdp_backend, "OVERRIDEABLE", None)
    if not callable(sdpa_kernel) or overrideable is None:
        return None, "SDPBackend.OVERRIDEABLE is unavailable"
    return (sdpa_kernel, overrideable), None


def _wrap_pytorch_attention(base_fn, attn_mod, sdpa_kernel, overrideable_backend, min_seq):
    """Wrap ``attention_pytorch`` to run long XPU float32 SDPA in bfloat16."""
    batch_limit = getattr(attn_mod, "SDP_BATCH_LIMIT", 2**31)

    @attn_mod.wrap_attn
    def attention_te_long_bf16(
        q,
        k,
        v,
        heads,
        mask=None,
        attn_precision=None,
        skip_reshape=False,
        skip_output_reshape=False,
        **kwargs,
    ):
        global _LOGGED_CALLS, _FALLBACK_LOGGED

        compiler = getattr(torch, "compiler", None)
        is_compiling = bool(compiler is not None and compiler.is_compiling())
        if (
            is_compiling
            or q.device.type != "xpu"
            or k.device != q.device
            or v.device != q.device
            or q.dtype != torch.float32
            or k.dtype != torch.float32
            or v.dtype != torch.float32
            or (
                mask is not None
                and (
                    mask.device != q.device
                    or (mask.dtype != torch.bool and not mask.is_floating_point())
                )
            )
        ):
            return base_fn(
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )

        if skip_reshape:
            b, _, q_len, dim_head = q.shape
            kv_len = k.shape[2]
        else:
            b, q_len, inner_dim = q.shape
            kv_len = k.shape[1]
            dim_head = inner_dim // heads

        if q_len < min_seq or kv_len < min_seq or b > batch_limit:
            return base_fn(
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )

        try:
            q_in, k_in, v_in = q, k, v
            if not skip_reshape:
                q_in, k_in, v_in = attn_mod._reshape_qkv_to_heads(
                    q_in,
                    k_in,
                    v_in,
                    b,
                    heads,
                    dim_head,
                    kwargs.get("enable_gqa", False),
                    expand_kv=False,
                )
                q_in, k_in, v_in = (t.transpose(1, 2) for t in (q_in, k_in, v_in))

            mask_in = mask
            if mask_in is not None:
                if mask_in.ndim == 2:
                    mask_in = mask_in.unsqueeze(0)
                if mask_in.ndim == 3:
                    mask_in = mask_in.unsqueeze(1)
                if mask_in.is_floating_point():
                    mask_in = mask_in.to(torch.bfloat16)

            sdpa_extra = {
                key: val for key, val in kwargs.items() if key in ("scale", "enable_gqa")
            }

            with sdpa_kernel([overrideable_backend]):
                out = torch.nn.functional.scaled_dot_product_attention(
                    q_in.to(torch.bfloat16),
                    k_in.to(torch.bfloat16),
                    v_in.to(torch.bfloat16),
                    attn_mask=mask_in,
                    dropout_p=0.0,
                    is_causal=False,
                    **sdpa_extra,
                )

            if not skip_output_reshape:
                out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            out = out.to(q.dtype)

            if _LOGGED_CALLS < 3:
                _LOGGED_CALLS += 1
                log.info(
                    "[OmniXPU] te_long_sdpa_bf16 #%d: heads=%d kv_heads=%d q=%d kv=%d (fp32->bf16 OVERRIDEABLE)",
                    _LOGGED_CALLS,
                    heads,
                    k_in.shape[1],
                    q_len,
                    kv_len,
                )
            return out
        except Exception as error:
            if is_fatal_accelerator_error(error):
                raise
            if not _FALLBACK_LOGGED:
                _FALLBACK_LOGGED = True
                log.warning(
                    "[OmniXPU] te_long_sdpa_bf16 failed for q=%d kv=%d (%s); falling back to stock SDPA",
                    q_len,
                    kv_len,
                    error,
                )
            return base_fn(
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )

    return attention_te_long_bf16


def patch_text_encoder_module(target_mod, attn_mod, sdpa_kernel, overrideable_backend, min_seq):
    """Wrap ``target_mod.optimized_attention_for_device`` idempotently."""
    orig_selector = getattr(target_mod, "optimized_attention_for_device", None)
    if not callable(orig_selector):
        return False
    if hasattr(orig_selector, _MARKER):
        return True

    base_fn = getattr(attn_mod, "attention_pytorch", None)
    if base_fn is None:
        return False
    wrapped_attention = _wrap_pytorch_attention(
        base_fn, attn_mod, sdpa_kernel, overrideable_backend, min_seq
    )

    @functools.wraps(orig_selector)
    def select_attention(device, mask=False, small_input=False):
        selected = orig_selector(device, mask=mask, small_input=small_input)
        if selected is base_fn:
            return wrapped_attention
        return selected

    setattr(select_attention, _MARKER, orig_selector)
    target_mod.optimized_attention_for_device = select_attention
    return True


def apply():
    capability, reason = _xpu_overrideable_bf16_supported()
    if capability is None:
        return False, reason
    sdpa_kernel, overrideable_backend = capability

    try:
        import comfy.ldm.modules.attention as attn_mod
        from comfy.text_encoders import llama as llama_mod
    except ImportError as exc:
        return False, f"ComfyUI text-encoder attention unavailable: {exc}"

    min_seq = _min_seq_len()
    patched_any = patch_text_encoder_module(
        llama_mod, attn_mod, sdpa_kernel, overrideable_backend, min_seq
    )
    try:
        from comfy.text_encoders import qwen35 as qwen35_mod
    except ImportError:
        qwen35_mod = None
    if qwen35_mod is not None:
        patched_any = (
            patch_text_encoder_module(
                qwen35_mod, attn_mod, sdpa_kernel, overrideable_backend, min_seq
            )
            or patched_any
        )

    if not patched_any:
        return False, "optimized_attention_for_device hook target not found"
    return True, None
