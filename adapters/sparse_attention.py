"""Let upstream block-sparse eligibility recognize Kitchen's native XPU API."""
from __future__ import annotations

import functools
import inspect
import sys
from pathlib import Path

_MARKER = "__omnixpu_sparse_eligibility_original__"


def _make_ineligible_wrapper(original, upstream, ck):
    @functools.wraps(original)
    def _ineligible(q, k, v, dim_head):
        if q.device.type != "xpu":
            return original(q, k, v, dim_head)
        head_dim = upstream.HEAD_DIM
        if not ck.sol_attn_is_available(q.device):
            return "no compiled sol_attn kernel for this GPU"
        import torch

        if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return f"dtype {q.dtype} (kernel takes bf16/fp16)"
        if dim_head != head_dim:
            return f"head_dim {dim_head} != {head_dim}"
        if q.shape != k.shape or q.shape != v.shape:
            return "cross-attention or GQA (kept dense)"
        if k.dtype != q.dtype or v.dtype != q.dtype:
            return f"mixed dtypes {q.dtype}/{k.dtype}/{v.dtype}"
        return None

    setattr(_ineligible, _MARKER, original)
    return _ineligible


def _make_h3_eligible_wrapper(original, upstream, ck):
    @functools.wraps(original)
    def h3_eligible(attn, x, rope_freqs, transformer_options, patch, block_index):
        if x.device.type != "xpu":
            return original(attn, x, rope_freqs, transformer_options, patch, block_index)
        import torch

        head_dim = upstream.HEAD_DIM
        n_tokens = x.shape[0]
        if rope_freqs is None or x.dtype != torch.bfloat16 or attn.head_dim != head_dim:
            return False
        reason = patch.dense_reason(transformer_options, n_tokens, block_index)
        if reason is None and not ck.sol_attn_is_available(x.device):
            reason = "no compiled sol_attn kernel for this GPU"
        if reason is not None:
            patch.log_once(("dense", n_tokens, reason), f"dense ({n_tokens} tokens): {reason}")
            return False
        if patch.vsa:
            layout = transformer_options.get("minimax_h3_layout")
            if layout is None or layout.seq_len != n_tokens:
                patch.log_once("no_layout", "no H3 layout for this call; running dense")
                return False
        return True

    setattr(h3_eligible, _MARKER, original)
    return h3_eligible


def _upstream_module():
    # ComfyUI's file loader registers this built-in as nodes_sparse_attention,
    # whereas a normal package import creates a separate module object.
    # Follow the registered node class so its closures use the adapted guards.
    nodes = sys.modules.get("nodes")
    node = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get("BlockSparseAttention")
    if node is not None:
        module = sys.modules.get(node.__module__)
        if module is None or getattr(module, "BlockSparseAttention", None) is not node:
            raise ValueError("registered sparse node has no owning module")
        expected = Path(nodes.__file__).resolve().parent / "comfy_extras/nodes_sparse_attention.py"
        if Path(module.__file__).resolve() != expected:
            raise ValueError("registered sparse node is not the upstream built-in")
        return module
    from comfy_extras import nodes_sparse_attention
    return nodes_sparse_attention


def apply():
    try:
        import comfy_kitchen as ck
        if not ck.sol_attn_is_available():
            return False, "complete native XPU Sol API is unavailable"
        upstream = _upstream_module()
        functions = (upstream._ineligible, upstream.h3_eligible)
        if all(hasattr(f, _MARKER) for f in functions):
            return True, "already patched"
        if any(hasattr(f, _MARKER) for f in functions):
            return False, "upstream sparse eligibility is partially patched"
        if tuple(inspect.signature(functions[0]).parameters) != ("q", "k", "v", "dim_head"):
            raise ValueError("unsupported _ineligible signature")
        if tuple(inspect.signature(functions[1]).parameters) != (
            "attn", "x", "rope_freqs", "transformer_options", "patch", "block_index"
        ):
            raise ValueError("unsupported h3_eligible signature")
        generic = _make_ineligible_wrapper(functions[0], upstream, ck)
        h3 = _make_h3_eligible_wrapper(functions[1], upstream, ck)
    except (ImportError, AttributeError, OSError, TypeError, ValueError) as exc:
        return False, f"upstream sparse eligibility is unsupported: {exc}"
    upstream._ineligible, upstream.h3_eligible = generic, h3
    return True, ""


__all__ = ["apply"]
