"""Bound SeedVR2 Ada modulation and apply its explicit embedding reshape.

The patch is intentionally source guarded. It changes only known
upstream contracts and declines to patch an unknown implementation.
"""

import functools
import inspect
import textwrap

import torch

_LEGACY_RESHAPE = "emb.reshape(emb.shape[0], -1, len(self.layers), 3)"
_EXPLICIT_RESHAPE = "emb.reshape(emb.shape[0], self.dim, -1, 3)"
_PATCH_MARKER = "_omnixpu_seedvr_ada_reshape_patched"


def apply():
    try:
        from comfy.ldm.seedvr import model as seedvr_model
    except ModuleNotFoundError as exc:
        if exc.name in {"comfy.ldm.seedvr", "comfy.ldm.seedvr.model"}:
            return False, "ComfyUI SeedVR2 model is not available"
        raise

    ada_single = getattr(seedvr_model, "AdaSingle", None)
    if ada_single is None or not hasattr(ada_single, "forward"):
        return False, "ComfyUI SeedVR2 AdaSingle is not available"
    if getattr(ada_single, _PATCH_MARKER, False):
        return False, "SeedVR2 Ada reshape patch is already applied"

    forward = ada_single.forward
    if not hasattr(ada_single, "_modulate"):
        return False, "SeedVR2 AdaSingle._modulate is unavailable"
    if tuple(inspect.signature(forward).parameters) != (
        "self", "hid", "emb", "layer", "mode", "cache", "branch_tag", "hid_len", "norm_eps"
    ):
        return False, "unsupported AdaSingle.forward signature"

    try:
        source = textwrap.dedent(inspect.getsource(forward))
    except (OSError, TypeError) as exc:
        return False, f"AdaSingle.forward source unavailable: {exc}"

    if source.count(_LEGACY_RESHAPE) != 1:
        if _EXPLICIT_RESHAPE in source:
            return False, "upstream Ada reshape and broadcast are already bounded"
        return False, "unsupported AdaSingle.forward reshape contract"

    cache_cls = getattr(seedvr_model, "Cache", None)
    expand_dims = getattr(seedvr_model, "expand_dims", None)
    if not callable(cache_cls) or not callable(expand_dims):
        return False, "SeedVR2 Ada helpers are unavailable"

    @functools.wraps(forward)
    def patched(
        self,
        hid: torch.FloatTensor,
        emb: torch.FloatTensor,
        layer: str,
        mode: str,
        cache=None,
        branch_tag: str = "",
        hid_len=None,
        norm_eps=None,
    ) -> torch.FloatTensor:
        if cache is None:
            cache = cache_cls(disable=True)
        idx = self.layers.index(layer)

        def sliced():
            e = emb.reshape(emb.shape[0], self.dim, -1, 3)[:, :, idx, :]
            return expand_dims(e, 1, hid.ndim + 1)

        if hid_len is None:
            mod = sliced()
        else:
            mod = cache(f"emb_repeat_{idx}_{branch_tag}", sliced)

        shiftA, scaleA, gateA = mod.unbind(-1)
        if hid_len is None or mod.shape[0] == 1:
            return self._modulate(hid, layer, mode, shiftA, scaleA, gateA, norm_eps=norm_eps)

        out = torch.empty_like(hid) if norm_eps is not None else hid
        offset = 0
        for i, length in enumerate(cache(f"hid_len_list_{branch_tag}", hid_len.tolist)):
            modulated = self._modulate(
                hid[offset:offset + length],
                layer,
                mode,
                shiftA[i],
                scaleA[i],
                gateA[i],
                norm_eps=norm_eps,
            )
            if norm_eps is not None:
                out[offset:offset + length] = modulated
            offset += length
        return out

    setattr(patched, _PATCH_MARKER, forward)
    ada_single.forward = patched
    setattr(ada_single, _PATCH_MARKER, True)
    return True, None
