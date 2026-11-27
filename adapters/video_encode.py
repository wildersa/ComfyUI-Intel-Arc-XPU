"""Idempotent Intel Quick Sync (h264_qsv) video encode adapter for ComfyUI SaveVideo."""

from __future__ import annotations

import io
import logging
import os
import threading

log = logging.getLogger("ComfyUI-OmniXPU")

_PATCHED = False
_PROBE_LOCK = threading.Lock()
_QSV_ENCODER = "h264_qsv"
_QSV_PIX_FMT = "nv12"
_PRESET_ENV = "OMNIXPU_QSV_PRESET"
_VALID_QSV_PRESETS = frozenset(
    {"veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"}
)


def _qsv_preset() -> str:
    val = os.environ.get(_PRESET_ENV, "medium").strip().lower()
    return val if val in _VALID_QSV_PRESETS else "medium"


def _probe_h264_qsv() -> tuple[bool, str]:
    try:
        import av
        import numpy as np
    except Exception as exc:
        return False, f"PyAV/numpy unavailable: {exc}"

    if _QSV_ENCODER not in getattr(av, "codecs_available", ()):
        return False, f"{_QSV_ENCODER} not compiled into PyAV"

    try:
        buf = io.BytesIO()
        with av.open(buf, mode="w", format="mp4") as out:
            stream = out.add_stream(_QSV_ENCODER, rate=24)
            stream.width = 256
            stream.height = 256
            stream.pix_fmt = _QSV_PIX_FMT
            stream.options = {"preset": _qsv_preset(), "global_quality": "18"}
            frame = av.VideoFrame.from_ndarray(
                np.zeros((256, 256, 3), dtype=np.uint8), format="rgb24"
            ).reformat(format=_QSV_PIX_FMT)
            for pkt in stream.encode(frame):
                out.mux(pkt)
            for pkt in stream.encode(None):
                out.mux(pkt)
        if not buf.getvalue():
            return False, f"{_QSV_ENCODER} produced empty probe stream"
        return True, ""
    except Exception as exc:
        return False, f"{_QSV_ENCODER} probe failed: {exc}"


def _supports_qsv_h264(
    codec,
    bit_depth: int | None = 8,
    chroma_subsampling: str | None = None,
    has_alpha: bool = False,
) -> bool:
    if str(getattr(codec, "value", codec)).lower() != "h264":
        return False
    if bit_depth is not None and int(bit_depth) > 8:
        return False
    if has_alpha:
        return False
    if chroma_subsampling is not None and str(chroma_subsampling) != "420":
        return False
    return True


def apply() -> tuple[bool, str]:
    """Patch comfy_api VideoFromComponents/VideoFromFile H.264 encodes to use h264_qsv."""
    global _PATCHED
    if _PATCHED:
        return True, "already patched"

    with _PROBE_LOCK:
        if _PATCHED:
            return True, "already patched"
        ok, reason = _probe_h264_qsv()
        if not ok:
            return False, reason

        try:
            from comfy_api.latest._input_impl import video_types
            from comfy_api.latest._util import VideoCodec
        except Exception as exc:
            return False, f"video_types unavailable: {exc}"

        if getattr(video_types, "_omnixpu_qsv_patched", False):
            _PATCHED = True
            return True, "already patched"

        orig_encoders = dict(video_types.VIDEO_ENCODERS)
        orig_encoder_options = video_types.video_encoder_options
        orig_pixel_format = video_types.video_pixel_format
        orig_components_save = video_types.VideoFromComponents.save_to
        orig_file_save = video_types.VideoFromFile.save_to

        local_state = threading.local()

        def _force_cpu() -> bool:
            return bool(getattr(local_state, "force_cpu", False))

        def _active_h264_encoder() -> str:
            return orig_encoders[VideoCodec.H264] if _force_cpu() else _QSV_ENCODER

        def wrapped_pixel_format(
            codec: VideoCodec,
            bit_depth: int,
            chroma_subsampling: str | None = None,
            has_alpha: bool = False,
        ) -> str:
            if not _force_cpu() and _supports_qsv_h264(
                codec, bit_depth=bit_depth, chroma_subsampling=chroma_subsampling, has_alpha=has_alpha
            ):
                return _QSV_PIX_FMT
            return orig_pixel_format(
                codec, bit_depth, chroma_subsampling=chroma_subsampling, has_alpha=has_alpha
            )

        def wrapped_encoder_options(
            codec: VideoCodec,
            crf: float | None,
            preset: str | None = None,
            pix_fmt: str | None = None,
        ) -> dict[str, str]:
            if not _force_cpu() and codec == VideoCodec.H264 and pix_fmt == _QSV_PIX_FMT:
                quality = 18 if crf is None else int(round(float(crf)))
                return {
                    "preset": preset or _qsv_preset(),
                    "global_quality": str(max(1, min(51, quality))),
                }
            return orig_encoder_options(codec, crf, preset=preset, pix_fmt=pix_fmt)

        def _call_with_qsv_fallback(orig_fn, self, path, *args, **kwargs):
            codec = kwargs.get("codec", VideoCodec.AUTO)
            bit_depth = kwargs.get("bit_depth")
            chroma = kwargs.get("chroma_subsampling")
            if codec not in (VideoCodec.AUTO, VideoCodec.H264) or not _supports_qsv_h264(
                VideoCodec.H264, bit_depth=bit_depth, chroma_subsampling=chroma
            ):
                prev = _force_cpu()
                local_state.force_cpu = True
                prev_enc = video_types.VIDEO_ENCODERS[VideoCodec.H264]
                video_types.VIDEO_ENCODERS[VideoCodec.H264] = orig_encoders[VideoCodec.H264]
                try:
                    return orig_fn(self, path, *args, **kwargs)
                finally:
                    video_types.VIDEO_ENCODERS[VideoCodec.H264] = prev_enc
                    local_state.force_cpu = prev

            prev = _force_cpu()
            local_state.force_cpu = False
            video_types.VIDEO_ENCODERS[VideoCodec.H264] = _active_h264_encoder()
            try:
                return orig_fn(self, path, *args, **kwargs)
            except Exception as exc:
                log.warning(
                    "[OmniXPU] h264_qsv save failed (%s); falling back to CPU h264", exc
                )
                local_state.force_cpu = True
                video_types.VIDEO_ENCODERS[VideoCodec.H264] = orig_encoders[VideoCodec.H264]
                return orig_fn(self, path, *args, **kwargs)
            finally:
                local_state.force_cpu = prev
                video_types.VIDEO_ENCODERS[VideoCodec.H264] = (
                    orig_encoders[VideoCodec.H264] if prev else _QSV_ENCODER
                )

        def wrapped_components_save(self, path, *args, **kwargs):
            return _call_with_qsv_fallback(orig_components_save, self, path, *args, **kwargs)

        def wrapped_file_save(self, path, *args, **kwargs):
            return _call_with_qsv_fallback(orig_file_save, self, path, *args, **kwargs)

        video_types.VIDEO_ENCODERS[VideoCodec.H264] = _QSV_ENCODER
        video_types.video_pixel_format = wrapped_pixel_format
        video_types.video_encoder_options = wrapped_encoder_options
        video_types.VideoFromComponents.save_to = wrapped_components_save
        video_types.VideoFromFile.save_to = wrapped_file_save
        video_types._omnixpu_qsv_patched = True
        _PATCHED = True
        log.info(
            "[OmniXPU] QSV video encode enabled: H264 -> %s (pix_fmt=%s, preset=%s)",
            _QSV_ENCODER,
            _QSV_PIX_FMT,
            _qsv_preset(),
        )
        return True, ""
