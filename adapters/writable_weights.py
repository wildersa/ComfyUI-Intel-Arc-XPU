"""Bridge ComfyUI writable backing and its stream boundary to XPU."""
from __future__ import annotations

import ctypes
import dataclasses
import functools
import inspect
import textwrap

import torch

_MARKER = "__omnixpu_writable_weight_original__"


def _current_stream(device):
    return (torch.xpu.current_stream(device) if device.type == "xpu"
            else torch.cuda.current_stream(device))


def _stream_pointer(stream, device):
    if stream is None:
        return 0
    return int(stream.sycl_queue if device is not None and device.type == "xpu"
               else stream.cuda_stream)


def _check_contract(function, expected_fragments):
    if function.__closure__:
        raise ValueError(f"{function.__name__} unexpectedly captures closure state")
    source = textwrap.dedent(inspect.getsource(function))
    for fragment, count in expected_fragments:
        if source.count(fragment) != count:
            raise ValueError(f"unsupported {function.__name__} writable contract")


def apply():
    import comfy.memory_management as memory
    import comfy.model_management as management
    import comfy.ops as ops
    import comfy.pinned_memory as pinned_memory
    from comfy.quant_ops import QuantizedTensor
    from comfy_aimdo import host_buffer

    if management.get_torch_device().type != "xpu":
        return False, "writable weight adapter requires XPU"
    if not hasattr(ops, "WritableWeight"):
        return False, "ComfyUI writable weights are unavailable"
    originals = (memory.write_tensor_file_slice, memory.read_tensor_file_slice_into,
                 ops.WritableWeight.writeback)
    if all(hasattr(f, _MARKER) for f in originals):
        return True, "already patched"
    if any(hasattr(f, _MARKER) for f in originals):
        return False, "writable weight adapter is partially applied"
    if not callable(getattr(host_buffer, "write_device_to_file", None)):
        return False, "AIMDO device-to-file writeback is unavailable"
    write, read, writeback = originals
    try:
        if tuple(inspect.signature(write).parameters) != ("tensor", "source", "stream"):
            raise ValueError("unsupported write_tensor_file_slice signature")
        if tuple(inspect.signature(read).parameters) != ("tensor", "destination", "stream", "destination2"):
            raise ValueError("unsupported read_tensor_file_slice_into signature")
        if tuple(inspect.signature(writeback).parameters) != ("self", "value"):
            raise ValueError("unsupported WritableWeight.writeback signature")
        _check_contract(write, (
            ('if source.device.type == "cuda":', 1),
            ("torch.cuda.current_stream(source.device)", 1),
            ("stream.cuda_stream,", 1),
            ("CPU or CUDA source", 1),
        ))
        _check_contract(read, (
            ('getattr(stream, "cuda_stream", 0) if stream is not None else 0', 2),
        ))
        _check_contract(writeback, (
            ("torch.cuda.current_stream(value.device).synchronize()", 1),
        ))
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        return False, str(exc)

    @functools.wraps(write)
    def write_file(tensor, source, stream=None):
        if source.device.type != "xpu":
            return write(tensor, source, stream=stream)
        info = getattr(tensor.untyped_storage(), "_comfy_tensor_file_slice", None)
        size = source.numel() * source.element_size()
        if (info is None or tensor.storage_offset() != 0 or not tensor.is_contiguous()
                or tensor.numel() * tensor.element_size() != info.size or not source.is_contiguous()
                or size != info.size):
            raise ValueError("Writable tensors require an exact file descriptor and a complete contiguous source")
        with info.lock:
            active_stream = _current_stream(source.device) if stream is None else stream
            host_buffer.write_device_to_file(
                info.file_ref,
                info.offset,
                size,
                _stream_pointer(active_stream, source.device),
                source.data_ptr(),
                source.device.index,
            )

    @functools.wraps(read)
    def read_file(tensor, destination, stream=None, destination2=None):
        xpu = any(t is not None and t.device.type == "xpu" for t in (destination, destination2))
        if not xpu:
            return read(tensor, destination, stream=stream, destination2=destination2)

        if isinstance(tensor, QuantizedTensor):
            if not read_file(
                tensor._qdata,
                destination._qdata if destination is not None else None,
                stream=stream,
                destination2=(destination2._qdata if destination2 is not None else None),
            ):
                return False
            if destination is not None:
                dst_orig_dtype = destination._params.orig_dtype
                destination._params.copy_from(tensor._params, non_blocking=False)
                destination._params = dataclasses.replace(destination._params, orig_dtype=dst_orig_dtype)
            if destination2 is not None:
                dst_orig_dtype = destination2._params.orig_dtype
                destination2._params.copy_from(
                    destination._params if destination is not None else tensor._params,
                    non_blocking=True,
                )
                destination2._params = dataclasses.replace(destination2._params, orig_dtype=dst_orig_dtype)
            return True

        info = getattr(tensor.untyped_storage(), "_comfy_tensor_file_slice", None)
        if info is None:
            return False

        if destination is not None and destination.device.type != "cpu" and destination2 is None:
            destination2 = destination
            destination = None

        file_obj = info.file_ref
        if (file_obj is None
                or (destination is None and destination2 is None)
                or (destination is not None and (destination.device.type != "cpu" or destination.numel() * destination.element_size() < info.size))
                or (destination2 is not None and (destination2.device.type == "cpu" or destination2.numel() * destination2.element_size() < info.size))
                or tensor.numel() * tensor.element_size() != info.size
                or tensor.storage_offset() != 0
                or not tensor.is_contiguous()):
            return False

        if info.size == 0:
            return True

        dev = destination2.device if destination2 is not None else None
        stream_ptr = _stream_pointer(stream, dev)

        if destination is None:
            host_buffer.read_file_to_device(
                file_obj, info.offset, info.size, stream_ptr,
                destination2.data_ptr(), destination2.device.index, mark_cold=False,
            )
            return True

        hostbuf = getattr(destination.untyped_storage(), "_comfy_hostbuf", None)
        if hostbuf is not None:
            device_ptr = destination2.data_ptr() if destination2 is not None else 0
            with info.lock:
                hostbuf.read_file_slice(
                    file_obj, info.offset, info.size,
                    offset=destination.data_ptr() - hostbuf.get_raw_address(),
                    stream=stream_ptr,
                    device_ptr=device_ptr,
                    device=None if destination2 is None else destination2.device.index,
                )
            return True

        if not hasattr(file_obj, "seek") or not hasattr(file_obj, "readinto"):
            return False

        buf_type = ctypes.c_ubyte * info.size
        view = memoryview(buf_type.from_address(destination.data_ptr()))
        try:
            with info.lock:
                file_obj.seek(info.offset)
                done = 0
                while done < info.size:
                    try:
                        n = file_obj.readinto(view[done:])
                    except OSError:
                        return False
                    if n <= 0:
                        return False
                    done += n
            return True
        finally:
            view.release()

    @functools.wraps(writeback)
    def save(self, value):
        if value.device.type != "xpu":
            return writeback(self, value)
        subset = self._pin_subset
        pin = pinned_memory.get_pin(self, subset) if subset is not None else None
        if pin is None:
            subset = (
                "weights-fast"
                if self._pin_state["fast_disk"]
                else "weights-loaded"
                if self._v_signature is not None
                else "weights"
            )
            pinned_memory.pin_memory(self, subset=subset)
            pin = pinned_memory.get_pin(self, subset)
        if pin is not None:
            dest = memory.interpret_gathered_like([self.weight], pin)[0]
            dest.copy_(value, non_blocking=True)
            _current_stream(value.device).synchronize()
            self._pins[subset]["dirty"] = True
        else:
            memory.write_tensor_file_slice(self.weight, value)

    for replacement, original in zip((write_file, read_file, save), originals):
        setattr(replacement, _MARKER, original)
    memory.write_tensor_file_slice, memory.read_tensor_file_slice_into = write_file, read_file
    ops.WritableWeight.writeback = save
    return True, None
