# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Small helpers for benchmarking/testing the MXFP8 grouped-GEMM kernel.

Kept intentionally minimal — for production use the equivalents in torchao
(``torchao.prototype.mx_formats.mx_tensor.to_mx``,
``torchao.prototype.moe_training.utils.generate_jagged_offs``) are more
robust; this module trades fidelity for a zero-torchao-dep benchmark.
"""

import random
from typing import Callable

import torch

try:
    import triton  # noqa: F401
    from triton.testing import do_bench
    _has_triton = True
except ImportError:
    _has_triton = False


def benchmark_cuda_function_in_microseconds(fn: Callable, *args, **kwargs) -> float:
    """Median runtime of ``fn(*args, **kwargs)`` in microseconds."""
    if not _has_triton:
        raise RuntimeError("triton is required for benchmarking")
    return do_bench(lambda: fn(*args, **kwargs), return_mode="median") * 1e3


def generate_jagged_offs(
    E: int, M: int, multiple_of: int = 32, dtype=torch.int32, device="cuda"
) -> torch.Tensor:
    """Random sorted cumulative offsets summing to M, each a multiple of
    ``multiple_of``. Last value is always M. Matches torchao's
    ``generate_jagged_offs`` semantics.
    """
    if M % multiple_of != 0:
        raise ValueError(f"M must be divisible by {multiple_of}")
    possible_values = list(range(multiple_of, M + 1, multiple_of))
    if E > len(possible_values):
        raise ValueError("E cannot be larger than the number of possible values")
    selected = torch.tensor(random.sample(possible_values[:-1], E - 1))
    selected = torch.cat((selected, torch.tensor([M])))
    selected, _ = torch.sort(selected)
    return selected.to(dtype).to(device)


# MXFP8 quantization -------------------------------------------------------
#
# Matches the FLOOR mode of torchao.prototype.mx_formats.mx_tensor.to_mx:
#   scale_e8m0_unbiased = floor(log2(max_abs)) - F8E4M3_MAX_POW2
#   stored_u8 = scale_e8m0_unbiased + 127 (e8m0 bias), clamped to [0, 254]
#
# F8E4M3_MAX_POW2 = 8 because floor(log2(448)) = 8 (2^8 = 256 <= 448 < 512).
# e8m0 byte 0 is reserved as NaN; valid range 1..254 (stored as uint8).
# fp8_e4m3fn max representable = 448.0.

_FP8_E4M3_MAX = 448.0
_F8E4M3_MAX_POW2 = 8


def to_mx(
    data: torch.Tensor, elem_dtype=torch.float8_e4m3fn, block_size: int = 32
):
    """Quantize ``data`` to MXFP8 (float8_e4m3fn + per-block e8m0 scales)
    along the last dim. Returns ``(scales_e8m0_as_uint8, data_fp8)``.

    Mirrors torchao's FLOOR scaling mode (the default in
    ``torchao.prototype.mx_formats.mx_tensor.to_mx``). Extracts the
    power-of-2 exponent directly from the fp32 bit pattern — same formula
    ao uses.
    """
    assert elem_dtype is torch.float8_e4m3fn, "only e4m3fn supported here"
    assert data.shape[-1] % block_size == 0, (
        f"last dim {data.shape[-1]} must be divisible by block_size {block_size}"
    )

    orig_shape = data.shape
    data_blocked = data.reshape(-1, data.shape[-1] // block_size, block_size)

    max_abs = data_blocked.abs().amax(dim=-1, keepdim=True).to(torch.float32)
    # Avoid log2(0); keep in normal-range fp32 for clean exponent extraction.
    max_abs = max_abs.clamp(min=torch.finfo(torch.float32).tiny)

    # floor(log2(max_abs)) via fp32 bit pattern (bits 30..23 = biased exp, bias 127).
    max_abs_int = max_abs.view(torch.int32)
    extracted_pow2 = ((max_abs_int >> 23) & 0xFF) - 127
    scale_e8m0_unbiased = extracted_pow2 - _F8E4M3_MAX_POW2

    scale_u8 = (scale_e8m0_unbiased + 127).clamp(0, 254).to(torch.uint8)

    scale_f32 = torch.exp2(scale_u8.to(torch.float32) - 127)
    scaled = (data_blocked.to(torch.float32) / scale_f32).clamp(
        -_FP8_E4M3_MAX, _FP8_E4M3_MAX
    )
    data_fp8 = scaled.to(elem_dtype).reshape(orig_shape)

    scale_shape = list(orig_shape)
    scale_shape[-1] = orig_shape[-1] // block_size
    return scale_u8.reshape(scale_shape), data_fp8


def is_MI350() -> bool:
    if getattr(torch.version, "hip", None) is None:
        return False
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False
    return "gfx950" in arch
