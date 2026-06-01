##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################

"""MX quantization and jagged-offset helpers for benchmarking and testing the
grouped-GEMM kernels (zero external quantization dependency)."""

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
    ``multiple_of``. Last value is always M.
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


# MXFP8 quantization (FLOOR scaling) ---------------------------------------
#   scale_e8m0_unbiased = floor(log2(max_abs)) - F8E4M3_MAX_POW2
#   stored_u8           = scale_e8m0_unbiased + 127, clamped to [0, 254]
# F8E4M3_MAX_POW2 = 8 since floor(log2(448)) = 8; fp8_e4m3fn max = 448.0.

_FP8_E4M3_MAX = 448.0
_F8E4M3_MAX_POW2 = 8


def to_mx(
    data: torch.Tensor, elem_dtype=torch.float8_e4m3fn, block_size: int = 32
):
    """Quantize ``data`` to MXFP8 (float8_e4m3fn + per-block e8m0 scales)
    along the last dim. Returns ``(scales_e8m0_as_uint8, data_fp8)``. The
    power-of-2 exponent is read directly from the fp32 bit pattern.
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


# MXFP4 quantization (FLOOR scaling) ---------------------------------------
# OCP MX FP4 = e2m1 elements (4 bits: sign/exp2/mant1) + per-32-block e8m0
# scales. element max = 6.0, floor(log2(6)) = 2, so the exponent is
# floor(log2(max_abs)) - 2. e2m1 has 8 magnitudes per sign; values are rounded
# to the nearest via the midpoint thresholds below. Packing follows
# tl.dot_scaled: two fp4 codes per uint8, even-K element in the low nibble.

_FP4_E2M1_MAX = 6.0
_F4E2M1_MAX_POW2 = 2

# Positive e2m1 magnitudes indexed by 3-bit code 0..7.
_E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
# Round-to-nearest thresholds = midpoints between consecutive magnitudes.
_E2M1_THRESH = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]


def to_mx_fp4(data: torch.Tensor, block_size: int = 32):
    """Quantize ``data`` to MXFP4 (packed e2m1 + per-block e8m0 scales) along
    the last dim. Returns ``(scales_e8m0_as_uint8, data_fp4_packed_u8)`` to
    mirror ``to_mx``'s ``(scales, data)`` ordering.

    ``data`` last dim must be divisible by ``block_size`` (32) AND even (it is,
    since 32 is even). Output packed tensor has last dim ``K // 2`` uint8, with
    the even-K element in the low nibble — exactly what ``tl.dot_scaled``'s
    ``e2m1`` format and ``lhs_k_pack=True`` expect.
    """
    assert data.shape[-1] % block_size == 0, (
        f"last dim {data.shape[-1]} must be divisible by block_size {block_size}"
    )
    orig_shape = data.shape
    K = orig_shape[-1]
    data_blocked = data.reshape(-1, K // block_size, block_size).to(torch.float32)

    max_abs = data_blocked.abs().amax(dim=-1, keepdim=True)
    max_abs = max_abs.clamp(min=torch.finfo(torch.float32).tiny)

    max_abs_int = max_abs.view(torch.int32)
    extracted_pow2 = ((max_abs_int >> 23) & 0xFF) - 127
    scale_e8m0_unbiased = extracted_pow2 - _F4E2M1_MAX_POW2
    scale_u8 = (scale_e8m0_unbiased + 127).clamp(0, 254).to(torch.uint8)

    scale_f32 = torch.exp2(scale_u8.to(torch.float32) - 127)
    scaled = (data_blocked / scale_f32).clamp(-_FP4_E2M1_MAX, _FP4_E2M1_MAX)

    sign = (scaled < 0).to(torch.uint8)
    thresh = torch.tensor(_E2M1_THRESH, dtype=torch.float32, device=data.device)
    code = torch.bucketize(scaled.abs(), thresh).to(torch.uint8)  # 0..7
    code = (code | (sign << 3)).reshape(orig_shape)  # 4-bit e2m1 code

    # Pack two codes per byte: even-K element -> low nibble.
    code2 = code.reshape(*orig_shape[:-1], K // 2, 2)
    packed = (code2[..., 0] | (code2[..., 1] << 4)).contiguous()

    scale_shape = list(orig_shape)
    scale_shape[-1] = K // block_size
    return scale_u8.reshape(scale_shape), packed


def mxfp4_dequant(packed: torch.Tensor, scale_u8: torch.Tensor, block_size: int = 32):
    """Inverse of ``to_mx_fp4`` to fp32 — for building bf16 references in tests."""
    *lead, Khalf = packed.shape
    K = Khalf * 2
    lo = (packed & 0x0F).to(torch.long)
    hi = (packed >> 4).to(torch.long)
    code = torch.stack((lo, hi), dim=-1).reshape(*lead, K)
    sign = (code >> 3) & 1
    mag = torch.tensor(_E2M1_MAG, dtype=torch.float32, device=packed.device)[code & 7]
    val = mag * torch.where(sign.bool(), -1.0, 1.0)
    val = val.reshape(-1, K // block_size, block_size)
    scale_f32 = torch.exp2(
        scale_u8.reshape(-1, K // block_size, 1).to(torch.float32) - 127
    )
    return (val * scale_f32).reshape(*lead, K)


def is_MI350() -> bool:
    if getattr(torch.version, "hip", None) is None:
        return False
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False
    return "gfx950" in arch
