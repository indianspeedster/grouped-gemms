# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Basic correctness sanity: MXFP8 grouped GEMM vs bf16 reference on a small
# Llama4-like shape. Tolerance is loose (MXFP8 is lossy by design); the goal
# is to catch indexing / layout regressions, not numerical fidelity.

import torch

from rocm_mxfp8_mm import triton_mxfp8_grouped_mm
from utils import generate_jagged_offs, is_MI350, to_mx


def reference_grouped_mm_bf16(
    A: torch.Tensor, B_t: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    """bf16 grouped mm: out[s:e, :] = A[s:e] @ B_t[g] for g in range(E)."""
    E = offs.shape[0]
    Mg, K = A.shape
    _, _, N = B_t.shape
    out = torch.empty((Mg, N), dtype=torch.bfloat16, device=A.device)
    start = 0
    for g in range(E):
        end = int(offs[g].item())
        if end > start:
            out[start:end] = A[start:end].to(torch.float32) @ B_t[g].to(torch.float32)
        start = end
    return out


def dequantize_mxfp8(
    data_fp8: torch.Tensor, scales_u8: torch.Tensor, block_size: int = 32
) -> torch.Tensor:
    """Inverse of ``to_mx``: fp8 data * e8m0 scale, broadcast across block."""
    scale_f32 = torch.exp2(scales_u8.to(torch.float32) - 127)
    shape = list(data_fp8.shape)
    shape[-1] = data_fp8.shape[-1] // block_size
    scale_f32 = scale_f32.reshape(*shape).unsqueeze(-1)
    data_blocked = data_fp8.to(torch.float32).reshape(
        *shape, block_size
    )
    return (data_blocked * scale_f32).reshape(data_fp8.shape)


def test_grouped_mm_shape(E: int, M: int, N: int, K: int, block_size: int = 32):
    print(f"\n=== E={E}, M={M}, N={N}, K={K} ===")
    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    B_t = torch.randn((E, N, K), dtype=torch.bfloat16, device="cuda").transpose(-2, -1)

    # MXFP8 path
    A_scales, A_fp8 = to_mx(A, elem_dtype=torch.float8_e4m3fn, block_size=block_size)
    B_nkK = B_t.transpose(-2, -1).contiguous()
    B_scales, B_fp8 = to_mx(B_nkK, elem_dtype=torch.float8_e4m3fn, block_size=block_size)

    offs_mxfp8 = generate_jagged_offs(E, M, multiple_of=block_size)
    out_mxfp8 = triton_mxfp8_grouped_mm(A_fp8, B_fp8, A_scales, B_scales, offs_mxfp8)

    # Reference: bf16 on dequantized inputs (so we compare like-for-like)
    A_deq = dequantize_mxfp8(A_fp8, A_scales, block_size).to(torch.bfloat16)
    B_deq = dequantize_mxfp8(B_fp8, B_scales, block_size).to(torch.bfloat16)
    B_t_deq = B_deq.transpose(-2, -1)
    out_ref = reference_grouped_mm_bf16(A_deq, B_t_deq, offs_mxfp8)

    diff = (out_mxfp8.to(torch.float32) - out_ref.to(torch.float32)).abs()
    rel = diff / out_ref.abs().clamp(min=1e-3).to(torch.float32)
    print(f"max abs diff: {diff.max().item():.3e}")
    print(f"max rel diff: {rel.max().item():.3e}")
    print(f"mean rel diff: {rel.mean().item():.3e}")

    # MXFP8 dequantizes losslessly for well-conditioned blocks; the MM itself
    # introduces fp32 accumulator rounding. Budget is generous — we key on
    # mean rel diff (structural correctness); max abs diff grows with K and
    # is an fp32-rounding signal, not a bug signal.
    assert rel.mean().item() < 0.01, "mean rel diff too large"
    assert diff.max().item() < 4.0, "abs diff unreasonably large"
    print("PASS")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    if not is_MI350():
        print("WARNING: not on MI350+ (gfx950). Kernel will not run.")
        raise SystemExit(1)

    # A handful of small shapes — full 36-shape sweep lives in bench.py.
    test_grouped_mm_shape(E=1, M=256, N=2048, K=2048)
    test_grouped_mm_shape(E=4, M=1024, N=2048, K=2048)
    test_grouped_mm_shape(E=8, M=2048, N=5120, K=2048)
    print("\nAll tests passed.")
