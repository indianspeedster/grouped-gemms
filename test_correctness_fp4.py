# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Correctness sanity for the MXFP4 grouped GEMM. MXFP4 (e2m1) is far coarser
# than MXFP8 — only 8 magnitudes per sign — so the SQNR floor vs a bf16
# reference is much lower. This test is report-first: it prints the SQNR for
# every shape and asserts a lenient floor so a real regression (NaN / wrong
# layout / dead kernel) still trips, but expected fp4 quantization noise does
# not. Run it to see the actual numbers, then tighten MIN_SQNR_DB if desired.

import torch

from kernels import triton_mxfp4_grouped_mm
from utils import generate_jagged_offs, is_MI350, mxfp4_dequant, to_mx_fp4


def compute_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    Ps = torch.linalg.vector_norm(x.to(torch.float32))
    Pn = torch.linalg.vector_norm((x - y).to(torch.float32))
    return 20 * torch.log10(Ps / Pn)


# Lenient gate — see module docstring. Real MXFP4 SQNR on random Gaussian
# inputs lands well above this; anything below means the kernel is broken.
MIN_SQNR_DB = 10.0


def test_grouped_mm_shape(E: int, M: int, N: int, K: int, block_size: int = 32):
    print(f"\n=== E={E}, M={M}, N={N}, K={K} ===")
    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    B_nkK_hp = torch.randn((E, N, K), dtype=torch.bfloat16, device="cuda")

    A_scales, A_fp4 = to_mx_fp4(A, block_size=block_size)
    B_scales, B_fp4 = to_mx_fp4(B_nkK_hp, block_size=block_size)

    offs = generate_jagged_offs(E, M, multiple_of=block_size)
    out_mxfp4 = triton_mxfp4_grouped_mm(A_fp4, B_fp4, A_scales, B_scales, offs)

    # Reference: bf16 grouped_mm on the DEQUANTIZED fp4 operands. This isolates
    # the kernel's arithmetic from quantization error — the MXFP4 path should
    # match a faithful dequant-then-bf16-matmul closely. (We also print the
    # SQNR vs the original hp inputs for context.)
    A_deq = mxfp4_dequant(A_fp4, A_scales, block_size).to(torch.bfloat16)
    B_deq = mxfp4_dequant(B_fp4, B_scales, block_size).to(torch.bfloat16)
    B_deq_t = B_deq.transpose(-2, -1)  # (E, K, N)
    out_ref_deq = torch._grouped_mm(
        A_deq, B_deq_t, offs=offs.to(torch.int32), out_dtype=torch.bfloat16,
    )
    out_ref_hp = torch._grouped_mm(
        A, B_nkK_hp.transpose(-2, -1), offs=offs.to(torch.int32),
        out_dtype=torch.bfloat16,
    )

    sqnr_deq = compute_error(out_ref_deq, out_mxfp4).item()
    sqnr_hp = compute_error(out_ref_hp, out_mxfp4).item()
    print(f"SQNR vs dequant-bf16 ref: {sqnr_deq:.2f} dB  (kernel arithmetic)")
    print(f"SQNR vs hp-bf16 ref:      {sqnr_hp:.2f} dB  (incl. fp4 quant noise)")

    assert sqnr_deq >= MIN_SQNR_DB, (
        f"SQNR {sqnr_deq:.2f} dB below lenient floor {MIN_SQNR_DB:.1f} dB — "
        f"kernel arithmetic looks broken, not just fp4 noise."
    )
    print("PASS")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    if not is_MI350():
        print("WARNING: not on MI350+ (gfx950). Kernel will not run.")
        raise SystemExit(1)

    test_grouped_mm_shape(E=1, M=1024, N=1024, K=1024)
    test_grouped_mm_shape(E=8, M=1024, N=4096, K=2048)
    test_grouped_mm_shape(E=4, M=1024, N=2048, K=2048)
    test_grouped_mm_shape(E=8, M=2048, N=5120, K=2048)
    print("\nAll tests passed.")
