# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Correctness sanity: MXFP8 grouped GEMM vs a bf16 reference (torch._grouped_mm
# on the original hp inputs). Checks SQNR against the same threshold used in
# torchao's test_mxfp8_grouped_mm.py:
#     min_sqnr = 27.0 dB  (forward output)

import torch

from kernels import triton_mxfp8_grouped_mm
from utils import generate_jagged_offs, is_MI350, to_mx


# Matches torchao.float8.float8_utils.compute_error — SQNR in dB.
# Ps = ||signal||,  Pn = ||signal - approx||,  SQNR = 20 * log10(Ps / Pn).
def compute_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    Ps = torch.linalg.vector_norm(x.to(torch.float32))
    Pn = torch.linalg.vector_norm((x - y).to(torch.float32))
    return 20 * torch.log10(Ps / Pn)


MIN_SQNR_DB = 27.0  # same threshold as ao test_mxfp8_grouped_mm.py forward tests


def test_grouped_mm_shape(E: int, M: int, N: int, K: int, block_size: int = 32):
    print(f"\n=== E={E}, M={M}, N={N}, K={K} ===")
    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    # keep hp (bf16) reference copies before quantizing — ao pattern
    B_nkK_hp = torch.randn((E, N, K), dtype=torch.bfloat16, device="cuda")
    B_t_hp = B_nkK_hp.transpose(-2, -1).contiguous().transpose(-2, -1)  # (E, K, N) view
    # equivalent, cleaner:
    B_t_hp = B_nkK_hp.transpose(-2, -1)  # (E, K, N)

    # MXFP8 path
    A_scales, A_fp8 = to_mx(A, elem_dtype=torch.float8_e4m3fn, block_size=block_size)
    B_scales, B_fp8 = to_mx(B_nkK_hp, elem_dtype=torch.float8_e4m3fn, block_size=block_size)

    offs = generate_jagged_offs(E, M, multiple_of=block_size)
    out_mxfp8 = triton_mxfp8_grouped_mm(A_fp8, B_fp8, A_scales, B_scales, offs)

    # Reference: torch._grouped_mm on ORIGINAL hp inputs — matches ao's
    # reference_grouped_mm in test_mxfp8_grouped_mm.py.
    out_ref = torch._grouped_mm(
        A, B_t_hp, offs=offs.to(torch.int32), out_dtype=torch.bfloat16,
    )

    sqnr = compute_error(out_ref, out_mxfp8).item()
    print(f"SQNR: {sqnr:.2f} dB  (threshold: >= {MIN_SQNR_DB:.1f} dB)")

    assert sqnr >= MIN_SQNR_DB, (
        f"SQNR {sqnr:.2f} dB below threshold {MIN_SQNR_DB:.1f} dB — "
        f"accuracy regression (compare vs ao test_mxfp8_grouped_mm.py)"
    )
    print("PASS")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    if not is_MI350():
        print("WARNING: not on MI350+ (gfx950). Kernel will not run.")
        raise SystemExit(1)

    # Same shape pattern as ao test_emulate_mxfp8_grouped_gemm_2d_3d plus a
    # couple of our Llama4 shapes. Full 36-shape sweep lives in bench.py.
    test_grouped_mm_shape(E=1, M=1024, N=1024, K=1024)
    test_grouped_mm_shape(E=8, M=1024, N=4096, K=2048)
    test_grouped_mm_shape(E=4, M=1024, N=2048, K=2048)
    test_grouped_mm_shape(E=8, M=2048, N=5120, K=2048)
    print("\nAll tests passed.")
