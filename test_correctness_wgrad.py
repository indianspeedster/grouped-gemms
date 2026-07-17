##############################################################################
# Correctness sanity: MXFP8 grouped wgrad vs bf16 reference on hp inputs.
# Asserts SQNR >= 27.0 dB for both native and gluon wgrad kernels.
#
# Same pattern as test_correctness.py (forward).
##############################################################################

import torch

from kernels.mxfp8.backward import triton_mxfp8_wgrad
from kernels.mxfp8.wgrad_gluon import triton_mxfp8_wgrad_gluon
from utils import generate_jagged_offs, is_MI350, to_mx


# SQNR in dB: 20 * log10(||signal|| / ||signal - approx||)
def compute_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    Ps = torch.linalg.vector_norm(x.to(torch.float32))
    Pn = torch.linalg.vector_norm((x - y).to(torch.float32))
    return 20 * torch.log10(Ps / Pn)


MIN_SQNR_DB = 27.0


def compute_bf16_reference(go_bf16, ia_bf16, offsets):
    """Grouped bf16 wgrad reference: per-group GO @ IA^T."""
    E = offsets.shape[0]
    N, M = go_bf16.shape
    K = ia_bf16.shape[0]
    out = torch.zeros((E, N, K), dtype=torch.float32, device=go_bf16.device)
    for e in range(E):
        s = offsets[e-1].item() if e > 0 else 0
        e_end = offsets[e].item()
        go_g = go_bf16[:, s:e_end].float()
        ia_g = ia_bf16[:, s:e_end].float()
        out[e] = go_g @ ia_g.T
    return out


def test_wgrad_shape(E: int, M: int, N: int, K: int, block_size: int = 32):
    print(f"\n=== E={E}, M={M}, N={N}, K={K} ===")
    torch.manual_seed(0)

    go_bf16 = torch.randn((N, M), dtype=torch.bfloat16, device="cuda")
    ia_bf16 = torch.randn((K, M), dtype=torch.bfloat16, device="cuda")
    offsets = generate_jagged_offs(E, M, multiple_of=block_size)

    # Quantize
    go_scale, go_fp8 = to_mx(go_bf16, torch.float8_e4m3fn, block_size)
    ia_scale, ia_fp8 = to_mx(ia_bf16, torch.float8_e4m3fn, block_size)

    # Reference on hp inputs
    ref = compute_bf16_reference(go_bf16, ia_bf16, offsets)

    # Native wgrad
    out_native = triton_mxfp8_wgrad(go_fp8, go_scale, ia_fp8, ia_scale, offsets)
    sqnr_native = compute_error(ref, out_native).item()
    print(f"  native SQNR: {sqnr_native:.2f} dB  (threshold: >= {MIN_SQNR_DB:.1f} dB)")
    assert sqnr_native >= MIN_SQNR_DB, (
        f"native SQNR {sqnr_native:.2f} dB below {MIN_SQNR_DB:.1f} dB"
    )

    # Gluon wgrad
    out_gluon = triton_mxfp8_wgrad_gluon(go_fp8, go_scale, ia_fp8, ia_scale, offsets)
    sqnr_gluon = compute_error(ref, out_gluon).item()
    print(f"  gluon  SQNR: {sqnr_gluon:.2f} dB  (threshold: >= {MIN_SQNR_DB:.1f} dB)")
    assert sqnr_gluon >= MIN_SQNR_DB, (
        f"gluon SQNR {sqnr_gluon:.2f} dB below {MIN_SQNR_DB:.1f} dB"
    )

    # Native vs gluon agreement
    agreement = compute_error(out_native, out_gluon).item()
    print(f"  native-gluon agreement: {agreement:.2f} dB")

    print("  PASS")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    if not is_MI350():
        print("WARNING: not on MI350+ (gfx950). Kernel will not run.")
        raise SystemExit(1)

    # Same shape pattern as test_correctness.py
    test_wgrad_shape(E=1, M=1024, N=1024, K=1024)
    test_wgrad_shape(E=8, M=1024, N=4096, K=2048)
    test_wgrad_shape(E=4, M=1024, N=2048, K=2048)
    test_wgrad_shape(E=8, M=2048, N=5120, K=2048)
    print("\nAll tests passed.")
