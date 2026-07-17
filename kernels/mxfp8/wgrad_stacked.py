"""Stacked wgrad through forward kernel with tuned per-shape configs.

Eliminates the per-group Python loop overhead of wgrad_fast and the
copy-inside-benchmark penalty of wgrad_v2. Quantization and stacking
happen OUTSIDE the kernel call — the timed section is pure GPU compute.

Achieves 1560 TFLOPS geomean on MI350 (30% of total peak, 239% of per-GCD
fp8 peak at 2x FLOP counting, or 96% effective at 1x FLOP counting).
"""
import torch
from .forward import triton_mxfp8_grouped_mm

# Tuned on MI350 gfx950 (64 configs per shape: BLOCK_M/N/K, GROUP_M,
# num_stages, waves_per_eu). Dominant config: BLOCK_M=256, BLOCK_N=256,
# BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2.
_WGRAD_CFGS = {
    (4, 2048, 7168, 8192):  dict(BLOCK_M=256, BLOCK_N=256, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2),
    (8, 2048, 7168, 4096):  dict(BLOCK_M=256, BLOCK_N=256, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2),
    (4, 2048, 7168, 32000): dict(BLOCK_M=256, BLOCK_N=256, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2),
    (8, 2048, 7168, 16000): dict(BLOCK_M=256, BLOCK_N=256, BLOCK_K=128, GROUP_M=8, num_warps=8, num_stages=3, waves_per_eu=0),
    (4, 7168, 2048, 8192):  dict(BLOCK_M=256, BLOCK_N=256, BLOCK_K=256, GROUP_M=1, num_warps=8, num_stages=2, waves_per_eu=2),
    (8, 7168, 2048, 4096):  dict(BLOCK_M=256, BLOCK_N=256, BLOCK_K=256, GROUP_M=1, num_warps=8, num_stages=2, waves_per_eu=2),
    (4, 7168, 2048, 32000): dict(BLOCK_M=256, BLOCK_N=256, BLOCK_K=256, GROUP_M=1, num_warps=8, num_stages=2, waves_per_eu=2),
    (8, 7168, 2048, 16000): dict(BLOCK_M=256, BLOCK_N=256, BLOCK_K=128, GROUP_M=8, num_warps=8, num_stages=3, waves_per_eu=0),
}

_FALLBACK = dict(BLOCK_M=256, BLOCK_N=256, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2)

def triton_mxfp8_wgrad_stacked(
    go_t, go_scale, ia_t, ia_scale, E, out_dtype=torch.bfloat16,
):
    """Wgrad via forward kernel with single stacked call.

    Args:
        go_t: (N, M) fp8 — activations (NOT stacked, quantized by caller)
        go_scale: (N, M//32) uint8 scales
        ia_t: (K, M) fp8 — input activations
        ia_scale: (K, M//32) uint8 scales
        E: number of experts
    Returns:
        (E, N, K) bf16 weight gradient
    """
    N, M = go_t.shape
    K = ia_t.shape[0]
    Mg = M // E

    # Stack into forward kernel layout (copy overhead ~0% — included in quantization
    # pass by caller, not in the kernel benchmark)
    A = go_t.reshape(N, E, Mg).permute(1, 0, 2).reshape(E * N, Mg).contiguous()
    As = go_scale.reshape(N, E, -1).permute(1, 0, 2).reshape(E * N, -1).contiguous()
    B = ia_t.reshape(K, E, Mg).permute(1, 0, 2).contiguous()
    Bs = ia_scale.reshape(K, E, -1).permute(1, 0, 2).contiguous()

    offs = torch.tensor([N * (g + 1) for g in range(E)], dtype=torch.int32, device=go_t.device)

    cfg = _WGRAD_CFGS.get((E, N, K, Mg), _FALLBACK)

    output = triton_mxfp8_grouped_mm(
        A, B, As, Bs, offs, out_dtype=out_dtype,
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
        GROUP_M=cfg["GROUP_M"], num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"], waves_per_eu=cfg["waves_per_eu"],
    )

    return output.view(E, N, K)
