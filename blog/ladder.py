# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Optimization ladder for the blog: a single parameterized MXFP8 grouped-GEMM
whose rungs each toggle exactly ONE optimization so the per-step speedup is
measurable in isolation.

Rungs (each builds on the previous):
  L0  naive          host-side torch routing · GROUP_M=1 · XCD=1 ·
                     small tiles (64/128/128) · 1 stage · plain scales
  L1  +fused-routing replace the host op-chain with _expt_data_kernel
  L2  +tiles         256/128/256 tiles · num_warps=8 · num_stages=2 (pipeline)
  L3  +scheduling    GROUP_M=8 + XCD_SWIZZLE=8 (L2 / per-XCD locality)
  L4  +cdna4-scales  pre-shuffled MFMA-native scale layout
  L5  +autotune      per-shape best config from kernels.forward._pick_config

Tiles are enlarged (L2) *before* the L2-reuse scheduling (L3): GROUP_M/XCD
only pay off once tiles are large and plentiful — applied first, on tiny
64-row tiles, they regress on big shapes.

It reuses the production Triton kernel and host shuffles from
``kernels.forward`` verbatim; the only new code is the *naive* host router and
the rung dispatch, so the measured deltas reflect real kernel behavior.
"""
import torch
import triton

from kernels.forward import (
    _mxfp8_grouped_mm_kernel,
    _build_expt_data,            # fused, sync-free router (one Triton launch)
    _shuffle_w_scales_cdna4_nonkdim16,
    _shuffle_x_scales_cdna4_nonkdim16,
    _shuffle_w_scales_cdna4_nonkdim32,
    _shuffle_x_scales_cdna4_nonkdim32,
    _pick_config,
)


# --- naive host-side router (the op-chain L1 removes) --------------------
def _build_expt_data_host(group_end_offsets, M, E, block_m):
    """Naive routing metadata built from torch ops on the host timeline:
    cat / diff / cumsum / arange / searchsorted / clamp / shift / where.

    Produces the identical (hist, offs_raw, offs_pad_sum, block_pid_map,
    grid_m_ub) tuple as the fused _build_expt_data kernel — only the *cost*
    differs (a chain of small launches vs one fused launch)."""
    device = group_end_offsets.device
    grid_m_ub = triton.cdiv(M, block_m) + max(E - 1, 0)

    offs = group_end_offsets.to(torch.int64)
    zero = torch.zeros(1, dtype=torch.int64, device=device)
    starts = torch.cat([zero, offs[:-1]])               # cat + shift  -> offs_raw
    hist = offs - starts                                # diff
    blocks_e = (hist + block_m - 1) // block_m          # cdiv
    cum = torch.cumsum(blocks_e, 0)                     # cumsum (inclusive)
    cum_excl = cum - blocks_e                           # exclusive prefix
    total = cum[-1]

    tiles = torch.arange(grid_m_ub, device=device)      # arange
    expert = torch.searchsorted(cum, tiles, right=True) # searchsorted
    expert_c = expert.clamp(max=E - 1)                  # clamp
    block = tiles - cum_excl[expert_c]                  # shift (gather + sub)
    packed = (block << 16) | expert_c
    block_pid_map = torch.where(                        # where
        tiles < total, packed, torch.full_like(packed, -1)
    ).to(torch.int32)

    return (
        hist.to(torch.int32),
        starts.to(torch.int32),
        total.to(torch.int32),
        block_pid_map,
        grid_m_ub,
    )


# --- rung definitions ----------------------------------------------------
# Each rung is the previous one plus a single changed field.
_BASE = dict(
    routing="host", BLOCK_M=64, BLOCK_N=128, BLOCK_K=128,
    GROUP_M=1, XCD_SWIZZLE=1, num_warps=4, num_stages=1,
    matrix_instr_nonkdim=16, waves_per_eu=0, scales="plain", autotune=False,
)

def _rung_cfg(rung, E, N, K):
    c = dict(_BASE)
    if rung >= 1: c["routing"] = "fused"
    if rung >= 2:
        c.update(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, num_warps=8, num_stages=2)
    if rung >= 3:
        c["GROUP_M"] = 8
        c["XCD_SWIZZLE"] = 8
    if rung >= 4: c["scales"] = "cdna4"
    if rung >= 5:
        best = _pick_config(E, N, K)
        c.update(
            BLOCK_M=best["BLOCK_M"], BLOCK_N=best["BLOCK_N"], BLOCK_K=best["BLOCK_K"],
            GROUP_M=best["GROUP_M"], num_warps=best["num_warps"],
            num_stages=best["num_stages"], waves_per_eu=best["waves_per_eu"],
            matrix_instr_nonkdim=best.get("matrix_instr_nonkdim", 16),
            XCD_SWIZZLE=best.get("xcd_swizzle", 8),
        )
        c["w_cache_modifier"] = best.get("w_cache_modifier", None)
        c["x_evict_policy"] = best.get("x_evict_policy", "")
        c["autotune"] = True
    return c

RUNGS = {
    0: "L0 naive",
    1: "L1 +fused-routing",
    2: "L2 +tiles/pipeline",
    3: "L3 +scheduling",
    4: "L4 +cdna4-scales",
    5: "L5 +per-shape-autotune",
}


def ladder_grouped_mm(input_act, weight, input_act_scales, weight_scales,
                      group_end_offsets, rung, out_dtype=torch.bfloat16):
    """MXFP8 grouped GEMM at a given optimization `rung` (0..6)."""
    M, K = input_act.shape
    E, N, K2 = weight.shape
    assert K == K2
    c = _rung_cfg(rung, E, N, K)

    BLOCK_M, BLOCK_N, BLOCK_K = c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_K"]
    nonkdim = c["matrix_instr_nonkdim"]
    w_cache_modifier = c.get("w_cache_modifier", None)
    x_evict_policy = c.get("x_evict_policy", "")

    # gate the CDNA4 scale path exactly like production, but only if this rung
    # asked for it.
    want_cdna4 = c["scales"] == "cdna4"
    use_cdna4_scale = (
        want_cdna4 and BLOCK_K >= 256 and K % 256 == 0 and N % 32 == 0 and M % 32 == 0
    )

    w_kn = weight.permute(0, 2, 1)                  # (E, K, N) col-major view
    x_scales_u8 = input_act_scales.view(torch.uint8)
    w_scales_u8 = weight_scales.view(torch.uint8)

    if use_cdna4_scale:
        if nonkdim == 32:
            w_scales_shuf = _shuffle_w_scales_cdna4_nonkdim32(w_scales_u8)
            x_scales_shuf = _shuffle_x_scales_cdna4_nonkdim32(x_scales_u8)
        else:
            w_scales_shuf = _shuffle_w_scales_cdna4_nonkdim16(w_scales_u8)
            x_scales_shuf = _shuffle_x_scales_cdna4_nonkdim16(x_scales_u8)
        w_scales_arg = w_scales_shuf
        # shuffled layout is (E, N//32, K): dim1 is the N-block dim, dim2 is K,
        # so stride_k = stride(2), stride_n = stride(1)  (matches forward.py).
        w_se, w_sk, w_sn = (w_scales_shuf.stride(0), w_scales_shuf.stride(2),
                            w_scales_shuf.stride(1))
        x_scales_arg = x_scales_shuf
        x_sm, x_sk = x_scales_shuf.stride(0), x_scales_shuf.stride(1)
        swizzle_mx_scale = "CDNA4_SCALE"
    else:
        w_scales_kn = w_scales_u8.permute(0, 2, 1)
        w_scales_arg = w_scales_kn
        w_se, w_sk, w_sn = (w_scales_kn.stride(0), w_scales_kn.stride(1),
                            w_scales_kn.stride(2))
        x_scales_arg = x_scales_u8
        x_sm, x_sk = x_scales_u8.stride(0), x_scales_u8.stride(1)
        swizzle_mx_scale = None

    if c["routing"] == "host":
        hist, offs_raw, offs_pad_sum, block_pid_map, grid_m = \
            _build_expt_data_host(group_end_offsets, M, E, BLOCK_M)
    else:
        hist, offs_raw, offs_pad_sum, block_pid_map, grid_m = \
            _build_expt_data(group_end_offsets, M, E, BLOCK_M)

    grid_n = triton.cdiv(N, BLOCK_N)
    grid = (grid_m * grid_n,)
    output = torch.empty((M, N), dtype=out_dtype, device=input_act.device)

    _mxfp8_grouped_mm_kernel[grid](
        output, output.stride(0), output.stride(1),
        input_act, input_act.stride(0), input_act.stride(1),
        x_scales_arg, x_sm, x_sk,
        w_kn, w_kn.stride(0), w_kn.stride(1), w_kn.stride(2),
        w_scales_arg, w_se, w_sk, w_sn,
        N, K,
        hist, offs_raw, offs_pad_sum, block_pid_map,
        grid_m, grid_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=c["GROUP_M"], XCD_SWIZZLE=c["XCD_SWIZZLE"],
        SWIZZLE_MX_SCALE=swizzle_mx_scale,
        SCALE_NONKDIM=nonkdim,
        EVEN_K=(K % BLOCK_K == 0), MASK_K_LIMIT=(K % BLOCK_K),
        W_CACHE_MODIFIER=w_cache_modifier,
        X_EVICT_POLICY=x_evict_policy,
        UPCAST_INDICES=False,
        num_warps=c["num_warps"], num_stages=c["num_stages"],
        matrix_instr_nonkdim=nonkdim, kpack=1, waves_per_eu=c["waves_per_eu"],
    )
    return output
