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

"""Forward / dgrad MXFP4 (e2m1) grouped-GEMM kernel for ROCm gfx950+.

Exports ``triton_mxfp4_grouped_mm``: computes ``A @ B^T`` per expert group.
The MXFP4 twin of the MXFP8 kernel, sharing its scheduling (XCD swizzle,
GROUP_M L2 reuse, packed per-tile expert lookup, CDNA4-native scale layout).

Operands are e2m1: two fp4 codes packed per uint8 along K, so operand tiles
are ``BLOCK_K // 2`` wide/step and the tail-K mask is in packed units. The
e8m0 scales are byte-for-byte identical to MXFP8 (one per 32 logical K
elements), so the host-side shuffle and in-kernel unshuffle helpers are reused
unchanged from the mxfp8 package.
"""

import torch

from .._common import _rocm_mxfp8_available

if _rocm_mxfp8_available:
    import triton
    import triton.language as tl

    # Scheduling + scale-layout helpers are element-type agnostic — reuse the
    # MXFP8 implementations rather than duplicating them.
    from ..mxfp8.forward import (
        _build_expt_data,
        _pid_grid,
        _shuffle_w_scales_cdna4_nonkdim16,
        _shuffle_w_scales_cdna4_nonkdim32,
        _shuffle_x_scales_cdna4_nonkdim16,
        _shuffle_x_scales_cdna4_nonkdim32,
        _unswizzle_mx_scale_cdna4,
        _unswizzle_mx_scale_cdna4_nonkdim32,
        _xcd_swizzle,
    )

    @triton.jit
    def _mxfp4_grouped_mm_kernel(
        Y, stride_y_m, stride_y_n,
        X, stride_x_m, stride_x_k,             # X packed (M, K//2) uint8
        XMxScale, stride_x_mx_m, stride_x_mx_k,
        W, stride_w_e, stride_w_k, stride_w_n,  # W packed col-major (E, K//2, N)
        WMxScale, stride_w_mx_e, stride_w_mx_k, stride_w_mx_n,
        N, K,                                   # K is the LOGICAL fp4 length
        ExptHist,
        ExptOffs,
        ExptOffsSum,
        ExptData,
        grid_m, grid_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,                  # logical K per tile (multiple of 2)
        GROUP_M: tl.constexpr,
        XCD_SWIZZLE: tl.constexpr,
        SWIZZLE_MX_SCALE: tl.constexpr,
        SCALE_NONKDIM: tl.constexpr,
        EVEN_K: tl.constexpr,
        MASK_K_LIMIT: tl.constexpr,             # K % BLOCK_K, in logical elements
        W_CACHE_MODIFIER: tl.constexpr,
        X_EVICT_POLICY: tl.constexpr,
        UPCAST_INDICES: tl.constexpr = False,
    ):
        MX_PACK_DIVISOR: tl.constexpr = 32      # logical K elements per e8m0 scale
        # Two fp4 codes per byte -> operand tiles are half as wide along K.
        PACKED_BLOCK_K: tl.constexpr = BLOCK_K // 2

        pid = tl.program_id(0)
        if ExptOffsSum is not None and XCD_SWIZZLE > 1:
            padding_m = grid_m - tl.load(ExptOffsSum)
        else:
            padding_m: tl.constexpr = 0

        index_type: tl.constexpr = tl.int64 if UPCAST_INDICES else tl.int32
        unpadded_m = grid_m - padding_m
        tl.assume(unpadded_m >= 0)
        total_actual_tiles = unpadded_m * grid_n
        if padding_m > 0 and pid >= total_actual_tiles:
            return

        pid_emn = pid
        if XCD_SWIZZLE != 1:
            pid_emn = _xcd_swizzle(pid_emn, total_actual_tiles, XCD_SWIZZLE)
        pid_m, pid_n = _pid_grid(pid_emn, unpadded_m, grid_n, GROUP_M)

        expt_data = tl.load(ExptData + pid_m)
        if expt_data == -1:
            return
        expt_id = expt_data & 0x0000FFFF
        block_id = expt_data >> 16
        M = tl.load(ExptHist + expt_id)
        start_m = tl.load(ExptOffs + expt_id)
        expt_id = expt_id.to(index_type)
        block_id = block_id.to(index_type)
        start_m = start_m.to(index_type)
        pid_n = pid_n.to(index_type)

        # X pointers (A, per-expert slice). K axis is packed -> PACKED_BLOCK_K.
        offs_x_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
        offs_x_m = tl.max_contiguous(tl.multiple_of(offs_x_m % M, BLOCK_M), BLOCK_M)
        X += start_m * stride_x_m
        offs_x_k = tl.arange(0, PACKED_BLOCK_K)
        XPtrs = (
            X
            + offs_x_m.to(index_type)[:, None] * stride_x_m
            + offs_x_k.to(index_type)[None, :] * stride_x_k
        )

        MX_SCALE_BLOCK_K: tl.constexpr = BLOCK_K // MX_PACK_DIVISOR

        # W scale pointers (identical layout to MXFP8).
        WMxScale += expt_id * stride_w_mx_e
        if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
            NON_K_PRESHUFFLE_BLOCK_SIZE: tl.constexpr = 32
            PACKED_MX_BLOCK: tl.constexpr = MX_SCALE_BLOCK_K * NON_K_PRESHUFFLE_BLOCK_SIZE
            SCALE_BLOCK_N: tl.constexpr = BLOCK_N // NON_K_PRESHUFFLE_BLOCK_SIZE
            SCALE_BLOCK_M: tl.constexpr = BLOCK_M // NON_K_PRESHUFFLE_BLOCK_SIZE
        else:
            PACKED_MX_BLOCK: tl.constexpr = MX_SCALE_BLOCK_K
            SCALE_BLOCK_N: tl.constexpr = BLOCK_N
            SCALE_BLOCK_M: tl.constexpr = BLOCK_M
        offs_w_n_scale = (pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)) % N
        offs_w_n_scale = tl.max_contiguous(
            tl.multiple_of(offs_w_n_scale, SCALE_BLOCK_N), SCALE_BLOCK_N
        )
        offs_w_k_scale = tl.arange(0, PACKED_MX_BLOCK)
        WMxScalePtrs = (
            WMxScale
            + offs_w_k_scale.to(index_type)[None, :] * stride_w_mx_k
            + offs_w_n_scale.to(index_type)[:, None] * stride_w_mx_n
        )

        # W pointers (col-major (E, K//2, N) view). K axis packed.
        offs_w_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_w_n = tl.max_contiguous(tl.multiple_of(offs_w_n % N, BLOCK_N), BLOCK_N)
        offs_w_k = tl.arange(0, PACKED_BLOCK_K)
        W += expt_id * stride_w_e
        WPtrs = W + (
            offs_w_k.to(index_type)[:, None] * stride_w_k
            + offs_w_n.to(index_type)[None, :] * stride_w_n
        )

        # X scale pointers (identical layout to MXFP8).
        if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
            XMxScale += (start_m // 32) * stride_x_mx_m
            offs_x_m_scale = BLOCK_M // NON_K_PRESHUFFLE_BLOCK_SIZE * block_id + tl.arange(0, SCALE_BLOCK_M)
            offs_x_k_scale = tl.arange(0, PACKED_MX_BLOCK)
        else:
            XMxScale += start_m * stride_x_mx_m
            offs_x_m_scale = offs_x_m
            offs_x_k_scale = tl.arange(0, MX_SCALE_BLOCK_K)
        XMxScalePtrs = (
            XMxScale
            + offs_x_m_scale.to(index_type)[:, None] * stride_x_mx_m
            + offs_x_k_scale.to(index_type)[None, :] * stride_x_mx_k
        )

        num_k_iter = tl.cdiv(K, BLOCK_K)
        if not EVEN_K:
            num_k_iter -= 1

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(num_k_iter):
            x = tl.load(XPtrs, eviction_policy=X_EVICT_POLICY)
            w = tl.load(WPtrs, cache_modifier=W_CACHE_MODIFIER)
            if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
                if SCALE_NONKDIM == 32:
                    x_scales = _unswizzle_mx_scale_cdna4_nonkdim32(
                        tl.load(XMxScalePtrs, eviction_policy=X_EVICT_POLICY),
                        BLOCK_M, MX_SCALE_BLOCK_K,
                    )
                    w_scales = _unswizzle_mx_scale_cdna4_nonkdim32(
                        tl.load(WMxScalePtrs, cache_modifier=W_CACHE_MODIFIER),
                        BLOCK_N, MX_SCALE_BLOCK_K,
                    )
                else:
                    x_scales = _unswizzle_mx_scale_cdna4(
                        tl.load(XMxScalePtrs, eviction_policy=X_EVICT_POLICY),
                        BLOCK_M, MX_SCALE_BLOCK_K,
                    )
                    w_scales = _unswizzle_mx_scale_cdna4(
                        tl.load(WMxScalePtrs, cache_modifier=W_CACHE_MODIFIER),
                        BLOCK_N, MX_SCALE_BLOCK_K,
                    )
            else:
                x_scales = tl.load(XMxScalePtrs, eviction_policy=X_EVICT_POLICY)
                w_scales = tl.load(WMxScalePtrs)

            acc = tl.dot_scaled(
                x, x_scales, "e2m1", w, w_scales, "e2m1", acc=acc, fast_math=True
            )

            WMxScalePtrs += PACKED_MX_BLOCK * stride_w_mx_k
            XMxScalePtrs += PACKED_MX_BLOCK * stride_x_mx_k
            XPtrs += PACKED_BLOCK_K * stride_x_k
            WPtrs += PACKED_BLOCK_K * stride_w_k

        if not EVEN_K:
            # offs_*_k are packed; logical position of packed col j is 2*j.
            mask_x_k = offs_x_k * 2 < MASK_K_LIMIT
            mask_w_k = offs_w_k * 2 < MASK_K_LIMIT
            if SWIZZLE_MX_SCALE is None:
                mask_w_k_scale = offs_w_k_scale * MX_PACK_DIVISOR < MASK_K_LIMIT
                mask_x_k_scale = offs_x_k_scale * MX_PACK_DIVISOR < MASK_K_LIMIT

            x = tl.load(XPtrs, mask=mask_x_k[None, :], other=0)
            w = tl.load(WPtrs, mask=mask_w_k[:, None], other=0,
                        cache_modifier=W_CACHE_MODIFIER)
            if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
                if SCALE_NONKDIM == 32:
                    x_scales = _unswizzle_mx_scale_cdna4_nonkdim32(
                        tl.load(XMxScalePtrs), BLOCK_M, MX_SCALE_BLOCK_K,
                    )
                    w_scales = _unswizzle_mx_scale_cdna4_nonkdim32(
                        tl.load(WMxScalePtrs, cache_modifier=W_CACHE_MODIFIER),
                        BLOCK_N, MX_SCALE_BLOCK_K,
                    )
                else:
                    x_scales = _unswizzle_mx_scale_cdna4(
                        tl.load(XMxScalePtrs), BLOCK_M, MX_SCALE_BLOCK_K,
                    )
                    w_scales = _unswizzle_mx_scale_cdna4(
                        tl.load(WMxScalePtrs, cache_modifier=W_CACHE_MODIFIER),
                        BLOCK_N, MX_SCALE_BLOCK_K,
                    )
            else:
                x_scales = tl.load(XMxScalePtrs, mask=mask_x_k_scale[None, :])
                w_scales = tl.load(WMxScalePtrs, mask=mask_w_k_scale[None, :])

            acc = tl.dot_scaled(
                x, x_scales, "e2m1", w, w_scales, "e2m1", acc=acc, fast_math=True
            )

        # Write-back
        offs_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
        offs_y_n = BLOCK_N * pid_n + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_y_n < N
        Y += start_m * stride_y_m
        YPtrs = (
            Y
            + offs_m.to(index_type)[:, None] * stride_y_m
            + offs_y_n.to(index_type)[None, :] * stride_y_n
        )
        tl.store(YPtrs, acc.to(Y.dtype.element_ty),
                 mask=mask_m[:, None] & mask_n[None, :])

    # Per-shape best configs from a 36-shape × 576-config MXFP4 sweep on
    # MI355X (8-GPU parallel, tune_driver_fp4.py / tune_worker_fp4.py). Search
    # space: BLOCK_M ∈ {64,128,256}, BLOCK_N ∈ {128,256},
    # BLOCK_K ∈ {128,256,512}, GROUP_M ∈ {4,8}, num_warps ∈ {4,8},
    # num_stages ∈ {1,2}, waves_per_eu ∈ {0,2}, nonkdim ∈ {16,32}, kpack=1.
    # Comments show the swept median runtime. Patterns: BLOCK_N=128 always
    # wins; nonkdim=32 wins 32/36 shapes; BLOCK_K=256 dominates for K≤5120 but
    # BLOCK_K=512 wins 8 of the largest shapes (fp4 packs 2 elems/byte, so a
    # 512-logical-K tile is the LDS footprint of an fp8 256 tile).
    _BEST_CFGS_FP4 = {
        (1, 2048, 2048): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=128, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 92.2us
        (1, 2048, 5120): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 137.5us
        (1, 2048, 8192): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 208.0us
        (1, 5120, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 143.2us
        (1, 5120, 5120): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 294.9us
        (1, 5120, 8192): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 450.5us
        (1, 8192, 2048): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=1, waves_per_eu=2, matrix_instr_nonkdim=32),  # 208.2us
        (1, 8192, 5120): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=1, waves_per_eu=2, matrix_instr_nonkdim=32),  # 445.3us
        (1, 8192, 8192): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=16),  # 729.4us
        (2, 2048, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 73.3us
        (2, 2048, 5120): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 138.3us
        (2, 2048, 8192): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 207.4us
        (2, 5120, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 146.9us
        (2, 5120, 5120): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 302.3us
        (2, 5120, 8192): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=512, GROUP_M=4, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 447.4us
        (2, 8192, 2048): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=1, waves_per_eu=2, matrix_instr_nonkdim=32),  # 218.4us
        (2, 8192, 5120): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=512, GROUP_M=4, num_warps=4, num_stages=1, waves_per_eu=0, matrix_instr_nonkdim=16),  # 533.5us
        (2, 8192, 8192): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=512, GROUP_M=4, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 707.5us
        (4, 2048, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 72.6us
        (4, 2048, 5120): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 140.1us
        (4, 2048, 8192): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 212.3us
        (4, 5120, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 148.4us
        (4, 5120, 5120): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 304.5us
        (4, 5120, 8192): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=512, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 465.8us
        (4, 8192, 2048): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=1, waves_per_eu=2, matrix_instr_nonkdim=16),  # 257.0us
        (4, 8192, 5120): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=4, num_stages=1, waves_per_eu=2, matrix_instr_nonkdim=32),  # 484.2us
        (4, 8192, 8192): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=512, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 740.4us
        (8, 2048, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 78.3us
        (8, 2048, 5120): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 145.3us
        (8, 2048, 8192): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 225.9us
        (8, 5120, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 159.4us
        (8, 5120, 5120): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=512, GROUP_M=4, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 330.7us
        (8, 5120, 8192): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=512, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 558.9us
        (8, 8192, 2048): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=4, num_stages=1, waves_per_eu=2, matrix_instr_nonkdim=32),  # 238.4us
        (8, 8192, 5120): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=512, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 519.2us
        (8, 8192, 8192): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=512, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 753.4us
    }

    # DSv3 671B shapes (N=2048, K=7168, E∈{4,8}, M∈{32768,128000}) from the
    # same 576-config sweep (tune_driver_fp4.py --shapes dsv3). Keyed on
    # (E, M, N, K) because the two M values share N/K. Unlike the Llama4 grid,
    # the narrow N=2048 + huge M regime prefers BLOCK_N=256; the M=128000
    # shapes also flip to nonkdim=16 + BLOCK_M=256.
    _BEST_CFGS_FP4_DSV3 = {
        (4, 32768, 2048, 7168): dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, GROUP_M=4, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 407.2us
        (8, 32768, 2048, 7168): dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=512, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 358.7us
        (4, 128000, 2048, 7168): dict(BLOCK_M=256, BLOCK_N=256, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=16),  # 1155.3us
        (8, 128000, 2048, 7168): dict(BLOCK_M=256, BLOCK_N=256, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=16),  # 1152.8us
    }

    # Fallback for shapes outside both swept grids, distilled from the Llama4
    # table: BLOCK_N=128 + nonkdim=32 always; small K wants BLOCK_K=256/
    # GROUP_M=8, large K wants a bigger M tile. BLOCK_K stays 256 (not 512) as
    # the safe default since 512 only wins on specific large E≥2 shapes.
    _FALLBACK_FP4_SMALL_K = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32)
    _FALLBACK_FP4_LARGE_K = dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32)

    def _pick_config(E: int, M: int, N: int, K: int) -> dict:
        """Per-shape best config from the swept grids; coarse fallback for
        unseen shapes. The DSv3 table is keyed (E,M,N,K) and tried first; the
        Llama4 table is keyed (E,N,K) (single M=16640 regime)."""
        cfg = _BEST_CFGS_FP4_DSV3.get((E, M, N, K))
        if cfg is not None:
            return cfg
        cfg = _BEST_CFGS_FP4.get((E, N, K))
        if cfg is not None:
            return cfg
        return _FALLBACK_FP4_SMALL_K if K <= 2048 else _FALLBACK_FP4_LARGE_K

    def triton_mxfp4_grouped_mm(
        input_act: torch.Tensor,
        weight: torch.Tensor,
        input_act_scales: torch.Tensor,
        weight_scales: torch.Tensor,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
        BLOCK_M: int = None,
        BLOCK_N: int = None,
        BLOCK_K: int = None,
        GROUP_M: int = None,
        XCD_SWIZZLE: int = None,
        num_warps: int = None,
        num_stages: int = None,
        matrix_instr_nonkdim: int = None,
        waves_per_eu: int = None,
        kpack: int = None,
        w_cache_modifier: str = None,
        x_evict_policy: str = None,
    ) -> torch.Tensor:
        """MXFP4 grouped GEMM: ``output[g] = input_act[group_g] @ weight[g]^T``.

        Forward + dgrad twin of ``triton_mxfp8_grouped_mm``. Operands are e2m1,
        packed two fp4 codes per uint8 along K (first/even-K element in the low
        nibble — ``tl.dot_scaled``'s convention).

        Args:
            input_act: ``(M, K//2)`` uint8 packed e2m1, row-major.
            weight: ``(E, N, K//2)`` uint8 packed e2m1, row-major; internally
                viewed column-major (``E, K//2, N``) before launch.
            input_act_scales: ``(M, K//32)`` e8m0-viewed-as-uint8.
            weight_scales: ``(E, N, K//32)`` e8m0-viewed-as-uint8.
            group_end_offsets: ``(E,)`` int32, cumulative token counts per expert.

        ``K`` (the logical fp4 length) is inferred as ``2 * input_act.shape[1]``.
        """
        M, K_half = input_act.shape
        E, N, K_half_w = weight.shape
        assert K_half == K_half_w, f"packed-K mismatch: A={K_half}, B={K_half_w}"
        K = K_half * 2

        _cfg = _pick_config(E, M, N, K)
        if BLOCK_M is None: BLOCK_M = _cfg["BLOCK_M"]
        if BLOCK_N is None: BLOCK_N = _cfg["BLOCK_N"]
        if BLOCK_K is None: BLOCK_K = _cfg["BLOCK_K"]
        if GROUP_M is None: GROUP_M = _cfg["GROUP_M"]
        if num_warps is None: num_warps = _cfg["num_warps"]
        if num_stages is None: num_stages = _cfg["num_stages"]
        if waves_per_eu is None: waves_per_eu = _cfg["waves_per_eu"]
        if matrix_instr_nonkdim is None:
            matrix_instr_nonkdim = _cfg.get("matrix_instr_nonkdim", 32)
        if kpack is None:
            kpack = _cfg.get("kpack", 1)
        if XCD_SWIZZLE is None:
            XCD_SWIZZLE = _cfg.get("XCD_SWIZZLE", 8)
        if w_cache_modifier is None:
            w_cache_modifier = _cfg.get("w_cache_modifier", None)
        if x_evict_policy is None:
            x_evict_policy = _cfg.get("x_evict_policy", "")

        # CDNA4 native-scale path. Same gates as MXFP8 but checked on the
        # logical K (scales are per-32-logical-element, layout-identical).
        use_cdna4_scale = (
            BLOCK_K >= 256 and K % 256 == 0 and N % 32 == 0 and M % 32 == 0
        )

        # Column-major view of packed W (E, K//2, N) with stride(-2)==1.
        w_kn = weight.permute(0, 2, 1)
        x_scales_u8 = input_act_scales.view(torch.uint8)
        w_scales_u8 = weight_scales.view(torch.uint8)

        if use_cdna4_scale:
            if matrix_instr_nonkdim == 32:
                w_scales_shuf = _shuffle_w_scales_cdna4_nonkdim32(w_scales_u8)
                x_scales_shuf = _shuffle_x_scales_cdna4_nonkdim32(x_scales_u8)
                nonkdim = 32
            else:
                w_scales_shuf = _shuffle_w_scales_cdna4_nonkdim16(w_scales_u8)
                x_scales_shuf = _shuffle_x_scales_cdna4_nonkdim16(x_scales_u8)
                nonkdim = 16

            w_scales_arg = w_scales_shuf
            w_scales_stride_e = w_scales_shuf.stride(0)
            w_scales_stride_n = w_scales_shuf.stride(1)
            w_scales_stride_k = w_scales_shuf.stride(2)

            x_scales_arg = x_scales_shuf
            x_scales_stride_m = x_scales_shuf.stride(0)
            x_scales_stride_k = x_scales_shuf.stride(1)

            swizzle_mx_scale = "CDNA4_SCALE"
        else:
            w_scales_kn = w_scales_u8.permute(0, 2, 1)
            w_scales_arg = w_scales_kn
            w_scales_stride_e = w_scales_kn.stride(0)
            w_scales_stride_k = w_scales_kn.stride(1)
            w_scales_stride_n = w_scales_kn.stride(2)

            x_scales_arg = x_scales_u8
            x_scales_stride_m = x_scales_u8.stride(0)
            x_scales_stride_k = x_scales_u8.stride(1)

            swizzle_mx_scale = None
            nonkdim = matrix_instr_nonkdim

        hist, offs_raw, offs_pad_sum, block_pid_map, grid_m = _build_expt_data(
            group_end_offsets, M, E, BLOCK_M
        )
        grid_n = triton.cdiv(N, BLOCK_N)
        grid = (grid_m * grid_n,)

        output = torch.empty((M, N), dtype=out_dtype, device=input_act.device)

        _mxfp4_grouped_mm_kernel[grid](
            output, output.stride(0), output.stride(1),
            input_act, input_act.stride(0), input_act.stride(1),
            x_scales_arg, x_scales_stride_m, x_scales_stride_k,
            w_kn, w_kn.stride(0), w_kn.stride(1), w_kn.stride(2),
            w_scales_arg, w_scales_stride_e, w_scales_stride_k, w_scales_stride_n,
            N, K,
            hist, offs_raw, offs_pad_sum, block_pid_map,
            grid_m, grid_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M, XCD_SWIZZLE=XCD_SWIZZLE,
            SWIZZLE_MX_SCALE=swizzle_mx_scale,
            SCALE_NONKDIM=nonkdim,
            EVEN_K=(K % BLOCK_K == 0), MASK_K_LIMIT=(K % BLOCK_K),
            W_CACHE_MODIFIER=w_cache_modifier,
            X_EVICT_POLICY=x_evict_policy,
            UPCAST_INDICES=False,
            num_warps=num_warps, num_stages=num_stages,
            matrix_instr_nonkdim=nonkdim, kpack=kpack,
            waves_per_eu=waves_per_eu,
        )
        return output

else:
    _UNAVAILABLE_MSG = "ROCm MXFP4 kernels require gfx950 or later and triton"

    def triton_mxfp4_grouped_mm(*args, **kwargs):
        raise NotImplementedError(_UNAVAILABLE_MSG)
