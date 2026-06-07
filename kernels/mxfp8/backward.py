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

"""Backward (weight-gradient) MXFP8 grouped-GEMM kernel for ROCm gfx950+.

Exports ``triton_mxfp8_wgrad`` — computes ``grad_W[g] =
grad_output[group_g]^T @ input_act[group_g]`` per expert group (A^T @ B).

Two execution modes:
  - Direct: one CTA per (BLOCK_N, BLOCK_K, group).
  - Split-M + reduce: partition the per-group M-loop across multiple CTAs,
    write fp32 partials, then reduce to bf16. Helps low-CTA small-E shapes
    where each CTA otherwise runs a very long M loop.
"""

import torch

from .._common import _rocm_mxfp8_available

if _rocm_mxfp8_available:
    import triton
    import triton.language as tl

    # Reuse forward's CDNA4 scale-layout helpers: the shuffle is dim-agnostic,
    # mapping (non_reduction, reduction//32) -> (non_reduction//32, reduction).
    # Both GO and IA scales here play the role of forward's X scale.
    from .forward import (
        _unswizzle_mx_scale_cdna4,
        _unswizzle_mx_scale_cdna4_nonkdim32,
        _shuffle_x_scales_cdna4_nonkdim16,
        _shuffle_x_scales_cdna4_nonkdim32,
    )

    @triton.jit
    def _wgrad_pid_grid(
        pid,
        num_pid_n,
        num_pid_k,
        GROUP_N: tl.constexpr,
        GROUP_K: tl.constexpr,
        SCHED_MODE: tl.constexpr,
    ):
        if SCHED_MODE == "GROUP_K":
            # Per N tile, run a contiguous block of K tiles: favors GO reuse.
            num_pid_k_groups = tl.cdiv(num_pid_k, GROUP_K)
            pids_per_n = num_pid_k_groups * GROUP_K
            pid_n = pid // pids_per_n
            pid_in_n = pid % pids_per_n
            pid_k = (pid_in_n // GROUP_K) * GROUP_K + (pid_in_n % GROUP_K)
        elif SCHED_MODE == "GROUP_NK" or SCHED_MODE == "GROUP_NK_K":
            # Rectangular N x K cluster: GO reused across K tiles, IA across N
            # tiles, within one L2 working set.
            num_pid_n_groups = tl.cdiv(num_pid_n, GROUP_N)
            num_pid_k_groups = tl.cdiv(num_pid_k, GROUP_K)
            pids_per_group = GROUP_N * GROUP_K
            group_id = pid // pids_per_group
            pid_in_group = pid % pids_per_group
            group_n = group_id % num_pid_n_groups
            group_k = group_id // num_pid_n_groups
            if SCHED_MODE == "GROUP_NK_K":
                pid_n = group_n * GROUP_N + (pid_in_group // GROUP_K)
                pid_k = group_k * GROUP_K + (pid_in_group % GROUP_K)
            else:
                pid_n = group_n * GROUP_N + (pid_in_group % GROUP_N)
                pid_k = group_k * GROUP_K + (pid_in_group // GROUP_N)
        elif SCHED_MODE == "GROUP_N":
            # Per K tile, run a contiguous block of N tiles: favors IA reuse.
            num_pid_n_groups = tl.cdiv(num_pid_n, GROUP_N)
            pids_per_k = num_pid_n_groups * GROUP_N
            pid_k = pid // pids_per_k
            pid_in_k = pid % pids_per_k
            pid_n = (pid_in_k // GROUP_N) * GROUP_N + (pid_in_k % GROUP_N)
        else:
            pid_n = pid % num_pid_n
            pid_k = pid // num_pid_n
        return pid_n, pid_k

    @triton.jit
    def _mxfp8_wgrad_direct_kernel(
        GO_ptr, GO_stride_n, GO_stride_m,
        GO_scales_ptr, GO_scales_stride_n, GO_scales_stride_mb,
        IA_ptr, IA_stride_k, IA_stride_m,
        IA_scales_ptr, IA_scales_stride_k, IA_scales_stride_mb,
        C_ptr, C_stride_e, C_stride_n, C_stride_k,
        group_end_offsets_ptr,
        M, N, K,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        SCALE_BLOCK: tl.constexpr,
        # None = plain (N, M//32) / (K, M//32) scale layout.
        # "CDNA4_SCALE" = pre-shuffled (N//32, M) / (K//32, M); needs
        # BLOCK_M >= 256 so MX_SCALE_BLOCK_M >= 8 (unswizzle requires //8).
        SWIZZLE_MX_SCALE: tl.constexpr,
        # CDNA4 unswizzle variant (16 or 32); must match matrix_instr_nonkdim.
        SCALE_NONKDIM: tl.constexpr,
        GROUP_N: tl.constexpr,
        GROUP_K: tl.constexpr,
        SCHED_MODE: tl.constexpr,
    ):
        pid_tile = tl.program_id(0)
        pid_g = tl.program_id(1)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_k = tl.cdiv(K, BLOCK_K)
        pid_n, pid_k = _wgrad_pid_grid(
            pid_tile, num_pid_n, num_pid_k, GROUP_N, GROUP_K, SCHED_MODE
        )

        group_start = tl.load(group_end_offsets_ptr + pid_g - 1, mask=pid_g > 0, other=0)
        group_end = tl.load(group_end_offsets_ptr + pid_g)
        M_g = group_end - group_start

        n_base = pid_n * BLOCK_N
        k_base = pid_k * BLOCK_K
        if n_base >= N or k_base >= K:
            return

        n_offs = n_base + tl.arange(0, BLOCK_N)
        k_offs = k_base + tl.arange(0, BLOCK_K)
        n_mask = n_offs < N
        k_mask = k_offs < K

        MX_SCALE_BLOCK_M: tl.constexpr = BLOCK_M // SCALE_BLOCK
        M_SCALES = M // SCALE_BLOCK

        if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
            NON_K_PRESHUFFLE_BLOCK_SIZE: tl.constexpr = 32
            PACKED_MX_BLOCK_M: tl.constexpr = MX_SCALE_BLOCK_M * NON_K_PRESHUFFLE_BLOCK_SIZE  # == BLOCK_M
            SCALE_BLOCK_N: tl.constexpr = BLOCK_N // NON_K_PRESHUFFLE_BLOCK_SIZE
            SCALE_BLOCK_K_OUT: tl.constexpr = BLOCK_K // NON_K_PRESHUFFLE_BLOCK_SIZE
        else:
            PACKED_MX_BLOCK_M: tl.constexpr = MX_SCALE_BLOCK_M
            SCALE_BLOCK_N: tl.constexpr = BLOCK_N
            SCALE_BLOCK_K_OUT: tl.constexpr = BLOCK_K

        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        for m_iter in range(0, tl.cdiv(M_g, BLOCK_M)):
            m_base = group_start + m_iter * BLOCK_M
            m_offs = m_base + tl.arange(0, BLOCK_M)
            m_mask = m_offs < group_end

            go_tile = tl.load(
                GO_ptr + n_offs[:, None] * GO_stride_n + m_offs[None, :] * GO_stride_m,
                mask=n_mask[:, None] & m_mask[None, :], other=0.0,
            )
            ia_tile = tl.load(
                IA_ptr + k_offs[None, :] * IA_stride_k + m_offs[:, None] * IA_stride_m,
                mask=m_mask[:, None] & k_mask[None, :], other=0.0,
            )

            if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
                # Shuffled scales: GO (N//32, M), IA (K//32, M); per-iter tiles
                # (BLOCK_N//32, BLOCK_M) / (BLOCK_K//32, BLOCK_M).
                offs_go_n_s = pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)
                offs_go_m_s = m_base + tl.arange(0, PACKED_MX_BLOCK_M)
                go_scale_tile = tl.load(
                    GO_scales_ptr + offs_go_n_s[:, None] * GO_scales_stride_n
                                  + offs_go_m_s[None, :] * GO_scales_stride_mb,
                    mask=offs_go_m_s[None, :] < M, other=127,
                )
                offs_ia_k_s = pid_k * SCALE_BLOCK_K_OUT + tl.arange(0, SCALE_BLOCK_K_OUT)
                offs_ia_m_s = m_base + tl.arange(0, PACKED_MX_BLOCK_M)
                ia_scale_tile = tl.load(
                    IA_scales_ptr + offs_ia_k_s[:, None] * IA_scales_stride_k
                                  + offs_ia_m_s[None, :] * IA_scales_stride_mb,
                    mask=offs_ia_m_s[None, :] < M, other=127,
                )
                if SCALE_NONKDIM == 32:
                    go_scale = _unswizzle_mx_scale_cdna4_nonkdim32(
                        go_scale_tile, BLOCK_N, MX_SCALE_BLOCK_M,
                    )
                    ia_scale = _unswizzle_mx_scale_cdna4_nonkdim32(
                        ia_scale_tile, BLOCK_K, MX_SCALE_BLOCK_M,
                    )
                else:
                    go_scale = _unswizzle_mx_scale_cdna4(
                        go_scale_tile, BLOCK_N, MX_SCALE_BLOCK_M,
                    )
                    ia_scale = _unswizzle_mx_scale_cdna4(
                        ia_scale_tile, BLOCK_K, MX_SCALE_BLOCK_M,
                    )
            else:
                # Plain (N, M//32) / (K, M//32) layout.
                mb_base = m_base // SCALE_BLOCK
                mb_offs = mb_base + tl.arange(0, MX_SCALE_BLOCK_M)
                mb_mask = mb_offs < M_SCALES
                # other=127: e8m0 bias 127 = 2^0 = 1.0 (neutral scale).
                go_scale = tl.load(
                    GO_scales_ptr + n_offs[:, None] * GO_scales_stride_n + mb_offs[None, :] * GO_scales_stride_mb,
                    mask=n_mask[:, None] & mb_mask[None, :], other=127,
                )
                ia_scale = tl.load(
                    IA_scales_ptr + k_offs[:, None] * IA_scales_stride_k + mb_offs[None, :] * IA_scales_stride_mb,
                    mask=k_mask[:, None] & mb_mask[None, :], other=127,
                )

            acc = tl.dot_scaled(
                go_tile, go_scale, "e4m3",
                ia_tile, ia_scale, "e4m3",
                acc=acc, out_dtype=tl.float32, fast_math=True,
            )

        c_mask = n_mask[:, None] & k_mask[None, :]
        # int64 pid_g: for E*N*K > 2^31 (e.g. E=128, N=4096, K=7168),
        # pid_g*C_stride_e overflows int32 into a wild address.
        tl.store(
            C_ptr + pid_g.to(tl.int64) * C_stride_e
                  + n_offs[:, None] * C_stride_n
                  + k_offs[None, :] * C_stride_k,
            acc.to(tl.bfloat16), mask=c_mask,
        )

    @triton.jit
    def _mxfp8_wgrad_direct_k2_kernel(
        GO_ptr, GO_stride_n, GO_stride_m,
        GO_scales_ptr, GO_scales_stride_n, GO_scales_stride_mb,
        IA_ptr, IA_stride_k, IA_stride_m,
        IA_scales_ptr, IA_scales_stride_k, IA_scales_stride_mb,
        C_ptr, C_stride_e, C_stride_n, C_stride_k,
        group_end_offsets_ptr,
        M, N, K,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        SCALE_BLOCK: tl.constexpr,
    ):
        pid_tile = tl.program_id(0)
        pid_g = tl.program_id(1)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        pid_n = pid_tile % num_pid_n
        pid_k_pair = pid_tile // num_pid_n

        group_start = tl.load(group_end_offsets_ptr + pid_g - 1, mask=pid_g > 0, other=0)
        group_end = tl.load(group_end_offsets_ptr + pid_g)
        M_g = group_end - group_start

        n_base = pid_n * BLOCK_N
        k_base0 = (pid_k_pair * 2) * BLOCK_K
        k_base1 = k_base0 + BLOCK_K
        if n_base >= N or k_base0 >= K:
            return

        n_offs = n_base + tl.arange(0, BLOCK_N)
        k_offs0 = k_base0 + tl.arange(0, BLOCK_K)
        k_offs1 = k_base1 + tl.arange(0, BLOCK_K)
        n_mask = n_offs < N
        k_mask0 = k_offs0 < K
        k_mask1 = k_offs1 < K

        MX_SCALE_BLOCK_M: tl.constexpr = BLOCK_M // SCALE_BLOCK
        M_SCALES = M // SCALE_BLOCK
        acc0 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        for m_iter in range(0, tl.cdiv(M_g, BLOCK_M)):
            m_base = group_start + m_iter * BLOCK_M
            m_offs = m_base + tl.arange(0, BLOCK_M)
            m_mask = m_offs < group_end

            go_tile = tl.load(
                GO_ptr + n_offs[:, None] * GO_stride_n + m_offs[None, :] * GO_stride_m,
                mask=n_mask[:, None] & m_mask[None, :], other=0.0,
            )
            ia_tile0 = tl.load(
                IA_ptr + k_offs0[None, :] * IA_stride_k + m_offs[:, None] * IA_stride_m,
                mask=m_mask[:, None] & k_mask0[None, :], other=0.0,
            )
            ia_tile1 = tl.load(
                IA_ptr + k_offs1[None, :] * IA_stride_k + m_offs[:, None] * IA_stride_m,
                mask=m_mask[:, None] & k_mask1[None, :], other=0.0,
            )

            mb_base = m_base // SCALE_BLOCK
            mb_offs = mb_base + tl.arange(0, MX_SCALE_BLOCK_M)
            mb_mask = mb_offs < M_SCALES
            go_scale = tl.load(
                GO_scales_ptr + n_offs[:, None] * GO_scales_stride_n + mb_offs[None, :] * GO_scales_stride_mb,
                mask=n_mask[:, None] & mb_mask[None, :], other=127,
            )
            ia_scale0 = tl.load(
                IA_scales_ptr + k_offs0[:, None] * IA_scales_stride_k + mb_offs[None, :] * IA_scales_stride_mb,
                mask=k_mask0[:, None] & mb_mask[None, :], other=127,
            )
            ia_scale1 = tl.load(
                IA_scales_ptr + k_offs1[:, None] * IA_scales_stride_k + mb_offs[None, :] * IA_scales_stride_mb,
                mask=k_mask1[:, None] & mb_mask[None, :], other=127,
            )

            acc0 = tl.dot_scaled(
                go_tile, go_scale, "e4m3",
                ia_tile0, ia_scale0, "e4m3",
                acc=acc0, out_dtype=tl.float32, fast_math=True,
            )
            acc1 = tl.dot_scaled(
                go_tile, go_scale, "e4m3",
                ia_tile1, ia_scale1, "e4m3",
                acc=acc1, out_dtype=tl.float32, fast_math=True,
            )

        tl.store(
            C_ptr + pid_g.to(tl.int64) * C_stride_e
                  + n_offs[:, None] * C_stride_n
                  + k_offs0[None, :] * C_stride_k,
            acc0.to(tl.bfloat16), mask=n_mask[:, None] & k_mask0[None, :],
        )
        tl.store(
            C_ptr + pid_g.to(tl.int64) * C_stride_e
                  + n_offs[:, None] * C_stride_n
                  + k_offs1[None, :] * C_stride_k,
            acc1.to(tl.bfloat16), mask=n_mask[:, None] & k_mask1[None, :],
        )

    @triton.jit
    def _mxfp8_wgrad_partial_kernel(
        GO_ptr, GO_stride_n, GO_stride_m,
        GO_scales_ptr, GO_scales_stride_n, GO_scales_stride_mb,
        IA_ptr, IA_stride_k, IA_stride_m,
        IA_scales_ptr, IA_scales_stride_k, IA_scales_stride_mb,
        P_ptr, P_stride_e, P_stride_s, P_stride_n, P_stride_k,
        group_end_offsets_ptr,
        M, N, K,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        SPLIT_M: tl.constexpr,
        SCALE_BLOCK: tl.constexpr,
        SWIZZLE_MX_SCALE: tl.constexpr,
        SCALE_NONKDIM: tl.constexpr,
        GROUP_N: tl.constexpr,
        GROUP_K: tl.constexpr,
        SCHED_MODE: tl.constexpr,
    ):
        pid_tile = tl.program_id(0)
        pid_eg = tl.program_id(1)
        pid_split = pid_eg % SPLIT_M
        pid_g = pid_eg // SPLIT_M
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_k = tl.cdiv(K, BLOCK_K)
        pid_n, pid_k = _wgrad_pid_grid(
            pid_tile, num_pid_n, num_pid_k, GROUP_N, GROUP_K, SCHED_MODE
        )

        group_start = tl.load(group_end_offsets_ptr + pid_g - 1, mask=pid_g > 0, other=0)
        group_end = tl.load(group_end_offsets_ptr + pid_g)
        M_g = group_end - group_start

        n_base = pid_n * BLOCK_N
        k_base = pid_k * BLOCK_K
        if n_base >= N or k_base >= K:
            return

        n_offs = n_base + tl.arange(0, BLOCK_N)
        k_offs = k_base + tl.arange(0, BLOCK_K)
        n_mask = n_offs < N
        k_mask = k_offs < K

        MX_SCALE_BLOCK_M: tl.constexpr = BLOCK_M // SCALE_BLOCK
        M_SCALES = M // SCALE_BLOCK

        if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
            NON_K_PRESHUFFLE_BLOCK_SIZE: tl.constexpr = 32
            PACKED_MX_BLOCK_M: tl.constexpr = MX_SCALE_BLOCK_M * NON_K_PRESHUFFLE_BLOCK_SIZE
            SCALE_BLOCK_N: tl.constexpr = BLOCK_N // NON_K_PRESHUFFLE_BLOCK_SIZE
            SCALE_BLOCK_K_OUT: tl.constexpr = BLOCK_K // NON_K_PRESHUFFLE_BLOCK_SIZE
        else:
            PACKED_MX_BLOCK_M: tl.constexpr = MX_SCALE_BLOCK_M
            SCALE_BLOCK_N: tl.constexpr = BLOCK_N
            SCALE_BLOCK_K_OUT: tl.constexpr = BLOCK_K

        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        num_m_iters = tl.cdiv(M_g, BLOCK_M)
        for m_iter in range(pid_split, num_m_iters, SPLIT_M):
            m_base = group_start + m_iter * BLOCK_M
            m_offs = m_base + tl.arange(0, BLOCK_M)
            m_mask = m_offs < group_end

            go_tile = tl.load(
                GO_ptr + n_offs[:, None] * GO_stride_n + m_offs[None, :] * GO_stride_m,
                mask=n_mask[:, None] & m_mask[None, :], other=0.0,
            )
            ia_tile = tl.load(
                IA_ptr + k_offs[None, :] * IA_stride_k + m_offs[:, None] * IA_stride_m,
                mask=m_mask[:, None] & k_mask[None, :], other=0.0,
            )

            if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
                offs_go_n_s = pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)
                offs_go_m_s = m_base + tl.arange(0, PACKED_MX_BLOCK_M)
                go_scale_tile = tl.load(
                    GO_scales_ptr + offs_go_n_s[:, None] * GO_scales_stride_n
                                  + offs_go_m_s[None, :] * GO_scales_stride_mb,
                    mask=offs_go_m_s[None, :] < M, other=127,
                )
                offs_ia_k_s = pid_k * SCALE_BLOCK_K_OUT + tl.arange(0, SCALE_BLOCK_K_OUT)
                offs_ia_m_s = m_base + tl.arange(0, PACKED_MX_BLOCK_M)
                ia_scale_tile = tl.load(
                    IA_scales_ptr + offs_ia_k_s[:, None] * IA_scales_stride_k
                                  + offs_ia_m_s[None, :] * IA_scales_stride_mb,
                    mask=offs_ia_m_s[None, :] < M, other=127,
                )
                if SCALE_NONKDIM == 32:
                    go_scale = _unswizzle_mx_scale_cdna4_nonkdim32(
                        go_scale_tile, BLOCK_N, MX_SCALE_BLOCK_M,
                    )
                    ia_scale = _unswizzle_mx_scale_cdna4_nonkdim32(
                        ia_scale_tile, BLOCK_K, MX_SCALE_BLOCK_M,
                    )
                else:
                    go_scale = _unswizzle_mx_scale_cdna4(
                        go_scale_tile, BLOCK_N, MX_SCALE_BLOCK_M,
                    )
                    ia_scale = _unswizzle_mx_scale_cdna4(
                        ia_scale_tile, BLOCK_K, MX_SCALE_BLOCK_M,
                    )
            else:
                mb_base = m_base // SCALE_BLOCK
                mb_offs = mb_base + tl.arange(0, MX_SCALE_BLOCK_M)
                mb_mask = mb_offs < M_SCALES
                go_scale = tl.load(
                    GO_scales_ptr + n_offs[:, None] * GO_scales_stride_n + mb_offs[None, :] * GO_scales_stride_mb,
                    mask=n_mask[:, None] & mb_mask[None, :], other=127,
                )
                ia_scale = tl.load(
                    IA_scales_ptr + k_offs[:, None] * IA_scales_stride_k + mb_offs[None, :] * IA_scales_stride_mb,
                    mask=k_mask[:, None] & mb_mask[None, :], other=127,
                )

            acc = tl.dot_scaled(
                go_tile, go_scale, "e4m3",
                ia_tile, ia_scale, "e4m3",
                acc=acc, out_dtype=tl.float32, fast_math=True,
            )

        p_mask = n_mask[:, None] & k_mask[None, :]
        tl.store(
            P_ptr + pid_g.to(tl.int64) * P_stride_e
                  + pid_split * P_stride_s
                  + n_offs[:, None] * P_stride_n
                  + k_offs[None, :] * P_stride_k,
            acc, mask=p_mask,
        )

    @triton.jit
    def _mxfp8_wgrad_reduce_kernel(
        P_ptr, P_stride_e, P_stride_s, P_stride_n, P_stride_k,
        C_ptr, C_stride_e, C_stride_n, C_stride_k,
        N, K,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_M: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_g = tl.program_id(2)

        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        n_mask = n_offs < N
        k_mask = k_offs < K
        c_mask = n_mask[:, None] & k_mask[None, :]

        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for s in range(SPLIT_M):
            acc += tl.load(
                P_ptr + pid_g.to(tl.int64) * P_stride_e
                      + s * P_stride_s
                      + n_offs[:, None] * P_stride_n
                      + k_offs[None, :] * P_stride_k,
                mask=c_mask, other=0.0,
            )

        tl.store(
            C_ptr + pid_g.to(tl.int64) * C_stride_e
                  + n_offs[:, None] * C_stride_n
                  + k_offs[None, :] * C_stride_k,
            acc.to(tl.bfloat16), mask=c_mask,
        )

    @triton.jit
    def _mxfp8_wgrad_atomic_split_kernel(
        GO_ptr, GO_stride_n, GO_stride_m,
        GO_scales_ptr, GO_scales_stride_n, GO_scales_stride_mb,
        IA_ptr, IA_stride_k, IA_stride_m,
        IA_scales_ptr, IA_scales_stride_k, IA_scales_stride_mb,
        A_ptr, A_stride_e, A_stride_n, A_stride_k,
        group_end_offsets_ptr,
        M, N, K,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        SPLIT_M: tl.constexpr,
        SCALE_BLOCK: tl.constexpr,
        SWIZZLE_MX_SCALE: tl.constexpr,
        SCALE_NONKDIM: tl.constexpr,
        GROUP_N: tl.constexpr,
        GROUP_K: tl.constexpr,
        SCHED_MODE: tl.constexpr,
    ):
        pid_tile = tl.program_id(0)
        pid_eg = tl.program_id(1)
        pid_split = pid_eg % SPLIT_M
        pid_g = pid_eg // SPLIT_M
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_k = tl.cdiv(K, BLOCK_K)
        pid_n, pid_k = _wgrad_pid_grid(
            pid_tile, num_pid_n, num_pid_k, GROUP_N, GROUP_K, SCHED_MODE
        )

        group_start = tl.load(group_end_offsets_ptr + pid_g - 1, mask=pid_g > 0, other=0)
        group_end = tl.load(group_end_offsets_ptr + pid_g)
        M_g = group_end - group_start

        n_base = pid_n * BLOCK_N
        k_base = pid_k * BLOCK_K
        if n_base >= N or k_base >= K:
            return

        n_offs = n_base + tl.arange(0, BLOCK_N)
        k_offs = k_base + tl.arange(0, BLOCK_K)
        n_mask = n_offs < N
        k_mask = k_offs < K

        MX_SCALE_BLOCK_M: tl.constexpr = BLOCK_M // SCALE_BLOCK
        M_SCALES = M // SCALE_BLOCK

        if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
            NON_K_PRESHUFFLE_BLOCK_SIZE: tl.constexpr = 32
            PACKED_MX_BLOCK_M: tl.constexpr = MX_SCALE_BLOCK_M * NON_K_PRESHUFFLE_BLOCK_SIZE
            SCALE_BLOCK_N: tl.constexpr = BLOCK_N // NON_K_PRESHUFFLE_BLOCK_SIZE
            SCALE_BLOCK_K_OUT: tl.constexpr = BLOCK_K // NON_K_PRESHUFFLE_BLOCK_SIZE
        else:
            PACKED_MX_BLOCK_M: tl.constexpr = MX_SCALE_BLOCK_M
            SCALE_BLOCK_N: tl.constexpr = BLOCK_N
            SCALE_BLOCK_K_OUT: tl.constexpr = BLOCK_K

        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        num_m_iters = tl.cdiv(M_g, BLOCK_M)
        for m_iter in range(pid_split, num_m_iters, SPLIT_M):
            m_base = group_start + m_iter * BLOCK_M
            m_offs = m_base + tl.arange(0, BLOCK_M)
            m_mask = m_offs < group_end

            go_tile = tl.load(
                GO_ptr + n_offs[:, None] * GO_stride_n + m_offs[None, :] * GO_stride_m,
                mask=n_mask[:, None] & m_mask[None, :], other=0.0,
            )
            ia_tile = tl.load(
                IA_ptr + k_offs[None, :] * IA_stride_k + m_offs[:, None] * IA_stride_m,
                mask=m_mask[:, None] & k_mask[None, :], other=0.0,
            )

            if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
                offs_go_n_s = pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)
                offs_go_m_s = m_base + tl.arange(0, PACKED_MX_BLOCK_M)
                go_scale_tile = tl.load(
                    GO_scales_ptr + offs_go_n_s[:, None] * GO_scales_stride_n
                                  + offs_go_m_s[None, :] * GO_scales_stride_mb,
                    mask=offs_go_m_s[None, :] < M, other=127,
                )
                offs_ia_k_s = pid_k * SCALE_BLOCK_K_OUT + tl.arange(0, SCALE_BLOCK_K_OUT)
                offs_ia_m_s = m_base + tl.arange(0, PACKED_MX_BLOCK_M)
                ia_scale_tile = tl.load(
                    IA_scales_ptr + offs_ia_k_s[:, None] * IA_scales_stride_k
                                  + offs_ia_m_s[None, :] * IA_scales_stride_mb,
                    mask=offs_ia_m_s[None, :] < M, other=127,
                )
                if SCALE_NONKDIM == 32:
                    go_scale = _unswizzle_mx_scale_cdna4_nonkdim32(
                        go_scale_tile, BLOCK_N, MX_SCALE_BLOCK_M,
                    )
                    ia_scale = _unswizzle_mx_scale_cdna4_nonkdim32(
                        ia_scale_tile, BLOCK_K, MX_SCALE_BLOCK_M,
                    )
                else:
                    go_scale = _unswizzle_mx_scale_cdna4(
                        go_scale_tile, BLOCK_N, MX_SCALE_BLOCK_M,
                    )
                    ia_scale = _unswizzle_mx_scale_cdna4(
                        ia_scale_tile, BLOCK_K, MX_SCALE_BLOCK_M,
                    )
            else:
                mb_base = m_base // SCALE_BLOCK
                mb_offs = mb_base + tl.arange(0, MX_SCALE_BLOCK_M)
                mb_mask = mb_offs < M_SCALES
                go_scale = tl.load(
                    GO_scales_ptr + n_offs[:, None] * GO_scales_stride_n + mb_offs[None, :] * GO_scales_stride_mb,
                    mask=n_mask[:, None] & mb_mask[None, :], other=127,
                )
                ia_scale = tl.load(
                    IA_scales_ptr + k_offs[:, None] * IA_scales_stride_k + mb_offs[None, :] * IA_scales_stride_mb,
                    mask=k_mask[:, None] & mb_mask[None, :], other=127,
                )

            acc = tl.dot_scaled(
                go_tile, go_scale, "e4m3",
                ia_tile, ia_scale, "e4m3",
                acc=acc, out_dtype=tl.float32, fast_math=True,
            )

        mask = n_mask[:, None] & k_mask[None, :]
        tl.atomic_add(
            A_ptr + pid_g.to(tl.int64) * A_stride_e
                  + n_offs[:, None] * A_stride_n
                  + k_offs[None, :] * A_stride_k,
            acc, sem="relaxed", mask=mask,
        )

    @triton.jit
    def _mxfp8_wgrad_finalize_kernel(
        A_ptr, A_stride_e, A_stride_n, A_stride_k,
        C_ptr, C_stride_e, C_stride_n, C_stride_k,
        N, K,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_g = tl.program_id(2)

        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = (n_offs[:, None] < N) & (k_offs[None, :] < K)
        acc = tl.load(
            A_ptr + pid_g.to(tl.int64) * A_stride_e
                  + n_offs[:, None] * A_stride_n
                  + k_offs[None, :] * A_stride_k,
            mask=mask, other=0.0,
        )
        tl.store(
            C_ptr + pid_g.to(tl.int64) * C_stride_e
                  + n_offs[:, None] * C_stride_n
                  + k_offs[None, :] * C_stride_k,
            acc.to(tl.bfloat16), mask=mask,
        )

    # Per-shape best configs from a 15-shape x 192-config DSv3 sweep on MI355X.
    # Search space: BLOCK_M in {32,64,128}, BLOCK_N,BLOCK_K in {128,256},
    # num_warps in {4,8}, num_stages in {1,2}, nonkdim in {16,32},
    # waves_per_eu in {0,2}. BLOCK_M=64 and nonkdim=32 won every shape.
    # Trailing annotations are the winning config's latency / throughput.
    _BEST_CFGS_WGRAD = {
        (8, 2048, 7168): dict(BLOCK_M=64, BLOCK_N=256, BLOCK_K=256, num_warps=8, num_stages=2, matrix_instr_nonkdim=32, waves_per_eu=0),  # 387.6us 620.5TF
        (8, 4096, 7168): dict(BLOCK_M=64, BLOCK_N=256, BLOCK_K=256, num_warps=8, num_stages=2, matrix_instr_nonkdim=32, waves_per_eu=0),  # 734.3us 655.1TF
        (8, 7168, 2048): dict(BLOCK_M=64, BLOCK_N=256, BLOCK_K=256, num_warps=8, num_stages=2, matrix_instr_nonkdim=32, waves_per_eu=0),  # 364.6us 659.7TF
        (16, 2048, 7168): dict(BLOCK_M=64, BLOCK_N=256, BLOCK_K=128, num_warps=4, num_stages=1, matrix_instr_nonkdim=32, waves_per_eu=2),  # 731.8us 657.3TF
        (16, 4096, 7168): dict(BLOCK_M=64, BLOCK_N=256, BLOCK_K=128, num_warps=4, num_stages=1, matrix_instr_nonkdim=32, waves_per_eu=0),  # 1410.2us 682.2TF
        (16, 7168, 2048): dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=256, num_warps=4, num_stages=1, matrix_instr_nonkdim=32, waves_per_eu=2),  # 718.0us 669.9TF
        (32, 2048, 7168): dict(BLOCK_M=64, BLOCK_N=256, BLOCK_K=128, num_warps=4, num_stages=1, matrix_instr_nonkdim=32, waves_per_eu=2),  # 1544.7us 622.8TF
        (32, 4096, 7168): dict(BLOCK_M=64, BLOCK_N=256, BLOCK_K=128, num_warps=4, num_stages=1, matrix_instr_nonkdim=32, waves_per_eu=2),  # 2898.4us 663.9TF
        (32, 7168, 2048): dict(BLOCK_M=64, BLOCK_N=256, BLOCK_K=256, num_warps=8, num_stages=2, matrix_instr_nonkdim=32, waves_per_eu=0),  # 1485.8us 647.5TF
        (64, 2048, 7168): dict(BLOCK_M=64, BLOCK_N=256, BLOCK_K=128, num_warps=4, num_stages=1, matrix_instr_nonkdim=32, waves_per_eu=2),  # 3046.8us 631.5TF
        (64, 4096, 7168): dict(BLOCK_M=64, BLOCK_N=256, BLOCK_K=128, num_warps=4, num_stages=1, matrix_instr_nonkdim=32, waves_per_eu=2),  # 5800.5us 663.4TF
        (64, 7168, 2048): dict(BLOCK_M=64, BLOCK_N=256, BLOCK_K=256, num_warps=8, num_stages=2, matrix_instr_nonkdim=32, waves_per_eu=0),  # 2887.4us 666.4TF
        (128, 2048, 7168): dict(BLOCK_M=64, BLOCK_N=256, BLOCK_K=256, num_warps=8, num_stages=2, matrix_instr_nonkdim=32, waves_per_eu=0),  # 6296.1us 611.2TF
        (128, 4096, 7168): dict(BLOCK_M=64, BLOCK_N=256, BLOCK_K=128, num_warps=4, num_stages=1, matrix_instr_nonkdim=32, waves_per_eu=2),  # 12020.3us 640.3TF
        (128, 7168, 2048): dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=256, num_warps=4, num_stages=1, matrix_instr_nonkdim=32, waves_per_eu=2),  # 5938.5us 648.0TF
    }

    # Fallback for unseen shapes. Two clusters from the sweep:
    # - Balanced (small-E): large blocks, deep pipeline (warps=8, stages=2).
    # - Asymmetric (large-E): smaller K block, shallow pipeline (warps=4,
    #   stages=1, wpe=2).
    _FALLBACK_BALANCED = dict(BLOCK_M=64, BLOCK_N=256, BLOCK_K=256, num_warps=8, num_stages=2, matrix_instr_nonkdim=32, waves_per_eu=0)
    _FALLBACK_ASYM = dict(BLOCK_M=64, BLOCK_N=256, BLOCK_K=128, num_warps=4, num_stages=1, matrix_instr_nonkdim=32, waves_per_eu=2)

    def _pick_config_wgrad(E: int, N: int, K: int) -> dict:
        cfg = _BEST_CFGS_WGRAD.get((E, N, K))
        if cfg is not None:
            return cfg
        # Large E or N != K -> asymmetric cluster; else balanced.
        return _FALLBACK_ASYM if (E >= 16 or N != K) else _FALLBACK_BALANCED

    def _pick_split_m_wgrad(
        E: int,
        M: int,
        N: int,
        K: int,
        BLOCK_N: int,
        BLOCK_K: int,
        BLOCK_M: int,
        max_partial_bytes: int = 1 << 30,
    ) -> int:
        base_ctas = triton.cdiv(N, BLOCK_N) * triton.cdiv(K, BLOCK_K) * E
        m_iters_per_group = triton.cdiv(triton.cdiv(M, E), BLOCK_M)
        if base_ctas < 1024 and m_iters_per_group >= 64:
            split_m = 4
        elif base_ctas < 512 and m_iters_per_group >= 32:
            split_m = 2
        else:
            split_m = 1

        while split_m > 1 and E * split_m * N * K * 4 > max_partial_bytes:
            split_m //= 2
        return split_m

    def triton_mxfp8_wgrad(
        go_t: torch.Tensor,
        go_scale: torch.Tensor,
        ia_t: torch.Tensor,
        ia_scale: torch.Tensor,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
        # None tunables are looked up via _pick_config_wgrad(E,N,K); an explicit
        # value overrides the lookup.
        BLOCK_N: int = None,
        BLOCK_K: int = None,
        BLOCK_M: int = None,
        num_warps: int = None,
        num_stages: int = None,
        matrix_instr_nonkdim: int = None,
        kpack: int = 1,
        waves_per_eu: int = None,
        GROUP_N: int = None,
        GROUP_K: int = None,
        sched_mode: str = None,
        split_m: int = None,
        cta_group_k: int = 1,
        split_reduce_mode: str = "partials",
        max_split_m_partial_bytes: int = 1 << 30,
    ) -> torch.Tensor:
        """MXFP8 weight gradient: ``grad_W[g] = grad_output[group_g]^T @ input_act[group_g]``.

        Both inputs must be dim1-quantized (scales along the M / token dim).
        Uses a single-pass CTA grid by default for large grids, and can split
        the M reduction across CTAs for low-CTA shapes.

        Args:
            go_t: ``(N, M)`` fp8.
            ia_t: ``(K, M)`` fp8.

        Returns:
            ``(E, N, K)`` bf16.
        """
        N, M = go_t.shape
        K, _ = ia_t.shape
        E = group_end_offsets.shape[0]
        SCALE_BLOCK = 32

        _cfg = _pick_config_wgrad(E, N, K)
        if BLOCK_M is None: BLOCK_M = _cfg["BLOCK_M"]
        if BLOCK_N is None: BLOCK_N = _cfg["BLOCK_N"]
        if BLOCK_K is None: BLOCK_K = _cfg["BLOCK_K"]
        if num_warps is None: num_warps = _cfg["num_warps"]
        if num_stages is None: num_stages = _cfg["num_stages"]
        if matrix_instr_nonkdim is None: matrix_instr_nonkdim = _cfg["matrix_instr_nonkdim"]
        if waves_per_eu is None: waves_per_eu = _cfg["waves_per_eu"]
        if split_m is None:
            split_m = _pick_split_m_wgrad(
                E, M, N, K, BLOCK_N, BLOCK_K, BLOCK_M,
                max_partial_bytes=max_split_m_partial_bytes,
            )
        if split_m < 1 or split_m & (split_m - 1):
            raise ValueError(f"split_m must be a power of two >= 1, got {split_m}")
        if cta_group_k not in (1, 2):
            raise ValueError(f"cta_group_k must be 1 or 2, got {cta_group_k}")
        if split_reduce_mode not in ("partials", "atomic"):
            raise ValueError(
                f"split_reduce_mode must be 'partials' or 'atomic', got {split_reduce_mode!r}"
            )
        if sched_mode is None:
            num_pid_n = triton.cdiv(N, BLOCK_N)
            num_pid_k = triton.cdiv(K, BLOCK_K)
            # Two L2-reuse directions: IA across N tiles, GO across K tiles.
            # A rectangular cluster exploits both before walking far in either
            # axis; pick the cluster shape from the grid aspect ratio.
            if num_pid_n > 1 and num_pid_k > 1:
                if GROUP_N is None:
                    GROUP_N = 2 if num_pid_n <= 8 else 4
                    GROUP_N = min(GROUP_N, num_pid_n)
                if GROUP_K is None:
                    GROUP_K = 4
                    GROUP_K = min(GROUP_K, num_pid_k)
                if num_pid_k > 2 * num_pid_n and GROUP_K > 1:
                    sched_mode = "GROUP_K"
                elif num_pid_n > 2 * num_pid_k and (GROUP_N > 1 or GROUP_K > 1):
                    sched_mode = "GROUP_NK"
                elif GROUP_N > 1 or GROUP_K > 1:
                    sched_mode = "GROUP_NK_K"
                else:
                    sched_mode = "NONE"
            else:
                sched_mode = "NONE"
        if GROUP_N is None: GROUP_N = 1
        if GROUP_K is None: GROUP_K = 1

        # CDNA4_SCALE: pre-shuffle scales into the native MFMA layout so each
        # thread loads one coalesced block instead of a permute chain.
        # The shuffle packs 8 scale-rows x 32 N-pack = 256 bytes into one outer
        # M-block, so BLOCK_M and every group_start must be multiples of 256 to
        # land on outer-block boundaries (non-256 starts corrupt g>=1; g=0 is
        # safe since group_start=0). Cheap scalar gates short-circuit before the
        # ~30us offs sync, which matters for the smallest shapes.
        use_cdna4_scale = (
            BLOCK_M % 256 == 0
            and M % 256 == 0
            and N % 32 == 0
            and K % 32 == 0
            and bool(((group_end_offsets % 256) == 0).all().item())
        )
        go_scale_u8 = go_scale.view(torch.uint8)
        ia_scale_u8 = ia_scale.view(torch.uint8)

        if use_cdna4_scale:
            if matrix_instr_nonkdim == 32:
                go_scale_arg = _shuffle_x_scales_cdna4_nonkdim32(go_scale_u8)
                ia_scale_arg = _shuffle_x_scales_cdna4_nonkdim32(ia_scale_u8)
            else:
                go_scale_arg = _shuffle_x_scales_cdna4_nonkdim16(go_scale_u8)
                ia_scale_arg = _shuffle_x_scales_cdna4_nonkdim16(ia_scale_u8)
            swizzle = "CDNA4_SCALE"
        else:
            go_scale_arg = go_scale_u8
            ia_scale_arg = ia_scale_u8
            swizzle = None

        output = torch.empty((E, N, K), dtype=out_dtype, device=go_t.device)

        num_pid_n = triton.cdiv(N, BLOCK_N)
        num_pid_k = triton.cdiv(K, BLOCK_K)
        if sched_mode == "GROUP_K":
            grid_tiles = num_pid_n * triton.cdiv(num_pid_k, GROUP_K) * GROUP_K
        elif sched_mode == "GROUP_NK" or sched_mode == "GROUP_NK_K":
            grid_tiles = (
                triton.cdiv(num_pid_n, GROUP_N)
                * triton.cdiv(num_pid_k, GROUP_K)
                * GROUP_N
                * GROUP_K
            )
        elif sched_mode == "GROUP_N":
            grid_tiles = num_pid_k * triton.cdiv(num_pid_n, GROUP_N) * GROUP_N
        else:
            grid_tiles = num_pid_n * num_pid_k
        grid = (grid_tiles, E)
        if split_m > 1:
            if split_reduce_mode == "atomic":
                accum = torch.empty((E, N, K), dtype=torch.float32, device=go_t.device)
                accum.zero_()
                _mxfp8_wgrad_atomic_split_kernel[(grid_tiles, E * split_m)](
                    go_t, go_t.stride(0), go_t.stride(1),
                    go_scale_arg,
                    go_scale_arg.stride(0), go_scale_arg.stride(1),
                    ia_t, ia_t.stride(0), ia_t.stride(1),
                    ia_scale_arg,
                    ia_scale_arg.stride(0), ia_scale_arg.stride(1),
                    accum,
                    accum.stride(0), accum.stride(1), accum.stride(2),
                    group_end_offsets,
                    M, N, K,
                    BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_M=BLOCK_M,
                    SPLIT_M=split_m,
                    SCALE_BLOCK=SCALE_BLOCK,
                    SWIZZLE_MX_SCALE=swizzle,
                    SCALE_NONKDIM=matrix_instr_nonkdim,
                    GROUP_N=GROUP_N,
                    GROUP_K=GROUP_K,
                    SCHED_MODE=sched_mode,
                    num_warps=num_warps, num_stages=num_stages,
                    matrix_instr_nonkdim=matrix_instr_nonkdim,
                    kpack=kpack,
                    waves_per_eu=waves_per_eu,
                )
                _mxfp8_wgrad_finalize_kernel[(num_pid_n, num_pid_k, E)](
                    accum,
                    accum.stride(0), accum.stride(1), accum.stride(2),
                    output,
                    output.stride(0), output.stride(1), output.stride(2),
                    N, K,
                    BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                    num_warps=num_warps, num_stages=num_stages,
                )
                return output

            partials = torch.empty(
                (E, split_m, N, K), dtype=torch.float32, device=go_t.device
            )
            _mxfp8_wgrad_partial_kernel[(grid_tiles, E * split_m)](
                go_t, go_t.stride(0), go_t.stride(1),
                go_scale_arg,
                go_scale_arg.stride(0), go_scale_arg.stride(1),
                ia_t, ia_t.stride(0), ia_t.stride(1),
                ia_scale_arg,
                ia_scale_arg.stride(0), ia_scale_arg.stride(1),
                partials,
                partials.stride(0), partials.stride(1),
                partials.stride(2), partials.stride(3),
                group_end_offsets,
                M, N, K,
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_M=BLOCK_M,
                SPLIT_M=split_m,
                SCALE_BLOCK=SCALE_BLOCK,
                SWIZZLE_MX_SCALE=swizzle,
                SCALE_NONKDIM=matrix_instr_nonkdim,
                GROUP_N=GROUP_N,
                GROUP_K=GROUP_K,
                SCHED_MODE=sched_mode,
                num_warps=num_warps, num_stages=num_stages,
                matrix_instr_nonkdim=matrix_instr_nonkdim,
                kpack=kpack,
                waves_per_eu=waves_per_eu,
            )
            _mxfp8_wgrad_reduce_kernel[(num_pid_n, num_pid_k, E)](
                partials,
                partials.stride(0), partials.stride(1),
                partials.stride(2), partials.stride(3),
                output,
                output.stride(0), output.stride(1), output.stride(2),
                N, K,
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SPLIT_M=split_m,
                num_warps=num_warps, num_stages=num_stages,
            )
            return output

        if cta_group_k == 2 and swizzle is None:
            grid_k2 = (num_pid_n * triton.cdiv(num_pid_k, 2), E)
            _mxfp8_wgrad_direct_k2_kernel[grid_k2](
                go_t, go_t.stride(0), go_t.stride(1),
                go_scale_arg,
                go_scale_arg.stride(0), go_scale_arg.stride(1),
                ia_t, ia_t.stride(0), ia_t.stride(1),
                ia_scale_arg,
                ia_scale_arg.stride(0), ia_scale_arg.stride(1),
                output, output.stride(0), output.stride(1), output.stride(2),
                group_end_offsets,
                M, N, K,
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_M=BLOCK_M,
                SCALE_BLOCK=SCALE_BLOCK,
                num_warps=num_warps, num_stages=num_stages,
                matrix_instr_nonkdim=matrix_instr_nonkdim,
                kpack=kpack,
                waves_per_eu=waves_per_eu,
            )
            return output

        _mxfp8_wgrad_direct_kernel[grid](
            go_t, go_t.stride(0), go_t.stride(1),
            go_scale_arg,
            go_scale_arg.stride(0), go_scale_arg.stride(1),
            ia_t, ia_t.stride(0), ia_t.stride(1),
            ia_scale_arg,
            ia_scale_arg.stride(0), ia_scale_arg.stride(1),
            output, output.stride(0), output.stride(1), output.stride(2),
            group_end_offsets,
            M, N, K,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_M=BLOCK_M,
            SCALE_BLOCK=SCALE_BLOCK,
            SWIZZLE_MX_SCALE=swizzle,
            SCALE_NONKDIM=matrix_instr_nonkdim,
            GROUP_N=GROUP_N,
            GROUP_K=GROUP_K,
            SCHED_MODE=sched_mode,
            num_warps=num_warps, num_stages=num_stages,
            matrix_instr_nonkdim=matrix_instr_nonkdim,
            kpack=kpack,
            waves_per_eu=waves_per_eu,
        )
        return output

else:
    _UNAVAILABLE_MSG = "ROCm MXFP8 kernels require gfx950 or later and triton"

    def triton_mxfp8_wgrad(*args, **kwargs):
        raise NotImplementedError(_UNAVAILABLE_MSG)

# ---------------------------------------------------------------------------
# Fast wgrad: computes grad_W via the forward kernel, one group at a time.
# The forward kernel is ~2x faster than the native backward kernel because
# it distributes work across CTAs more efficiently.
# ---------------------------------------------------------------------------
def triton_mxfp8_wgrad_fast(
    go_t: torch.Tensor,
    go_scale: torch.Tensor,
    ia_t: torch.Tensor,
    ia_scale: torch.Tensor,
    group_end_offsets: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Fast MXFP8 weight gradient using the forward kernel.

    Computes grad_W[g] = GO[g] @ IA[g]^T by calling the forward
    kernel E times (once per group). Each call maps:
      A = GO[g]  (N, M_g)   -> forward M_f=N, K_f=M_g
      B = IA[g]  (K, M_g)   -> forward B=(1, K, M_g)
    so the forward computes (N, M_g) @ (K, M_g)^T = (N, K) = grad_W[g].

    Returns:
        ``(E, N, K)`` bf16.
    """
    from kernels.mxfp8.forward import triton_mxfp8_grouped_mm

    N, M = go_t.shape
    K = ia_t.shape[0]
    E = group_end_offsets.shape[0]

    offs_cpu = group_end_offsets.cpu().tolist()
    out = go_t.new_empty((E, N, K), dtype=out_dtype)

    for g in range(E):
        g_start = offs_cpu[g - 1] if g > 0 else 0
        g_end = offs_cpu[g]
        M_g = g_end - g_start

        # A = GO[g]: (N, M_g) row-major fp8
        A_g = torch.narrow(go_t, 1, g_start, M_g)
        As_g = torch.narrow(go_scale, 1, g_start // 32, M_g // 32)

        # B = IA[g]: (1, K, M_g) for forward
        B_g = torch.narrow(ia_t, 1, g_start, M_g).unsqueeze(0)
        Bs_g = torch.narrow(ia_scale, 1, g_start // 32, M_g // 32).unsqueeze(0)

        # Single-group forward
        grp_offs = A_g.new_tensor([N], dtype=torch.int32)

        out[g] = triton_mxfp8_grouped_mm(
            A_g, B_g, As_g, Bs_g, grp_offs, out_dtype=out_dtype,
        )

    return out

# ---------------------------------------------------------------------------
# Fast wgrad v2: computes grad_W via the forward kernel.
#
# grad_W[g] = GO[g] @ IA[g]^T   maps to forward: A = stacked GO, B = stacked IA
#
# For DSv3 (uniform groups, Mg = M/E):
#   A = GO reshaped as (E*N, Mg) fp8
#   B = IA reshaped as (E, K, Mg) fp8
#   group_end_offsets = [N, 2N, ..., E*N]
# Forward computes: output = (E*N, K), which is exactly grad_W reshaped.
# ---------------------------------------------------------------------------
def triton_mxfp8_wgrad_v2(
    go_t: torch.Tensor,
    go_scale: torch.Tensor,
    ia_t: torch.Tensor,
    ia_scale: torch.Tensor,
    group_end_offsets: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Fast MXFP8 weight gradient via forward kernel.

    Re-formulates grad_W[g] = GO[g] @ IA[g]^T as a grouped forward call.
    When groups are uniform (Mg == M//E for all groups), uses a single
    batched forward call achieving ~0.95x of forward kernel performance.

    Returns:
        ``(E, N, K)`` bf16.
    """
    from kernels.mxfp8.forward import triton_mxfp8_grouped_mm

    N, M = go_t.shape
    K = ia_t.shape[0]
    E = group_end_offsets.shape[0]

    offs_cpu = group_end_offsets.cpu().tolist()

    # Check if all groups have uniform M_g
    Mg_first = offs_cpu[0] - 0
    uniform = all(
        (offs_cpu[g] - (offs_cpu[g - 1] if g > 0 else 0)) == Mg_first
        for g in range(E)
    )

    if uniform:
        Mg = Mg_first
        # Single forward call: stack per-group GO along dim 0, IA along dim 0
        A = go_t.new_empty((E * N, Mg))
        As = go_scale.new_empty((E * N, Mg // 32))
        B = ia_t.new_empty((E, K, Mg))
        Bs = ia_scale.new_empty((E, K, Mg // 32))
        for g in range(E):
            gs = g * Mg
            A[g * N : (g + 1) * N] = go_t[:, gs : gs + Mg]
            As[g * N : (g + 1) * N] = go_scale[:, gs // 32 : gs // 32 + Mg // 32]
            B[g] = ia_t[:, gs : gs + Mg]
            Bs[g] = ia_scale[:, gs // 32 : gs // 32 + Mg // 32]
        offs = go_t.new_tensor([N * (g + 1) for g in range(E)], dtype=torch.int32)

        result = triton_mxfp8_grouped_mm(A, B, As, Bs, offs, out_dtype=out_dtype)
        # result shape: (E*N, K) → reshape to (E, N, K)
        return result.view(E, N, K).contiguous()
    else:
        # Non-uniform groups: per-group fallback
        out = go_t.new_empty((E, N, K), dtype=out_dtype)
        for g in range(E):
            gs = offs_cpu[g - 1] if g > 0 else 0
            ge = offs_cpu[g]
            Mg = ge - gs
            A_g = go_t.narrow(1, gs, Mg)
            As_g = go_scale.narrow(1, gs // 32, Mg // 32)
            B_g = ia_t.narrow(1, gs, Mg).unsqueeze(0)
            Bs_g = ia_scale.narrow(1, gs // 32, Mg // 32).unsqueeze(0)
            offs_g = go_t.new_tensor([N], dtype=torch.int32)
            out[g] = triton_mxfp8_grouped_mm(
                A_g, B_g, As_g, Bs_g, offs_g, out_dtype=out_dtype,
            )
        return out

