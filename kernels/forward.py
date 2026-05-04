# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Forward / dgrad MXFP8 grouped-GEMM kernel for ROCm gfx950+.

Exports ``triton_mxfp8_grouped_mm`` — a persistent grouped GEMM used for the
MoE forward pass AND the dgrad (input gradient) pass, both of which are
A @ B^T per expert group. The weight-gradient path lives in ``backward.py``.

Scheduling (adapted from AMD aiter's moe_op_gemm_a8w8, which derives from
triton-lang's triton_kernels matmul_ogs):
  - XCD swizzle (ordered per-XCD launch for MI300/MI350 8-XCD parts)
  - GROUP_M pid reordering for L2 reuse
  - Per-tile expert lookup via a packed (block_id << 16) | expt_id map
  - CDNA4-native pre-shuffled scale layout (removes the #blocked -> #linear1
    permute chain the lowering would otherwise emit for MFMA scale loads)
"""

import torch

from ._common import _rocm_mxfp8_available

if _rocm_mxfp8_available:
    import triton
    import triton.language as tl

    @triton.jit
    def _xcd_swizzle(pid, domain_size, XCD_SWIZZLE: tl.constexpr):
        pids_per_group = domain_size // XCD_SWIZZLE
        extra_pid_groups = domain_size % XCD_SWIZZLE
        group = pid % XCD_SWIZZLE
        local_pid = pid // XCD_SWIZZLE
        return group * pids_per_group + min(group, extra_pid_groups) + local_pid

    @triton.jit
    def _pid_grid(pid, num_pid_m, num_pid_n, GROUP_M: tl.constexpr = 1):
        if GROUP_M == 1:
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n
        else:
            num_pid_in_group = GROUP_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            tl.assume(group_size_m >= 0)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
        return pid_m, pid_n

    @triton.jit
    def _unswizzle_mx_scale_cdna4(
        x,
        BLOCK_N: tl.constexpr,
        MX_SCALE_BLOCK_K: tl.constexpr,
        N_PRESHUFFLE_FACTOR: tl.constexpr = 32,
    ):
        """Inverse of host-side shuffle for nonkdim=16 MFMA."""
        x = x.reshape(
            BLOCK_N // N_PRESHUFFLE_FACTOR, MX_SCALE_BLOCK_K // 8, 4, 16, 2, 2, 1
        )
        x = x.permute(0, 5, 3, 1, 4, 2, 6)
        return x.reshape(BLOCK_N, MX_SCALE_BLOCK_K)

    @triton.jit
    def _unswizzle_mx_scale_cdna4_nonkdim32(
        x,
        BLOCK_N: tl.constexpr,
        MX_SCALE_BLOCK_K: tl.constexpr,
        N_PRESHUFFLE_FACTOR: tl.constexpr = 32,
    ):
        """Inverse of host-side shuffle for nonkdim=32 MFMA (from tutorial 10)."""
        x = x.reshape(
            BLOCK_N // N_PRESHUFFLE_FACTOR, MX_SCALE_BLOCK_K // 8, 2, 32, 4, 1
        )
        x = x.permute(0, 3, 1, 4, 2, 5)
        return x.reshape(BLOCK_N, MX_SCALE_BLOCK_K)

    @triton.jit
    def _mxfp8_grouped_mm_kernel(
        Y, stride_y_m, stride_y_n,
        X, stride_x_m, stride_x_k,
        XMxScale, stride_x_mx_m, stride_x_mx_k,
        W, stride_w_e, stride_w_k, stride_w_n,
        WMxScale, stride_w_mx_e, stride_w_mx_k, stride_w_mx_n,
        N, K,
        ExptHist,       # (E,) int32 - tokens per expert
        ExptOffs,       # (E,) int32 - start offset per expert (expert-sorted)
        ExptOffsSum,    # 0-d int32 - total tile blocks (sum of cdiv(hist[e], BLOCK_M))
        ExptData,       # (grid_m_max,) int32 - packed (block_id<<16)|expt_id, -1 pad
        grid_m, grid_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        XCD_SWIZZLE: tl.constexpr,
        # "CDNA4_SCALE" enables pre-shuffled MX scale layout; None = plain.
        SWIZZLE_MX_SCALE: tl.constexpr,
        # Which CDNA4 shuffle formula: 16 or 32. Must match matrix_instr_nonkdim.
        SCALE_NONKDIM: tl.constexpr,
        EVEN_K: tl.constexpr,
        MASK_K_LIMIT: tl.constexpr,
        W_CACHE_MODIFIER: tl.constexpr,
        UPCAST_INDICES: tl.constexpr = False,
    ):
        MX_PACK_DIVISOR: tl.constexpr = 32

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

        # X pointers (A, per-expert slice)
        offs_x_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
        offs_x_m = tl.max_contiguous(tl.multiple_of(offs_x_m % M, BLOCK_M), BLOCK_M)
        X += start_m * stride_x_m
        offs_x_k = tl.arange(0, BLOCK_K)
        XPtrs = (
            X
            + offs_x_m.to(index_type)[:, None] * stride_x_m
            + offs_x_k.to(index_type)[None, :] * stride_x_k
        )

        MX_SCALE_BLOCK_K: tl.constexpr = BLOCK_K // MX_PACK_DIVISOR

        # W scale pointers
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

        # W pointers
        offs_w_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_w_n = tl.max_contiguous(tl.multiple_of(offs_w_n % N, BLOCK_N), BLOCK_N)
        offs_w_k = tl.arange(0, BLOCK_K)
        W += expt_id * stride_w_e
        WPtrs = W + (
            offs_w_k.to(index_type)[:, None] * stride_w_k
            + offs_w_n.to(index_type)[None, :] * stride_w_n
        )

        # X scale pointers
        # Plain path: tile shape (BLOCK_M, MX_SCALE_BLOCK_K), stride_x_mx_m
        #   indexes the M-dim of the original (total_M, K//32) scale tensor.
        # CDNA4_SCALE path: scales are pre-shuffled to (total_M//32, K) and
        #   stride_x_mx_m indexes the M//32-dim. We load a (SCALE_BLOCK_M,
        #   PACKED_MX_BLOCK) = (BLOCK_M/32, BLOCK_K) tile and unshuffle it
        #   back to (BLOCK_M, MX_SCALE_BLOCK_K) before tl.dot_scaled.
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
            x = tl.load(XPtrs)
            w = tl.load(WPtrs, cache_modifier=W_CACHE_MODIFIER)
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
                x_scales = tl.load(XMxScalePtrs)
                w_scales = tl.load(WMxScalePtrs)

            acc = tl.dot_scaled(
                x, x_scales, "e4m3", w, w_scales, "e4m3", acc=acc, fast_math=True
            )

            WMxScalePtrs += PACKED_MX_BLOCK * stride_w_mx_k
            XMxScalePtrs += PACKED_MX_BLOCK * stride_x_mx_k
            XPtrs += BLOCK_K * stride_x_k
            WPtrs += BLOCK_K * stride_w_k

        if not EVEN_K:
            mask_x_k = offs_x_k < MASK_K_LIMIT
            mask_w_k = offs_w_k < MASK_K_LIMIT
            if SWIZZLE_MX_SCALE is None:
                mask_w_k_scale = offs_w_k_scale * MX_PACK_DIVISOR < MASK_K_LIMIT
                mask_x_k_scale = offs_x_k_scale * MX_PACK_DIVISOR < MASK_K_LIMIT

            x = tl.load(XPtrs, mask=mask_x_k[None, :], other=0.0)
            w = tl.load(WPtrs, mask=mask_w_k[:, None], other=0.0,
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
                x, x_scales, "e4m3", w, w_scales, "e4m3", acc=acc, fast_math=True
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

    @triton.jit
    def _expt_data_kernel(
        OffsetsPtr,             # (E,) int32 - group_end_offsets
        HistPtr,                # (E,) int32 - output: tokens per expert
        OffsRawPtr,             # (E,) int32 - output: prefix start offset per expert
        OffsPadSumPtr,          # 0-d int32 - output: sum of cdiv(hist[e], BLOCK_M)
        BlockPidMapPtr,         # (GRID_M_UB,) int32 - output: packed (block<<16)|e or -1
        E: tl.constexpr,
        BLOCK_M: tl.constexpr,
        GRID_M_UB: tl.constexpr,
    ):
        """One-launch build of all routing tensors the MM kernel needs.

        Replaces a chain of cat/diff/cumsum/arange/searchsorted/clamp/shift/where
        (~30-40 us of host-visible launch overhead) with a single ~2-3 us launch.
        """
        pid = tl.program_id(0)

        offs_prev = tl.zeros((), dtype=tl.int32)
        cum = tl.zeros((), dtype=tl.int32)
        target_e = tl.zeros((), dtype=tl.int32)
        target_block = tl.zeros((), dtype=tl.int32)
        valid = tl.full((), 0, dtype=tl.int1)

        for e in tl.static_range(E):
            off = tl.load(OffsetsPtr + e).to(tl.int32)
            h = off - offs_prev
            if pid == e:
                tl.store(HistPtr + e, h)
                tl.store(OffsRawPtr + e, offs_prev)
            blocks_e = (h + BLOCK_M - 1) // BLOCK_M
            cum_next = cum + blocks_e
            owned = (pid >= cum) & (pid < cum_next)
            target_e = tl.where(owned, e, target_e)
            target_block = tl.where(owned, pid - cum, target_block)
            valid = valid | owned
            cum = cum_next
            offs_prev = off

        value = tl.where(valid, (target_block << 16) | target_e, tl.full((), -1, dtype=tl.int32))
        tl.store(BlockPidMapPtr + pid, value)
        if pid == 0:
            tl.store(OffsPadSumPtr, cum)

    def _build_expt_data(
        group_end_offsets: torch.Tensor, M: int, E: int, block_m: int
    ):
        """Single-kernel build of (hist, offs_raw, offs_pad_sum, block_pid_map,
        grid_m_ub) from ``group_end_offsets``. Sync-free, torch.compile-clean.
        """
        device = group_end_offsets.device
        grid_m_ub = triton.cdiv(M, block_m) + max(E - 1, 0)

        total = 2 * E + 1 + grid_m_ub
        buf = torch.empty(total, dtype=torch.int32, device=device)
        hist = buf[:E]
        offs_raw = buf[E:2 * E]
        offs_pad_sum = buf[2 * E]                      # 0-d view
        block_pid_map = buf[2 * E + 1:]

        offs_i32 = group_end_offsets if group_end_offsets.dtype == torch.int32 \
            else group_end_offsets.to(torch.int32)

        _expt_data_kernel[(grid_m_ub,)](
            offs_i32,
            hist, offs_raw, offs_pad_sum, block_pid_map,
            E=E, BLOCK_M=block_m, GRID_M_UB=grid_m_ub,
            num_warps=1,
        )
        return hist, offs_raw, offs_pad_sum, block_pid_map, grid_m_ub

    def _shuffle_w_scales_cdna4_nonkdim16(w_scales: torch.Tensor) -> torch.Tensor:
        """Pre-shuffle (E, N, K//32) uint8 W scales into the CDNA4 nonkdim=16
        native layout. Requires N % 32 == 0 and K % 256 == 0."""
        E, N, Ks = w_scales.shape
        x = w_scales.reshape(E, N // 32, 2, 16, Ks // 8, 2, 4, 1)
        x = x.permute(0, 1, 4, 6, 3, 5, 2, 7).contiguous()
        return x.reshape(E, N // 32, Ks * 32)

    def _shuffle_x_scales_cdna4_nonkdim16(x_scales: torch.Tensor) -> torch.Tensor:
        """Pre-shuffle (M, K//32) uint8 X (activation) scales into CDNA4
        nonkdim=16 layout. Output (M//32, K). Requires M % 32 == 0, K % 256 == 0."""
        M, Ks = x_scales.shape
        x = x_scales.reshape(M // 32, 2, 16, Ks // 8, 2, 4, 1)
        x = x.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
        return x.reshape(M // 32, Ks * 32)

    def _shuffle_w_scales_cdna4_nonkdim32(w_scales: torch.Tensor) -> torch.Tensor:
        """nonkdim=32 variant of the W-scale shuffle. Output (E, N//32, K).
        Requires N % 32 == 0, K % 256 == 0."""
        E, N, Ks = w_scales.shape
        x = w_scales.reshape(E, N // 32, 32, Ks // 8, 4, 2, 1)
        x = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        return x.reshape(E, N // 32, Ks * 32)

    def _shuffle_x_scales_cdna4_nonkdim32(x_scales: torch.Tensor) -> torch.Tensor:
        """nonkdim=32 variant of the X-scale shuffle. Output (M//32, K).
        Requires M % 32 == 0, K % 256 == 0."""
        M, Ks = x_scales.shape
        x = x_scales.reshape(M // 32, 32, Ks // 8, 4, 2, 1)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.reshape(M // 32, Ks * 32)

    # Per-shape best configs from a 36-shape × 576-config sweep on MI355X
    # (8-GPU parallel, see tune_driver.py / tune_worker.py). Search space:
    # BLOCK_M ∈ {64,128,256}, BLOCK_N ∈ {128,256}, BLOCK_K ∈ {128,256},
    # GROUP_M ∈ {1,4,8}, num_warps ∈ {4,8}, num_stages ∈ {1,2},
    # waves_per_eu ∈ {0,2}, matrix_instr_nonkdim ∈ {16,32}. Comments show
    # the swept median runtime. nk32 wins 27/36 shapes (5-9% on K=2048,
    # 1-3% elsewhere); nk16 wins remaining 9 shapes (mostly large-N+K).
    _BEST_CFGS = {
        (1, 2048, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 99.4us
        (1, 2048, 5120): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 236.4us
        (1, 2048, 8192): dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 361.3us
        (1, 5120, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 218.7us
        (1, 5120, 5120): dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, GROUP_M=4, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 496.1us
        (1, 5120, 8192): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=16),  # 777.2us
        (1, 8192, 2048): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 339.1us
        (1, 8192, 5120): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 766.4us
        (1, 8192, 8192): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=16),  # 1168.9us
        (2, 2048, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 97.4us
        (2, 2048, 5120): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 238.0us
        (2, 2048, 8192): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 377.6us
        (2, 5120, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 221.3us
        (2, 5120, 5120): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 509.2us
        (2, 5120, 8192): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=16),  # 783.0us
        (2, 8192, 2048): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 347.5us
        (2, 8192, 5120): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 775.0us
        (2, 8192, 8192): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16),  # 1180.6us
        (4, 2048, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 99.8us
        (4, 2048, 5120): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 244.9us
        (4, 2048, 8192): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 375.6us
        (4, 5120, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 235.8us
        (4, 5120, 5120): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 520.8us
        (4, 5120, 8192): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16),  # 795.2us
        (4, 8192, 2048): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 355.8us
        (4, 8192, 5120): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 789.4us
        (4, 8192, 8192): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16),  # 1199.9us
        (8, 2048, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 110.8us
        (8, 2048, 5120): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 250.8us
        (8, 2048, 8192): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 380.2us
        (8, 5120, 2048): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 244.3us
        (8, 5120, 5120): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 529.6us
        (8, 5120, 8192): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=16),  # 813.4us
        (8, 8192, 2048): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 370.9us
        (8, 8192, 5120): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16),  # 816.2us
        (8, 8192, 8192): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16),  # 1288.5us
    }

    # Heuristic for shapes outside the swept grid. The sweep showed BLOCK_N=128
    # always wins; BLOCK_M/BLOCK_K/num_warps split cleanly on K (small K wants
    # 128/128/4-warp, large K wants 256/256/8-warp). nk32 is the better
    # geomean default; the very-large-N+K corners that prefer nk16 are a
    # minority and not derivable from N/K alone.
    _FALLBACK_SMALL_K = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32)
    _FALLBACK_LARGE_K = dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32)

    def _pick_config(E: int, N: int, K: int) -> dict:
        """Per-shape best config from the swept Llama4 grid; coarse fallback
        for unseen shapes."""
        cfg = _BEST_CFGS.get((E, N, K))
        if cfg is not None:
            return cfg
        return _FALLBACK_SMALL_K if K <= 2048 else _FALLBACK_LARGE_K

    def triton_mxfp8_grouped_mm(
        input_act: torch.Tensor,
        weight: torch.Tensor,
        input_act_scales: torch.Tensor,
        weight_scales: torch.Tensor,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
        # All tunables default to None → looked up via _pick_config(E, N, K).
        # An explicit value overrides the lookup (used by the per-shape sweep).
        BLOCK_M: int = None,
        BLOCK_N: int = None,
        BLOCK_K: int = None,
        GROUP_M: int = None,
        XCD_SWIZZLE: int = 8,
        num_warps: int = None,
        num_stages: int = None,
        matrix_instr_nonkdim: int = None,
        waves_per_eu: int = None,
        kpack: int = 1,
    ) -> torch.Tensor:
        """MXFP8 grouped GEMM: ``output[g] = input_act[group_g] @ weight[g]^T``.

        Same kernel is used for the forward pass and the input-gradient
        (``dgrad``) pass of the MoE layer — both are A @ B^T per group.

        Args:
            input_act: ``(M, K)`` fp8, row-major.
            weight: ``(E, N, K)`` fp8 row-major; internally viewed as column-major
                (``E, K, N``) before launch (``stride(-2)`` must become 1).
            input_act_scales: ``(M, K//32)`` e8m0-viewed-as-uint8.
            weight_scales: ``(E, N, K//32)`` e8m0-viewed-as-uint8.
            group_end_offsets: ``(E,)`` int32, cumulative token counts per expert.
        """
        M, K = input_act.shape
        E, N, K2 = weight.shape
        assert K == K2, f"K mismatch: A={K}, B={K2}"

        _cfg = _pick_config(E, N, K)
        if BLOCK_M is None: BLOCK_M = _cfg["BLOCK_M"]
        if BLOCK_N is None: BLOCK_N = _cfg["BLOCK_N"]
        if BLOCK_K is None: BLOCK_K = _cfg["BLOCK_K"]
        if GROUP_M is None: GROUP_M = _cfg["GROUP_M"]
        if num_warps is None: num_warps = _cfg["num_warps"]
        if num_stages is None: num_stages = _cfg["num_stages"]
        if waves_per_eu is None: waves_per_eu = _cfg["waves_per_eu"]
        # nk16 is the default for the swept Llama4 grid; per-shape entries
        # in _BEST_CFGS may override via "matrix_instr_nonkdim".
        if matrix_instr_nonkdim is None:
            matrix_instr_nonkdim = _cfg.get("matrix_instr_nonkdim", 16)

        # CDNA4_SCALE path: pre-shuffle W and X scales into the layout that
        # v_mfma_scale_f32_16x16x128_f8f6f4 consumes natively, so the kernel
        # loads one coalesced block per thread instead of the 6x ds_read_u8 +
        # 3x v_perm_b32 chain the #blocked->#linear1 lowering produces.
        # Requires BLOCK_K >= 256, N/M % 32 == 0, K % 256 == 0.
        use_cdna4_scale = (
            BLOCK_K >= 256 and K % 256 == 0 and N % 32 == 0 and M % 32 == 0
        )

        # Column-major view of W (E, K, N) with stride(-2)==1.
        w_kn = weight.permute(0, 2, 1)
        x_scales_u8 = input_act_scales.view(torch.uint8)
        w_scales_u8 = weight_scales.view(torch.uint8)

        if use_cdna4_scale:
            # Pick nk16 or nk32 host-side shuffle based on caller's
            # matrix_instr_nonkdim. nk16 is geomean-best across the swept
            # Llama4 shapes; nk32 wins ~24% on some DSv3 shapes (per
            # forward.py historical comments) and is now selectable per-shape
            # via the tuned _BEST_CFGS table.
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
            # Plain path: W (E, K//32, N), X (M, K//32).
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

        _mxfp8_grouped_mm_kernel[grid](
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
            W_CACHE_MODIFIER=None,
            UPCAST_INDICES=False,
            num_warps=num_warps, num_stages=num_stages,
            matrix_instr_nonkdim=nonkdim, kpack=kpack,
            waves_per_eu=waves_per_eu,
        )
        return output

else:
    _UNAVAILABLE_MSG = "ROCm MXFP8 kernels require gfx950 or later and triton"

    def triton_mxfp8_grouped_mm(*args, **kwargs):
        raise NotImplementedError(_UNAVAILABLE_MSG)
