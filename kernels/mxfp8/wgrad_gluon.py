##############################################################################
# Gluon Wgrad — persistent cooperative kernel for MXFP8 grouped wgrad.
##############################################################################

import torch

from .._common import _rocm_mxfp8_available

if _rocm_mxfp8_available:
    import triton
    import triton.language as tl

    from .forward import (
        _unswizzle_mx_scale_cdna4,
        _unswizzle_mx_scale_cdna4_nonkdim32,
        _shuffle_x_scales_cdna4_nonkdim16,
        _shuffle_x_scales_cdna4_nonkdim32,
    )

    @triton.jit
    def _gluon_wgrad_kernel(
        GO_ptr, GO_stride_n, GO_stride_m,
        GO_scales_ptr, GO_scales_stride_n, GO_scales_stride_mb,
        IA_ptr, IA_stride_k, IA_stride_m,
        IA_scales_ptr, IA_scales_stride_k, IA_scales_stride_mb,
        C_ptr, C_stride_e, C_stride_n, C_stride_k,
        group_end_offsets_ptr,
        work_counter_ptr,
        M, N, K, E,
        TOTAL_WORK_ITEMS: tl.constexpr,
        MAX_CTAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        SCALE_BLOCK: tl.constexpr,
        SWIZZLE_MX_SCALE: tl.constexpr,
        SCALE_NONKDIM: tl.constexpr,
        K_TILES_PER_CTA: tl.constexpr,
    ):
        MX_SBM: tl.constexpr = BLOCK_M // SCALE_BLOCK
        M_SCALES = M // SCALE_BLOCK
        CDNA4: tl.constexpr = (SWIZZLE_MX_SCALE == "CDNA4_SCALE")
        num_n = tl.cdiv(N, BLOCK_N)
        num_k = tl.cdiv(K, BLOCK_K)
        max_work = num_n * num_k * E

        # Persistent loop: each CTA steals at most MAX_CTAS work items
        for _ in range(TOTAL_WORK_ITEMS // MAX_CTAS + 1):
            work_id = tl.atomic_add(work_counter_ptr, 1)
            if work_id >= max_work:
                work_id = max_work

            pid_g = work_id // (num_n * num_k)
            rest = work_id % (num_n * num_k)
            pid_n = rest // num_k
            pid_k_base = rest % num_k

            if pid_g >= E:
                pid_g = E

            group_start = tl.load(group_end_offsets_ptr + pid_g - 1, mask=pid_g > 0, other=0)
            group_end = tl.load(group_end_offsets_ptr + pid_g)
            M_g = group_end - group_start

            n_base = pid_n * BLOCK_N
            k_base = pid_k_base * BLOCK_K * K_TILES_PER_CTA
            valid = (pid_g < E) & (n_base < N) & (k_base < K) & (M_g > 0)

            if valid:
                n_offs = n_base + tl.arange(0, BLOCK_N)
                n_mask = n_offs < N

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

                    if CDNA4:
                        SN: tl.constexpr = 32
                        offs_go_n_s = pid_n * (BLOCK_N // SN) + tl.arange(0, BLOCK_N // SN)
                        offs_go_m_s = m_base + tl.arange(0, MX_SBM * SN)
                        go_st = tl.load(
                            GO_scales_ptr + offs_go_n_s[:, None] * GO_scales_stride_n
                                          + offs_go_m_s[None, :] * GO_scales_stride_mb,
                            mask=offs_go_m_s[None, :] < M, other=127,
                        )
                        if SCALE_NONKDIM == 32:
                            go_sc = _unswizzle_mx_scale_cdna4_nonkdim32(go_st, BLOCK_N, MX_SBM)
                        else:
                            go_sc = _unswizzle_mx_scale_cdna4(go_st, BLOCK_N, MX_SBM)
                    else:
                        mb_base = m_base // SCALE_BLOCK
                        mb_offs = mb_base + tl.arange(0, MX_SBM)
                        mb_mask = mb_offs < M_SCALES
                        go_sc = tl.load(
                            GO_scales_ptr + n_offs[:, None] * GO_scales_stride_n
                                          + mb_offs[None, :] * GO_scales_stride_mb,
                            mask=n_mask[:, None] & mb_mask[None, :], other=127,
                        )

                    # K-tile 0
                    k_offs0 = k_base + tl.arange(0, BLOCK_K)
                    k_mask0 = k_offs0 < K
                    ia_tile0 = tl.load(
                        IA_ptr + k_offs0[None, :] * IA_stride_k + m_offs[:, None] * IA_stride_m,
                        mask=m_mask[:, None] & k_mask0[None, :], other=0.0,
                    )
                    if CDNA4:
                        SN2: tl.constexpr = 32
                        k_pid0 = k_base // BLOCK_K
                        offs_ia_k_s0 = k_pid0 * (BLOCK_K // SN2) + tl.arange(0, BLOCK_K // SN2)
                        offs_ia_m_s = m_base + tl.arange(0, MX_SBM * SN2)
                        ia_st0 = tl.load(
                            IA_scales_ptr + offs_ia_k_s0[:, None] * IA_scales_stride_k
                                          + offs_ia_m_s[None, :] * IA_scales_stride_mb,
                            mask=offs_ia_m_s[None, :] < M, other=127,
                        )
                        if SCALE_NONKDIM == 32:
                            ia_sc0 = _unswizzle_mx_scale_cdna4_nonkdim32(ia_st0, BLOCK_K, MX_SBM)
                        else:
                            ia_sc0 = _unswizzle_mx_scale_cdna4(ia_st0, BLOCK_K, MX_SBM)
                    else:
                        ia_sc0 = tl.load(
                            IA_scales_ptr + k_offs0[:, None] * IA_scales_stride_k
                                          + mb_offs[None, :] * IA_scales_stride_mb,
                            mask=k_mask0[:, None] & mb_mask[None, :], other=127,
                        )
                    acc0 = tl.dot_scaled(go_tile, go_sc, "e4m3", ia_tile0, ia_sc0, "e4m3",
                                         acc=acc0, out_dtype=tl.float32, fast_math=True)

                    # K-tile 1: reuse GO/scales (gluon optimization)
                    if K_TILES_PER_CTA >= 2:
                        k_base1 = k_base + BLOCK_K
                        k_offs1 = k_base1 + tl.arange(0, BLOCK_K)
                        k_mask1 = k_offs1 < K
                        active1 = k_base1 < K
                        ia_tile1 = tl.load(
                            IA_ptr + k_offs1[None, :] * IA_stride_k + m_offs[:, None] * IA_stride_m,
                            mask=m_mask[:, None] & k_mask1[None, :] & active1, other=0.0,
                        )
                        if CDNA4:
                            k_pid1 = k_base1 // BLOCK_K
                            offs_ia_k_s1 = k_pid1 * (BLOCK_K // SN2) + tl.arange(0, BLOCK_K // SN2)
                            ia_st1 = tl.load(
                                IA_scales_ptr + offs_ia_k_s1[:, None] * IA_scales_stride_k
                                              + offs_ia_m_s[None, :] * IA_scales_stride_mb,
                                mask=offs_ia_m_s[None, :] < M, other=127,
                            )
                            if SCALE_NONKDIM == 32:
                                ia_sc1 = _unswizzle_mx_scale_cdna4_nonkdim32(ia_st1, BLOCK_K, MX_SBM)
                            else:
                                ia_sc1 = _unswizzle_mx_scale_cdna4(ia_st1, BLOCK_K, MX_SBM)
                        else:
                            ia_sc1 = tl.load(
                                IA_scales_ptr + k_offs1[:, None] * IA_scales_stride_k
                                              + mb_offs[None, :] * IA_scales_stride_mb,
                                mask=k_mask1[:, None] & mb_mask[None, :] & active1, other=127,
                            )
                        acc1 = tl.dot_scaled(go_tile, go_sc, "e4m3", ia_tile1, ia_sc1, "e4m3",
                                             acc=acc1, out_dtype=tl.float32, fast_math=True)

                # Store
                k_offs0 = k_base + tl.arange(0, BLOCK_K)
                k_mask0 = k_offs0 < K
                tl.store(
                    C_ptr + pid_g.to(tl.int64) * C_stride_e
                          + n_offs[:, None] * C_stride_n
                          + k_offs0[None, :] * C_stride_k,
                    acc0.to(tl.bfloat16), mask=n_mask[:, None] & k_mask0[None, :],
                )

                if K_TILES_PER_CTA >= 2:
                    k_base1 = k_base + BLOCK_K
                    k_offs1 = k_base1 + tl.arange(0, BLOCK_K)
                    k_mask1 = k_offs1 < K
                    tl.store(
                        C_ptr + pid_g.to(tl.int64) * C_stride_e
                              + n_offs[:, None] * C_stride_n
                              + k_offs1[None, :] * C_stride_k,
                        acc1.to(tl.bfloat16), mask=n_mask[:, None] & k_mask1[None, :],
                    )

    def triton_mxfp8_wgrad_gluon(
        go_t: torch.Tensor,
        go_scale: torch.Tensor,
        ia_t: torch.Tensor,
        ia_scale: torch.Tensor,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
        BLOCK_N: int = None,
        BLOCK_K: int = None,
        BLOCK_M: int = 64,
        num_warps: int = 8,
        num_stages: int = 2,
        matrix_instr_nonkdim: int = 32,
        kpack: int = 1,
        waves_per_eu: int = 0,
        k_tiles_per_cta: int = 1,
    ) -> torch.Tensor:
        N, M = go_t.shape
        K, _ = ia_t.shape
        E = group_end_offsets.shape[0]
        SCALE_BLOCK = 32

        if BLOCK_N is None:
            BLOCK_N = min(256, triton.next_power_of_2(N)) if N > 0 else 128
        if BLOCK_K is None:
            BLOCK_K = min(256, triton.next_power_of_2(K)) if K > 0 else 64
        k_tiles_per_cta = max(1, min(k_tiles_per_cta, 2))

        use_cdna4_scale = (
            BLOCK_M % 256 == 0 and M % 256 == 0
            and N % 32 == 0 and K % 32 == 0
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
            swizzle = "NONE"

        output = torch.empty((E, N, K), dtype=out_dtype, device=go_t.device)

        total_work = triton.cdiv(N, BLOCK_N) * triton.cdiv(K, BLOCK_K * k_tiles_per_cta) * E
        num_ctas = 8 * 64 * 4  # MI350X: 8 GCDs × 64 CUs × 4 waves
        work_counter = torch.zeros(1, dtype=torch.int32, device=go_t.device)

        _gluon_wgrad_kernel[(num_ctas, 1, 1)](
            go_t, go_t.stride(-2), go_t.stride(-1),
            go_scale_arg, go_scale_arg.stride(-2), go_scale_arg.stride(-1),
            ia_t, ia_t.stride(-2), ia_t.stride(-1),
            ia_scale_arg, ia_scale_arg.stride(-2), ia_scale_arg.stride(-1),
            output, output.stride(-3), output.stride(-2), output.stride(-1),
            group_end_offsets,
            work_counter,
            M, N, K, E,
            TOTAL_WORK_ITEMS=total_work,
            MAX_CTAS=num_ctas,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_M=BLOCK_M,
            K_TILES_PER_CTA=k_tiles_per_cta,
            SCALE_BLOCK=SCALE_BLOCK,
            SWIZZLE_MX_SCALE=swizzle,
            SCALE_NONKDIM=matrix_instr_nonkdim,
            num_warps=num_warps, num_stages=num_stages,
            matrix_instr_nonkdim=matrix_instr_nonkdim,
            kpack=kpack, waves_per_eu=waves_per_eu,
        )
        return output

else:
    def triton_mxfp8_wgrad_gluon(*args, **kwargs):
        raise NotImplementedError("ROCm MXFP8 kernels require gfx950+")
