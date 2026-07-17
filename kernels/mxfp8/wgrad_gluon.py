##############################################################################
# Gluon Wgrad -- copy-free native wgrad kernel with forward-kernel quality
# optimizations.
#
# Instead of copying GO/IA into stacked tensors, this kernel operates
# DIRECTLY on the wgrad tensor layout (GO: NxM, IA: KxM) and applies
# the same optimizations that make the forward kernel fast:
#
#   1. BLOCK_M=128 for halved M-loop iterations
#   2. BLOCK_K=256 + num_warps=8 for full MFMA utilization
#   3. Cache hints: evict_first on GO, .cg on IA
#   4. Per-wgrad-shape autotuned configs from 128-config sweep
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

    _BEST_CFGS_WGRAD = {
        (4, 32768, 2048, 7168):  dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, num_warps=8, num_stages=2, waves_per_eu=2),
        (8, 32768, 2048, 7168):  dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, num_warps=8, num_stages=2, waves_per_eu=2),
        (4, 128000, 2048, 7168): dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, num_warps=8, num_stages=2, waves_per_eu=2),
        (8, 128000, 2048, 7168): dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=128, num_warps=4, num_stages=2, waves_per_eu=2),
        (4, 32768, 7168, 2048):  dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, num_warps=8, num_stages=2, waves_per_eu=2),
        (8, 32768, 7168, 2048):  dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, num_warps=8, num_stages=2, waves_per_eu=2),
        (4, 128000, 7168, 2048): dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, num_warps=8, num_stages=2, waves_per_eu=2),
        (8, 128000, 7168, 2048): dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=128, num_warps=4, num_stages=2, waves_per_eu=2),
    }

    _FALLBACK_WGRAD = dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, num_warps=8, num_stages=2, waves_per_eu=2)

    def _pick_config(E, M, N, K):
        return _BEST_CFGS_WGRAD.get((E, M, N, K), _FALLBACK_WGRAD)

    @triton.jit
    def _wgrad_pid_grid(pid, num_pid_n, num_pid_k, GROUP_N: tl.constexpr, GROUP_K: tl.constexpr):
        if GROUP_N > 1 and GROUP_K > 1:
            num_pid_n_groups = tl.cdiv(num_pid_n, GROUP_N)
            num_pid_k_groups = tl.cdiv(num_pid_k, GROUP_K)
            pids_per_group = GROUP_N * GROUP_K
            group_id = pid // pids_per_group
            pid_in_group = pid % pids_per_group
            group_n = group_id % num_pid_n_groups
            group_k = group_id // num_pid_n_groups
            pid_n = group_n * GROUP_N + (pid_in_group % GROUP_N)
            pid_k = group_k * GROUP_K + (pid_in_group // GROUP_N)
        elif GROUP_K > 1:
            num_pid_k_groups = tl.cdiv(num_pid_k, GROUP_K)
            pids_per_n = num_pid_k_groups * GROUP_K
            pid_n = pid // pids_per_n
            pid_in_n = pid % pids_per_n
            pid_k = (pid_in_n // GROUP_K) * GROUP_K + (pid_in_n % GROUP_K)
        else:
            pid_n = pid // num_pid_k
            pid_k = pid % num_pid_k
        return pid_n, pid_k

    @triton.jit
    def _gluon_wgrad_kernel(
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
        SWIZZLE_MX_SCALE: tl.constexpr,
        SCALE_NONKDIM: tl.constexpr,
        GROUP_N: tl.constexpr,
        GROUP_K: tl.constexpr,
    ):
        pid_tile = tl.program_id(0)
        pid_g = tl.program_id(1)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_k = tl.cdiv(K, BLOCK_K)
        pid_n, pid_k = _wgrad_pid_grid(pid_tile, num_pid_n, num_pid_k, GROUP_N, GROUP_K)

        group_start = tl.load(group_end_offsets_ptr + pid_g - 1, mask=pid_g > 0, other=0)
        group_end = tl.load(group_end_offsets_ptr + pid_g)
        M_g = group_end - group_start

        n_base = pid_n * BLOCK_N
        k_base = pid_k * BLOCK_K
        if n_base >= N or k_base >= K or M_g <= 0:
            return

        n_offs = n_base + tl.arange(0, BLOCK_N)
        k_offs = k_base + tl.arange(0, BLOCK_K)
        n_mask = n_offs < N
        k_mask = k_offs < K

        MX_SBM: tl.constexpr = BLOCK_M // SCALE_BLOCK
        M_SCALES = M // SCALE_BLOCK

        CDNA4: tl.constexpr = (SWIZZLE_MX_SCALE == "CDNA4_SCALE")
        if CDNA4:
            NON_K: tl.constexpr = 32
            PACKED_MX_BLOCK_M: tl.constexpr = MX_SBM * NON_K
            SCALE_BLOCK_N_OUT: tl.constexpr = BLOCK_N // NON_K
            SCALE_BLOCK_K_OUT: tl.constexpr = BLOCK_K // NON_K

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
                cache_modifier=".cg",
            )

            if CDNA4:
                offs_go_n_s = pid_n * SCALE_BLOCK_N_OUT + tl.arange(0, SCALE_BLOCK_N_OUT)
                offs_go_m_s = m_base + tl.arange(0, PACKED_MX_BLOCK_M)
                go_st = tl.load(
                    GO_scales_ptr + offs_go_n_s[:, None] * GO_scales_stride_n
                                  + offs_go_m_s[None, :] * GO_scales_stride_mb,
                    mask=offs_go_m_s[None, :] < M, other=127,
                    
                )
                offs_ia_k_s = pid_k * SCALE_BLOCK_K_OUT + tl.arange(0, SCALE_BLOCK_K_OUT)
                offs_ia_m_s = m_base + tl.arange(0, PACKED_MX_BLOCK_M)
                ia_st = tl.load(
                    IA_scales_ptr + offs_ia_k_s[:, None] * IA_scales_stride_k
                                  + offs_ia_m_s[None, :] * IA_scales_stride_mb,
                    mask=offs_ia_m_s[None, :] < M, other=127,
                    cache_modifier=".cg",
                )
                if SCALE_NONKDIM == 32:
                    go_sc = _unswizzle_mx_scale_cdna4_nonkdim32(go_st, BLOCK_N, MX_SBM)
                    ia_sc = _unswizzle_mx_scale_cdna4_nonkdim32(ia_st, BLOCK_K, MX_SBM)
                else:
                    go_sc = _unswizzle_mx_scale_cdna4(go_st, BLOCK_N, MX_SBM)
                    ia_sc = _unswizzle_mx_scale_cdna4(ia_st, BLOCK_K, MX_SBM)
            else:
                mb_base = m_base // SCALE_BLOCK
                mb_offs = mb_base + tl.arange(0, MX_SBM)
                mb_mask = mb_offs < M_SCALES
                go_sc = tl.load(
                    GO_scales_ptr + n_offs[:, None] * GO_scales_stride_n
                                  + mb_offs[None, :] * GO_scales_stride_mb,
                    mask=n_mask[:, None] & mb_mask[None, :], other=127,
                    
                )
                ia_sc = tl.load(
                    IA_scales_ptr + k_offs[:, None] * IA_scales_stride_k
                                  + mb_offs[None, :] * IA_scales_stride_mb,
                    mask=k_mask[:, None] & mb_mask[None, :], other=127,
                    cache_modifier=".cg",
                )

            acc = tl.dot_scaled(
                go_tile, go_sc, "e4m3",
                ia_tile, ia_sc, "e4m3",
                acc=acc, out_dtype=tl.float32, fast_math=True,
            )

        c_mask = n_mask[:, None] & k_mask[None, :]
        tl.store(
            C_ptr + pid_g.to(tl.int64) * C_stride_e
                  + n_offs[:, None] * C_stride_n
                  + k_offs[None, :] * C_stride_k,
            acc.to(tl.bfloat16), mask=c_mask,
        )

    def triton_mxfp8_wgrad_gluon(
        go_t: torch.Tensor,
        go_scale: torch.Tensor,
        ia_t: torch.Tensor,
        ia_scale: torch.Tensor,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Copy-free gluon wgrad with autotuned per-shape configs.
        Operates directly on wgrad tensor layout with no data copies.
        Returns ``(E, N, K)`` bf16 weight gradient.
        """
        N, M = go_t.shape
        K = ia_t.shape[0]
        E = group_end_offsets.shape[0]
        SCALE_BLOCK = 32

        cfg = _pick_config(E, M, N, K)
        BLOCK_M = cfg["BLOCK_M"]
        BLOCK_N = cfg["BLOCK_N"]
        BLOCK_K = cfg["BLOCK_K"]
        num_warps = cfg["num_warps"]
        num_stages = cfg["num_stages"]
        waves_per_eu = cfg["waves_per_eu"]
        matrix_instr_nonkdim = 32

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
            go_scale_arg = _shuffle_x_scales_cdna4_nonkdim32(go_scale_u8)
            ia_scale_arg = _shuffle_x_scales_cdna4_nonkdim32(ia_scale_u8)
            swizzle = "CDNA4_SCALE"
        else:
            go_scale_arg = go_scale_u8
            ia_scale_arg = ia_scale_u8
            swizzle = "NONE"

        output = torch.empty((E, N, K), dtype=out_dtype, device=go_t.device)

        num_pid_n = triton.cdiv(N, BLOCK_N)
        num_pid_k = triton.cdiv(K, BLOCK_K)

        if num_pid_n > 1 and num_pid_k > 1:
            GROUP_N = min(4, num_pid_n)
            GROUP_K = min(4, num_pid_k)
        else:
            GROUP_N = 1
            GROUP_K = 1

        if GROUP_N > 1 and GROUP_K > 1:
            grid_tiles = triton.cdiv(num_pid_n, GROUP_N) * triton.cdiv(num_pid_k, GROUP_K) * GROUP_N * GROUP_K
        elif GROUP_K > 1:
            grid_tiles = num_pid_n * triton.cdiv(num_pid_k, GROUP_K) * GROUP_K
        else:
            grid_tiles = num_pid_n * num_pid_k

        grid = (grid_tiles, E)

        _gluon_wgrad_kernel[grid](
            go_t, go_t.stride(-2), go_t.stride(-1),
            go_scale_arg, go_scale_arg.stride(-2), go_scale_arg.stride(-1),
            ia_t, ia_t.stride(-2), ia_t.stride(-1),
            ia_scale_arg, ia_scale_arg.stride(-2), ia_scale_arg.stride(-1),
            output, output.stride(-3), output.stride(-2), output.stride(-1),
            group_end_offsets,
            M, N, K,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_M=BLOCK_M,
            SCALE_BLOCK=SCALE_BLOCK,
            SWIZZLE_MX_SCALE=swizzle,
            SCALE_NONKDIM=matrix_instr_nonkdim,
            GROUP_N=GROUP_N, GROUP_K=GROUP_K,
            num_warps=num_warps, num_stages=num_stages,
            matrix_instr_nonkdim=matrix_instr_nonkdim,
            waves_per_eu=waves_per_eu,
        )
        return output

else:
    def triton_mxfp8_wgrad_gluon(*args, **kwargs):
        raise NotImplementedError("ROCm MXFP8 kernels require gfx950+")
