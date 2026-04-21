# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Backward (weight-gradient) MXFP8 grouped-GEMM kernel for ROCm gfx950+.

Exports ``triton_mxfp8_wgrad`` — computes ``grad_W[g] = grad_output[group_g]^T
@ input_act[group_g]`` for each expert group. Distinct from the forward
kernel in ``forward.py``, which covers the fwd and dgrad (input gradient)
passes (both A @ B^T).

Two kernel variants:
  - Direct (E >= 2): one CTA per (BLOCK_N, BLOCK_K, group) — the natural
    grid already saturates the device, write bf16 straight out.
  - Partial + reduce (E == 1): partition the per-group M-loop across SPLIT_M
    CTAs, write fp32 partials to an (E, SPLIT_M, N, K) buffer, reduce in a
    second pass. Needed because a single group's (N/BN, K/BK) grid leaves
    too few CTAs to fill all SMs.
"""

import torch

from ._common import _rocm_mxfp8_available

if _rocm_mxfp8_available:
    import triton
    import triton.language as tl

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
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_g = tl.program_id(2)

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

        SUB_PER_BLOCK_M: tl.constexpr = BLOCK_M // SCALE_BLOCK
        M_SCALES = M // SCALE_BLOCK
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

            mb_base = m_base // SCALE_BLOCK
            mb_offs = mb_base + tl.arange(0, SUB_PER_BLOCK_M)
            mb_mask = mb_offs < M_SCALES
            # other=127: e8m0 bias 127 = 2^0 = 1.0 (neutral).
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
                acc=acc, out_dtype=tl.float32,
            )

        c_mask = n_mask[:, None] & k_mask[None, :]
        tl.store(
            C_ptr + pid_g * C_stride_e + n_offs[:, None] * C_stride_n + k_offs[None, :] * C_stride_k,
            acc.to(tl.bfloat16), mask=c_mask,
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
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_eg = tl.program_id(2)
        pid_split = pid_eg % SPLIT_M
        pid_g = pid_eg // SPLIT_M

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

        SUB_PER_BLOCK_M: tl.constexpr = BLOCK_M // SCALE_BLOCK
        M_SCALES = M // SCALE_BLOCK
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

            mb_base = m_base // SCALE_BLOCK
            mb_offs = mb_base + tl.arange(0, SUB_PER_BLOCK_M)
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
                acc=acc, out_dtype=tl.float32,
            )

        p_mask = n_mask[:, None] & k_mask[None, :]
        tl.store(
            P_ptr + pid_g * P_stride_e + pid_split * P_stride_s
                  + n_offs[:, None] * P_stride_n + k_offs[None, :] * P_stride_k,
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
                P_ptr + pid_g * P_stride_e + s * P_stride_s
                      + n_offs[:, None] * P_stride_n + k_offs[None, :] * P_stride_k,
                mask=c_mask, other=0.0,
            )

        tl.store(
            C_ptr + pid_g * C_stride_e + n_offs[:, None] * C_stride_n + k_offs[None, :] * C_stride_k,
            acc.to(tl.bfloat16), mask=c_mask,
        )

    def triton_mxfp8_wgrad(
        go_t: torch.Tensor,
        go_scale: torch.Tensor,
        ia_t: torch.Tensor,
        ia_scale: torch.Tensor,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
        BLOCK_N: int = 256,
        BLOCK_K: int = 256,
        BLOCK_M: int = 64,
        num_warps: int = 8,
        num_stages: int = 2,
    ) -> torch.Tensor:
        """MXFP8 weight gradient: ``grad_W[g] = grad_output[group_g]^T @ input_act[group_g]``.

        Both inputs must be dim1-quantized (scales along the M / token dim).
        For a single group (``E == 1``) the (N/BN, K/BK) grid may not saturate
        the device, so we partition the per-group M-loop across SPLIT_M CTAs
        and reduce their fp32 partials in a second pass; for E >= 2 the natural
        grid is enough and we write bf16 directly.

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
        SPLIT_M = 2 if E == 1 else 1

        output = torch.empty((E, N, K), dtype=out_dtype, device=go_t.device)

        if SPLIT_M == 1:
            grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K), E)
            _mxfp8_wgrad_direct_kernel[grid](
                go_t, go_t.stride(0), go_t.stride(1),
                go_scale.view(torch.uint8),
                go_scale.stride(0), go_scale.stride(1),
                ia_t, ia_t.stride(0), ia_t.stride(1),
                ia_scale.view(torch.uint8),
                ia_scale.stride(0), ia_scale.stride(1),
                output, output.stride(0), output.stride(1), output.stride(2),
                group_end_offsets,
                M, N, K,
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_M=BLOCK_M,
                SCALE_BLOCK=SCALE_BLOCK,
                num_warps=num_warps, num_stages=num_stages,
                matrix_instr_nonkdim=0, kpack=1,
            )
            return output

        partials = torch.empty(
            (E, SPLIT_M, N, K), dtype=torch.float32, device=go_t.device
        )
        partial_grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K), E * SPLIT_M)
        _mxfp8_wgrad_partial_kernel[partial_grid](
            go_t, go_t.stride(0), go_t.stride(1),
            go_scale.view(torch.uint8),
            go_scale.stride(0), go_scale.stride(1),
            ia_t, ia_t.stride(0), ia_t.stride(1),
            ia_scale.view(torch.uint8),
            ia_scale.stride(0), ia_scale.stride(1),
            partials,
            partials.stride(0), partials.stride(1),
            partials.stride(2), partials.stride(3),
            group_end_offsets,
            M, N, K,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_M=BLOCK_M,
            SPLIT_M=SPLIT_M, SCALE_BLOCK=SCALE_BLOCK,
            num_warps=num_warps, num_stages=num_stages,
            matrix_instr_nonkdim=0, kpack=1,
        )

        reduce_grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K), E)
        _mxfp8_wgrad_reduce_kernel[reduce_grid](
            partials,
            partials.stride(0), partials.stride(1),
            partials.stride(2), partials.stride(3),
            output,
            output.stride(0), output.stride(1), output.stride(2),
            N, K,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SPLIT_M=SPLIT_M,
            num_warps=num_warps, num_stages=num_stages,
        )
        return output

else:
    _UNAVAILABLE_MSG = "ROCm MXFP8 kernels require gfx950 or later and triton"

    def triton_mxfp8_wgrad(*args, **kwargs):
        raise NotImplementedError(_UNAVAILABLE_MSG)
