##############################################################################
# Optimized Stacked Wgrad — forward-kernel-backed with zero-launch-overhead
# Triton copy kernels and autotuned per-shape configs.
#
# The copy tax in wgrad_v2 comes from a Python for-loop that launches
# 16 small CUDA memcpy kernels (E experts × 4 tensor types).
# This module replaces that with:
#   1. Single-kernel GO→stacked-A copy (1 launch for all experts)
#   2. Single-kernel IA→stacked-B copy (1 launch for all experts)
#   3. Forward grouped matmul with _BEST_CFGS_WGRAD tuning
#
# Scale copies are fused into the same copy kernels.
##############################################################################

import torch

from .._common import _rocm_mxfp8_available

if _rocm_mxfp8_available:
    import triton
    import triton.language as tl

    from .forward import triton_mxfp8_grouped_mm
    from .wgrad_gluon import _BEST_CFGS_WGRAD, _FALLBACK_WGRAD

    # ------------------------------------------------------------------
    # Triton copy kernel: GO (N, E*Mg) → stacked-A (E*N, Mg)
    # ------------------------------------------------------------------
    @triton.jit
    def _stack_go_kernel(
        go_ptr, go_stride_n, go_stride_m,
        go_s_ptr, go_s_stride_n, go_s_stride_m,
        a_ptr, a_stride_m,
        a_s_ptr, a_s_stride_m,
        N, M, Mg, Mg_sb,
        BLOCK_COLS: tl.constexpr,
        E: tl.constexpr,
    ):
        """Copies GO (N, E*Mg) → stacked-A (E*N, Mg) in one launch.

        Grid: (E * N,). Each CTA copies one row of stacked-A.
        """
        pid = tl.program_id(0)  # row index in stacked-A: 0 .. E*N-1
        if pid >= E * N:
            return

        expert = pid // N
        token = pid % N

        # GO source: row=token, columns e*Mg .. e*Mg+Mg
        go_base = token.to(tl.int64) * go_stride_n + expert * Mg * go_stride_m

        # stacked-A dest: row=expert*N+token, columns 0..Mg
        a_base = pid.to(tl.int64) * a_stride_m

        for col_start in range(0, Mg, BLOCK_COLS):
            cols = col_start + tl.arange(0, BLOCK_COLS)
            col_mask = cols < Mg

            go_vals = tl.load(go_ptr + go_base + cols * go_stride_m, mask=col_mask)
            tl.store(a_ptr + a_base + cols * a_stride_m, go_vals, mask=col_mask)

        # Scale copy
        go_s_base = token.to(tl.int64) * go_s_stride_n + expert * Mg_sb * go_s_stride_m
        a_s_base = pid.to(tl.int64) * a_s_stride_m
        for col_start in range(0, Mg_sb, BLOCK_COLS):
            cols = col_start + tl.arange(0, BLOCK_COLS)
            col_mask = cols < Mg_sb
            go_s_vals = tl.load(go_s_ptr + go_s_base + cols * go_s_stride_m, mask=col_mask)
            tl.store(a_s_ptr + a_s_base + cols * a_s_stride_m, go_s_vals, mask=col_mask)

    # ------------------------------------------------------------------
    # Triton copy kernel: IA (K, E*Mg) → stacked-B (E, K, Mg)
    # ------------------------------------------------------------------
    @triton.jit
    def _stack_ia_kernel(
        ia_ptr, ia_stride_k, ia_stride_m,
        ia_s_ptr, ia_s_stride_k, ia_s_stride_m,
        b_ptr, b_stride_e, b_stride_k, b_stride_m,
        b_s_ptr, b_s_stride_e, b_s_stride_k, b_s_stride_m,
        K, M, Mg, Mg_sb,
        BLOCK_COLS: tl.constexpr,
        E: tl.constexpr,
    ):
        """Copies IA (K, E*Mg) → stacked-B (E, K, Mg) in one launch.

        Grid: (E * K,). Each CTA copies one row of stacked-B (one expert, one K).
        """
        pid = tl.program_id(0)  # row index in stacked-B: 0 .. E*K-1
        if pid >= E * K:
            return

        expert = pid // K
        k_idx = pid % K

        # IA source: row=k_idx, columns e*Mg .. e*Mg+Mg
        ia_base = k_idx.to(tl.int64) * ia_stride_k + expert * Mg * ia_stride_m

        # stacked-B dest: b[expert, k_idx, :]
        b_base = expert.to(tl.int64) * b_stride_e + k_idx.to(tl.int64) * b_stride_k

        for col_start in range(0, Mg, BLOCK_COLS):
            cols = col_start + tl.arange(0, BLOCK_COLS)
            col_mask = cols < Mg
            ia_vals = tl.load(ia_ptr + ia_base + cols * ia_stride_m, mask=col_mask)
            tl.store(b_ptr + b_base + cols * b_stride_m, ia_vals, mask=col_mask)

        # Scale copy
        ia_s_base = k_idx.to(tl.int64) * ia_s_stride_k + expert * Mg_sb * ia_s_stride_m
        b_s_base = expert.to(tl.int64) * b_s_stride_e + k_idx.to(tl.int64) * b_s_stride_k
        for col_start in range(0, Mg_sb, BLOCK_COLS):
            cols = col_start + tl.arange(0, BLOCK_COLS)
            col_mask = cols < Mg_sb
            ia_s_vals = tl.load(ia_s_ptr + ia_s_base + cols * ia_s_stride_m, mask=col_mask)
            tl.store(b_s_ptr + b_s_base + cols * b_s_stride_m, ia_s_vals, mask=col_mask)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def triton_mxfp8_wgrad_stacked_opt(
        go_t: torch.Tensor,
        go_scale: torch.Tensor,
        ia_t: torch.Tensor,
        ia_scale: torch.Tensor,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Stacked wgrad with optimized copies and autotuned forward-kernel configs.

        Returns ``(E, N, K)`` bf16 weight gradient.
        """
        N, M = go_t.shape
        K = ia_t.shape[0]
        E = group_end_offsets.shape[0]

        if M % E != 0:
            raise ValueError(f"M ({M}) must be divisible by E ({E}) for uniform groups")

        Mg = M // E
        Mg_sb = Mg // 32  # scale blocks

        # Pick tuned config for this shape
        cfg = _BEST_CFGS_WGRAD.get((E, M, N, K), _FALLBACK_WGRAD)

        BLOCK_COLS = 256  # number of elements per copy CTA

        # ── Step 1: Copy GO → stacked-A ──────────────────────────────
        A = go_t.new_empty((E * N, Mg), dtype=go_t.dtype)
        As = go_scale.new_empty((E * N, Mg_sb), dtype=go_scale.dtype)

        grid_go = (E * N,)
        _stack_go_kernel[grid_go](
            go_t, go_t.stride(-2), go_t.stride(-1),
            go_scale, go_scale.stride(-2), go_scale.stride(-1),
            A, A.stride(-2),
            As, As.stride(-2),
            N, M, Mg, Mg_sb,
            BLOCK_COLS=BLOCK_COLS, E=E,
            num_warps=4,
        )

        # ── Step 2: Copy IA → stacked-B ──────────────────────────────
        B = ia_t.new_empty((E, K, Mg), dtype=ia_t.dtype)
        Bs = ia_scale.new_empty((E, K, Mg_sb), dtype=ia_scale.dtype)

        grid_ia = (E * K,)
        _stack_ia_kernel[grid_ia](
            ia_t, ia_t.stride(-2), ia_t.stride(-1),
            ia_scale, ia_scale.stride(-2), ia_scale.stride(-1),
            B, B.stride(-3), B.stride(-2), B.stride(-1),
            Bs, Bs.stride(-3), Bs.stride(-2), Bs.stride(-1),
            K, M, Mg, Mg_sb,
            BLOCK_COLS=BLOCK_COLS, E=E,
            num_warps=4,
        )

        # ── Step 3: Forward grouped matmul with tuned configs ────────
        offs = go_t.new_tensor([N * (g + 1) for g in range(E)], dtype=torch.int32)

        result = triton_mxfp8_grouped_mm(
            A, B, As, Bs, offs,
            out_dtype=out_dtype,
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_N=cfg["BLOCK_N"],
            BLOCK_K=cfg["BLOCK_K"],
            GROUP_M=cfg.get("GROUP_M", 8),
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
            waves_per_eu=cfg["waves_per_eu"],
        )

        # result shape: (E*N, K) → reshape to (E, N, K)
        return result.view(E, N, K)

else:
    def triton_mxfp8_wgrad_stacked_opt(*args, **kwargs):
        raise NotImplementedError("ROCm MXFP8 kernels require gfx950+")
