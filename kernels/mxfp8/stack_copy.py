##############################################################################
# Stacked-copy kernel: copies fp8 data from wgrad layout to stacked layout
# in a single Triton kernel per tensor (replaces Python for-loop).
#
# GO  (N, E*Mg)  → stacked-A  (E*N, Mg)
# IA  (K, E*Mg)  → stacked-B  (E, K, Mg)
#
# Pure memcpy — no fp8 conversion. Quantization is done upstream via to_mx().
##############################################################################

import torch

try:
    from .._common import _rocm_mxfp8_available
except ImportError:
    from kernels._common import _rocm_mxfp8_available

if _rocm_mxfp8_available:
    import triton
    import triton.language as tl

    @triton.jit
    def _stack_kernel(
        src_ptr,
        src_stride_row,
        src_stride_col,
        dst_ptr,
        dst_stride_row,
        dst_stride_col,
        E: tl.constexpr,
        N: tl.constexpr,   # rows per expert in wgrad layout
        D: tl.constexpr,    # destination rows per expert (N for A, K for B)
        Mg: tl.constexpr,   # M per expert
        BLOCK_SIZE: tl.constexpr,
    ):
        """Generic copy: wgrad-layout → stacked-layout.

        Grid: (E * D,).  One CTA per destination row.

        Source layout: each expert occupies a contiguous column-slice of Mg
        elements.  Row r in the destination maps to (expert, local_idx) where
        expert = r // D  and  local_idx = r % D.

        For GO → A:  N=N_tokens, D=N_tokens
        For IA → B:  N=K_wgrad,  D=K_wgrad
        """
        pid = tl.program_id(0)
        if pid >= E * D:
            return

        expert = pid // D
        local = pid % D

        # Source: src[local, expert*Mg : expert*Mg + Mg]
        src_base = local.to(tl.int64) * src_stride_row + expert * Mg
        # Destination: dst[pid, :]
        dst_base = pid.to(tl.int64) * dst_stride_row

        for off in range(0, Mg, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < Mg
            vals = tl.load(src_ptr + src_base + cols, mask=mask)
            tl.store(dst_ptr + dst_base + cols, vals, mask=mask)


    @triton.jit
    def _stack_scales_kernel(
        src_ptr,
        src_stride_row,
        src_stride_col,
        dst_ptr,
        dst_stride_row,
        dst_stride_col,
        E: tl.constexpr,
        D: tl.constexpr,
        Mg_sb: tl.constexpr,  # M // scale_block
        BLOCK_SIZE: tl.constexpr,
    ):
        """Copy float32 scales: same pattern as _stack_kernel."""
        pid = tl.program_id(0)
        if pid >= E * D:
            return

        expert = pid // D
        local = pid % D

        src_base = local.to(tl.int64) * src_stride_row + expert * Mg_sb
        dst_base = pid.to(tl.int64) * dst_stride_row

        for off in range(0, Mg_sb, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < Mg_sb
            vals = tl.load(src_ptr + src_base + cols, mask=mask)
            tl.store(dst_ptr + dst_base + cols, vals, mask=mask)


    def stack_go(
        go_fp8: torch.Tensor,
        go_scales: torch.Tensor,
        E: int,
        Mg: int,
        Mg_sb: int,
    ):
        """Copy GO (N, E*Mg) → stacked-A (E*N, Mg) + scales."""
        N = go_fp8.shape[0]

        a = go_fp8.new_empty((E * N, Mg))
        a_s = go_scales.new_empty((E * N, Mg_sb))

        grid = (E * N,)
        _stack_kernel[grid](
            go_fp8, go_fp8.stride(0), go_fp8.stride(1),
            a, a.stride(0), a.stride(1),
            E=E, N=N, D=N, Mg=Mg,
            BLOCK_SIZE=256,
            num_warps=4,
        )
        _stack_scales_kernel[grid](
            go_scales, go_scales.stride(0), go_scales.stride(1),
            a_s, a_s.stride(0), a_s.stride(1),
            E=E, D=N, Mg_sb=Mg_sb,
            BLOCK_SIZE=256,
            num_warps=4,
        )
        return a, a_s


    def stack_ia(
        ia_fp8: torch.Tensor,
        ia_scales: torch.Tensor,
        E: int,
        Mg: int,
        Mg_sb: int,
    ):
        """Copy IA (K, E*Mg) → stacked-B (E, K, Mg) + scales."""
        K = ia_fp8.shape[0]

        b = ia_fp8.new_empty((E, K, Mg))
        b_s = ia_scales.new_empty((E, K, Mg_sb))

        grid = (E * K,)
        _stack_kernel[grid](
            ia_fp8, ia_fp8.stride(0), ia_fp8.stride(1),
            b.reshape(E * K, Mg), (E * K, Mg)[0], (E * K, Mg)[1],
            E=E, N=K, D=K, Mg=Mg,
            BLOCK_SIZE=256,
            num_warps=4,
        )
        _stack_scales_kernel[grid](
            ia_scales, ia_scales.stride(0), ia_scales.stride(1),
            b_s.reshape(E * K, Mg_sb), (E * K, Mg_sb)[0], (E * K, Mg_sb)[1],
            E=E, D=K, Mg_sb=Mg_sb,
            BLOCK_SIZE=256,
            num_warps=4,
        )
        return b, b_s

    def triton_mxfp8_wgrad_stacked(
        go_t: torch.Tensor,
        go_scale: torch.Tensor,
        ia_t: torch.Tensor,
        ia_scale: torch.Tensor,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Stacked wgrad via forward kernel with Triton copy kernels.

        Copies GO/IA into stacked layout using fused Triton kernels
        (2 launches instead of 16), then calls the forward grouped matmul.

        Returns ``(E, N, K)`` bf16 weight gradient.
        """
        from .forward import triton_mxfp8_grouped_mm

        N, M = go_t.shape
        K = ia_t.shape[0]
        E = group_end_offsets.shape[0]

        if M % E != 0:
            raise ValueError(f"M ({M}) must be divisible by E ({E})")

        Mg = M // E
        Mg_sb = Mg // 32

        # Copy GO → stacked-A (1 launch)
        A, As = stack_go(go_t, go_scale, E, Mg, Mg_sb)

        # Copy IA → stacked-B (1 launch)
        B, Bs = stack_ia(ia_t, ia_scale, E, Mg, Mg_sb)

        # Forward grouped matmul
        offs = go_t.new_tensor([N * (g + 1) for g in range(E)], dtype=torch.int32)
        result = triton_mxfp8_grouped_mm(A, B, As, Bs, offs, out_dtype=out_dtype)

        # result shape: (E*N, K) → reshape to (E, N, K)
        return result.view(E, N, K)

else:
    def stack_go(*args, **kwargs):
        raise NotImplementedError("ROCm MXFP8 kernels require gfx950+")

    def stack_ia(*args, **kwargs):
        raise NotImplementedError("ROCm MXFP8 kernels require gfx950+")

    def triton_mxfp8_wgrad_stacked(*args, **kwargs):
        raise NotImplementedError("ROCm MXFP8 kernels require gfx950+")
