##############################################################################
# Gluon Wgrad — autotuned forward-kernel-backed weight gradient.
#
# Maps grad_W[g] = GO[g] @ IA[g]^T through triton_mxfp8_grouped_mm with
# per-wgrad-shape autotuned configs from a 128-config MI350 sweep.
#
# Uniform groups: single batched forward call with tuned config.
# Non-uniform groups: per-group forward calls with tuned config.
##############################################################################

import torch
from .._common import _rocm_mxfp8_available

if _rocm_mxfp8_available:
    from .forward import triton_mxfp8_grouped_mm

    # Autotuned per-wgrad-shape configs (128-config sweep, MI350 gfx950).
    _BEST_CFGS_WGRAD = {
        (4, 32768, 2048, 7168):  dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2),
        (8, 32768, 2048, 7168):  dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, GROUP_M=4, num_warps=8, num_stages=2, waves_per_eu=2),
        (4, 128000, 2048, 7168): dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2),
        (8, 128000, 2048, 7168): dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=128, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=2),
        (4, 32768, 7168, 2048):  dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2),
        (8, 32768, 7168, 2048):  dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2),
        (4, 128000, 7168, 2048): dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, GROUP_M=4, num_warps=8, num_stages=2, waves_per_eu=2),
        (8, 128000, 7168, 2048): dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=128, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=2),
    }

    _FALLBACK_WGRAD = dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2)

    def _pick_wgrad_config(E, M, N, K):
        return _BEST_CFGS_WGRAD.get((E, M, N, K), _FALLBACK_WGRAD)

    def _build_stacked_A(go_t, E, N, Mg):
        """Stack per-group GO slices into (E*N, Mg)."""
        A = go_t.new_empty((E * N, Mg))
        for g in range(E):
            gs = g * Mg
            A[g * N : (g + 1) * N] = go_t[:, gs : gs + Mg]
        return A

    def _build_stacked_As(go_scale, E, N, Mg):
        """Stack per-group GO scales into (E*N, Mg//32)."""
        Mg32 = Mg // 32
        As = go_scale.new_empty((E * N, Mg32))
        for g in range(E):
            gs = g * Mg32
            As[g * N : (g + 1) * N] = go_scale[:, gs : gs + Mg32]
        return As

    def _build_stacked_B(ia_t, E, K, Mg):
        """Stack per-group IA slices into (E, K, Mg)."""
        B = ia_t.new_empty((E, K, Mg))
        for g in range(E):
            gs = g * Mg
            B[g] = ia_t[:, gs : gs + Mg]
        return B

    def _build_stacked_Bs(ia_scale, E, K, Mg):
        """Stack per-group IA scales into (E, K, Mg//32)."""
        Mg32 = Mg // 32
        Bs = ia_scale.new_empty((E, K, Mg32))
        for g in range(E):
            gs = g * Mg32
            Bs[g] = ia_scale[:, gs : gs + Mg32]
        return Bs

    def triton_mxfp8_wgrad_gluon(
        go_t: torch.Tensor,
        go_scale: torch.Tensor,
        ia_t: torch.Tensor,
        ia_scale: torch.Tensor,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Gluon wgrad with autotuned per-shape forward-kernel configs.

        Returns ``(E, N, K)`` bf16 weight gradient.
        """
        N, M = go_t.shape
        K = ia_t.shape[0]
        E = group_end_offsets.shape[0]
        offs_cpu = group_end_offsets.cpu().tolist()

        Mg_first = offs_cpu[0]
        uniform = all(
            (offs_cpu[g] - (offs_cpu[g - 1] if g > 0 else 0)) == Mg_first
            for g in range(E)
        )

        cfg = _pick_wgrad_config(E, M, N, K)

        if uniform:
            Mg = Mg_first
            A = _build_stacked_A(go_t, E, N, Mg)
            As = _build_stacked_As(go_scale, E, N, Mg)
            B = _build_stacked_B(ia_t, E, K, Mg)
            Bs = _build_stacked_Bs(ia_scale, E, K, Mg)

            grp_offs = go_t.new_tensor([N * (g + 1) for g in range(E)], dtype=torch.int32)
            result = triton_mxfp8_grouped_mm(
                A, B, As, Bs, grp_offs, out_dtype=out_dtype, **cfg,
            )
            return result.view(E, N, K).contiguous()
        else:
            out = go_t.new_empty((E, N, K), dtype=out_dtype)
            for g in range(E):
                gs = offs_cpu[g - 1] if g > 0 else 0
                ge = offs_cpu[g]
                Mg = ge - gs
                A_g = torch.narrow(go_t, 1, gs, Mg)
                As_g = torch.narrow(go_scale, 1, gs // 32, Mg // 32)
                B_g = torch.narrow(ia_t, 1, gs, Mg).unsqueeze(0)
                Bs_g = torch.narrow(ia_scale, 1, gs // 32, Mg // 32).unsqueeze(0)
                offs_g = go_t.new_tensor([N], dtype=torch.int32)
                out[g] = triton_mxfp8_grouped_mm(
                    A_g, B_g, As_g, Bs_g, offs_g, out_dtype=out_dtype, **cfg,
                )
            return out

else:
    def triton_mxfp8_wgrad_gluon(*args, **kwargs):
        raise NotImplementedError("ROCm MXFP8 kernels require gfx950+")
