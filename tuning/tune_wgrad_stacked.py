"""Quick sweep of BLOCK sizes for stacked wgrad shapes."""
import torch, math, itertools
from kernels.mxfp8.forward import triton_mxfp8_grouped_mm
from utils import generate_jagged_offs, to_mx, benchmark_cuda_function_in_microseconds

torch.manual_seed(42)
device = "cuda"

shapes = [
    (4, 32768, 2048, 7168),
    (8, 32768, 2048, 7168),
    (4, 128000, 2048, 7168),
    (8, 128000, 2048, 7168),
    (4, 32768, 7168, 2048),
    (8, 32768, 7168, 2048),
    (4, 128000, 7168, 2048),
    (8, 128000, 7168, 2048),
]

block_ms = [128, 256]
block_ns = [128, 256]
block_ks = [128, 256]
group_ms = [1, 8]
num_warps_list = [4, 8]
waves_list = [0, 2]

best_cfgs = {}

for e, m, n, k in shapes:
    Mg = m // e
    go = torch.randn((n, m), dtype=torch.bfloat16, device=device)
    ia = torch.randn((k, m), dtype=torch.bfloat16, device=device)
    offs = generate_jagged_offs(e, m, multiple_of=32)
    gs, gf = to_mx(go, torch.float8_e4m3fn, 32)
    iss, iaf = to_mx(ia, torch.float8_e4m3fn, 32)
    flops = 2 * n * k * m

    A = go.new_empty((e * n, Mg), dtype=gf.dtype)
    As = gs.new_empty((e * n, Mg // 32), dtype=gs.dtype)
    B = ia.new_empty((e, k, Mg), dtype=iaf.dtype)
    Bs = iss.new_empty((e, k, Mg // 32), dtype=iss.dtype)
    for g in range(e):
        gs_off = g * Mg
        A[g*n:(g+1)*n] = gf[:, gs_off:gs_off+Mg]
        As[g*n:(g+1)*n] = gs[:, gs_off//32:gs_off//32+Mg//32]
        B[g] = iaf[:, gs_off:gs_off+Mg]
        Bs[g] = iss[:, gs_off//32:gs_off//32+Mg//32]
    grp_offs = go.new_tensor([n * (g+1) for g in range(e)], dtype=torch.int32)

    best_t = float("inf")
    best_cfg = None
    for bm, bn, bk, gm, nw, wpe in itertools.product(block_ms, block_ns, block_ks, group_ms, num_warps_list, waves_list):
        if bk > Mg: continue
        if wpe == 2 and nw == 4: continue  # invalid combo
        ns = 2
        def run():
            return triton_mxfp8_grouped_mm(A, B, As, Bs, grp_offs,
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                GROUP_M=gm, num_warps=nw, num_stages=ns,
                waves_per_eu=wpe)
        try:
            t = benchmark_cuda_function_in_microseconds(run)
            if t < best_t:
                best_t = t
                best_cfg = (bm, bn, bk, gm, nw, ns, wpe)
        except Exception:
            pass

    best_tf = flops / best_t / 1e6
    # Store with forward-kernel key: (E, E*N_tokens, K_wgrad, Mg)
    best_cfgs[(e, e * n, k, Mg)] = best_cfg
    print(f"({e},{m//1000:>3d}K,{n},{k}): BLOCK=({best_cfg[0]},{best_cfg[1]},{best_cfg[2]}) GM={best_cfg[3]} nw={best_cfg[4]} ns={best_cfg[5]} wpe={best_cfg[6]} -> {best_tf:.0f} TFLOPS")

print("\n_BEST_CFGS_WGRAD_STACKED = {")
for (e, em, k_out, mg), cfg in best_cfgs.items():
    print(f"    ({e}, {em}, {k_out}, {mg}): dict(BLOCK_M={cfg[0]}, BLOCK_N={cfg[1]}, BLOCK_K={cfg[2]}, GROUP_M={cfg[3]}, num_warps={cfg[4]}, num_stages={cfg[5]}, waves_per_eu={cfg[6]}),")
print("}")
