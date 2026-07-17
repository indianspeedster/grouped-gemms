"""Tune forward kernel for wgrad-mapped shapes."""

import json, sys, time, torch
from kernels import triton_mxfp8_grouped_mm
from utils import benchmark_cuda_function_in_microseconds, generate_jagged_offs, to_mx

def build_search_space():
    cfgs = []
    for BLOCK_M in (64, 128):
        for BLOCK_N in (128, 256):
            for BLOCK_K in (128, 256):
                for GROUP_M in (4, 8):
                    for num_warps in (4, 8):
                        for num_stages in (1, 2):
                            for waves_per_eu in (0, 2):
                                cfgs.append({
                                    "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N,
                                    "BLOCK_K": BLOCK_K, "GROUP_M": GROUP_M,
                                    "num_warps": num_warps, "num_stages": num_stages,
                                    "waves_per_eu": waves_per_eu,
                                    "matrix_instr_nonkdim": 32, "kpack": 1,
                                })
    return cfgs

def wgrad_to_forward(E_w, M_w, N_w, K_w):
    return E_w, E_w * N_w, K_w, M_w // E_w

def tune_one(E_w, M_w, N_w, K_w, out_path):
    E, M, N, K = wgrad_to_forward(E_w, M_w, N_w, K_w)
    print("=" * 60)
    print("Wgrad (%d,%d,%d,%d) -> Forward (%d,%d,%d,%d)" % (E_w, M_w, N_w, K_w, E, M, N, K))
    print("=" * 60)

    device = torch.device("cuda")
    torch.manual_seed(123)

    A = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    B_t = torch.randn((E, N, K), dtype=torch.bfloat16, device=device).transpose(-2, -1)
    offs = generate_jagged_offs(E, M, multiple_of=32)

    A_scales, A_fp8 = to_mx(A, torch.float8_e4m3fn, 32)
    B_nkK = B_t.transpose(-2, -1).contiguous()
    B_scales, B_fp8 = to_mx(B_nkK, torch.float8_e4m3fn, 32)

    cfgs = build_search_space()
    best_us = float("inf")
    best_cfg = None
    flops = 2 * M * N * K
    t_start = time.time()

    for i, cfg in enumerate(cfgs):
        try:
            us = benchmark_cuda_function_in_microseconds(
                triton_mxfp8_grouped_mm,
                A_fp8, B_fp8, A_scales, B_scales, offs,
                out_dtype=torch.bfloat16,
                **cfg,
            )
            if us < best_us:
                best_us = us
                best_cfg = cfg
                tf = flops / us / 1e6
                print("  [%3d/%d] NEW BEST: %.1fus (%.0f TFLOPS) %s" % (i+1, len(cfgs), us, tf, str(cfg)))
        except Exception as e:
            print("  [%3d/%d] FAIL: %s" % (i+1, len(cfgs), str(e)[:80]))

    elapsed = time.time() - t_start
    result = {
        "wgrad_shape": [E_w, M_w, N_w, K_w],
        "forward_shape": [E, M, N, K],
        "best_us": round(best_us, 3) if best_us < float("inf") else None,
        "flops": flops,
        "tflops": round(flops / best_us / 1e6, 1) if best_us < float("inf") else 0,
        "best_cfg": best_cfg,
        "elapsed_s": round(elapsed, 1),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    if best_cfg:
        print("\n  BEST: %.1fus (%.0f TFLOPS)" % (best_us, result["tflops"]))
        print("  Config: %s" % best_cfg)
    else:
        print("\n  ALL CONFIGS FAILED")
    print("  Done in %.0fs -> %s" % (elapsed, out_path))
    return result

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("E", type=int)
    ap.add_argument("M", type=int)
    ap.add_argument("N", type=int)
    ap.add_argument("K", type=int)
    ap.add_argument("out_path")
    args = ap.parse_args()
    tune_one(args.E, args.M, args.N, args.K, args.out_path)

if __name__ == "__main__":
    main()
