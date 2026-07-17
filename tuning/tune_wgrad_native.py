"""Tune wgrad kernel directly (not forward-mapped)."""

import json, sys, time, torch
from kernels.mxfp8.wgrad_gluon import triton_mxfp8_wgrad_gluon
from utils import benchmark_cuda_function_in_microseconds, generate_jagged_offs, to_mx

def build_search_space():
    cfgs = []
    for BLOCK_N in (128, 256):
        for BLOCK_K in (128, 256):
            for num_warps in (4, 8):
                for num_stages in (1, 2):
                    for waves_per_eu in (0, 2):
                        cfgs.append({
                            "BLOCK_M": 64, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
                            "num_warps": num_warps, "num_stages": num_stages,
                            "waves_per_eu": waves_per_eu,
                        })
    return cfgs

def tune_one(E_w, M_w, N_w, K_w, out_path):
    print("=" * 60)
    print("Wgrad (%d,%d,%d,%d)" % (E_w, M_w, N_w, K_w))
    print("=" * 60)

    device = torch.device("cuda")
    torch.manual_seed(123)

    go = torch.randn((N_w, M_w), dtype=torch.bfloat16, device=device)
    ia = torch.randn((K_w, M_w), dtype=torch.bfloat16, device=device)
    offs = generate_jagged_offs(E_w, M_w, multiple_of=32)

    gs, gf = to_mx(go, torch.float8_e4m3fn, 32)
    iss, iaf = to_mx(ia, torch.float8_e4m3fn, 32)

    # Monkey-patch the config lookup
    import kernels.mxfp8.wgrad_gluon as gluon_mod
    orig_cfg = gluon_mod._BEST_CFGS_WGRAD.get((E_w, M_w, N_w, K_w))

    cfgs = build_search_space()
    best_us = float("inf")
    best_cfg = None
    flops = 2 * N_w * K_w * M_w
    t_start = time.time()

    for i, cfg in enumerate(cfgs):
        try:
            gluon_mod._BEST_CFGS_WGRAD[(E_w, M_w, N_w, K_w)] = cfg
            us = benchmark_cuda_function_in_microseconds(
                triton_mxfp8_wgrad_gluon, gf, gs, iaf, iss, offs,
            )
            if us < best_us:
                best_us = us
                best_cfg = dict(cfg)
                tf = flops / us / 1e6
                print("  [%2d/%d] NEW BEST: %.1fus (%.0f TFLOPS) %s" % (i+1, len(cfgs), us, tf, str(cfg)))
        except Exception as e:
            pass

    # Restore
    if orig_cfg:
        gluon_mod._BEST_CFGS_WGRAD[(E_w, M_w, N_w, K_w)] = orig_cfg

    elapsed = time.time() - t_start
    result = {
        "wgrad_shape": [E_w, M_w, N_w, K_w],
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
    print("  Done in %.0fs" % elapsed)
    return result

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("E", type=int); ap.add_argument("M", type=int)
    ap.add_argument("N", type=int); ap.add_argument("K", type=int)
    ap.add_argument("out_path")
    args = ap.parse_args()
    tune_one(args.E, args.M, args.N, args.K, args.out_path)

if __name__ == "__main__":
    main()
