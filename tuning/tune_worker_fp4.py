"""MXFP4 forward sweep worker. One shape, full config grid, writes JSON.

I/O contract mirrors tune_worker_kpack2.py: prints OK to stdout, writes
incremental JSON to <out_path>. Invalid configs (e.g. an nonkdim the fp4
MFMA doesn't support) are caught and recorded with us=None.
"""
import argparse
import json
import os
import random
import sys
import time
import traceback

# kpack is fixed per sweep (env var, default 1) so kpack=1 and kpack=2 passes
# stay directly comparable shape-by-shape under the same per-process offs draw.
KPACK = int(os.environ.get("KPACK", "1"))


def build_search_space():
    """576 configs at the env-selected kpack. BLOCK_K includes 512 (fp4 packs
    2 elems/byte, so a 512-logical-K tile is the same LDS footprint as fp8's
    BLOCK_K=256). GROUP_M=1 dropped — it never wins and kills L2 reuse."""
    cfgs = []
    for BLOCK_M in (64, 128, 256):
        for BLOCK_N in (128, 256):
            for BLOCK_K in (128, 256, 512):
                for GROUP_M in (4, 8):
                    for num_warps in (4, 8):
                        for num_stages in (1, 2):
                            for waves_per_eu in (0, 2):
                                for matrix_instr_nonkdim in (16, 32):
                                    cfgs.append({
                                        "BLOCK_M": BLOCK_M,
                                        "BLOCK_N": BLOCK_N,
                                        "BLOCK_K": BLOCK_K,
                                        "GROUP_M": GROUP_M,
                                        "num_warps": num_warps,
                                        "num_stages": num_stages,
                                        "waves_per_eu": waves_per_eu,
                                        "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                        "kpack": KPACK,
                                    })
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("E", type=int)
    ap.add_argument("M", type=int)
    ap.add_argument("N", type=int)
    ap.add_argument("K", type=int)
    ap.add_argument("out_path")
    args = ap.parse_args()

    import torch
    from kernels import triton_mxfp4_grouped_mm
    from utils import (
        benchmark_cuda_function_in_microseconds,
        generate_jagged_offs,
        to_mx_fp4,
    )

    torch.manual_seed(123)
    random.seed(123)

    E, M, N, K = args.E, args.M, args.N, args.K
    device = torch.device("cuda")

    A = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    B_t = torch.randn((E, N, K), dtype=torch.bfloat16, device=device).transpose(-2, -1)
    offs = generate_jagged_offs(E, M, multiple_of=32)

    A_scales, A_fp4 = to_mx_fp4(A, block_size=32)
    B_nkK = B_t.transpose(-2, -1).contiguous()
    B_scales, B_fp4 = to_mx_fp4(B_nkK, block_size=32)

    cfgs = build_search_space()
    results = []
    t_start = time.time()
    for i, cfg in enumerate(cfgs):
        try:
            us = benchmark_cuda_function_in_microseconds(
                triton_mxfp4_grouped_mm,
                A_fp4, B_fp4, A_scales, B_scales, offs,
                **cfg,
            )
            results.append({"cfg": cfg, "us": float(us), "err": None})
        except Exception as ex:
            results.append({
                "cfg": cfg,
                "us": None,
                "err": f"{type(ex).__name__}: {ex}",
            })
        if (i + 1) % 32 == 0 or (i + 1) == len(cfgs):
            with open(args.out_path, "w") as f:
                json.dump({
                    "shape": [E, M, N, K],
                    "results": results,
                    "elapsed_s": time.time() - t_start,
                    "n_done": i + 1,
                    "n_total": len(cfgs),
                }, f)

    print("OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
