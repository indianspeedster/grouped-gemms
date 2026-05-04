"""Per-shape sweep worker. One shape, all configs, prints JSON to stdout.

Invoked as: python tune_worker.py <E> <M> <N> <K> <out_path>
With CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES pinning a single GPU.

Writes a JSON {"shape": [E,M,N,K], "results": [{"cfg": {...}, "us": float|null,
"err": str|null}, ...]} to <out_path>. Prints "OK" on stdout when done.

Per CLAUDE.md gotcha #1, one shape per process avoids the multi-shape GPU
memory-fault crash, so this is intentionally one-and-done.
"""
import argparse
import json
import os
import random
import sys
import time
import traceback


def build_search_space():
    """288 configs, structured with cheapest-first ordering so any partial
    result is still informative."""
    cfgs = []
    for BLOCK_M in (64, 128, 256):
        for BLOCK_N in (128, 256):
            for BLOCK_K in (128, 256):
                for GROUP_M in (1, 4, 8):
                    for num_warps in (4, 8):
                        for num_stages in (1, 2):
                            for waves_per_eu in (0, 2):
                                cfgs.append({
                                    "BLOCK_M": BLOCK_M,
                                    "BLOCK_N": BLOCK_N,
                                    "BLOCK_K": BLOCK_K,
                                    "GROUP_M": GROUP_M,
                                    "num_warps": num_warps,
                                    "num_stages": num_stages,
                                    "waves_per_eu": waves_per_eu,
                                })
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("E", type=int)
    ap.add_argument("M", type=int)
    ap.add_argument("N", type=int)
    ap.add_argument("K", type=int)
    ap.add_argument("out_path")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max configs (smoke test).")
    args = ap.parse_args()

    import torch
    from kernels import triton_mxfp8_grouped_mm
    from utils import (
        benchmark_cuda_function_in_microseconds,
        generate_jagged_offs,
        to_mx,
    )

    torch.manual_seed(123)
    random.seed(123)

    E, M, N, K = args.E, args.M, args.N, args.K
    device = torch.device("cuda")

    A = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    B_t = torch.randn((E, N, K), dtype=torch.bfloat16, device=device).transpose(-2, -1)
    offs = generate_jagged_offs(E, M, multiple_of=32)

    A_scales, A_fp8 = to_mx(A, elem_dtype=torch.float8_e4m3fn, block_size=32)
    B_nkK = B_t.transpose(-2, -1).contiguous()  # (E, N, K)
    B_scales, B_fp8 = to_mx(B_nkK, elem_dtype=torch.float8_e4m3fn, block_size=32)

    cfgs = build_search_space()
    if args.limit is not None:
        cfgs = cfgs[: args.limit]

    results = []
    t_start = time.time()
    for i, cfg in enumerate(cfgs):
        try:
            us = benchmark_cuda_function_in_microseconds(
                triton_mxfp8_grouped_mm,
                A_fp8, B_fp8, A_scales, B_scales, offs,
                **cfg,
            )
            results.append({"cfg": cfg, "us": float(us), "err": None})
        except Exception as ex:
            results.append({
                "cfg": cfg,
                "us": None,
                "err": f"{type(ex).__name__}: {ex}",
            })

        # Periodic flush so a crash mid-sweep still leaves partial data.
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
