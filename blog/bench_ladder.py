# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark each optimization rung of blog/ladder.py on representative Llama4
shapes. For every (shape, rung): verify SQNR vs the bf16 reference, time it,
and report us / TFLOPS / speedup-over-naive and -over-bf16.

One shape per subprocess (pinned to its own GPU) to dodge the
many-shapes-per-process GPU memory fault noted in CLAUDE.md. The driver
fans the representative shapes across GPUs and merges to blog/ladder_results.json.

    python blog/bench_ladder.py            # run the full sweep + write JSON
    python blog/bench_ladder.py --one E N K  # single shape, prints one JSON line
"""
import argparse
import json
import math
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
MIN_SQNR_DB = 27.0
M = 16640

# Representative subset spanning the regimes.
SHAPES = [
    (8, 2048, 2048),   # biggest MXFP8 win (small K, bf16 starved)
    (1, 8192, 8192),   # peak-TFLOPS large square
    (2, 8192, 2048),   # worst observed speedup
    (4, 5120, 5120),   # mid
    (8, 5120, 8192),   # large N+K (nk16 regime)
    (1, 2048, 2048),   # the BLOCK_K=128 special case
]


def run_one(E, N, K):
    """Bench all rungs for a single shape in THIS process. Returns a dict."""
    import torch
    sys.path.insert(0, REPO)
    from blog.ladder import ladder_grouped_mm, RUNGS
    from utils import (benchmark_cuda_function_in_microseconds,
                       generate_jagged_offs, to_mx)
    import random
    random.seed(123)
    torch.manual_seed(0)
    device = torch.device("cuda")

    A = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    B_nkK = torch.randn((E, N, K), dtype=torch.bfloat16, device=device)
    B_t_hp = B_nkK.transpose(-2, -1)
    A_scales, A_fp8 = to_mx(A, block_size=32)
    B_scales, B_fp8 = to_mx(B_nkK, block_size=32)
    offs = generate_jagged_offs(E, M, multiple_of=32)

    out_ref = torch._grouped_mm(A, B_t_hp, offs=offs.to(torch.int32),
                                out_dtype=torch.bfloat16)
    bf16_us = benchmark_cuda_function_in_microseconds(
        torch._grouped_mm, A, B_t_hp, offs.to(torch.int32),
        out_dtype=torch.bfloat16)

    def sqnr(x, y):
        Ps = torch.linalg.vector_norm(x.float())
        Pn = torch.linalg.vector_norm((x - y).float())
        return (20 * torch.log10(Ps / Pn)).item()

    flops = 2 * M * N * K
    rungs = []
    naive_us = None
    for r in sorted(RUNGS):
        out = ladder_grouped_mm(A_fp8, B_fp8, A_scales, B_scales, offs, rung=r)
        s = sqnr(out_ref, out)
        us = benchmark_cuda_function_in_microseconds(
            ladder_grouped_mm, A_fp8, B_fp8, A_scales, B_scales, offs, r)
        if naive_us is None:
            naive_us = us
        rungs.append(dict(rung=r, name=RUNGS[r], us=round(us, 2),
                          tflops=round((flops / 1e12) / (us / 1e6), 1),
                          sqnr=round(s, 2), vs_naive=round(naive_us / us, 3),
                          vs_bf16=round(bf16_us / us, 3),
                          sqnr_ok=bool(s >= MIN_SQNR_DB)))
    return dict(E=E, M=M, N=N, K=K, bf16_us=round(bf16_us, 2), rungs=rungs)


def driver():
    procs = []
    for i, (E, N, K) in enumerate(SHAPES):
        gpu = i % 8
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   HIP_VISIBLE_DEVICES=str(gpu), PYTHONPATH=REPO)
        p = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--one",
             str(E), str(N), str(K)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
        procs.append(((E, N, K), p))

    results = []
    for (E, N, K), p in procs:
        out, err = p.communicate()
        line = next((l for l in out.splitlines() if l.startswith("{")), None)
        if line is None:
            print(f"!! shape E{E} N{N} K{K} failed:\n{err[-600:]}")
            continue
        res = json.loads(line)
        results.append(res)
        print(f"\n=== E={res['E']} M={M} N={res['N']} K={res['K']}  "
              f"(bf16 {res['bf16_us']} us) ===")
        for r in res["rungs"]:
            flag = "" if r["sqnr_ok"] else "  !! SQNR FAIL"
            print(f"  {r['name']:<26} {r['us']:>9.2f} us  {r['tflops']:>6.0f} TF  "
                  f"vs_naive {r['vs_naive']:>5.2f}x  vs_bf16 {r['vs_bf16']:>5.2f}x  "
                  f"sqnr {r['sqnr']:.1f}{flag}")

    if not results:
        print("no results")
        return
    from blog.ladder import RUNGS
    agg = []
    for i, r in enumerate(sorted(RUNGS)):
        gn = math.exp(sum(math.log(s["rungs"][i]["vs_naive"]) for s in results) / len(results))
        gb = math.exp(sum(math.log(s["rungs"][i]["vs_bf16"]) for s in results) / len(results))
        agg.append(dict(rung=r, name=RUNGS[r],
                        geomean_vs_naive=round(gn, 3), geomean_vs_bf16=round(gb, 3)))

    out = dict(M=M, shapes=results, geomean=agg,
               note=f"representative {len(results)}-shape subset, MI355X")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ladder_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== geomean across representative shapes ===")
    for a in agg:
        print(f"  {a['name']:<26} vs_naive {a['geomean_vs_naive']:>5.2f}x   "
              f"vs_bf16 {a['geomean_vs_bf16']:>5.2f}x")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", nargs=3, type=int, metavar=("E", "N", "K"))
    args = ap.parse_args()
    if args.one:
        print(json.dumps(run_one(*args.one)))
    else:
        driver()
