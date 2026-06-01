##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################

"""Bench MXFP4 grouped GEMM vs a bf16 ``torch._grouped_mm`` baseline on gfx950.
``--shapes {llama4,dsv3}`` selects the shape set."""

import argparse
import itertools
import math
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm

from kernels import triton_mxfp4_grouped_mm
from utils import (
    benchmark_cuda_function_in_microseconds,
    generate_jagged_offs,
    is_MI350,
    to_mx_fp4,
)

device = torch.device("cuda")


@dataclass(frozen=True)
class Cfg:
    e: int
    m: int
    n: int
    k: int


_LLAMA4_M = [16640]
_LLAMA4_K = [2048, 5120, 8192]
_LLAMA4_N = [2048, 5120, 8192]
_LLAMA4_E = [1, 2, 4, 8]

# DSv3 671B: N=2048, K=7168, E∈{4,8}, M∈{32768,128000} (mirrors bench.py).
_DSV3_EMNK = [
    (4, 32768, 2048, 7168),
    (8, 32768, 2048, 7168),
    (4, 128000, 2048, 7168),
    (8, 128000, 2048, 7168),
]


def get_configs(shape_set: str = "llama4") -> List[Cfg]:
    if shape_set == "llama4":
        return [
            Cfg(e=e, m=m, n=n, k=k)
            for e, m, n, k in itertools.product(
                _LLAMA4_E, _LLAMA4_M, _LLAMA4_N, _LLAMA4_K)
        ]
    if shape_set == "dsv3":
        return [Cfg(e=e, m=m, n=n, k=k) for e, m, n, k in _DSV3_EMNK]
    raise ValueError(f"unknown shape set: {shape_set}")


def bench_mxfp4(A, B_t, offs, block_size: int = 32) -> float:
    A_scales, A_fp4 = to_mx_fp4(A, block_size=block_size)
    B_nkK = B_t.transpose(-2, -1).contiguous()  # (E, N, K)
    B_scales, B_fp4 = to_mx_fp4(B_nkK, block_size=block_size)
    return benchmark_cuda_function_in_microseconds(
        triton_mxfp4_grouped_mm, A_fp4, B_fp4, A_scales, B_scales, offs,
    )


def run_experiment(cfg: Cfg):
    e, m, n, k = cfg.e, cfg.m, cfg.n, cfg.k
    A = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    B_t = torch.randn((e, n, k), dtype=torch.bfloat16, device=device).transpose(-2, -1)
    offs = generate_jagged_offs(e, m, multiple_of=32)

    bf16_us = benchmark_cuda_function_in_microseconds(
        torch._grouped_mm, A, B_t, offs, out_dtype=torch.bfloat16,
    )
    mxfp4_us = bench_mxfp4(A, B_t, offs) if is_MI350() else float("inf")
    return bf16_us, mxfp4_us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="llama4", choices=("llama4", "dsv3"))
    args = parser.parse_args()
    torch.random.manual_seed(123)
    import random
    random.seed(123)

    rows, geo_x, geo_tf, n_ok = [], 0.0, 0.0, 0
    for cfg in tqdm(get_configs(args.shapes)):
        bf16_us, mxfp4_us = run_experiment(cfg)
        flops = 2 * cfg.m * cfg.n * cfg.k
        bf16_tf = (flops / 1e12) / (bf16_us / 1e6)
        mxfp4_tf = (flops / 1e12) / (mxfp4_us / 1e6)
        x = bf16_us / mxfp4_us
        if x > 0 and x != float("inf"):
            geo_x += math.log(x); geo_tf += math.log(mxfp4_tf); n_ok += 1
        rows.append([cfg.e, cfg.m, cfg.n, cfg.k, round(bf16_us, 1),
                     round(mxfp4_us, 1), f"{x:.2f}x", round(bf16_tf, 1),
                     round(mxfp4_tf, 1)])

    print(tabulate(rows, headers=["E", "M", "N", "K", "bf16_us", "mxfp4_us",
                                  "mxfp4_x", "bf16_TFLOPS", "mxfp4_TFLOPS"]))
    if n_ok:
        print(f"\nGeomean speedup vs bf16 ({n_ok} shapes): "
              f"{math.exp(geo_x / n_ok):.3f}x")
        print(f"Geomean MXFP4 TFLOPS ({n_ok} shapes): "
              f"{math.exp(geo_tf / n_ok):.1f}")


if __name__ == "__main__":
    main()
