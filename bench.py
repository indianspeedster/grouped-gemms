# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Minimal replica of torchao benchmarks/prototype/moe_training/
# bench_2d_3d_grouped_gemm.py — Llama4 shapes, bf16 baseline vs MXFP8 grouped
# mm on MI350+. The FP8 rowwise arm (Hopper-only) and CUDA SM100 MXFP8 arm
# are dropped; only the ROCm triton path is benched here.

import argparse
import itertools
import logging
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm

from rocm_mxfp8_mm import triton_mxfp8_grouped_mm
from utils import (
    benchmark_cuda_function_in_microseconds,
    generate_jagged_offs,
    is_MI350,
    to_mx,
)

device = torch.device("cuda")


@dataclass(frozen=True)
class ExperimentConfig:
    e: int
    m: int
    n: int
    k: int


@dataclass(frozen=True)
class ExperimentResult:
    bf16_us: float
    mxfp8_us: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    # Llama4 shapes (same 36-shape sweep as the ao CI bench).
    M = [16640]
    K = [2048, 5120, 8192]
    N = [2048, 5120, 8192]
    E = [1, 2, 4, 8]
    return [
        ExperimentConfig(e=e, m=m, n=n, k=k)
        for e, m, n, k in itertools.product(E, M, N, K)
    ]


def bench_mxfp8_grouped_mm_rocm(A, B_t, offs, block_size: int = 32) -> float:
    A_scales, A_fp8 = to_mx(A, elem_dtype=torch.float8_e4m3fn, block_size=block_size)
    B_nkK = B_t.transpose(-2, -1).contiguous()  # (E, N, K)
    B_scales, B_fp8 = to_mx(
        B_nkK, elem_dtype=torch.float8_e4m3fn, block_size=block_size
    )

    E = offs.shape[0]
    Mg = A.shape[0]
    offs_mxfp8 = generate_jagged_offs(E, Mg, multiple_of=block_size)

    return benchmark_cuda_function_in_microseconds(
        triton_mxfp8_grouped_mm,
        A_fp8,
        B_fp8,
        A_scales,
        B_scales,
        offs_mxfp8,
    )


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    e, m, n, k = config.e, config.m, config.n, config.k

    A = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    B_t = torch.randn(
        (e, n, k), dtype=torch.bfloat16, device=device, requires_grad=True
    ).transpose(-2, -1)

    # jagged offsets for bf16 baseline (16-aligned — torch._grouped_mm's requirement)
    Mg = A.shape[0]
    offs = generate_jagged_offs(e, Mg, multiple_of=16)

    bf16_us = benchmark_cuda_function_in_microseconds(
        torch._grouped_mm, A, B_t, offs, out_dtype=torch.bfloat16,
    )

    if is_MI350():
        mxfp8_us = bench_mxfp8_grouped_mm_rocm(A, B_t, offs)
    else:
        logging.warning(
            "MXFP8 path only runs on gfx950+ (MI350+). Got %s",
            torch.cuda.get_device_properties(0).gcnArchName
            if torch.cuda.is_available() else "no CUDA device",
        )
        mxfp8_us = float("inf")

    return ExperimentResult(bf16_us=round(bf16_us, 3), mxfp8_us=round(mxfp8_us, 3))


def print_results(experiments: List[Experiment]):
    headers = [
        "E", "M", "N", "K",
        "bf16_us", "mxfp8_us",
        "bf16_tflops", "mxfp8_tflops",
        "mxfp8_speedup",
    ]
    rows = []
    geo_speedup_log = 0.0
    geo_n = 0
    for exp in experiments:
        m, n, k = exp.config.m, exp.config.n, exp.config.k
        flops = 2 * m * n * k
        bf16_tflops = (flops / 1e12) / (exp.result.bf16_us / 1e6)
        mxfp8_tflops = (flops / 1e12) / (exp.result.mxfp8_us / 1e6)
        speedup = exp.result.bf16_us / exp.result.mxfp8_us
        if speedup > 0 and speedup != float("inf"):
            geo_speedup_log += torch.log(torch.tensor(speedup)).item()
            geo_n += 1
        rows.append([
            exp.config.e, m, n, k,
            exp.result.bf16_us, exp.result.mxfp8_us,
            round(bf16_tflops, 3), round(mxfp8_tflops, 3),
            f"{speedup:.2f}x",
        ])
    print(tabulate(rows, headers=headers))
    if geo_n:
        import math
        print(f"\nGeomean MXFP8 speedup vs bf16: {math.exp(geo_speedup_log / geo_n):.3f}x "
              f"({geo_n} shapes)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="llama4",
                        help="Shape set (currently only 'llama4' — 36 shapes)")
    args = parser.parse_args()
    del args  # only one shape set for now

    torch.random.manual_seed(123)
    configs = get_configs()
    results = [Experiment(config=c, result=run_experiment(c)) for c in tqdm(configs)]
    print_results(results)


if __name__ == "__main__":
    main()
