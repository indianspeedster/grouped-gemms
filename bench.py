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

from kernels import triton_mxfp8_grouped_mm
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
    rowwise_us: float
    tensorwise_us: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


_FP8_E4M3_MAX = 448.0


def _quant_rowwise_fp8(A_bf16: torch.Tensor, B_nk_bf16: torch.Tensor):
    """Rowwise FP8 quant: per-row scale on A (over K), per-N-feature scale
    on B (over K). Returns (A_fp8, B_t_fp8 col-major, a_scale (M,), b_scale (E,N))."""
    a_amax = A_bf16.abs().amax(dim=-1).to(torch.float32).clamp(min=1e-12)
    a_scale = a_amax / _FP8_E4M3_MAX
    A_fp8 = (A_bf16.float() / a_scale.unsqueeze(-1)).clamp(
        -_FP8_E4M3_MAX, _FP8_E4M3_MAX
    ).to(torch.float8_e4m3fn)

    b_amax = B_nk_bf16.abs().amax(dim=-1).to(torch.float32).clamp(min=1e-12)
    b_scale = b_amax / _FP8_E4M3_MAX  # (E, N)
    B_fp8_nk = (B_nk_bf16.float() / b_scale.unsqueeze(-1)).clamp(
        -_FP8_E4M3_MAX, _FP8_E4M3_MAX
    ).to(torch.float8_e4m3fn)
    return A_fp8, B_fp8_nk.transpose(-2, -1), a_scale, b_scale


def _quant_tensorwise_fp8(A_bf16: torch.Tensor, B_nk_bf16: torch.Tensor):
    """Tensorwise FP8 quant: one scalar for A, one scalar for B (broadcast
    into the (M,) and (E,N) tensors the op requires)."""
    M = A_bf16.shape[0]
    E, N = B_nk_bf16.shape[0], B_nk_bf16.shape[1]
    a_scalar = (A_bf16.abs().amax().to(torch.float32) / _FP8_E4M3_MAX).clamp(min=1e-12)
    A_fp8 = (A_bf16.float() / a_scalar).clamp(
        -_FP8_E4M3_MAX, _FP8_E4M3_MAX
    ).to(torch.float8_e4m3fn)

    b_scalar = (B_nk_bf16.abs().amax().to(torch.float32) / _FP8_E4M3_MAX).clamp(min=1e-12)
    B_fp8_nk = (B_nk_bf16.float() / b_scalar).clamp(
        -_FP8_E4M3_MAX, _FP8_E4M3_MAX
    ).to(torch.float8_e4m3fn)

    a_scale = a_scalar.expand(M).contiguous()
    b_scale = b_scalar.expand(E, N).contiguous()
    return A_fp8, B_fp8_nk.transpose(-2, -1), a_scale, b_scale


def bench_rowwise_fp8_grouped_mm(A_bf16, B_nk_bf16, offs) -> float:
    A_fp8, B_t_fp8, a_scale, b_scale = _quant_rowwise_fp8(A_bf16, B_nk_bf16)
    return benchmark_cuda_function_in_microseconds(
        torch._scaled_grouped_mm,
        A_fp8, B_t_fp8, a_scale, b_scale,
        offs, None, None, torch.bfloat16,
    )


def bench_tensorwise_fp8_grouped_mm(A_bf16, B_nk_bf16, offs) -> float:
    A_fp8, B_t_fp8, a_scale, b_scale = _quant_tensorwise_fp8(A_bf16, B_nk_bf16)
    return benchmark_cuda_function_in_microseconds(
        torch._scaled_grouped_mm,
        A_fp8, B_t_fp8, a_scale, b_scale,
        offs, None, None, torch.bfloat16,
    )


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

    # Use the caller's offs directly so bf16 and MXFP8 see the same jagged
    # partition. The caller is responsible for 32-alignment (MX block_size);
    # the bench's run_experiment enforces that.
    return benchmark_cuda_function_in_microseconds(
        triton_mxfp8_grouped_mm,
        A_fp8,
        B_fp8,
        A_scales,
        B_scales,
        offs,
    )


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    e, m, n, k = config.e, config.m, config.n, config.k

    A = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    B_t = torch.randn(
        (e, n, k), dtype=torch.bfloat16, device=device, requires_grad=True
    ).transpose(-2, -1)

    # Single jagged-offs draw, 32-aligned (MX block_size). 32 is a multiple
    # of 16, so torch._grouped_mm accepts it too — both paths see the
    # identical per-expert token partition.
    Mg = A.shape[0]
    offs = generate_jagged_offs(e, Mg, multiple_of=32)

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

    # torch._scaled_grouped_mm consumes the same (E, N, K) bf16 weight pre-quant.
    B_nk = B_t.transpose(-2, -1).contiguous()
    rowwise_us = bench_rowwise_fp8_grouped_mm(A, B_nk, offs)
    tensorwise_us = bench_tensorwise_fp8_grouped_mm(A, B_nk, offs)

    return ExperimentResult(
        bf16_us=round(bf16_us, 3),
        mxfp8_us=round(mxfp8_us, 3),
        rowwise_us=round(rowwise_us, 3),
        tensorwise_us=round(tensorwise_us, 3),
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "E", "M", "N", "K",
        "bf16_us", "mxfp8_us", "rowwise_us", "tensorwise_us",
        "mxfp8_x", "rowwise_x", "tensorwise_x",
        "mxfp8_TFLOPS", "rowwise_TFLOPS", "tensorwise_TFLOPS",
    ]
    rows = []
    import math
    geo_x = {"mxfp8": 0.0, "rowwise": 0.0, "tensorwise": 0.0}
    geo_tflops = {"bf16": 0.0, "mxfp8": 0.0, "rowwise": 0.0, "tensorwise": 0.0}
    geo_n = 0
    for exp in experiments:
        m, n, k = exp.config.m, exp.config.n, exp.config.k
        flops = 2 * m * n * k
        bf16_tflops = (flops / 1e12) / (exp.result.bf16_us / 1e6)
        mxfp8_tflops = (flops / 1e12) / (exp.result.mxfp8_us / 1e6)
        rowwise_tflops = (flops / 1e12) / (exp.result.rowwise_us / 1e6)
        tensorwise_tflops = (flops / 1e12) / (exp.result.tensorwise_us / 1e6)
        mxfp8_x = exp.result.bf16_us / exp.result.mxfp8_us
        rowwise_x = exp.result.bf16_us / exp.result.rowwise_us
        tensorwise_x = exp.result.bf16_us / exp.result.tensorwise_us
        if all(s > 0 and s != float("inf") for s in (mxfp8_x, rowwise_x, tensorwise_x)):
            geo_x["mxfp8"] += math.log(mxfp8_x)
            geo_x["rowwise"] += math.log(rowwise_x)
            geo_x["tensorwise"] += math.log(tensorwise_x)
            geo_tflops["bf16"] += math.log(bf16_tflops)
            geo_tflops["mxfp8"] += math.log(mxfp8_tflops)
            geo_tflops["rowwise"] += math.log(rowwise_tflops)
            geo_tflops["tensorwise"] += math.log(tensorwise_tflops)
            geo_n += 1
        rows.append([
            exp.config.e, m, n, k,
            exp.result.bf16_us, exp.result.mxfp8_us,
            exp.result.rowwise_us, exp.result.tensorwise_us,
            f"{mxfp8_x:.2f}x", f"{rowwise_x:.2f}x", f"{tensorwise_x:.2f}x",
            round(mxfp8_tflops, 1),
            round(rowwise_tflops, 1),
            round(tensorwise_tflops, 1),
        ])
    print(tabulate(rows, headers=headers))
    if geo_n:
        print(
            f"\nGeomean speedup vs bf16 ({geo_n} shapes): "
            f"MXFP8={math.exp(geo_x['mxfp8']/geo_n):.3f}x  "
            f"rowwise={math.exp(geo_x['rowwise']/geo_n):.3f}x  "
            f"tensorwise={math.exp(geo_x['tensorwise']/geo_n):.3f}x"
        )
        print(
            f"Geomean TFLOPS ({geo_n} shapes): "
            f"bf16={math.exp(geo_tflops['bf16']/geo_n):.1f}  "
            f"MXFP8={math.exp(geo_tflops['mxfp8']/geo_n):.1f}  "
            f"rowwise={math.exp(geo_tflops['rowwise']/geo_n):.1f}  "
            f"tensorwise={math.exp(geo_tflops['tensorwise']/geo_n):.1f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="llama4",
                        help="Shape set (currently only 'llama4' — 36 shapes)")
    args = parser.parse_args()
    del args  # only one shape set for now

    torch.random.manual_seed(123)
    # generate_jagged_offs uses Python's random.sample — seed that too so
    # runs are deterministic across invocations.
    import random
    random.seed(123)
    configs = get_configs()
    results = [Experiment(config=c, result=run_experiment(c)) for c in tqdm(configs)]
    print_results(results)


if __name__ == "__main__":
    main()
