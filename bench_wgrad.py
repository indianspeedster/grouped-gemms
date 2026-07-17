##############################################################################
# Bench MXFP8 grouped wgrad variants (native, gluon) vs bf16 reference.
# Uses DSv3 shapes — same pattern as bench.py for forward.
##############################################################################

import argparse
import itertools
import math
import torch
from dataclasses import dataclass
from typing import List

from kernels.mxfp8.backward import triton_mxfp8_wgrad
from kernels.mxfp8.wgrad_gluon import triton_mxfp8_wgrad_gluon
from utils import benchmark_cuda_function_in_microseconds, generate_jagged_offs, to_mx

device = torch.device("cuda")

# DSv3 671B wgrad shapes. GO: (N,M) = grad_output, IA: (K,M) = input.
# For gate/up proj (w1/w3): wgrad is (E, N, K) with N=2048, K=7168.
# For down proj (w2): wgrad is (E, N, K) with N=7168, K=2048.
_DSV3_WGRAD = [
    # gate/up proj
    (4, 32768, 2048, 7168),
    (8, 32768, 2048, 7168),
    (4, 128000, 2048, 7168),
    (8, 128000, 2048, 7168),
    # down proj
    (4, 32768, 7168, 2048),
    (8, 32768, 7168, 2048),
    (4, 128000, 7168, 2048),
    (8, 128000, 7168, 2048),
]

# Llama4 shapes for wgrad: M=16640, N×K combos
_LLAMA4_WGRAD = [
    (e, 16640, n, k)
    for e, n, k in itertools.product([1, 2, 4, 8], [2048, 5120, 8192], [2048, 5120, 8192])
]

SHAPE_SETS = ("dsv3", "llama4")


@dataclass(frozen=True)
class WgradResult:
    bf16_us: float
    native_us: float
    gluon_us: float


def bench_bf16_wgrad(go_bf16, ia_bf16, offsets):
    """bf16 reference: grouped matmul -> wgrad."""
    E = offsets.shape[0]
    M = go_bf16.shape[1]
    N = go_bf16.shape[0]
    K = ia_bf16.shape[0]

    def _ref():
        out = torch.zeros((E, N, K), dtype=torch.bfloat16, device=go_bf16.device)
        for e in range(E):
            s = offsets[e-1].item() if e > 0 else 0
            e_end = offsets[e].item()
            go_g = go_bf16[:, s:e_end].float()
            ia_g = ia_bf16[:, s:e_end].float()
            out[e] = (go_g @ ia_g.T).bfloat16()
        return out

    return benchmark_cuda_function_in_microseconds(_ref)


def bench_native_wgrad(go_fp8, go_scale, ia_fp8, ia_scale, offsets):
    return benchmark_cuda_function_in_microseconds(
        triton_mxfp8_wgrad, go_fp8, go_scale, ia_fp8, ia_scale, offsets
    )


def bench_gluon_wgrad(go_fp8, go_scale, ia_fp8, ia_scale, offsets):
    return benchmark_cuda_function_in_microseconds(
        triton_mxfp8_wgrad_gluon, go_fp8, go_scale, ia_fp8, ia_scale, offsets
    )


def run_wgrad_bench(e, m, n, k):
    torch.manual_seed(0)
    go_bf16 = torch.randn((n, m), dtype=torch.bfloat16, device=device)
    ia_bf16 = torch.randn((k, m), dtype=torch.bfloat16, device=device)
    offsets = generate_jagged_offs(e, m, multiple_of=32)

    go_scale, go_fp8 = to_mx(go_bf16, torch.float8_e4m3fn, 32)
    ia_scale, ia_fp8 = to_mx(ia_bf16, torch.float8_e4m3fn, 32)

    bf16_us = bench_bf16_wgrad(go_bf16, ia_bf16, offsets)
    native_us = bench_native_wgrad(go_fp8, go_scale, ia_fp8, ia_scale, offsets)
    gluon_us = bench_gluon_wgrad(go_fp8, go_scale, ia_fp8, ia_scale, offsets)

    return WgradResult(bf16_us=round(bf16_us, 3), native_us=round(native_us, 3), gluon_us=round(gluon_us, 3))


def print_results(results: List, shapes: List):
    print(f"{'E':>3s} {'M':>7s} {'N':>5s} {'K':>5s}  "
          f"{"bf16_us":>10s} {"native_us":>10s} {"gluon_us":>10s}  "
          f"{"nat_x":>7s} {"glu_x":>7s}  "
          f"{"nat_TF":>7s} {"glu_TF":>7s}")
    print("-" * 100)

    geo_nat = []  # log speedups
    geo_glu = []
    geo_bf16_tf = []
    geo_nat_tf = []
    geo_glu_tf = []

    for (e, m, n, k), r in zip(shapes, results):
        flops = 2 * n * k * m
        bf16_tf = (flops / 1e12) / (r.bf16_us / 1e6)
        nat_tf = (flops / 1e12) / (r.native_us / 1e6)
        glu_tf = (flops / 1e12) / (r.gluon_us / 1e6)
        nat_x = r.bf16_us / r.native_us
        glu_x = r.bf16_us / r.gluon_us

        if all(s > 0 and s != float("inf") for s in (nat_x, glu_x)):
            geo_nat.append(math.log(nat_x))
            geo_glu.append(math.log(glu_x))
            geo_bf16_tf.append(math.log(bf16_tf))
            geo_nat_tf.append(math.log(nat_tf))
            geo_glu_tf.append(math.log(glu_tf))

        print(f"{e:3d} {m:7d} {n:5d} {k:5d}  "
              f"{r.bf16_us:>10.1f} {r.native_us:>10.1f} {r.gluon_us:>10.1f}  "
              f"{nat_x:>5.2f}x {glu_x:>5.2f}x  "
              f"{nat_tf:>6.0f} {glu_tf:>6.0f}")

    if geo_nat:
        n = len(geo_nat)
        print(f"\nGeomean ({n} shapes):")
        print(f"  native vs bf16: {math.exp(sum(geo_nat)/n):.2f}x  "
              f"gluon vs bf16: {math.exp(sum(geo_glu)/n):.2f}x")
        print(f"  gluon vs native: {math.exp(sum(geo_glu)/n - sum(geo_nat)/n):.2f}x")
        print(f"  native TFLOPS: {math.exp(sum(geo_nat_tf)/n):.0f}  "
              f"gluon TFLOPS: {math.exp(sum(geo_glu_tf)/n):.0f}")


def main():
    parser = argparse.ArgumentParser(description="Bench wgrad kernels")
    parser.add_argument("--shapes", default="dsv3", choices=SHAPE_SETS)
    args = parser.parse_args()

    shapes = _DSV3_WGRAD if args.shapes == "dsv3" else _LLAMA4_WGRAD

    results = []
    for e, m, n, k in shapes:
        print(f"Benchmarking ({e},{m},{n},{k})...")
        results.append(run_wgrad_bench(e, m, n, k))

    print_results(results, shapes)


if __name__ == "__main__":
    main()
