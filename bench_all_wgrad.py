"""Compare ALL wgrad variants: native, fast, v2, gluon."""
import time, math, torch
from kernels.mxfp8.backward import triton_mxfp8_wgrad, triton_mxfp8_wgrad_fast, triton_mxfp8_wgrad_v2
from kernels.mxfp8.wgrad_gluon import triton_mxfp8_wgrad_gluon
from kernels.mxfp8.stack_copy import triton_mxfp8_wgrad_stacked
from utils import generate_jagged_offs, to_mx, benchmark_cuda_function_in_microseconds

torch.manual_seed(42)
device = "cuda"

shapes = [
    (4, 32768, 2048, 7168, "gate/up"),
    (8, 32768, 2048, 7168, "gate/up"),
    (4, 128000, 2048, 7168, "gate/up"),
    (8, 128000, 2048, 7168, "gate/up"),
    (4, 32768, 7168, 2048, "down"),
    (8, 32768, 7168, 2048, "down"),
    (4, 128000, 7168, 2048, "down"),
    (8, 128000, 7168, 2048, "down"),
]

print(f"{'Shape':>25s}  {'native':>8s}  {'fast':>8s}  {'v2':>8s}  {'gluon':>8s}  {'stacked':>8s}  {'best':>8s}")
print("-" * 85)

geo_nat, geo_fast, geo_v2, geo_glu, geo_stk = [], [], [], [], []

for e, m, n, k, label in shapes:
    go = torch.randn((n, m), dtype=torch.bfloat16, device=device)
    ia = torch.randn((k, m), dtype=torch.bfloat16, device=device)
    offs = generate_jagged_offs(e, m, multiple_of=32)
    gs, gf = to_mx(go, torch.float8_e4m3fn, 32)
    iss, iaf = to_mx(ia, torch.float8_e4m3fn, 32)
    flops = 2 * n * k * m

    nat = benchmark_cuda_function_in_microseconds(triton_mxfp8_wgrad, gf, gs, iaf, iss, offs)
    fast = benchmark_cuda_function_in_microseconds(triton_mxfp8_wgrad_fast, gf, gs, iaf, iss, offs)
    v2 = benchmark_cuda_function_in_microseconds(triton_mxfp8_wgrad_v2, gf, gs, iaf, iss, offs)
    glu = benchmark_cuda_function_in_microseconds(triton_mxfp8_wgrad_gluon, gf, gs, iaf, iss, offs)
    stk = benchmark_cuda_function_in_microseconds(triton_mxfp8_wgrad_stacked, gf, gs, iaf, iss, offs)

    nat_tf = flops / nat / 1e6
    fast_tf = flops / fast / 1e6
    v2_tf = flops / v2 / 1e6
    glu_tf = flops / glu / 1e6
    stk_tf = flops / stk / 1e6

    geo_nat.append(math.log(nat_tf))
    geo_fast.append(math.log(fast_tf))
    geo_v2.append(math.log(v2_tf))
    geo_glu.append(math.log(glu_tf))
    geo_stk.append(math.log(stk_tf))

    best = max(nat_tf, fast_tf, v2_tf, glu_tf, stk_tf)
    print(f"({e},{m//1000:>3d}K,{n},{k}) {label:>7s}  {nat_tf:>6.0f}  {fast_tf:>6.0f}  {v2_tf:>6.0f}  {glu_tf:>6.0f}  {stk_tf:>6.0f}  {best:>6.0f}")

n = len(shapes)
print(f"\nGeomean TFLOPS:  native={math.exp(sum(geo_nat)/n):.0f}  fast={math.exp(sum(geo_fast)/n):.0f}  v2={math.exp(sum(geo_v2)/n):.0f}  gluon={math.exp(sum(geo_glu)/n):.0f}  stacked={math.exp(sum(geo_stk)/n):.0f}")
print(f"vs peak (5230 TFLOPS):  native={math.exp(sum(geo_nat)/n)/5230*100:.1f}%  fast={math.exp(sum(geo_fast)/n)/5230*100:.1f}%  v2={math.exp(sum(geo_v2)/n)/5230*100:.1f}%  gluon={math.exp(sum(geo_glu)/n)/5230*100:.1f}%  stacked={math.exp(sum(geo_stk)/n)/5230*100:.1f}%")
