# grouped-gemms

Standalone ROCm **MXFP8 grouped GEMM** Triton kernel for AMD MI350+ (gfx950 / CDNA4),
extracted from the `rocm-mxfp8-aiter-port` branch of
[torchao](https://github.com/pytorch/ao) for isolated iteration.

The kernel is a drop-in stand-in for `torch._scaled_grouped_mm`'s MXFP8 path
until that ships on ROCm. It uses `tl.dot_scaled` to consume per-block e8m0
scales directly; scheduling follows AMD aiter's MoE matmul (XCD swizzle +
GROUP_M L2 reuse + packed expert-to-tile routing), with a CDNA4-specific
pre-shuffled scale layout that removes the `#blocked → #linear1` permute
chain on the MFMA scale load.

## Quickstart

```bash
git clone https://github.com/indianspeedster/grouped-gemms.git
cd grouped-gemms

# Creates a clean venv with the exact tested torch / triton-rocm versions.
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Sanity-check correctness (SQNR vs torch._grouped_mm, 27 dB threshold).
python test_correctness.py

# Run the 36-shape Llama4 bench (bf16 vs MXFP8 + geomean speedup).
python bench.py
```

Expected on a healthy MI355X: all 4 correctness shapes pass at ~27.6 dB
SQNR, bench geomean ≈ 1.4× over bf16.

## Contents

| Path | Purpose |
| --- | --- |
| `kernels/forward.py` | `triton_mxfp8_grouped_mm` — forward + dgrad (A @ B^T per group) |
| `kernels/backward.py` | `triton_mxfp8_wgrad` — weight gradient (A^T @ B per group) |
| `kernels/_common.py` | ROCm availability probe shared by both kernels |
| `kernels/__init__.py` | Re-exports both entry points |
| `utils.py` | Minimal `to_mx`, `generate_jagged_offs`, bench helper |
| `bench.py` | 36-shape Llama4 bf16-vs-MXFP8 bench (mirrors torchao CI) |
| `test_correctness.py` | Sanity check vs bf16 reference on small shapes |

No torchao dependency at runtime. Import from the package:

```python
from kernels import triton_mxfp8_grouped_mm, triton_mxfp8_wgrad
```

## Requirements

- AMD MI350+ (gfx950) — the kernel's CDNA4_SCALE fast-path needs
  `v_mfma_scale_f32_16x16x128_f8f6f4`; older arches will fail to compile
- PyTorch ROCm build with `torch.float8_e4m3fn` support (tested on torch
  `2.13.0.dev20260416+rocm7.1`)
- Triton with ROCm backend that supports `tl.dot_scaled` and
  `matrix_instr_nonkdim=16/32` (tested on `triton-rocm 3.7.0+gitb4e20bbe`,
  pulled in automatically as a torch dep)

`pip install -r requirements.txt` resolves the torch nightly wheel via the
`--extra-index-url` line in the file; no separate PyTorch install step is
needed. See `requirements.txt` for the full tested-against versions.

## Bench

```bash
python bench.py
```

Walks the same 36 Llama4 shapes the torchao CI bench uses
(`E ∈ {1, 2, 4, 8}`, `M = 16640`, `N, K ∈ {2048, 5120, 8192}`) and prints
per-shape bf16 µs, MXFP8 µs, TFLOPs, speedup, plus a geomean. Latest run
on MI355X: **1.394× geomean**, best 2.18× (E=8, N=K=2048), worst 1.16×
(E=2, K=8192).

## Correctness test

```bash
python test_correctness.py
```

Compares `triton_mxfp8_grouped_mm` output against `torch._grouped_mm` on the
original bf16 inputs using the same **SQNR metric and 27.0 dB threshold** as
torchao's `test/prototype/moe_training/test_mxfp8_grouped_mm.py`:

```
compute_error(x, y) = 20 * log10(||x|| / ||x - y||)   # dB
min_sqnr = 27.0                                        # matches ao forward test
```

Covers 4 shapes (E ∈ {1, 4, 8}); typical passing SQNR is ~27.6 dB.

## Kernel summary

`triton_mxfp8_grouped_mm(A, B, A_scales, B_scales, offsets)` where
- `A: (M, K)` fp8 e4m3fn, row-major
- `B: (E, N, K)` fp8 e4m3fn, row-major (viewed column-major internally)
- `A_scales: (M, K/32)` uint8 (e8m0 bias 127)
- `B_scales: (E, N, K/32)` uint8
- `offsets: (E,)` int32 cumulative token counts per expert; must be
  multiples of 32 (the MX block_size)

Returns `(M, N)` bf16 with `out[s:e] = A[s:e] @ B[g]^T` per expert group.

Tile selection: shape-aware (BLOCK_N, BLOCK_K) — 128x128 for small
`(N=K=2048)`, 256x256 otherwise (from an 864-run sweep on MI355X). XCD
swizzle=8 and GROUP_M=8 are the MI300/MI350 defaults.

## License

BSD 3-Clause — see [LICENSE](LICENSE). Original code © Meta Platforms, Inc.
