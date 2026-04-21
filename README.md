# groupedGemms

Standalone ROCm **MXFP8 grouped GEMM** Triton kernel for AMD MI350+ (gfx950 / CDNA4),
extracted from the `rocm-mxfp8-aiter-port` branch of
[torchao](https://github.com/pytorch/ao) for isolated iteration.

The kernel is a drop-in stand-in for `torch._scaled_grouped_mm`'s MXFP8 path
until that ships on ROCm. It uses `tl.dot_scaled` to consume per-block e8m0
scales directly; scheduling follows AMD aiter's MoE matmul (XCD swizzle +
GROUP_M L2 reuse + packed expert-to-tile routing), with a CDNA4-specific
pre-shuffled scale layout that removes the `#blocked → #linear1` permute
chain on the MFMA scale load.

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
- PyTorch ROCm build with `torch.float8_e4m3fn` support
- Triton with ROCm backend that supports `tl.dot_scaled` and
  `matrix_instr_nonkdim=16/32`

```bash
pip install -r requirements.txt
```

## Run the bench

```bash
python bench.py
```

This walks the same 36 Llama4 shapes the torchao CI bench uses
(`E ∈ {1, 2, 4, 8}`, `M = 16640`, `N, K ∈ {2048, 5120, 8192}`) and prints
per-shape bf16 µs, MXFP8 µs, TFLOPs, speedup, plus a geomean.

## Run the correctness test

```bash
python test_correctness.py
```

Compares `triton_mxfp8_grouped_mm` against a dequantized-input bf16 matmul
reference on three small shapes; tolerances are loose (MXFP8 is lossy) —
the goal is to catch indexing / layout regressions.

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
