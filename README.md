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
per-shape bf16 µs, MXFP8 µs, TFLOPs, speedup, plus a geomean.

### Current performance (MI355X, gfx950, ROCm 7.1.52802)

**Geomean MXFP8 speedup vs bf16: 1.389×** (36 shapes, bf16 and MXFP8
both using the same jagged-offset partition per shape). Best 2.14×
(E=8, N=K=2048), worst 1.17× (E=2, N=8192, K=2048).

| E | M | N | K | bf16 µs | mxfp8 µs | bf16 TFLOPs | mxfp8 TFLOPs | speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 16640 | 2048 | 2048 |  159.0 |  103.8 |  878 | 1344 | 1.53× |
| 1 | 16640 | 2048 | 5120 |  330.5 |  250.0 | 1056 | 1396 | 1.32× |
| 1 | 16640 | 2048 | 8192 |  483.5 |  373.7 | 1155 | 1494 | 1.29× |
| 1 | 16640 | 5120 | 2048 |  317.5 |  246.1 | 1099 | 1418 | 1.29× |
| 1 | 16640 | 5120 | 5120 |  698.0 |  516.1 | 1250 | 1690 | 1.35× |
| 1 | 16640 | 5120 | 8192 | 1113.7 |  792.2 | 1253 | 1762 | 1.41× |
| 1 | 16640 | 8192 | 2048 |  473.1 |  378.0 | 1180 | 1477 | 1.25× |
| 1 | 16640 | 8192 | 5120 | 1064.6 |  801.9 | 1311 | 1741 | 1.33× |
| 1 | 16640 | 8192 | 8192 | 1654.1 | 1248.8 | 1350 | 1788 | 1.32× |
| 2 | 16640 | 2048 | 2048 |  142.0 |  102.4 |  983 | 1363 | 1.39× |
| 2 | 16640 | 2048 | 5120 |  323.7 |  248.2 | 1078 | 1406 | 1.30× |
| 2 | 16640 | 2048 | 8192 |  463.4 |  383.0 | 1205 | 1458 | 1.21× |
| 2 | 16640 | 5120 | 2048 |  361.7 |  247.8 |  965 | 1408 | 1.46× |
| 2 | 16640 | 5120 | 5120 |  661.8 |  530.6 | 1318 | 1644 | 1.25× |
| 2 | 16640 | 5120 | 8192 | 1093.2 |  815.6 | 1277 | 1711 | 1.34× |
| 2 | 16640 | 8192 | 2048 |  460.7 |  393.8 | 1212 | 1418 | 1.17× |
| 2 | 16640 | 8192 | 5120 | 1086.3 |  835.4 | 1285 | 1671 | 1.30× |
| 2 | 16640 | 8192 | 8192 | 1576.4 | 1282.9 | 1417 | 1741 | 1.23× |
| 4 | 16640 | 2048 | 2048 |  180.8 |  102.2 |  772 | 1366 | 1.77× |
| 4 | 16640 | 2048 | 5120 |  376.4 |  252.9 |  927 | 1380 | 1.49× |
| 4 | 16640 | 2048 | 8192 |  563.4 |  390.3 |  991 | 1430 | 1.44× |
| 4 | 16640 | 5120 | 2048 |  324.5 |  259.4 | 1075 | 1345 | 1.25× |
| 4 | 16640 | 5120 | 5120 |  720.8 |  541.4 | 1210 | 1611 | 1.33× |
| 4 | 16640 | 5120 | 8192 | 1065.0 |  845.2 | 1311 | 1652 | 1.26× |
| 4 | 16640 | 8192 | 2048 |  487.4 |  400.2 | 1146 | 1395 | 1.22× |
| 4 | 16640 | 8192 | 5120 | 1112.1 |  865.3 | 1255 | 1613 | 1.29× |
| 4 | 16640 | 8192 | 8192 | 1683.1 | 1333.4 | 1327 | 1675 | 1.26× |
| 8 | 16640 | 2048 | 2048 |  243.0 |  113.6 |  574 | 1228 | 2.14× |
| 8 | 16640 | 2048 | 5120 |  493.6 |  272.2 |  707 | 1282 | 1.81× |
| 8 | 16640 | 2048 | 8192 |  713.5 |  414.8 |  783 | 1346 | 1.72× |
| 8 | 16640 | 5120 | 2048 |  458.2 |  266.0 |  762 | 1312 | 1.72× |
| 8 | 16640 | 5120 | 5120 |  861.8 |  582.4 | 1012 | 1498 | 1.48× |
| 8 | 16640 | 5120 | 8192 | 1267.9 |  875.5 | 1101 | 1594 | 1.45× |
| 8 | 16640 | 8192 | 2048 |  557.0 |  420.6 | 1002 | 1327 | 1.32× |
| 8 | 16640 | 8192 | 5120 | 1257.9 |  880.6 | 1110 | 1585 | 1.43× |
| 8 | 16640 | 8192 | 8192 | 1779.0 | 1327.0 | 1255 | 1683 | 1.34× |

> Run-to-run noise on this bench is ~5–15% (bf16 baseline itself swings
> that much); geomean is stable to within ±2%. Reproducible: seeds both
> `torch.random` and Python's `random` (the latter is what
> `generate_jagged_offs` uses).

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
