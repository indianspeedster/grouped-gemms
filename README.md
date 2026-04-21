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

**Geomean MXFP8 speedup vs bf16: 1.394×** (36 shapes). Best 2.18×
(E=8, N=K=2048), worst 1.16× (E=2, N=2048, K=8192).

| E | M | N | K | bf16 µs | mxfp8 µs | bf16 TFLOPs | mxfp8 TFLOPs | speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 16640 | 2048 | 2048 |  162.9 |  104.1 |  857 | 1341 | 1.57× |
| 1 | 16640 | 2048 | 5120 |  330.4 |  251.1 | 1056 | 1390 | 1.32× |
| 1 | 16640 | 2048 | 8192 |  478.2 |  373.8 | 1167 | 1494 | 1.28× |
| 1 | 16640 | 5120 | 2048 |  318.2 |  246.4 | 1097 | 1416 | 1.29× |
| 1 | 16640 | 5120 | 5120 |  694.8 |  515.3 | 1256 | 1693 | 1.35× |
| 1 | 16640 | 5120 | 8192 | 1120.5 |  789.4 | 1246 | 1768 | 1.42× |
| 1 | 16640 | 8192 | 2048 |  473.4 |  379.5 | 1179 | 1471 | 1.25× |
| 1 | 16640 | 8192 | 5120 | 1066.4 |  800.8 | 1309 | 1743 | 1.33× |
| 1 | 16640 | 8192 | 8192 | 1659.2 | 1251.1 | 1346 | 1785 | 1.33× |
| 2 | 16640 | 2048 | 2048 |  150.1 |  101.2 |  930 | 1379 | 1.48× |
| 2 | 16640 | 2048 | 5120 |  323.6 |  249.1 | 1078 | 1401 | 1.30× |
| 2 | 16640 | 2048 | 8192 |  441.9 |  381.2 | 1264 | 1465 | 1.16× |
| 2 | 16640 | 5120 | 2048 |  329.9 |  250.8 | 1058 | 1391 | 1.32× |
| 2 | 16640 | 5120 | 5120 |  668.4 |  534.1 | 1305 | 1633 | 1.25× |
| 2 | 16640 | 5120 | 8192 |  984.9 |  823.4 | 1417 | 1695 | 1.20× |
| 2 | 16640 | 8192 | 2048 |  472.4 |  383.9 | 1182 | 1455 | 1.23× |
| 2 | 16640 | 8192 | 5120 | 1010.0 |  858.2 | 1382 | 1626 | 1.18× |
| 2 | 16640 | 8192 | 8192 | 1546.2 | 1276.0 | 1444 | 1750 | 1.21× |
| 4 | 16640 | 2048 | 2048 |  184.9 |  103.5 |  755 | 1348 | 1.79× |
| 4 | 16640 | 2048 | 5120 |  406.2 |  259.6 |  859 | 1344 | 1.57× |
| 4 | 16640 | 2048 | 8192 |  577.1 |  404.2 |  967 | 1381 | 1.43× |
| 4 | 16640 | 5120 | 2048 |  387.4 |  259.6 |  901 | 1344 | 1.49× |
| 4 | 16640 | 5120 | 5120 |  774.1 |  550.2 | 1127 | 1586 | 1.41× |
| 4 | 16640 | 5120 | 8192 | 1072.8 |  843.3 | 1301 | 1655 | 1.27× |
| 4 | 16640 | 8192 | 2048 |  502.5 |  395.4 | 1111 | 1412 | 1.27× |
| 4 | 16640 | 8192 | 5120 | 1115.8 |  858.5 | 1251 | 1626 | 1.30× |
| 4 | 16640 | 8192 | 8192 | 1792.7 | 1311.1 | 1246 | 1703 | 1.37× |
| 8 | 16640 | 2048 | 2048 |  248.2 |  113.8 |  562 | 1227 | 2.18× |
| 8 | 16640 | 2048 | 5120 |  506.5 |  267.1 |  689 | 1307 | 1.90× |
| 8 | 16640 | 2048 | 8192 |  623.0 |  413.6 |  896 | 1350 | 1.51× |
| 8 | 16640 | 5120 | 2048 |  440.7 |  278.3 |  792 | 1254 | 1.58× |
| 8 | 16640 | 5120 | 5120 |  893.7 |  582.8 |  976 | 1497 | 1.53× |
| 8 | 16640 | 5120 | 8192 | 1213.3 |  874.7 | 1150 | 1596 | 1.39× |
| 8 | 16640 | 8192 | 2048 |  585.5 |  421.7 |  954 | 1324 | 1.39× |
| 8 | 16640 | 8192 | 5120 | 1280.4 |  879.6 | 1090 | 1587 | 1.46× |
| 8 | 16640 | 8192 | 8192 | 1830.5 | 1323.9 | 1220 | 1687 | 1.38× |

> Run-to-run noise on this bench is ~5–15% (bf16 baseline itself swings
> that much); geomean is stable to within ±2%.

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
