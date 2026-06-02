# Tuning

Per-shape config sweeps for the grouped-GEMM kernels. Each driver fans the
shapes of a set across the local GPUs (one shape per GPU), and each worker
benchmarks the full config grid for one shape, writing incremental JSON. The
driver then prints a ready-to-paste `_BEST_CFGS*` table.

| Script | Kernel | Grid |
| --- | --- | --- |
| `tune_driver_fp4.py` / `tune_worker_fp4.py` | MXFP4 (`triton_mxfp4_grouped_mm`) | 576 cfgs (`BLOCK_K ∈ {128,256,512}`) |
| `tune_driver_mxfp8.py` / `tune_worker_mxfp8.py` | MXFP8 (`triton_mxfp8_grouped_mm`) | 384 cfgs (`BLOCK_K ∈ {128,256}`) |

## Run

```bash
# from the repo root, with the kernels' venv active
python tuning/tune_driver_fp4.py   --shapes dsv3        # 671B gate/up + down
python tuning/tune_driver_mxfp8.py --shapes dsv3_16b    # 16B gate/up + down
python tuning/tune_driver_fp4.py   --shapes llama4      # 36-shape Llama4 grid
```

Shape sets: `llama4` (keyed `(E,N,K)`, single M), `dsv3` and `dsv3_16b`
(keyed `(E,M,N,K)`). The printed table goes into the matching kernel's
`_BEST_CFGS*` dict; for a new keyed table also add a lookup line to
`_pick_config`. The fp4 driver also takes `--kpack {1,2}` (kpack is a no-op on
gfx950 — the compiler forces it to 1).

## Environment overrides

- `TUNE_OUT` — scratch dir for result JSON (default `/tmp`). **Set this to a
  persistent path on hosts where `/tmp` is wiped mid-run**, e.g.
  `TUNE_OUT=$HOME/tune_out`.
- `GROUPED_GEMMS_REPO` — repo root on `PYTHONPATH` (default: parent of this dir).
- `VENV_PY` — interpreter for the worker subprocesses (default: same as driver).

Needs a gfx950 (MI350+) box with the kernels' triton/torch venv.
