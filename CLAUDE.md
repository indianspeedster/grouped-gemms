# CLAUDE.md — grouped-gemms

Operational notes for future Claude Code sessions in this repo. Pragmatic
only — the README has the user-facing explanation.

## Repo purpose

Standalone ROCm MXFP8 grouped-GEMM Triton kernel for AMD MI350+ (gfx950),
extracted from the `rocm-mxfp8-aiter-port` branch of `pytorch/ao` for
isolated iteration. Two kernels:
- `kernels/forward.py` → `triton_mxfp8_grouped_mm` (fwd + dgrad, A @ B^T per group)
- `kernels/backward.py` → `triton_mxfp8_wgrad` (weight grad, A^T @ B per group)

No torchao runtime dep — `utils.py` has a minimal `to_mx`,
`generate_jagged_offs`, and bench helper.

## Environment

Tested venv lives outside the repo:
```bash
source /it-share/shekhar/mxfp8/.venv/bin/activate
```
This has `torch 2.13.0.dev20260416+rocm7.1` and `triton-rocm 3.7.0+gitb4e20bbe`
pre-installed. The repo's `requirements.txt` reproduces this env in a
clean `python3 -m venv` via an `--extra-index-url` line; don't need it if
you already have the shared venv.

Hardware: 8× MI355X (gfx950) accessible on this host. Confirm with:
```bash
rocm-smi --showproductname | grep -c "MI355X"   # should be 8
```

## Running the bench and tests

```bash
cd /it-share/shekhar/grouped-gemms

python test_correctness.py   # 4 shapes, SQNR vs torch._grouped_mm at 27 dB
python bench.py              # 36-shape Llama4 sweep, prints geomean
```

`bench.py` ~10–15 s on a healthy machine. `test_correctness.py` ~20 s.

## Parallel 36-shape sweeps across 8 GPUs

For correctness or a wider autotune, dispatching shapes across all 8 GPUs
takes ~40 s instead of ~5–10 min serial. Driver template at
`/tmp/parallel_check.py`:
- `ProcessPoolExecutor(max_workers=8)` pool
- Each worker runs `/tmp/check_one.py` with
  `CUDA_VISIBLE_DEVICES=<gpu>`, `HIP_VISIBLE_DEVICES=<gpu>`
- **Must set `PYTHONPATH=/it-share/shekhar/grouped-gemms`** in the
  subprocess env — Python puts the script's dir on `sys.path`, NOT cwd,
  so `from kernels import ...` fails without this.

Reuse this pattern for any per-shape work (accuracy sweep, config sweep,
crash bisection).

## Gotchas that burn time

1. **GPU memory faults are driver state, not kernel bugs.** Running all
   36 shapes in one Python process crashes on shape ~10 with "Memory
   access fault by GPU". Running one-per-subprocess → 0 crashes. The
   fault predates the aiter port (exists in `ba33d1862`, state J, and
   even older commits).

2. **`generate_jagged_offs` uses Python's `random.sample`, not torch.**
   `torch.random.manual_seed` alone is not enough for reproducibility —
   also `import random; random.seed(...)`. `bench.py::main` seeds both.

3. **MX block_size = 32.** Group end offsets fed to the MXFP8 path must
   be multiples of 32. `bench.py` uses `multiple_of=32` for both bf16
   and mxfp8 so the comparison is fair (32 is also a multiple of 16,
   which is what `torch._grouped_mm` needs).

4. **CDNA4_SCALE fast-path gates on shape.** `use_cdna4_scale = BLOCK_K >= 256
   and K % 256 == 0 and N % 32 == 0 and M % 32 == 0`. Llama4 bench
   shapes hit it except `(N=K=2048)` which stays on BLOCK_K=128 per
   `_pick_block_nk`.

5. **`matrix_instr_nonkdim` is forced per path.** Caller passes 32 but
   the CDNA4_SCALE path overrides to 16 because the nonkdim=16 shuffle
   formula is what's wired up by default (nk16 geomean-beats nk32 by
   3.1% across Llama4; nk32 wins up to 24% on some dsv3 shapes — nk32
   helpers exist but are dormant).

## Current performance (baseline)

On MI355X, 36-shape Llama4 bench, bf16 vs MXFP8:
- **Geomean 1.389×** (with fair matching offsets for both paths)
- Peak MXFP8 1788 TFLOPS (E=1, N=K=8192)
- Best speedup 2.14× (E=8, N=K=2048 — bf16 starved, not mxfp8 outlier)
- Worst speedup 1.17× (E=2, N=8192, K=2048)
- SQNR 27.61–27.62 dB on all 36 shapes (matches quantization noise floor)

Run-to-run noise ~5–15% per shape, geomean stable to ±2%.

## Tuning status — important context

Only `(BLOCK_N, BLOCK_K)` is shape-aware, via `_pick_block_nk`:
- `(N=K=2048) → 128/128`
- else → `256/256`

Everything else is fixed defaults:
- `BLOCK_M=128`, `num_warps=8`, `num_stages=2`, `GROUP_M=8`,
  `XCD_SWIZZLE=8`, `matrix_instr_nonkdim=32`, `kpack=1`, `waves_per_eu=0`

**We have not done exhaustive per-shape autotuning.** If a coworker asks
"did you sweep configs per shape" — the honest answer is "no, we have a
coarse (BLOCK_N, BLOCK_K) heuristic from an 864-run sweep, but the full
config space (num_warps × num_stages × GROUP_M × nonkdim × BLOCK_M)
is unexplored". Pending task #4 (SPLIT_K for low-M-tile / E=1) and #6
(num_stages=3 with trimmed LDS) are in the TaskList.

## Commit etiquette (user-specific)

- **Do NOT commit without explicit user request** — standard.
- **No `Co-Authored-By` trailers** — saved in user memory, the user
  rejected them.
- **No auto-push** — push only when explicitly asked.
- Remote: `https://github.com/indianspeedster/grouped-gemms` (private).

## Related context

- Origin branch: `rocm-mxfp8-aiter-port` in `/it-share/shekhar/torch/ao`
  (torchao). Current head there: `b01c4a65b`.
- B200 reference numbers + aiter-port history are in a Notion page titled
  "ROCm MXFP8 Grouped GEMM — Tuning Findings (gfx950 / MI355X)".
- Prior session summaries (if they seem relevant) live in user memory.
