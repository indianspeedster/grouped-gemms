# Squeezing 1.49× out of an MXFP8 Grouped GEMM on AMD MI355X

*How we tuned a standalone Triton grouped-GEMM kernel for the Llama4 MoE forward
(and dgrad) pass on gfx950 / CDNA4 — optimization by optimization.*

---

## TL;DR

`triton_mxfp8_grouped_mm` is a persistent grouped GEMM that computes
`out[g] = A[group_g] @ B[g]^T` per expert, consuming MXFP8 (e4m3 data + per-32
e8m0 scales) directly through `tl.dot_scaled`. On a 36-shape Llama4 sweep
(`E ∈ {1,2,4,8}`, `M = 16640`, `N,K ∈ {2048,5120,8192}`) on a single MI355X it
reaches a **geomean 1.49× over bf16** (`torch._grouped_mm`), peaking near
**1924 TFLOPS**, while holding **27.6 dB SQNR** on every shape.

The wins came from seven distinct layers. Each section below has an editable
Excalidraw scene (`blog/NN-*.excalidraw`) — open it at
[excalidraw.com](https://excalidraw.com) (*File → Open*) and *Export* to PNG/SVG
to embed in the published post.

> **Diagram index**
> | # | Optimization | Scene file |
> |---|---|---|
> | 1 | Sync-free routing build | `01-routing-build.excalidraw` |
> | 2 | Packed expert→tile map | `02-packed-expert-map.excalidraw` |
> | 3 | GROUP_M L2 reuse | `03-group-m-l2-reuse.excalidraw` |
> | 4 | XCD swizzle | `04-xcd-swizzle.excalidraw` |
> | 5 | CDNA4 pre-shuffled scales | `05-cdna4-scale-shuffle.excalidraw` |
> | 6 | dot_scaled K-loop + EVEN_K | `06-dot-scaled-kloop.excalidraw` |
> | 7 | Per-shape autotuning | `07-per-shape-autotune.excalidraw` |

---

## The problem

An MoE layer routes a jagged number of tokens to each of `E` experts. The
forward GEMM is therefore not one matmul but a *grouped* one: a variable-height
`A` slice per expert multiplied by that expert's weight matrix. Two things make
this hard to run fast on a GPU:

1. **Routing.** You need to translate `group_end_offsets` (cumulative token
   counts) into "which expert and which row-block does program *p* compute?"
   without stalling the GPU on host-side tensor ops.
2. **MXFP8 scales.** Every 32 elements along K carry an e8m0 scale byte. Those
   scales have to reach the CDNA4 `v_mfma_scale_f32_16x16x128_f8f6f4`
   instruction in *exactly* the layout it wants, or the compiler emits an
   address-shuffle chain on every K iteration.

Everything below is about removing overhead from one of those two paths, or
about feeding the MFMA units more efficiently.

---

## Optimization 1 — Sync-free routing build

**Scene:** `01-routing-build.excalidraw`

The naive way to turn `group_end_offsets` into the per-tile routing metadata is a
chain of host-side tensor ops: `cat → diff → cumsum → arange → searchsorted →
clamp → shift → where`. That's ~8 kernel launches and host↔device syncs,
roughly **30–40 µs** of launch overhead that lands squarely on the critical path
and breaks `torch.compile` graphs.

We replace the whole chain with **one** Triton kernel, `_expt_data_kernel`,
launched with `num_warps=1`. It walks the `E` experts in a `tl.static_range`,
keeps a running prefix-sum in registers, and writes all four routing tensors in a
single pass:

- `ExptHist` — tokens per expert
- `ExptOffs` — start offset per expert (expert-sorted)
- `ExptOffsSum` — total padded tile-block count
- `ExptData` — the packed block→expert map (see Opt-2)

Result: **~2–3 µs, one launch, no host sync, torch.compile-clean** — a 10–15×
cut in routing overhead. The diagram shows the op-chain collapsing into a single
green box fanning out to the four output tensors.

---

## Optimization 2 — Packed expert→tile routing

**Scene:** `02-packed-expert-map.excalidraw`

Inside the GEMM, every program needs to know *which expert* and *which row-block*
it owns. Doing a binary search or expert scan per tile would burn the savings
from Opt-1. Instead we pack both numbers into a single `int32` per tile:

```
value = (block_id << 16) | expt_id     # -1 for a padding tile
```

The GEMM kernel does exactly one load and two bit-ops:

```python
expt_data = tl.load(ExptData + pid_m)
if expt_data == -1:        # padding tile from BLOCK_M round-up → bail out
    return
expt_id  = expt_data & 0x0000FFFF
block_id = expt_data >> 16
```

The diagram lays out the jagged expert-sorted M dimension chopped into `BLOCK_M`
tiles, each tile coloured by expert, with a trailing red `-1` pad tile that the
kernel skips. No search, no scan — a single coalesced int32 read decodes the
tile's identity.

---

## Optimization 3 — GROUP_M tile ordering for L2 reuse

**Scene:** `03-group-m-l2-reuse.excalidraw`

Program IDs map to `(pid_m, pid_n)` output tiles. If you iterate row-major
(`GROUP_M = 1`), consecutive programs sweep an entire N row before advancing M —
so the tiles running concurrently across the machine share almost no working set
and the L2 thrashes.

`_pid_grid` reorders pids into **super-blocks of `GROUP_M` rows traversed
column-first**. Now the tiles live at any instant form a compact `GROUP_M × n`
block that re-hits the *same* A rows and B columns out of L2. The diagram shows
two 6×6 tile grids side by side: the naive one highlights a single thin B-column
strip; the grouped one highlights a fat reuse block.

`GROUP_M` is one of the per-shape tuned knobs (mostly 8, occasionally 4 or 16).

---

## Optimization 4 — XCD swizzle

**Scene:** `04-xcd-swizzle.excalidraw`

MI355X is built from **8 XCDs** (accelerator complex dies), and the hardware
dispatches workgroups round-robin: program `p` lands on XCD `p % 8`. Each XCD has
its own L2 slice. Without intervention, the tiles that *should* share operands
(consecutive pids after the GROUP_M reorder) get scattered across all 8 XCDs, so
the GROUP_M locality never materializes in any single L2.

`_xcd_swizzle` pre-permutes the pid so that, after the hardware's `% 8`, **each
XCD receives a contiguous block of tiles**. The diagram contrasts the scattered
round-robin assignment (`0,8,16…` on XCD 0) with the post-swizzle contiguous
ranges (`0…2` on XCD 0, `3…5` on XCD 1, …). `XCD_SWIZZLE` defaults to 8 and is
tuned down to 4 on a couple of shapes.

> This scheduling trio (Opt 2–4) is adapted from AMD aiter's
> `moe_op_gemm_a8w8`, itself derived from triton-lang's `matmul_ogs`. It is a
> **forward-only** win: porting the same XCD/GROUP_M scheme to the wgrad
> (`A^T @ B`) kernel roughly *halved* its throughput, so wgrad keeps a plain
> grid.

---

## Optimization 5 — CDNA4-native pre-shuffled MX scale layout

**Scene:** `05-cdna4-scale-shuffle.excalidraw`

This is the CDNA4-specific win and the most subtle one. `tl.dot_scaled` lowers to
`v_mfma_scale_f32_16x16x128_f8f6f4`, which wants its e8m0 scale operands in a
specific swizzled register layout. If you hand it row-major `(M, K/32)` scales,
the Triton `#blocked → #linear1` lowering inserts, **on every K iteration**, an
address-shuffle chain: roughly **6× `ds_read_u8` + 3× `v_perm_b32`** per scale
tensor — pure overhead in the hottest loop in the kernel.

We pre-shuffle the scales **once, host-side** (`_shuffle_x_scales_cdna4_*` /
`_shuffle_w_scales_cdna4_*`) into the exact MFMA-native order. The in-kernel
`_unswizzle_*_cdna4` then becomes a pure `tl.reshape`/`tl.permute` on registers —
which the compiler folds away. We confirmed in the AMDGCN that **`v_perm_b32`
count drops to 0**.

Details captured in the diagram:
- **Gate:** `BLOCK_K ≥ 256 && K%256==0 && N%32==0 && M%32==0` (`use_cdna4_scale`).
  Llama4 shapes hit it except `(N=K=2048)`, which stays on `BLOCK_K=128`.
- **`nonkdim` 16 vs 32:** two shuffle formulas exist. nk16 was the original
  geomean-best; the per-shape sweep later found **nk32 wins 27/36 shapes**
  (5–9% on K=2048), so `matrix_instr_nonkdim` is now selected per shape and the
  matching host shuffle picked to suit.

---

## Optimization 6 — Direct `dot_scaled` K-loop with EVEN_K peel

**Scene:** `06-dot-scaled-kloop.excalidraw`

Two things in the inner loop:

**Scales ride into the MFMA.** Rather than dequantizing fp8→bf16 and multiplying
by scales in a separate pass over each tile, the e8m0 bytes feed straight into
the MFMA scale operand:

```python
acc = tl.dot_scaled(x, x_scales, "e4m3", w, w_scales, "e4m3",
                    acc=acc, fast_math=True)
```

**EVEN_K peeling.** When `K % BLOCK_K == 0` (the common case), the main loop runs
every iteration with no K-mask compare. When it doesn't divide evenly, we run
`num_k_iter − 1` unmasked iterations and peel a **single** masked tail iteration
(`offs_k < MASK_K_LIMIT`). The bounds check is paid once, not every iteration,
keeping the steady-state loop branch-free. The diagram shows the loop body
(load X, load W, load scales, `dot_scaled`, advance pointers) with the dashed
loop-back arrow, and the EVEN_K decision feeding the tail.

---

## Optimization 7 — Per-shape autotuning

**Scene:** `07-per-shape-autotune.excalidraw`

None of the above fixes a single config across shapes. We swept a
**576-config** space per shape:

```
BLOCK_M ∈ {64,128,256}   BLOCK_N ∈ {128,256}   BLOCK_K ∈ {128,256}
GROUP_M ∈ {1,4,8}        num_warps ∈ {4,8}      num_stages ∈ {1,2}
waves_per_eu ∈ {0,2}     matrix_instr_nonkdim ∈ {16,32}
```

across all **36 shapes**, dispatched **one-shape-per-GPU across 8 MI355X** with a
`ProcessPoolExecutor(8)` (each worker pinned via `CUDA/HIP_VISIBLE_DEVICES` and
`PYTHONPATH` set so `from kernels import …` resolves). A full sweep finishes in
**~13–18 minutes** of wall time instead of hours serial.

The winners are frozen into a `_BEST_CFGS[(E,N,K)]` table consulted at launch by
`_pick_config`, with a small/large-K fallback heuristic for unseen shapes. On top
of the block/warp/stage config, two cache hints are baked in per shape:

- **12 shapes** get `eviction_policy="evict_first"` on the X load,
- **4 shapes** get `cache_modifier=".cg"` on the W load.

The biggest single-shape cache-hint win was **+10%** on `(1, 2048, 2048)`.
End to end, tuning moved the geomean from **1.389× → 1.487×** over bf16
(**+7.0%**).

---

## What *didn't* work (and why)

Worth recording the dead ends — they bound the Triton-level ceiling:

| Lever | Result | Why |
|---|---|---|
| `num_stages=3` on K=2048 | −7% | K-loop only ~16 iters; pipeline fill/drain + LDS occupancy hit outweighs overlap |
| `N, K` as `tl.constexpr` | **−25%** | compiler emitted worse code with literal large ints |
| `kpack=2` | no-op | Triton-AMD silently forces 1 on gfx950 |
| `loop_unroll_factor=2` | LDS OOM | unrolled body needs 2× LDS, past the 160 KB cap |
| E=1 specialized path | ≈0% | per-call savings hidden under the noise floor |
| XCD/GROUP_M ported to wgrad | −50% | wgrad is operand-latency-bound, not schedule-bound |

The remaining gap to hipBLASLt's rowwise/tensorwise FP8 (~1.55×) is the
per-32-block scale-load cost in the inner loop, plus a known `s_waitcnt`
conservatism in our `release/3.7.x` Triton build. Next experiments: cherry-pick
the token-aware wait-count fix, evaluate Gluon, and the FlyDSL raw-intrinsic path
for the long-term ceiling.

---

## Reproduce

```bash
source /it-share/shekhar/mxfp8/.venv/bin/activate
cd /it-share/shekhar/grouped-gemms
python test_correctness.py   # SQNR vs torch._grouped_mm, 27 dB threshold
python bench.py              # 36-shape Llama4 sweep + geomean

# regenerate the diagrams
python blog/gen_diagrams.py
```

*Kernel: `kernels/forward.py`. Hardware: 8× AMD MI355X (gfx950). Software: PyTorch
2.13 nightly (rocm7.1) · triton-rocm 3.7.0.*
