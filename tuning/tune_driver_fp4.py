"""Driver for the MXFP4 forward sweep. Runs a shape set across 8 GPUs;
tune_worker_fp4.py exhausts the config grid per shape. Outputs:
  <out_dir>/<E>_<M>_<N>_<K>.json
  <out_dir>/summary.json
and prints a ready-to-paste config table.

  python tune_driver_fp4.py                 # 36 Llama4 shapes (default)
  python tune_driver_fp4.py --shapes dsv3   # 4 DSv3 671B shapes
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Portable defaults: repo root = parent of this tuning/ dir; the worker is the
# sibling script; same interpreter as the driver; scratch under $TUNE_OUT
# (default /tmp; override on hosts where /tmp is wiped).
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("GROUPED_GEMMS_REPO", os.path.dirname(_HERE))
VENV_PY = os.environ.get("VENV_PY", sys.executable)
WORKER = os.path.join(_HERE, "tune_worker_fp4.py")
TUNE_OUT = os.environ.get("TUNE_OUT", "/tmp")

SHAPES_LLAMA4 = [
    (e, m, n, k)
    for e, m, n, k in itertools.product(
        [1, 2, 4, 8], [16640], [2048, 5120, 8192], [2048, 5120, 8192],
    )
]

# DSv3 671B: hidden=7168, moe_inter=2048. Both MoE grouped GEMMs, separate
# gate/up: gate/up (N=2048, K=7168) and down (N=7168, K=2048). E in {4,8},
# M in {32768, 128000}. Keys on (E,M,N,K).
SHAPES_DSV3 = [
    (4, 32768, 2048, 7168),
    (8, 32768, 2048, 7168),
    (4, 128000, 2048, 7168),
    (8, 128000, 2048, 7168),
    (4, 32768, 7168, 2048),
    (8, 32768, 7168, 2048),
    (4, 128000, 7168, 2048),
    (8, 128000, 7168, 2048),
]

# DSv3 16B (torchtitan): hidden=2048, moe_inter=1408, 64 routed experts.
# Separate gate/up (N=1408, K=2048) and down (N=2048, K=1408), E in {4,8},
# M in {32768, 128000}. Keys on (E,M,N,K) like dsv3.
SHAPES_DSV3_16B = [
    (4, 32768, 1408, 2048),
    (8, 32768, 1408, 2048),
    (4, 128000, 1408, 2048),
    (8, 128000, 1408, 2048),
    (4, 32768, 2048, 1408),
    (8, 32768, 2048, 1408),
    (4, 128000, 2048, 1408),
    (8, 128000, 2048, 1408),
]

SHAPE_SETS = {
    "llama4": SHAPES_LLAMA4,
    "dsv3": SHAPES_DSV3,
    "dsv3_16b": SHAPES_DSV3_16B,
}


def run_one(shape_and_gpu):
    shape, gpu_id, out_dir, kpack = shape_and_gpu
    E, M, N, K = shape
    out_path = f"{out_dir}/{E}_{M}_{N}_{K}.json"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = REPO
    env["KPACK"] = str(kpack)
    t0 = time.time()
    res = subprocess.run(
        [VENV_PY, WORKER, str(E), str(M), str(N), str(K), out_path],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=3600,
    )
    return {
        "shape": shape, "gpu": gpu_id, "elapsed_s": time.time() - t0,
        "stderr_tail": res.stderr[-1000:] if res.stderr else "",
        "rc": res.returncode, "out_path": out_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="llama4", choices=list(SHAPE_SETS))
    ap.add_argument("--kpack", type=int, default=1, choices=(1, 2))
    args = ap.parse_args()

    shapes = SHAPE_SETS[args.shapes]
    base_dir = f"{TUNE_OUT}/tune_fp4_results_{args.shapes}"
    out_dir = base_dir + ("_kpack2" if args.kpack == 2 else "")
    # dsv3 keys on (E,M,N,K) since two M values share N/K; llama4 has a single
    # M so (E,N,K) suffices and matches forward_fp4._BEST_CFGS_FP4.
    key_with_m = args.shapes in ("dsv3", "dsv3_16b")
    table_name = {
        "dsv3": "_BEST_CFGS_FP4_DSV3",
        "dsv3_16b": "_BEST_CFGS_FP4_DSV3_16B",
    }.get(args.shapes, "_BEST_CFGS_FP4")

    os.makedirs(out_dir, exist_ok=True)
    n_gpus = 8
    work = [(shape, i % n_gpus, out_dir, args.kpack) for i, shape in enumerate(shapes)]
    print(f"Submitting {len(work)} {args.shapes} shapes across {n_gpus} GPUs "
          f"(MXFP4, kpack={args.kpack}) -> {out_dir}", flush=True)

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=n_gpus) as pool:
        futs = {pool.submit(run_one, w): w for w in work}
        for fut in as_completed(futs):
            shape, gpu, _, _ = futs[fut]
            r = fut.result()
            ok = (r["rc"] == 0)
            tag = "OK" if ok else f"FAIL(rc={r['rc']})"
            print(f"[{tag}] gpu={gpu} shape={shape} t={r['elapsed_s']:.1f}s", flush=True)
            if not ok:
                print(f"  stderr: {r['stderr_tail']}", flush=True)

    print(f"\nTotal wall: {time.time() - t_start:.1f}s\n", flush=True)

    def best_of(results_dir, shape):
        p = f"{results_dir}/{shape[0]}_{shape[1]}_{shape[2]}_{shape[3]}.json"
        if not os.path.exists(p):
            return None, None
        d = json.load(open(p))
        ok = [r for r in d["results"] if r["us"] is not None]
        if not ok:
            return None, d
        return min(ok, key=lambda r: r["us"]), d

    summary = {}
    # kpack=2 pass: compare against the kpack=1 baseline in base_dir and only
    # flag shapes where kpack=2 wins by > the noise threshold.
    WIN_THRESHOLD = 0.005  # 0.5%
    if args.kpack == 2:
        print("=== kpack=2 vs kpack=1 (threshold > 0.5%) ===")
        wins = []
        for shape in shapes:
            E, M, N, K = shape
            best2, d2 = best_of(out_dir, shape)
            best1, _ = best_of(base_dir, shape)
            if best2 is None:
                print(f"# NO_OK_CFG kp2: {shape}", flush=True); continue
            us2 = best2["us"]
            us1 = best1["us"] if best1 else float("inf")
            delta = (us1 - us2) / us1 if us1 != float("inf") else 0.0
            tag = "WIN " if delta > WIN_THRESHOLD else "    "
            print(f"  {tag}E={E} M={M} N={N} K={K}: kp1={us1:.1f}us "
                  f"kp2={us2:.1f}us  {delta*100:+.2f}%", flush=True)
            summary[f"{E}_{M}_{N}_{K}"] = {
                "best_kpack2": best2, "best_kpack1_us": us1, "delta": delta,
            }
            if delta > WIN_THRESHOLD:
                wins.append((shape, best2, us1, us2, delta))

        print(f"\n=== {len(wins)} shapes where kpack=2 wins — apply these ===")
        for shape, best2, us1, us2, delta in wins:
            E, M, N, K = shape
            c = best2["cfg"]
            key = f"({E}, {M}, {N}, {K})" if key_with_m else f"({E}, {N}, {K})"
            print(
                f"        {key}: dict("
                f"BLOCK_M={c['BLOCK_M']}, BLOCK_N={c['BLOCK_N']}, "
                f"BLOCK_K={c['BLOCK_K']}, GROUP_M={c['GROUP_M']}, "
                f"num_warps={c['num_warps']}, num_stages={c['num_stages']}, "
                f"waves_per_eu={c['waves_per_eu']}, "
                f"matrix_instr_nonkdim={c['matrix_instr_nonkdim']}, kpack=2),  "
                f"# {us2:.1f}us  (kp1 {us1:.1f}us, {delta*100:+.2f}%)",
                flush=True,
            )
    else:
        print("=== best MXFP4 cfg per shape ===")
        print(f"    {table_name} = {{")
        for shape in shapes:
            E, M, N, K = shape
            best, data = best_of(out_dir, shape)
            if best is None:
                print(f"# NO_OK_CFG: {shape}", flush=True); continue
            c = best["cfg"]
            summary[f"{E}_{M}_{N}_{K}"] = {
                "best": best, "n_total": data["n_total"]}
            key = f"({E}, {M}, {N}, {K})" if key_with_m else f"({E}, {N}, {K})"
            print(
                f"        {key}: dict("
                f"BLOCK_M={c['BLOCK_M']}, BLOCK_N={c['BLOCK_N']}, "
                f"BLOCK_K={c['BLOCK_K']}, GROUP_M={c['GROUP_M']}, "
                f"num_warps={c['num_warps']}, num_stages={c['num_stages']}, "
                f"waves_per_eu={c['waves_per_eu']}, "
                f"matrix_instr_nonkdim={c['matrix_instr_nonkdim']}),  "
                f"# {best['us']:.1f}us", flush=True,
            )
        print("    }")

    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_dir}/summary.json", flush=True)


if __name__ == "__main__":
    main()
