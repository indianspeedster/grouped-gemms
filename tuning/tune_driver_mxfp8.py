"""Driver for the MXFP8 forward sweep. Runs a shape set across 8 GPUs;
tune_worker_mxfp8.py exhausts the config grid per shape. Outputs:
  <out_dir>/<E>_<M>_<N>_<K>.json
  <out_dir>/summary.json
and prints a ready-to-paste config table.

  python tune_driver_mxfp8.py --shapes dsv3
  python tune_driver_mxfp8.py --shapes dsv3_16b
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
WORKER = os.path.join(_HERE, "tune_worker_mxfp8.py")
TUNE_OUT = os.environ.get("TUNE_OUT", "/tmp")

SHAPES_LLAMA4 = [
    (e, m, n, k)
    for e, m, n, k in itertools.product(
        [1, 2, 4, 8], [16640], [2048, 5120, 8192], [2048, 5120, 8192],
    )
]
SHAPES_DSV3 = [
    (4, 32768, 2048, 7168), (8, 32768, 2048, 7168),
    (4, 128000, 2048, 7168), (8, 128000, 2048, 7168),
    (4, 32768, 7168, 2048), (8, 32768, 7168, 2048),
    (4, 128000, 7168, 2048), (8, 128000, 7168, 2048),
]
SHAPES_DSV3_16B = [
    (4, 32768, 1408, 2048), (8, 32768, 1408, 2048),
    (4, 128000, 1408, 2048), (8, 128000, 1408, 2048),
    (4, 32768, 2048, 1408), (8, 32768, 2048, 1408),
    (4, 128000, 2048, 1408), (8, 128000, 2048, 1408),
]
SHAPE_SETS = {
    "llama4": SHAPES_LLAMA4,
    "dsv3": SHAPES_DSV3,
    "dsv3_16b": SHAPES_DSV3_16B,
}


def run_one(shape_and_gpu):
    shape, gpu_id, out_dir = shape_and_gpu
    E, M, N, K = shape
    out_path = f"{out_dir}/{E}_{M}_{N}_{K}.json"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = REPO
    t0 = time.time()
    res = subprocess.run(
        [VENV_PY, WORKER, str(E), str(M), str(N), str(K), out_path],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=3600,
    )
    return {"shape": shape, "gpu": gpu_id, "elapsed_s": time.time() - t0,
            "stderr_tail": res.stderr[-1000:] if res.stderr else "",
            "rc": res.returncode}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="llama4", choices=list(SHAPE_SETS))
    args = ap.parse_args()

    shapes = SHAPE_SETS[args.shapes]
    out_dir = f"{TUNE_OUT}/tune_mxfp8_results_{args.shapes}"
    key_with_m = args.shapes in ("dsv3", "dsv3_16b")
    table_name = {
        "dsv3": "_BEST_CFGS_DSV3",
        "dsv3_16b": "_BEST_CFGS_DSV3_16B",
    }.get(args.shapes, "_BEST_CFGS")

    os.makedirs(out_dir, exist_ok=True)
    n_gpus = 8
    work = [(shape, i % n_gpus, out_dir) for i, shape in enumerate(shapes)]
    print(f"Submitting {len(work)} {args.shapes} shapes across {n_gpus} GPUs "
          f"(MXFP8) -> {out_dir}", flush=True)

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=n_gpus) as pool:
        futs = {pool.submit(run_one, w): w for w in work}
        for fut in as_completed(futs):
            shape, gpu, _ = futs[fut]
            r = fut.result()
            tag = "OK" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
            print(f"[{tag}] gpu={gpu} shape={shape} t={r['elapsed_s']:.1f}s", flush=True)
            if r["rc"] != 0:
                print(f"  stderr: {r['stderr_tail']}", flush=True)

    print(f"\nTotal wall: {time.time() - t_start:.1f}s\n", flush=True)

    summary = {}
    print("=== best MXFP8 cfg per shape ===")
    print(f"    {table_name} = {{")
    for shape in shapes:
        E, M, N, K = shape
        path = f"{out_dir}/{E}_{M}_{N}_{K}.json"
        if not os.path.exists(path):
            print(f"# MISSING: {path}", flush=True); continue
        data = json.load(open(path))
        ok = [r for r in data["results"] if r["us"] is not None]
        if not ok:
            print(f"# NO_OK_CFG: {shape}", flush=True); continue
        best = min(ok, key=lambda r: r["us"])
        c = best["cfg"]
        summary[f"{E}_{M}_{N}_{K}"] = {"best": best, "n_total": data["n_total"]}
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
