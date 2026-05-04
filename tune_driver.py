"""8-GPU parallel sweep driver. Round-robins 36 Llama4 shapes across GPUs,
one shape per subprocess (CLAUDE.md gotcha #1: multi-shape-in-one-process
crashes the driver).

Outputs:
  /tmp/tune_results/<E>_<M>_<N>_<K>.json   per-shape sweep results
  /tmp/tune_results/best.json              {shape_str: best_cfg}

Run: python tune_driver.py
"""
import itertools
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = "/it-share/shekhar/grouped-gemms"
VENV_PY = "/it-share/shekhar/mxfp8/.venv/bin/python"
OUT_DIR = "/tmp/tune_results"

SHAPES = [
    (e, m, n, k)
    for e, m, n, k in itertools.product(
        [1, 2, 4, 8],         # E
        [16640],              # M
        [2048, 5120, 8192],   # N
        [2048, 5120, 8192],   # K
    )
]


def run_one(shape_and_gpu):
    shape, gpu_id = shape_and_gpu
    E, M, N, K = shape
    out_path = f"{OUT_DIR}/{E}_{M}_{N}_{K}.json"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = REPO
    t0 = time.time()
    res = subprocess.run(
        [VENV_PY, "tune_worker.py", str(E), str(M), str(N), str(K), out_path],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    elapsed = time.time() - t0
    return {
        "shape": shape,
        "gpu": gpu_id,
        "elapsed_s": elapsed,
        "stdout_tail": res.stdout[-500:] if res.stdout else "",
        "stderr_tail": res.stderr[-1000:] if res.stderr else "",
        "rc": res.returncode,
        "out_path": out_path,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n_gpus = 8
    # Round-robin assignment by index. With 36 shapes / 8 GPUs, each GPU gets
    # 4-5 shapes; the pool processes 8 in parallel and serializes the rest.
    work = [(shape, i % n_gpus) for i, shape in enumerate(SHAPES)]
    print(f"Submitting {len(work)} shapes across {n_gpus} GPUs", flush=True)

    t_start = time.time()
    summaries = []
    with ProcessPoolExecutor(max_workers=n_gpus) as pool:
        futs = {pool.submit(run_one, w): w for w in work}
        for fut in as_completed(futs):
            shape, gpu = futs[fut]
            try:
                r = fut.result()
                ok = (r["rc"] == 0)
                tag = "OK" if ok else f"FAIL(rc={r['rc']})"
                print(
                    f"[{tag}] gpu={gpu} shape={shape} t={r['elapsed_s']:.1f}s",
                    flush=True,
                )
                if not ok:
                    print(f"  stderr: {r['stderr_tail']}", flush=True)
                summaries.append(r)
            except Exception as ex:
                print(f"[CRASH] gpu={gpu} shape={shape}: {ex}", flush=True)
                summaries.append({"shape": shape, "gpu": gpu, "err": str(ex)})

    print(f"\nTotal wall time: {time.time() - t_start:.1f}s", flush=True)

    # Aggregate per-shape best.
    best = {}
    for shape in SHAPES:
        E, M, N, K = shape
        path = f"{OUT_DIR}/{E}_{M}_{N}_{K}.json"
        if not os.path.exists(path):
            print(f"MISSING: {path}", flush=True)
            continue
        with open(path) as f:
            data = json.load(f)
        ok_results = [r for r in data["results"] if r["us"] is not None]
        if not ok_results:
            print(f"NO_OK_CFG: {shape}", flush=True)
            continue
        best_r = min(ok_results, key=lambda r: r["us"])
        key = f"{E}_{M}_{N}_{K}"
        best[key] = {
            "cfg": best_r["cfg"],
            "us": best_r["us"],
            "n_ok": len(ok_results),
            "n_total": len(data["results"]),
        }
        flops = 2 * M * N * K
        tflops = (flops / 1e12) / (best_r["us"] / 1e6)
        print(
            f"  {key}: {best_r['us']:.1f}us ({tflops:.1f} TFLOPS) "
            f"cfg={best_r['cfg']} ({len(ok_results)}/{len(data['results'])} ok)",
            flush=True,
        )

    with open(f"{OUT_DIR}/best.json", "w") as f:
        json.dump(best, f, indent=2)
    print(f"\nWrote {OUT_DIR}/best.json with {len(best)} entries", flush=True)


if __name__ == "__main__":
    main()
