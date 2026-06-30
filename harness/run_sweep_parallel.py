#!/usr/bin/env python3
"""Parallel model x task sweep with a GPU worker pool.

Each (model, task) job is pinned to one free GPU via HARNESS_GPU (which flows into
both the agent subprocess and the scoring env). Concurrency = number of GPUs. Per-run
stdout/stderr go to runs/<task>/<run_id>.runlog; results stream to --out incrementally.

Usage (run on a multi-GPU box, e.g. worker-4):
  python run_sweep_parallel.py \
    --models claude-haiku-4-5,gpt-5.4-mini,deepseek-v4-flash \
    --tasks tasks/perf/gemm_optimize \
    --gpus 0,1,2,3,4,5,6,7 \
    --out runs/sweep_parallel.json \
    --api-base $LLM_API_BASE --api-key-env LLM_API_KEY
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
ROOT = HARNESS.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--tasks", default=None, help="comma-separated task dirs")
    ap.add_argument("--manifest", default=None, help="read task dirs from a manifest.json")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--out", default="runs/sweep_parallel.json")
    ap.add_argument("--api-base", default=os.getenv("LLM_API_BASE"))
    ap.add_argument("--api-key-env", default="LLM_API_KEY")
    ap.add_argument("--step-limit", type=int, default=30)
    ap.add_argument("--cost-limit", type=float, default=1.0)
    ap.add_argument("--repeats", type=int, default=1, help="runs per (model,task) for noise")
    ap.add_argument("--tag", default="", help="suffix for run_id/workspace, to isolate "
                    "concurrent sweeps that share the same NFS (e.g. a hostname)")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.manifest:
        man = json.loads(Path(args.manifest).read_text())
        tasks = [t["dir"] for t in man["tasks"]]
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    gpu_q: "queue.Queue[str]" = queue.Queue()
    for g in gpus:
        gpu_q.put(g)

    jobs = [(m, t, r) for t in tasks for m in models for r in range(args.repeats)]
    rows = []
    lock = threading.Lock()

    def flush():
        agg = defaultdict(list)
        for x in rows:
            agg[x["model"]].append(x["score"])
        board = sorted(((m, sum(v) / len(v), len(v)) for m, v in agg.items()),
                       key=lambda z: -z[1])
        out.write_text(json.dumps({"rows": rows, "leaderboard": board}, indent=2))

    def run_job(job):
        model, task, rep = job
        gpu = gpu_q.get()
        try:
            suffix = f"__r{rep}" if args.repeats > 1 else ""
            tag = f"__{args.tag}" if args.tag else ""
            run_id = f"{model.replace('/', '_')}__{Path(task).name}{suffix}{tag}"
            logp = ROOT / "runs" / Path(task).name / f"{run_id}.runlog"
            logp.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["HARNESS_GPU"] = gpu
            cmd = [sys.executable, str(HARNESS / "run_agent.py"),
                   "--task-dir", task, "--model", model, "--run-id", run_id,
                   "--step-limit", str(args.step_limit), "--cost-limit", str(args.cost_limit),
                   "--api-key-env", args.api_key_env]
            if args.api_base:
                cmd += ["--api-base", args.api_base]
            print(f">>> START {run_id} on GPU {gpu}", flush=True)
            with open(logp, "w") as lf:
                p = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, text=True)
            txt = logp.read_text()
            try:
                rec = json.loads(txt[txt.rindex("{"):txt.rindex("}") + 1])
            except Exception:
                rec = {"final_score": 0.0, "exit_status": f"runner_rc={p.returncode}"}
            row = {"model": model, "task": Path(task).name, "rep": rep,
                   "score": rec.get("final_score", 0.0), "exit": rec.get("exit_status"),
                   "calls": rec.get("n_model_calls"), "cost": rec.get("model_cost"),
                   "gpu": gpu, "log": str(logp)}
        finally:
            gpu_q.put(gpu)
        with lock:
            rows.append(row)
            flush()
        print(f"<<< DONE  {row['model']}  score={row['score']}  exit={row['exit']}  gpu={gpu}", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        list(ex.map(run_job, jobs))

    flush()
    print("\n=== LEADERBOARD (mean score) ===", flush=True)
    data = json.loads(out.read_text())
    for m, s, n in data["leaderboard"]:
        print(f"  {s:.4f}  (n={n})  {m}", flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
