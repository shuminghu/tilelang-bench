#!/usr/bin/env python3
"""Run a model x task sweep and aggregate a leaderboard.

Single GPU -> runs are serialized. Results are written incrementally so progress
can be monitored (read the --out file or per-run *.trajectory.json at any time).

Usage:
  python run_sweep.py --models claude-haiku-4-5,deepseek-v4-flash,gemini-3-flash \
      --tasks tasks/perf/gemm_optimize --out runs/sweep.json \
      --api-base https://opencode.ai/zen/v1 --api-key-env LLM_API_KEY
"""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HARNESS = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated model ids")
    ap.add_argument("--tasks", required=True, help="comma-separated task dirs")
    ap.add_argument("--out", default="runs/sweep.json")
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--api-key-env", default="LLM_API_KEY")
    ap.add_argument("--step-limit", type=int, default=30)
    ap.add_argument("--cost-limit", type=float, default=1.0)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for task in tasks:
        for model in models:
            run_id = f"{model.replace('/', '_')}__{Path(task).name}"
            cmd = [sys.executable, str(HARNESS / "run_agent.py"),
                   "--task-dir", task, "--model", model, "--run-id", run_id,
                   "--step-limit", str(args.step_limit), "--cost-limit", str(args.cost_limit),
                   "--api-key-env", args.api_key_env]
            if args.api_base:
                cmd += ["--api-base", args.api_base]
            print(f">>> {model}  x  {Path(task).name}", flush=True)
            p = subprocess.run(cmd, capture_output=True, text=True)
            try:
                rec = json.loads(p.stdout[p.stdout.index("{"):p.stdout.rindex("}") + 1])
            except Exception:
                rec = {"final_score": 0.0, "exit_status": "runner_error",
                       "stderr": p.stderr[-800:]}
            row = {"model": model, "task": Path(task).name,
                   "score": rec.get("final_score", 0.0), "exit": rec.get("exit_status"),
                   "calls": rec.get("n_model_calls"), "cost": rec.get("model_cost")}
            rows.append(row)
            print(f"    -> score={row['score']}  exit={row['exit']}  cost={row['cost']}", flush=True)
            out.write_text(json.dumps({"rows": rows}, indent=2))  # incremental

    # Leaderboard: mean score per model across tasks.
    agg = defaultdict(list)
    for r in rows:
        agg[r["model"]].append(r["score"])
    board = sorted(((m, sum(v) / len(v)) for m, v in agg.items()), key=lambda x: -x[1])
    print("\n=== LEADERBOARD (mean score over tasks) ===", flush=True)
    for m, s in board:
        print(f"  {s:.4f}  {m}", flush=True)
    out.write_text(json.dumps({"rows": rows, "leaderboard": board}, indent=2))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
