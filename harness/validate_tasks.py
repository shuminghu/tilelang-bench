#!/usr/bin/env python3
"""Validate generated tasks on GPU: are they well-formed and solvable?

perf task      PASS if baseline+target compile & are correct on every SHAPE and
               median(target) is faster than median(baseline) by > min_speedup.
implement task PASS if the private oracle (_build) compiles & is correct on every
               case and returns a tilelang kernel object.

Writes one JSON line per task to --out. Shard with --shard k/n to split across GPUs.

  CUDA_VISIBLE_DEVICES=6 .venv/bin/python harness/validate_tasks.py --shard 0/4 --out runs/val0.jsonl
"""
import argparse
import importlib.util
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "tasks"
MIN_SPEEDUP = 1.05


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def bench(fn, iters=20):
    import torch
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return st.median(ts)


def allclose(out, ref, tol):
    import torch
    return (hasattr(out, "shape") and tuple(out.shape) == tuple(ref.shape)
            and torch.allclose(out.float(), ref.float(), rtol=tol["rtol"], atol=tol["atol"]))


def validate(task_dir):
    import torch
    tj = json.load(open(task_dir / "task.json"))
    track, tol = tj.get("track", "perf"), tj["tolerance"]
    prob = load(task_dir / "problem.py", "problem_" + task_dir.name)
    res = {"id": tj["id"], "track": track, "dir": str(task_dir), "ok": False}
    try:
        if track == "perf":
            sp = []
            for shape in prob.SHAPES:
                inp = prob.make_inputs(*shape, seed=1234); rf = prob.ref(*inp)
                bk, tk = prob.baseline(*shape), prob.target(*shape)
                if not allclose(bk(*inp), rf, tol):
                    res["reason"] = f"baseline incorrect at {shape}"; return res
                if not allclose(tk(*inp), rf, tol):
                    res["reason"] = f"target incorrect at {shape}"; return res
                for _ in range(10):
                    bk(*inp); tk(*inp)
                torch.cuda.synchronize()
                tb, tt = bench(lambda: bk(*inp)), bench(lambda: tk(*inp))
                sp.append(tb / tt)
            res["speedup"] = round(st.median(sp), 3)
            res["ok"] = res["speedup"] > MIN_SPEEDUP
            if not res["ok"]:
                res["reason"] = f"insufficient speedup {res['speedup']}"
        else:  # implement -> validate oracle
            for shape in prob.SHAPES:
                inp = prob.make_inputs(*shape, seed=1234); rf = prob.ref(*inp)
                k = prob._build(*shape)
                if not type(k).__module__.split(".")[0].startswith("tilelang"):
                    res["reason"] = f"oracle not tilelang ({type(k).__module__})"; return res
                if not allclose(k(*inp), rf, tol):
                    res["reason"] = f"oracle incorrect at {shape}"; return res
            res["ok"] = True
    except Exception as e:
        import traceback
        res["reason"] = f"{type(e).__name__}: {str(e)[:160]}"
        res["trace"] = traceback.format_exc(limit=2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    k, n = (int(x) for x in args.shard.split("/"))

    cand = json.load(open(TASKS / "candidates.json"))
    dirs = []
    for track in ("perf", "implement", "debug"):
        for tid in cand.get(f"{track}_candidates", []):
            dirs.append(TASKS / track / tid)
    dirs = [d for i, d in enumerate(sorted(dirs)) if i % n == k]

    with open(args.out, "w") as f:
        for d in dirs:
            r = validate(d)
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"  [{'PASS' if r['ok'] else 'FAIL'}] {r['id']:28s} "
                  f"{r.get('speedup', '')} {r.get('reason', '')}", flush=True)


if __name__ == "__main__":
    main()
