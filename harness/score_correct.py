#!/usr/bin/env python3
"""Score an implement-track (correctness) solution in [0, 1].

score = (# cases passing allclose) / (# cases)

A "case" is one entry of problem.SHAPES. The agent's build(*shape) must return a
callable kernel whose output matches problem.ref on problem.make_inputs(*shape).

Anti-passthrough: by default the returned kernel must be a compiled TileLang object
(its type's module starts with "tilelang"), so a plain `lambda A,B: A@B` scores 0.
Disable via task.json:  "require_tilelang": false.

Usage:
    python score_correct.py --task <task.json> --problem <problem.py> --solution <solution.py>
"""
import argparse
import importlib.util
import json
import sys
import traceback


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def correct(out, ref_out, tol):
    import torch
    if not hasattr(out, "shape"):
        return False, f"non-tensor output: {type(out)}"
    if tuple(out.shape) != tuple(ref_out.shape):
        return False, f"shape {tuple(out.shape)} != {tuple(ref_out.shape)}"
    ok = torch.allclose(out.float(), ref_out.float(), rtol=tol["rtol"], atol=tol["atol"])
    err = (out.float() - ref_out.float()).abs().max().item()
    return ok, ("ok" if ok else f"max_abs_err={err:.4g}")


def score_case(problem, build_agent, shape, tol, require_tilelang):
    import torch
    try:
        inputs = problem.make_inputs(*shape, seed=1234)
        ref_out = problem.ref(*inputs)
        kernel = build_agent(*shape)
    except Exception:
        return {"shape": shape, "ok": False, "reason": "build/setup failed: "
                + traceback.format_exc(limit=2)}
    if require_tilelang and not type(kernel).__module__.split(".")[0].startswith("tilelang"):
        return {"shape": shape, "ok": False,
                "reason": f"not a tilelang kernel (type={type(kernel).__module__})"}
    try:
        out = kernel(*inputs)
        torch.cuda.synchronize()
    except Exception:
        return {"shape": shape, "ok": False, "reason": "exec failed: "
                + traceback.format_exc(limit=2)}
    ok, why = correct(out, ref_out, tol)
    return {"shape": shape, "ok": ok, "reason": why}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--solution", required=True)
    args = ap.parse_args()

    task = json.load(open(args.task))
    tol = task["tolerance"]
    require_tilelang = task.get("require_tilelang", True)
    problem = load_module(args.problem, "problem")
    solution = load_module(args.solution, "solution")

    results = [score_case(problem, solution.build, tuple(s), tol, require_tilelang)
               for s in problem.SHAPES]
    n_ok = sum(1 for r in results if r["ok"])
    final = n_ok / len(results) if results else 0.0
    print(json.dumps({"task": task["id"], "final_score": round(final, 4),
                      "n_ok": n_ok, "n_cases": len(results), "per_case": results}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print(json.dumps({"final_score": 0.0, "error": "harness failure"}))
        sys.exit(0)
