#!/usr/bin/env python3
"""Score a perf-track solution in [0, 1].

score = 0                              if the agent kernel fails to compile or is
                                       numerically incorrect (hard gate);
score = clamp01( (log t_base - log t_agent) / (log t_base - log t_target) )
                                       otherwise.

t_base   : naive reference kernel (the 0.0 anchor)
t_target : the repo's tuned tilelang kernel (the 1.0 anchor; normalization (a))
t_agent  : the candidate

Latencies are measured interleaved (base -> agent -> target each round) so that
unpinned-clock / thermal drift is common-mode and cancels in the ratio.

Usage:
    python score_perf.py --task <task.json> --problem <problem.py> --solution <solution.py>
"""
import argparse
import importlib.util
import json
import math
import statistics as st
import sys
import traceback


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bench(fn, iters):
    import torch
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return st.median(times)


def relstd(xs):
    return 100 * st.pstdev(xs) / st.mean(xs) if st.mean(xs) else float("inf")


def clamp01(x):
    return max(0.0, min(1.0, x))


def measure_triplet(base_fn, agent_fn, target_fn, m):
    """Interleaved measurement with re-measurement if agent timing is noisy."""
    rounds, iters = m["rounds"], m["iters"]
    rel_max, max_extra = m["rel_std_max"], m.get("remeasure", 2)
    base_t, agent_t, target_t = [], [], []
    attempts = 0
    while True:
        for _ in range(rounds):
            base_t.append(bench(base_fn, iters))
            agent_t.append(bench(agent_fn, iters))
            target_t.append(bench(target_fn, iters))
        if relstd(agent_t) <= rel_max or attempts >= max_extra:
            break
        attempts += 1
    return base_t, agent_t, target_t


def correct(fn, inputs, ref_out, tol):
    import torch
    out = fn(*inputs)
    if out.shape != ref_out.shape:
        return False, f"shape {tuple(out.shape)} != {tuple(ref_out.shape)}"
    ok = torch.allclose(out.float(), ref_out.float(), rtol=tol["rtol"], atol=tol["atol"])
    return ok, ("ok" if ok else f"max_abs_err={(out.float()-ref_out.float()).abs().max().item():.4g}")


def score_shape(problem, build_agent, shape, tol, m):
    import torch
    inputs = problem.make_inputs(*shape, seed=1234)
    ref_out = problem.ref(*inputs)

    # Build the three kernels.
    base_k = problem.baseline(*shape)
    target_k = problem.target(*shape)
    try:
        agent_k = build_agent(*shape)
    except Exception:
        return {"shape": shape, "score": 0.0, "correct": False,
                "reason": "agent build failed: " + traceback.format_exc(limit=2)}

    # Hard correctness gate on the agent (sanity-check anchors too).
    ok, why = correct(agent_k, inputs, ref_out, tol)
    if not ok:
        return {"shape": shape, "score": 0.0, "correct": False, "reason": f"incorrect: {why}"}
    base_ok, _ = correct(base_k, inputs, ref_out, tol)
    tgt_ok, _ = correct(target_k, inputs, ref_out, tol)
    if not (base_ok and tgt_ok):
        return {"shape": shape, "score": None, "correct": True,
                "reason": "ANCHOR kernel incorrect -- task definition bug"}

    base_fn = lambda: base_k(*inputs)
    agent_fn = lambda: agent_k(*inputs)
    target_fn = lambda: target_k(*inputs)
    for _ in range(m["warmup"]):
        base_fn(); agent_fn(); target_fn()
    torch.cuda.synchronize()

    bt, at, tt = measure_triplet(base_fn, agent_fn, target_fn, m)
    tb, ta, tg = st.median(bt), st.median(at), st.median(tt)
    denom = math.log(tb) - math.log(tg)
    raw = (math.log(tb) - math.log(ta)) / denom if denom > 0 else 0.0
    return {
        "shape": shape, "correct": True, "score": clamp01(raw), "raw_score": raw,
        "t_base_ms": round(tb, 4), "t_agent_ms": round(ta, 4), "t_target_ms": round(tg, 4),
        "speedup_vs_base": round(tb / ta, 3), "frac_of_target": round(tg / ta, 3),
        "relstd_agent_pct": round(relstd(at), 2), "relstd_base_pct": round(relstd(bt), 2),
        "relstd_target_pct": round(relstd(tt), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--solution", required=True)
    args = ap.parse_args()

    task = json.load(open(args.task))
    tol = task["tolerance"]
    m = task["measure"]
    problem = load_module(args.problem, "problem")
    solution = load_module(args.solution, "solution")
    build_agent = solution.build

    results = [score_shape(problem, build_agent, tuple(s), tol, m) for s in problem.SHAPES]
    scored = [r["score"] for r in results if r.get("score") is not None]
    final = sum(scored) / len(scored) if scored else 0.0
    out = {"task": task["id"], "final_score": round(final, 4), "per_shape": results}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print(json.dumps({"final_score": 0.0, "error": "harness failure"}))
        sys.exit(0)
