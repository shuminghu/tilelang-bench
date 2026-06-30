#!/usr/bin/env python3
"""Aggregate a sweep into a per-model / per-track leaderboard.

  python report.py --sweep runs/sweep_full.json --manifest tasks/manifest.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="runs/sweep_full.json")
    ap.add_argument("--manifest", default="tasks/manifest.json")
    args = ap.parse_args()

    rows = json.loads(Path(args.sweep).read_text())["rows"]
    track_of = {t["id"]: t["track"] for t in json.loads(Path(args.manifest).read_text())["tasks"]}

    # model -> track -> [scores]
    agg = defaultdict(lambda: defaultdict(list))
    exits = defaultdict(lambda: defaultdict(int))
    for r in rows:
        tr = track_of.get(r["task"], "unknown")
        agg[r["model"]][tr].append(r["score"])
        agg[r["model"]]["all"].append(r["score"])
        exits[r["model"]][str(r.get("exit"))] += 1

    tracks = ["perf", "implement", "debug", "all"]
    models = sorted(agg, key=lambda m: -(sum(agg[m]["all"]) / max(1, len(agg[m]["all"]))))

    w = 18
    print(f"\n{'model':<{w}}" + "".join(f"{t:>12}" for t in tracks) + f"{'runs':>8}")
    print("-" * (w + 12 * len(tracks) + 8))
    for m in models:
        cells = ""
        for t in tracks:
            v = agg[m][t]
            cells += (f"{sum(v)/len(v):>11.3f}" + " ") if v else f"{'-':>12}"
        print(f"{m:<{w}}{cells}{len(agg[m]['all']):>8}")
    print(f"\n(tracks show mean score in [0,1]; per-task n varies)")

    print("\nexit-status breakdown:")
    for m in models:
        tot = sum(exits[m].values())
        brk = ", ".join(f"{k}:{v}" for k, v in sorted(exits[m].items(), key=lambda x: -x[1]))
        print(f"  {m:<{w}} {brk}")


if __name__ == "__main__":
    main()
