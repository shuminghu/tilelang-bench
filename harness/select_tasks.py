#!/usr/bin/env python3
"""Select exactly 100 validated tasks into tasks/manifest.json.

Reads validation result JSONL (one {id,track,ok,...} per line) from the given files,
keeps only PASSing tasks, and composes a balanced 100-task benchmark.

  python select_tasks.py --results runs/validation_main.jsonl runs/valdebug.jsonl \
      --perf 28 --debug 12 --implement 60
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "tasks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--perf", type=int, default=28)
    ap.add_argument("--debug", type=int, default=12)
    ap.add_argument("--implement", type=int, default=60)
    ap.add_argument("--out", default=str(TASKS / "manifest.json"))
    args = ap.parse_args()

    passing = {"perf": [], "implement": [], "debug": []}
    seen = set()
    for fp in args.results:
        if not Path(fp).exists():
            continue
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["ok"] and r["id"] not in seen and r["track"] in passing:
                passing[r["track"]].append(r)
                seen.add(r["id"])

    want = {"perf": args.perf, "debug": args.debug, "implement": args.implement}
    selected = []
    # take perf/debug first (scarcer), then fill the rest from implement to hit 100.
    for tr in ("perf", "debug"):
        take = sorted(passing[tr], key=lambda r: r["id"])[:want[tr]]
        selected += take
    remaining = 100 - len(selected)
    selected += sorted(passing["implement"], key=lambda r: r["id"])[:remaining]

    if len(selected) < 100:
        # backfill from any leftover passing tasks if a track was short
        pool = [r for tr in passing for r in passing[tr]
                if r["id"] not in {s["id"] for s in selected}]
        selected += sorted(pool, key=lambda r: r["id"])[:100 - len(selected)]

    manifest = {"total": len(selected),
                "by_track": {tr: sum(1 for s in selected if s["track"] == tr)
                             for tr in ("perf", "implement", "debug")},
                "tasks": [{"id": s["id"], "track": s["track"], "dir": s["dir"],
                           "speedup": s.get("speedup")} for s in
                          sorted(selected, key=lambda r: (r["track"], r["id"]))]}
    Path(args.out).write_text(json.dumps(manifest, indent=2))
    print(f"selected {len(selected)} tasks: {manifest['by_track']}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
