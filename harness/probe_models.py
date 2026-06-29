#!/usr/bin/env python3
"""Probe candidate models for end-to-end usability on a tool-call gateway.

Runs a 2-turn exchange (so it catches the provider_specific_fields resend bug,
not just a single completion) using the same CleanLitellmModel + BASH_TOOL the
real harness uses. No GPU needed. Prints PASS/FAIL per model.

Usage:
  python probe_models.py --models a,b,c --api-base $LLM_API_BASE --api-key-env LLM_API_KEY
"""
import argparse
import os
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))
from zen_model import CleanLitellmModel  # noqa: E402
from minisweagent.exceptions import FormatError  # noqa: E402


def probe(name, api_base, api_key):
    model_name = name if "/" in name else (f"openai/{name}" if api_base else name)
    mk = {"num_retries": 0, "timeout": 90}
    if api_base:
        mk["api_base"] = api_base
    if api_key:
        mk["api_key"] = api_key
    m = CleanLitellmModel(model_name=model_name, model_kwargs=mk, cost_tracking="ignore_errors")

    sys_msg = {"role": "system", "content":
               "You are a shell agent. Use the bash tool to run exactly one command per turn."}
    msgs = [sys_msg, {"role": "user", "content": "Run `echo hi` using the bash tool."}]
    # Turn 1
    r1 = m.query(msgs)
    acts1 = r1.get("extra", {}).get("actions", [])
    if not acts1:
        return False, "turn1: no tool call returned"
    # Append assistant message (full, like the agent does) + a tool result, then turn 2.
    msgs.append(r1)
    tool_id = (r1.get("tool_calls") or [{}])[0].get("id", "call_0")
    msgs.append({"role": "tool", "tool_call_id": tool_id, "content": "<returncode>0</returncode>\nhi"})
    msgs.append({"role": "user", "content": "Now run `echo done` using the bash tool."})
    r2 = m.query(msgs)
    acts2 = r2.get("extra", {}).get("actions", [])
    if not acts2:
        return False, "turn2: no tool call returned"
    return True, f"ok (2 turns, calls parsed: {len(acts1)},{len(acts2)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--api-base", default=os.getenv("LLM_API_BASE"))
    ap.add_argument("--api-key-env", default="LLM_API_KEY")
    args = ap.parse_args()
    api_key = os.getenv(args.api_key_env)
    results = []
    for name in [x.strip() for x in args.models.split(",") if x.strip()]:
        try:
            ok, msg = probe(name, args.api_base, api_key)
        except FormatError as e:
            ok, msg = False, f"FormatError: {str(e)[:160]}"
        except Exception as e:
            ok, msg = False, f"{type(e).__name__}: {str(e)[:160]}"
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name:24s} {msg}", flush=True)
        results.append((name, ok))
    print("\nWORKING:", ",".join(n for n, ok in results if ok))


if __name__ == "__main__":
    main()
