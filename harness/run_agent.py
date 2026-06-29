#!/usr/bin/env python3
"""Drive one task with mini-swe-agent, then score it with the private grader.

Runs in the DRIVER venv (.venv-agent, has mini-swe-agent). The agent's bash
commands execute in a hardened LocalEnvironment:
  - cwd = the task workspace (only solution.py + prompt.txt visible)
  - PATH points at the GOLDEN venv (so `python` has tilelang)
  - caches redirected into the workspace; network forced offline; API keys blanked
The model API key stays in THIS process's env (for litellm) but is blanked in the
subprocess env, so the agent can't exfiltrate it or fetch the upstream fix.

Usage:
  python run_agent.py --task-dir tasks/perf/gemm_optimize --model deterministic
  python run_agent.py --task-dir tasks/perf/gemm_optimize --model anthropic/claude-sonnet-4-5 \
      --step-limit 40 --cost-limit 2.0
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from minisweagent import package_dir
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "harness"
GOLDEN_VENV = Path(os.getenv("GOLDEN_VENV", ROOT / ".venv"))

SENSITIVE = [
    "GH_TOKEN", "GITHUB_TOKEN", "GITHUB_API_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "WANDB_API_KEY",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "NETRC",
]


def build_exec_env(work: Path) -> dict:
    """Env overrides applied to the agent's subprocesses (merged over os.environ)."""
    gpu = os.getenv("HARNESS_GPU", "7")
    env = {
        "VIRTUAL_ENV": str(GOLDEN_VENV),
        "PATH": f"{GOLDEN_VENV}/bin:" + os.environ.get("PATH", ""),
        "HOME": str(work / "home"),
        "TMPDIR": str(work / "tmp"),
        "XDG_CACHE_HOME": str(work / "cache"),
        "TILELANG_CACHE_DIR": str(work / "tl_cache"),
        "TRITON_CACHE_DIR": str(work / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(work / "inductor"),
        "PYTHONPYCACHEPREFIX": str(work / "pyc"),
        "CUDA_VISIBLE_DEVICES": gpu,
        # network off (best-effort; no netns on this host)
        "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
        "ALL_PROXY": "http://127.0.0.1:9", "http_proxy": "http://127.0.0.1:9",
        "https_proxy": "http://127.0.0.1:9", "all_proxy": "http://127.0.0.1:9",
        "NO_PROXY": "", "no_proxy": "",
        "PIP_NO_INDEX": "1", "UV_OFFLINE": "1", "GIT_TERMINAL_PROMPT": "0",
        "GIT_ALLOW_PROTOCOL": "file", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PAGER": "cat", "MANPAGER": "cat", "TQDM_DISABLE": "1", "PIP_PROGRESS_BAR": "off",
    }
    for k in SENSITIVE:        # blank credentials for the subprocess only
        env[k] = ""
    return env


def make_model(name: str, api_base: str | None, api_key: str | None):
    if name == "deterministic":
        from minisweagent.models.test_models import DeterministicModel, make_output
        tuned = (TASK_DIR / "_oracle_solution.py")
        sol = tuned.read_text() if tuned.exists() else _DEFAULT_TUNED
        write_cmd = "cat > solution.py <<'PYEOF'\n" + sol + "\nPYEOF"
        return DeterministicModel(outputs=[
            make_output("Replacing solution.py with a tuned kernel.", [{"command": write_cmd}]),
            make_output("Submitting.", [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}]),
        ])
    sys.path.insert(0, str(HARNESS))
    from zen_model import CleanLitellmModel
    # Route bare model ids through litellm's OpenAI-compatible path when a custom
    # endpoint (e.g. OpenCode Zen) is supplied.
    model_name = name if "/" in name else (f"openai/{name}" if api_base else name)
    mk: dict = {"num_retries": 2, "timeout": 180}
    if api_base:
        mk["api_base"] = api_base
    if api_key:
        mk["api_key"] = api_key
    return CleanLitellmModel(model_name=model_name, model_kwargs=mk, cost_tracking="ignore_errors")


_DEFAULT_TUNED = """import tilelang
import tilelang.language as T

def build(M, N, K):
    bM, bN, bK, stages, threads = 128, 128, 64, 3, 128
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"),
             C: T.Tensor((M, N), "float16")):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            As = T.alloc_shared((bM, bK), "float16"); Bs = T.alloc_shared((bK, bN), "float16")
            Cl = T.alloc_fragment((bM, bN), "float"); T.clear(Cl)
            for ko in T.Pipelined(T.ceildiv(K, bK), num_stages=stages):
                T.copy(A[by*bM, ko*bK], As); T.copy(B[ko*bK, bx*bN], Bs)
                T.gemm(As, Bs, Cl)
            T.copy(Cl, C[by*bM, bx*bN])
    return tilelang.compile(main, out_idx=[2])
"""


def main():
    global TASK_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--model", required=True, help="'deterministic' or a litellm model name")
    ap.add_argument("--api-base", default=os.getenv("LLM_API_BASE"),
                    help="custom OpenAI-compatible base URL (e.g. https://opencode.ai/zen/v1)")
    ap.add_argument("--api-key-env", default="LLM_API_KEY",
                    help="env var holding the API key for the custom endpoint")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--step-limit", type=int, default=30)
    ap.add_argument("--cost-limit", type=float, default=2.0)
    ap.add_argument("--max-format-errors", type=int, default=10,
                    help="consecutive format errors tolerated before giving up")
    ap.add_argument("--timeout", type=int, default=600, help="per-command timeout (s)")
    args = ap.parse_args()

    TASK_DIR = Path(args.task_dir).resolve()
    run_id = args.run_id or f"{args.model.replace('/', '_')}_{os.getpid()}"

    # 1) Materialize a hardened workspace (grader stays private).
    out = subprocess.run(["bash", str(HARNESS / "setup_task.sh"), str(TASK_DIR), run_id],
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"setup_task failed:\n{out.stderr}")
    work = Path(out.stdout.strip().splitlines()[-1])
    prompt = (work / "prompt.txt").read_text().strip()

    # 2) Build hardened env + agent.
    cfg = yaml.safe_load((package_dir / "config" / "default.yaml").read_text())
    env = LocalEnvironment(cwd=str(work), env=build_exec_env(work), timeout=args.timeout)
    api_key = os.getenv(args.api_key_env) if args.model != "deterministic" else None
    model = make_model(args.model, args.api_base, api_key)
    # Keep the trajectory OUT of the agent's cwd so it can't read/confuse itself.
    traj_path = work.parent / f"{run_id}.trajectory.json"
    agent = DefaultAgent(
        model, env,
        system_template=cfg["agent"]["system_template"],
        instance_template=cfg["agent"]["instance_template"],
        step_limit=args.step_limit, cost_limit=args.cost_limit,
        max_consecutive_format_errors=args.max_format_errors,
        output_path=traj_path,
    )
    task = (prompt + "\n\nConstraints: edit ONLY solution.py in the current directory. "
            "You have no network access. Submit when done.")

    exit_status = "Unknown"
    try:
        result = agent.run(task)
        exit_status = result.get("exit_status", "Unknown")
    except Exception as e:
        exit_status = f"AgentError: {type(e).__name__}: {e}"

    # 3) Score with the private grader.
    sc = subprocess.run(["bash", str(HARNESS / "score_task.sh"), str(work), str(TASK_DIR)],
                        capture_output=True, text=True)
    try:
        score_json = json.loads(sc.stdout[sc.stdout.index("{"):sc.stdout.rindex("}") + 1])
    except Exception:
        score_json = {"final_score": 0.0, "error": "score parse failed", "stderr": sc.stderr[-2000:]}

    record = {
        "task": TASK_DIR.name, "model": args.model, "run_id": run_id,
        "exit_status": exit_status, "final_score": score_json.get("final_score", 0.0),
        "per_shape": score_json.get("per_shape"), "workspace": str(work),
        "trajectory": str(traj_path),
        "n_model_calls": agent.n_calls, "model_cost": round(agent.cost, 4),
    }
    (work / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({k: record[k] for k in
                      ["task", "model", "exit_status", "final_score", "n_model_calls", "model_cost"]},
                     indent=2))


if __name__ == "__main__":
    main()
