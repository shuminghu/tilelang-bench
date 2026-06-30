#!/usr/bin/env bash
# Shared configuration + helpers for the tilelang eval harness.
# Source this from other scripts:  source "$(dirname "$0")/common.sh"

set -euo pipefail

# Resolve repo-eval root (parent of harness/)
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HARNESS_DIR/.." && pwd)"

# Immutable "golden" assets.
GOLDEN_VENV="${GOLDEN_VENV:-$ROOT/.venv}"      # read-only env shared by perf tasks
TASKS_DIR="${TASKS_DIR:-$ROOT/tasks}"
RUNS_DIR="${RUNS_DIR:-$ROOT/runs}"

# Private answer-keys root, OUTSIDE the repo tree so a browsing agent can't discover
# the grader oracle/target via `ls ../..`. Open-book (docs/examples/installed package)
# is fine; only the per-task answer key lives here. Regenerable from gen_tasks.py.
export GRADERS_DIR="${GRADERS_DIR:-$HOME/.tl_graders}"

# Shared, content-addressed ccache (safe to share across tasks; speeds rebuilds).
export CCACHE_DIR="${CCACHE_DIR:-$ROOT/.ccache}"

# Pin the GPU the harness uses (single-GPU design; box is shared/multi-GPU).
# Forced to physical GPU 7 (in this user's 6,7 allocation, and idle).
# Override with HARNESS_GPU=<id> if needed.
export CUDA_VISIBLE_DEVICES="${HARNESS_GPU:-7}"

log()  { printf '[harness] %s\n' "$*" >&2; }
die()  { printf '[harness][ERROR] %s\n' "$*" >&2; exit 1; }

# Compute a stable hash of the golden venv's installed package set.
golden_freeze_hash() {
  ( . "$GOLDEN_VENV/bin/activate" && uv pip freeze 2>/dev/null | LC_ALL=C sort ) | sha256sum | cut -d' ' -f1
}
