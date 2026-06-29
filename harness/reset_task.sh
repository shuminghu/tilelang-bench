#!/usr/bin/env bash
# Tear down a task workspace created by setup_task.sh.
#
#   reset_task.sh <workspace_dir>
#
# Because all mutable state (HOME, TMPDIR, caches, solution edits) lives inside
# the workspace, a single rm -rf fully resets it. CCACHE_DIR is intentionally
# preserved (shared, content-addressed) to keep internals-track rebuilds fast.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

WORK="${1:?usage: reset_task.sh <workspace_dir>}"
case "$(cd "$WORK" 2>/dev/null && pwd)/" in
  "$RUNS_DIR"/*) : ;;                       # safety: only delete inside runs/
  *) die "refusing to delete path outside $RUNS_DIR: $WORK" ;;
esac

rm -rf "$WORK"
log "removed workspace: $WORK"
