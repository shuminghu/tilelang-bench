#!/usr/bin/env bash
# Score a solved workspace using the PRIVATE grader from the task source dir.
# The grader (problem.py / task.json) never enters the agent workspace, so the
# agent cannot read the target kernel or reference out of its own cwd.
#
#   score_task.sh <workspace_dir> <task_dir>
#
# Prints the scorer JSON on stdout.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

WORK="${1:?usage: score_task.sh <workspace_dir> <task_dir>}"
TASK_DIR="${2:?usage: score_task.sh <workspace_dir> <task_dir>}"
[ -f "$WORK/env.sh" ]        || die "no env.sh in workspace: $WORK"
[ -f "$TASK_DIR/task.json" ] || die "no task.json in task dir: $TASK_DIR"

# Resolve the grader (answer key). Prefer the PRIVATE graders dir (outside the repo,
# not discoverable by a browsing agent); fall back to the in-tree copy for back-compat.
TASK_JSON="$(cd "$TASK_DIR" && pwd)/task.json"
TID="$(python3 -c "import json;print(json.load(open('$TASK_JSON'))['id'])")"
if [ -f "$GRADERS_DIR/$TID/problem.py" ]; then
  PROBLEM="$GRADERS_DIR/$TID/problem.py"
elif [ -f "$TASK_DIR/problem.py" ]; then
  PROBLEM="$(cd "$TASK_DIR" && pwd)/problem.py"
else
  die "no problem.py in $GRADERS_DIR/$TID or $TASK_DIR"
fi
SOLUTION="$(cd "$WORK" && pwd)/solution.py"

# Dispatch to the scorer for this task's track.
TRACK="$(python3 -c "import json;print(json.load(open('$TASK_JSON')).get('track','perf'))")"
case "$TRACK" in
  perf)       SCORER="score_perf.py" ;;
  implement)  SCORER="score_correct.py" ;;
  debug)      SCORER="score_correct.py" ;;
  regression) SCORER="score_regression.py" ;;
  *)          SCORER="score_perf.py" ;;
esac

( source "$WORK/env.sh"
  python3 "$HARNESS_DIR/$SCORER" \
    --task "$TASK_JSON" --problem "$PROBLEM" --solution "$SOLUTION" )
