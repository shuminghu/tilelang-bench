#!/usr/bin/env bash
# Assert the immutable golden environment has not drifted (e.g. an agent that
# ran `uv pip install ...` against the shared venv). Run between tasks.
#
#   verify_clean.sh --record     # store the current golden hash as the baseline
#   verify_clean.sh              # compare against the baseline; nonzero exit on drift

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BASELINE="$HARNESS_DIR/golden_freeze.sha256"

if [ "${1:-}" = "--record" ]; then
  golden_freeze_hash > "$BASELINE"
  log "recorded golden baseline: $(cat "$BASELINE")"
  exit 0
fi

[ -f "$BASELINE" ] || die "no baseline; run: verify_clean.sh --record"
want="$(cat "$BASELINE")"
have="$(golden_freeze_hash)"
if [ "$want" != "$have" ]; then
  die "GOLDEN VENV DRIFTED  expected=$want  actual=$have  (restore from snapshot before continuing)"
fi
log "golden venv clean ($have)"
