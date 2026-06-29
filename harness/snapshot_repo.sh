#!/usr/bin/env bash
# Produce a flat, history-stripped checkout of a repo at a base commit, so an
# agent cannot recover the upstream fix via `git log/show/checkout/diff origin`.
#
#   snapshot_repo.sh <src_repo> <commit-ish> <dest_dir> [--fresh-git]
#
# By default the destination has NO .git at all (pure source tree). With
# --fresh-git it gets a brand-new single-commit repo with no remotes and no
# history, so agents/tools that require git still work but can see nothing
# beyond the base state.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SRC="${1:?usage: snapshot_repo.sh <src_repo> <commit> <dest> [--fresh-git]}"
COMMIT="${2:?missing <commit>}"
DEST="${3:?missing <dest>}"
FRESH_GIT="${4:-}"

[ -d "$SRC/.git" ] || die "not a git repo: $SRC"
[ -e "$DEST" ] && die "destination already exists: $DEST"
git -C "$SRC" rev-parse --verify -q "$COMMIT^{commit}" >/dev/null || die "bad commit: $COMMIT"

mkdir -p "$DEST"
# git archive emits ONLY the tree at <commit> -- no .git, no other commits.
git -C "$SRC" archive --format=tar "$COMMIT" | tar -x -C "$DEST"

if [ "$FRESH_GIT" = "--fresh-git" ]; then
  ( cd "$DEST"
    git init -q
    git -c user.email=eval@harness -c user.name=harness add -A
    git -c user.email=eval@harness -c user.name=harness commit -q -m "base state"
    git remote 2>/dev/null | while read -r r; do git remote remove "$r"; done )
  log "snapshot (fresh-git, no history/remotes): $DEST @ $(git -C "$SRC" rev-parse --short "$COMMIT")"
else
  log "snapshot (no .git): $DEST @ $(git -C "$SRC" rev-parse --short "$COMMIT")"
fi
echo "$DEST"
