#!/usr/bin/env bash
# =============================================================================
# Publish this project to your own GitHub repository.
#
# It refuses to push to the `upstream` remote (the original DiLA repository),
# so a typo cannot overwrite a repo you do not own.
#
# Usage:
#   bash push_to_github.sh <github-username> [repo-name]
#
# Example:
#   bash push_to_github.sh octocat disentangled-latent-action-world-models
#
# Prerequisites:
#   1. Create the EMPTY repo on GitHub first (no README / no .gitignore / no
#      license), or use the `gh repo create` variant printed at the end.
#   2. For HTTPS auth, GitHub no longer accepts account passwords: create a
#      Personal Access Token (classic, scope `repo`) and paste it at the
#      password prompt:  https://github.com/settings/tokens
#   3. data/, pretrained/ and checkpoints/ are gitignored and stay local.
# =============================================================================
set -euo pipefail

USER_NAME="${1:-}"
REPO_NAME="${2:-disentangled-latent-action-world-models}"
PROTOCOL="${PROTOCOL:-https}"     # https | ssh
BRANCH="${BRANCH:-main}"

die() { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

[ -n "$USER_NAME" ] || die "usage: bash push_to_github.sh <github-username> [repo-name]"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== preflight =="
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not a git repository"

UPSTREAM_URL=$(git remote get-url upstream 2>/dev/null || echo "")
if [ -n "$(git status --porcelain)" ]; then
  echo "WARNING: uncommitted changes present; they will NOT be pushed:"
  git status --short
fi

echo "branch : $BRANCH"
echo "files  : $(git ls-files | wc -l) tracked"
echo

# --- configure the destination remote ---------------------------------------
if [ "$PROTOCOL" = "ssh" ]; then
  NEW_URL="git@github.com:${USER_NAME}/${REPO_NAME}.git"
else
  NEW_URL="https://github.com/${USER_NAME}/${REPO_NAME}.git"
fi

if [ "$NEW_URL" = "$UPSTREAM_URL" ]; then
  die "refusing to push to the upstream repository (${UPSTREAM_URL}).
  Choose a different repo name, or push to your own fork instead."
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$NEW_URL"
else
  git remote add origin "$NEW_URL"
fi
echo "origin -> $NEW_URL"
echo

cat <<EOF
== checks before pushing ==
  * the GitHub repo must already exist and be EMPTY
  * never commit/push these (already gitignored):
      data/        -> 100k FD4MM frames
      pretrained/  -> 1.9 GB RAE/DINOv2 weights
      checkpoints/ -> training checkpoints (~1.4 GB each)

  Create it now if you haven't:
    web:  https://github.com/new
    gh :  gh auth login && gh repo create $REPO_NAME --public --source=. --remote=origin --push
EOF
echo

read -r -p "Push '$BRANCH' to origin now? [y/N] " ans
case "$ans" in
  [yY]*) ;;
  *) echo "aborted."; exit 0 ;;
esac

git push -u origin "$BRANCH"

echo
echo "Done. Repository: https://github.com/${USER_NAME}/${REPO_NAME}"
