#!/usr/bin/env bash
# One tick: rewrite RAPO hopper-walker scores → commit+push sweep_results only.
# Do not checkout train/launcher files; GPU workers spawn those paths.
set -euo pipefail

ROOT=/home/shchoi/AMO_td3-amo-bootrms
PY=/home/shchoi/miniconda3/envs/offrl/bin/python
RESULTS=sweep_results/td3_rapo_jax_tinit5_td3bc_hw
export PATH="/home/shchoi/miniconda3/bin:$PATH"
export GIT_EXEC_PATH=/home/shchoi/miniconda3/libexec/git-core
export GIT_TEMPLATE_DIR=/home/shchoi/miniconda3/share/git-core/templates
GIT_DIR=$(git -C "$ROOT" rev-parse --git-dir)
LOCK=$GIT_DIR/rapo_results_push.lock

exec 9>"$LOCK"
if ! flock -w 180 9; then
  echo "fatal: could not lock $LOCK"
  exit 1
fi

cd "$ROOT"
TS=$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S %Z')
echo "=== td3 rapo hw sweep_results push @ ${TS} ==="

AUTHOR=$(git -C "$ROOT" log -1 --format='%an' origin/main 2>/dev/null || git log -1 --format='%an')
EMAIL=$(git -C "$ROOT" log -1 --format='%ae' origin/main 2>/dev/null || git log -1 --format='%ae')
export GIT_AUTHOR_NAME="$AUTHOR" GIT_AUTHOR_EMAIL="$EMAIL"
export GIT_COMMITTER_NAME="$AUTHOR" GIT_COMMITTER_EMAIL="$EMAIL"

if [[ -d .git/rebase-merge || -d .git/rebase-apply ]]; then
  echo "[recover] aborting stuck rebase"
  git rebase --abort || true
fi

# Drop only previously committed sweep_results dirt before pull; keep RAPO train files.
if git ls-files --error-unmatch "$RESULTS" >/dev/null 2>&1; then
  if ! git diff --quiet -- "$RESULTS"; then
    echo "[park] checkout $RESULTS (rewritten after pull)"
    git checkout -- "$RESULTS"
  fi
fi

pull_main() {
  git fetch origin main
  if git pull --rebase origin main; then
    return 0
  fi
  echo "pull --rebase blocked; retry with --autostash"
  git rebase --abort 2>/dev/null || true
  git pull --rebase --autostash origin main
}

set +e
pull_main
pull_rc=$?
set -e
if [[ $pull_rc -ne 0 ]]; then
  echo "fatal: git pull --rebase failed rc=$pull_rc"
  git rebase --abort 2>/dev/null || true
  exit "$pull_rc"
fi

"$PY" "$ROOT/scripts/write_td3_rapo_jax_tinit5_td3bc_hw_results.py"

git add "$RESULTS"
if git diff --cached --quiet; then
  echo "no RAPO sweep_results changes; skip commit/push"
  exit 0
fi

git commit -m "$(cat <<'EOF'
Update TD3 RAPO hopper-walker 1M D4RL scores.

EOF
)"

set +e
git push origin HEAD:main
push_rc=$?
set -e
if [[ $push_rc -ne 0 ]]; then
  echo "push rejected; pull --rebase and retry"
  pull_main
  git push origin HEAD:main
fi
echo "pushed ok $(git rev-parse --short HEAD)"
