#!/usr/bin/env bash
set -u
OUT="${OUT:-/home/svcho/amo/results/iql_amo_lel2_seeds03}"
PY="${PY:-/home/svcho/anaconda3/envs/offrl/bin/python}"
SCRIPT="${SCRIPT:-/home/svcho/amo/scripts/refresh_iql_amo_lel2_canvas.py}"
INTERVAL="${INTERVAL:-300}"

queue_still_going() {
  [[ -f "$OUT/PEL2_LAUNCHED" && ! -f "$OUT/QUEUE_COMPLETE" ]] && return 0
  pgrep -f 'run_td3_iql_amo_pel2_svcho\.sh' >/dev/null 2>&1 && return 0
  pgrep -f 'wait_then_execonly_main_svcho\.sh' >/dev/null 2>&1 && return 0
  pgrep -f 'train.py --algorithm iql_amo' >/dev/null 2>&1 && return 0
  return 1
}

echo "[start] $(date -Is) OUT=$OUT interval=${INTERVAL}s"
while true; do
  if [[ -f "$OUT/QUEUE_COMPLETE" ]] && ! queue_still_going; then
    echo "[stop] QUEUE_COMPLETE $(date -Is)"
    "$PY" "$SCRIPT" >/dev/null || true
    break
  fi
  if "$PY" "$SCRIPT" >/dev/null; then
    echo "[ok] $(date -Is) refreshed"
  else
    echo "[err] $(date -Is) refresh failed rc=$?"
  fi
  sleep "$INTERVAL"
done
