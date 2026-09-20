#!/usr/bin/env bash
# Durable 5-minute Canvas refresher for TD3-AMO π_E-only main 540.
set -u
OUT="${OUT:-/home/svcho/amo/results/td3_amo_execonly_main_seeds03}"
PY="${PY:-/home/svcho/anaconda3/envs/offrl/bin/python}"
SCRIPT="${SCRIPT:-/home/svcho/amo/scripts/refresh_td3_amo_execonly_main_canvas.py}"
INTERVAL="${INTERVAL:-300}"

queue_still_going() {
  [[ -f "$OUT/EXECONLY_PENDING" ]] && return 0
  pgrep -f 'run_td3_amo_execonly_main_svcho\.sh' >/dev/null 2>&1 && return 0
  pgrep -f 'run_td3_iql_amo_pel2_svcho\.sh' >/dev/null 2>&1 && return 0
  pgrep -f 'wait_then_execonly_main_svcho\.sh' >/dev/null 2>&1 && return 0
  pgrep -f 'train.py --algorithm td3_amo' >/dev/null 2>&1 && return 0
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
