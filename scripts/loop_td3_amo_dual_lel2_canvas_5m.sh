#!/usr/bin/env bash
# Durable 5-minute Canvas refresher for dual L_E+L2_RMS vs BootRMS.
set -u
OUT="${OUT:-/home/svcho/amo/results/td3_amo_dual_lel2_seeds03}"
PY="${PY:-/home/svcho/anaconda3/envs/offrl/bin/python}"
SCRIPT="${SCRIPT:-/home/svcho/amo/scripts/refresh_td3_amo_dual_lel2_canvas.py}"
INTERVAL="${INTERVAL:-300}"

queue_still_going() {
  pgrep -f 'wait_then_dual_lel2_then_iql_ddpgbc_svcho\.sh' >/dev/null 2>&1 && return 0
  pgrep -f 'run_svcho_amo_wave\.sh' >/dev/null 2>&1 && return 0
  pgrep -f 'td3_amo_dual_lel2_seeds03/jobs' >/dev/null 2>&1 && return 0
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
