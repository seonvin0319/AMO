#!/usr/bin/env bash
# Refresh td3-rapo-jax-tinit5-td3bc-hw.canvas.tsx every 5 minutes.
set -uo pipefail

ROOT=/home/shchoi/AMO_td3-amo-bootrms
OUT=$ROOT/results/td3_rapo_jax_tinit5_td3bc_hw_s03
PY=/home/shchoi/miniconda3/bin/python
GEN=$ROOT/scripts/gen_td3_rapo_jax_tinit5_s03_canvas.py
INTERVAL_SEC="${INTERVAL_SEC:-300}"
PID_FILE=$OUT/canvas_5m.pid
LOG=$OUT/canvas_5m.log

mkdir -p "$OUT"
if [[ -f "$PID_FILE" ]]; then
  old="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    printf '[%s] already running pid=%s\n' "$(TZ=Asia/Seoul date '+%F %T %Z')" "$old" | tee -a "$LOG"
    exit 0
  fi
fi
echo $$ >"$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

printf '[%s] td3_rapo tinit5 canvas 5m watcher start\n' "$(TZ=Asia/Seoul date '+%F %T %Z')" | tee -a "$LOG"

while true; do
  if "$PY" -u "$GEN" >>"$LOG" 2>&1; then
    printf '[%s] canvas ok\n' "$(TZ=Asia/Seoul date '+%F %T %Z')" >>"$LOG"
  else
    printf '[%s] canvas FAILED\n' "$(TZ=Asia/Seoul date '+%F %T %Z')" >>"$LOG"
  fi
  sleep "$INTERVAL_SEC"
done
