#!/usr/bin/env bash
# Every 10 minutes: regenerate TD3+AMO JAX loco9 canvas from live store.
set -uo pipefail
ROOT=/home/ext_csv/AMO-main
PY=/home/ext_csv/miniconda3/envs/amo-jax/bin/python
LOG_DIR=/home/ext_csv/logs
LOG="$LOG_DIR/td3_amo_jax_loco9_canvas_10m.log"
PIDFILE="$LOG_DIR/td3_amo_jax_loco9_canvas_10m.pid"
INTERVAL="${AMO_JAX_LOCO9_CANVAS_REFRESH_SEC:-600}"

mkdir -p "$LOG_DIR"
echo $$ >"$PIDFILE"
echo "==== $(TZ=Asia/Seoul date -Is) loop start pid=$$ interval=${INTERVAL}s ====" >>"$LOG"

tick() {
  echo "$(TZ=Asia/Seoul date -Is) canvas tick start" >>"$LOG"
  "$PY" "$ROOT/scripts/gen_td3_amo_jax_loco9_canvas.py" >>"$LOG" 2>&1 || true
  echo "$(TZ=Asia/Seoul date -Is) canvas tick done" >>"$LOG"
}

while true; do
  tick
  sleep "$INTERVAL"
done
