#!/usr/bin/env bash
# Every 5 minutes: push RAPO hopper-walker sweep_results to AMO origin/main.
set -uo pipefail
trap '' HUP

ROOT=/home/shchoi/AMO_td3-amo-bootrms
OUT=$ROOT/results/td3_rapo_jax_tinit5_td3bc_hw_s03
LOG=$OUT/push_5m.log
PID_FILE=$OUT/push_5m.pid
TICK=$ROOT/scripts/push_td3_rapo_jax_tinit5_td3bc_hw_tick.sh
INTERVAL_SEC="${INTERVAL_SEC:-300}"

mkdir -p "$OUT"
if [[ -f "$PID_FILE" ]]; then
  old="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    printf '[%s] already running pid=%s\n' "$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S %Z')" "$old" | tee -a "$LOG"
    exit 0
  fi
fi
echo $$ >"$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT
exec >>"$LOG" 2>&1

log() { printf '[%s] %s\n' "$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S %Z')" "$*"; }

log "push_5m start pid=$$"
while true; do
  log "tick begin"
  if bash "$TICK"; then
    log "tick ok"
  else
    log "tick FAILED rc=$?"
  fi
  sleep "$INTERVAL_SEC"
done
