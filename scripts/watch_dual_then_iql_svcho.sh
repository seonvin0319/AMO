#!/usr/bin/env bash
# Restart the post-BootRMS chain if the waiter/wave dies. Never signals trainers.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WAIT="$ROOT/scripts/wait_then_dual_lel2_then_iql_ddpgbc_svcho.sh"
BOOT_OUT="${BOOT_OUT:-$ROOT/results/td3_amo_bootrms_maincand_seeds03}"
DUAL_OUT="${DUAL_OUT:-$ROOT/results/td3_amo_dual_lel2_seeds03}"
IQL_OUT="${IQL_OUT:-$ROOT/results/iql_ddpgbc_amo_seeds03}"
LOGDIR="$DUAL_OUT/queue_logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/watch_chain_$(date +%Y%m%d_%H%M%S).log"
POLL_SEC="${POLL_SEC:-30}"
PARALLEL="${PARALLEL:-2}"

alive() {
  pgrep -f 'wait_then_dual_lel2_then_iql_ddpgbc_svcho\.sh' >/dev/null 2>&1 && return 0
  pgrep -f 'run_svcho_amo_wave\.sh' >/dev/null 2>&1 && return 0
  return 1
}

exec >>"$LOG" 2>&1
echo "=== post-BootRMS chain watchdog start $(date -Is) PARALLEL=$PARALLEL ==="

while true; do
  if [[ -f "$IQL_OUT/QUEUE_COMPLETE" ]] && [[ -f "$DUAL_OUT/QUEUE_COMPLETE" ]]; then
    echo "[stop] both QUEUE_COMPLETE $(date -Is)"
    break
  fi
  if alive; then
    echo "[ok] chain live $(date -Is)"
    sleep "$POLL_SEC"
    continue
  fi
  echo "[RESTART] chain missing $(date -Is)"
  df -h /home/svcho | tail -1 || true
  setsid env PARALLEL="$PARALLEL" bash "$WAIT" </dev/null >/dev/null 2>&1 &
  sleep 5
  if alive; then
    echo "[RESTART] ok $(date -Is)"
  else
    echo "[WARN] chain did not stay up $(date -Is)"
  fi
  sleep "$POLL_SEC"
done
echo "=== chain watchdog exit $(date -Is) ==="
