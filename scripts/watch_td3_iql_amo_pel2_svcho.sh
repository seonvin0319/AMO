#!/usr/bin/env bash
# Restart only the PEL2 launcher if it dies. Never signals trainers (Error 1).
# The launcher adopts live BootRMS / TD3 / IQL PIDs and refills P=3.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LAUNCH="$ROOT/scripts/run_td3_iql_amo_pel2_svcho.sh"
TD3_OUT="${TD3_OUT:-$ROOT/results/td3_amo_execonly_main_seeds03}"
IQL_OUT="${IQL_OUT:-$ROOT/results/iql_amo_lel2_seeds03}"
BOOT_OUT="${BOOT_OUT:-$ROOT/results/td3_amo_bootrms_maincand_seeds03}"
LOGDIR="$TD3_OUT/queue_logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/watch_pel2_$(date +%Y%m%d_%H%M%S).log"
POLL_SEC="${POLL_SEC:-30}"

launcher_alive() {
  pgrep -f 'run_td3_iql_amo_pel2_svcho\.sh' >/dev/null 2>&1
}

queue_finished() {
  [[ -f "$TD3_OUT/QUEUE_COMPLETE" && -f "$IQL_OUT/QUEUE_COMPLETE" ]]
}

exec >>"$LOG" 2>&1
echo "=== PEL2 watchdog start $(date -Is) poll=${POLL_SEC}s ==="
echo "LAUNCH=$LAUNCH"

while true; do
  if launcher_alive; then
    echo "[ok] launcher live $(date -Is)"
    sleep "$POLL_SEC"
    continue
  fi
  if queue_finished; then
    echo "[stop] QUEUE_COMPLETE and no launcher $(date -Is)"
    break
  fi
  echo "[RESTART] pel2 launcher missing; start and adopt live trainers $(date -Is)"
  df -h /home/svcho | tail -1 || true
  setsid bash "$LAUNCH" </dev/null >/dev/null 2>&1 &
  sleep 5
  if launcher_alive; then
    echo "[RESTART] launcher pid=$(pgrep -f 'run_td3_iql_amo_pel2_svcho\.sh' | head -1)"
  else
    echo "[WARN] launcher did not stay up $(date -Is)"
  fi
  sleep "$POLL_SEC"
done
echo "=== PEL2 watchdog exit $(date -Is) ==="
