#!/usr/bin/env bash
# After BootRMS QUEUE_COMPLETE, run TD3 5→1, IQL 5→1, then both 2.
# TD3: execution_only L_E+L2_RMS. IQL: original π_E (V Bellman) with β_E = L_E+L2_RMS.
# Does not kill live GPU trainers (Error 1).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BOOT_OUT="${BOOT_OUT:-$ROOT/results/td3_amo_bootrms_maincand_seeds03}"
TD3_OUT="${TD3_OUT:-$ROOT/results/td3_amo_execonly_main_seeds03}"
IQL_OUT="${IQL_OUT:-$ROOT/results/iql_amo_lel2_seeds03}"
LAUNCH="$ROOT/scripts/run_td3_iql_amo_pel2_svcho.sh"
POLL="$ROOT/scripts/poll_iql_amo_cpu_eval_parallel.sh"
CANVAS_TD3="$ROOT/scripts/loop_td3_amo_execonly_main_canvas_5m.sh"
CANVAS_IQL="$ROOT/scripts/loop_iql_amo_lel2_canvas_5m.sh"
POLL_SEC="${POLL_SEC:-20}"
PY="${PY:-/home/svcho/anaconda3/envs/offrl/bin/python}"
LOGDIR="$TD3_OUT/queue_logs"
mkdir -p "$LOGDIR" "$TD3_OUT/jobs" "$IQL_OUT/jobs" "$IQL_OUT/queue_logs"
LOG="$LOGDIR/wait_then_pel2_$(date +%Y%m%d_%H%M%S).log"
MARKER="$TD3_OUT/EXECONLY_PENDING"
BOOT_MARKER="$BOOT_OUT/EXECONLY_NEXT"

exec > >(tee -a "$LOG") 2>&1
date -Is >"$MARKER"
date -Is >"$BOOT_MARKER"
echo "=== wait_then PEL2 TD3 5→1, IQL 5→1, then both 2 start $(date -Is) ==="
echo "Waiting for BootRMS, then exec $LAUNCH"

while true; do
  if pgrep -f 'run_td3_amo_bootrms_maincand_svcho\.sh' >/dev/null 2>&1; then
    echo "[WAIT] bootrms launcher still up $(date -Is)"
    sleep "$POLL_SEC"
    continue
  fi
  if pgrep -f 'td3_amo_bootrms_maincand_seeds03' >/dev/null 2>&1; then
    echo "[WAIT] BootRMS trainers still up $(date -Is)"
    sleep "$POLL_SEC"
    continue
  fi
  if [[ ! -f "$BOOT_OUT/QUEUE_COMPLETE" ]]; then
    echo "[WAIT] BootRMS QUEUE_COMPLETE missing $(date -Is)"
    sleep "$POLL_SEC"
    continue
  fi
  echo "[GO] BootRMS idle $(date -Is) — starting PEL2 1080"
  break
done

rm -f "$TD3_OUT/QUEUE_COMPLETE" "$IQL_OUT/QUEUE_COMPLETE"
setsid env PY="$PY" OUT="$TD3_OUT" TRAIN_MATCH='train.py --algorithm td3_amo' PARALLEL_EVAL=6 \
  bash "$POLL" </dev/null >>"$TD3_OUT/queue_logs/cpu_eval_poller.nohup.stdout" 2>&1 &
echo "[SPAWN] td3 cpu poller pid=$!"
setsid env PY="$PY" OUT="$IQL_OUT" TRAIN_MATCH='train.py --algorithm iql_amo' PARALLEL_EVAL=6 \
  bash "$POLL" </dev/null >>"$IQL_OUT/queue_logs/cpu_eval_poller.nohup.stdout" 2>&1 &
echo "[SPAWN] iql cpu poller pid=$!"
setsid bash "$CANVAS_TD3" </dev/null >>"$TD3_OUT/queue_logs/canvas_refresh_5m.log" 2>&1 &
echo "[SPAWN] td3 canvas loop pid=$!"
if [[ -f "$CANVAS_IQL" ]]; then
  setsid bash "$CANVAS_IQL" </dev/null >>"$IQL_OUT/queue_logs/canvas_refresh_5m.log" 2>&1 &
  echo "[SPAWN] iql canvas loop pid=$!"
fi
rm -f "$MARKER" "$BOOT_MARKER"
exec "$LAUNCH"
