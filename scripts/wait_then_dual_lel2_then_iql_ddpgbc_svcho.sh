#!/usr/bin/env bash
# After BootRMS QUEUE_COMPLETE, run dual-actor L_E+L2_RMS (α_init=5, 15 envs), then iql_ddpgbc_amo.
# Does not signal BootRMS trainers (Error 1). Does not use set -e (Error 7).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BOOT_OUT="${BOOT_OUT:-$ROOT/results/td3_amo_bootrms_maincand_seeds03}"
DUAL_OUT="${DUAL_OUT:-$ROOT/results/td3_amo_dual_lel2_seeds03}"
IQL_OUT="${IQL_OUT:-$ROOT/results/iql_ddpgbc_amo_seeds03}"
WAVE="$ROOT/scripts/run_svcho_amo_wave.sh"
POLL="$ROOT/scripts/poll_jobs_cpu_eval.sh"
POLL_SEC="${POLL_SEC:-20}"
PARALLEL="${PARALLEL:-2}"
LOGDIR="$DUAL_OUT/queue_logs"
mkdir -p "$LOGDIR" "$DUAL_OUT/jobs" "$IQL_OUT/jobs"
LOG="$LOGDIR/wait_then_dual_then_iql_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1
echo "=== wait_then BootRMS → dual_lel2 → iql_ddpgbc start $(date -Is) PARALLEL=$PARALLEL ==="

bootrms_busy() {
  pgrep -f 'run_td3_amo_bootrms_maincand_svcho\.sh' >/dev/null 2>&1 && return 0
  pgrep -f 'train.py.*td3_amo_bootrms_maincand_seeds03' >/dev/null 2>&1 && return 0
  return 1
}

disk_ok() {
  local avail
  avail="$(df -Pk /home/svcho | awk 'NR==2 {print $4}')"
  [[ "${avail:-0}" -ge 20000000 ]]
}

ensure_eval() {
  local out="$1"
  mkdir -p "$out/queue_logs"
  local pidfile="$out/queue_logs/eval_poller.pid"
  if [[ -f "$pidfile" ]]; then
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  fi
  nohup env OUT="$out" bash "$POLL" </dev/null >/dev/null 2>&1 &
  echo $! >"$pidfile"
  echo "[EVAL] poller pid=$! OUT=$out"
}

while true; do
  if [[ -f "$BOOT_OUT/QUEUE_COMPLETE" ]] && ! bootrms_busy; then
    echo "[GO] BootRMS idle $(date -Is)"
    break
  fi
  echo "[WAIT] BootRMS QUEUE_COMPLETE=$([[ -f $BOOT_OUT/QUEUE_COMPLETE ]] && echo yes || echo no) $(date -Is)"
  sleep "$POLL_SEC"
done

while ! disk_ok; do
  echo "[WAIT_DISK] $(df -h /home/svcho | tail -1) $(date -Is)"
  sleep 60
done

if [[ ! -f "$DUAL_OUT/QUEUE_COMPLETE" ]]; then
  ensure_eval "$DUAL_OUT"
  echo "[GO] dual_lel2 $(date -Is)"
  env WAVE=dual_lel2 OUT="$DUAL_OUT" PARALLEL="$PARALLEL" bash "$WAVE"
  echo "[DONE_WAVE] dual_lel2 rc=$? $(date -Is)"
fi

while pgrep -f 'train.py.*td3_amo_dual_lel2_seeds03' >/dev/null 2>&1; do
  echo "[WAIT] dual trainers still up $(date -Is)"
  sleep "$POLL_SEC"
done

while ! disk_ok; do
  echo "[WAIT_DISK] $(df -h /home/svcho | tail -1) $(date -Is)"
  sleep 60
done

if [[ ! -f "$IQL_OUT/QUEUE_COMPLETE" ]]; then
  ensure_eval "$IQL_OUT"
  echo "[GO] iql_ddpgbc $(date -Is)"
  env WAVE=iql_ddpgbc OUT="$IQL_OUT" PARALLEL="$PARALLEL" bash "$WAVE"
  echo "[DONE_WAVE] iql_ddpgbc rc=$? $(date -Is)"
fi

echo "=== chain complete $(date -Is) ==="
date -Is >"$DUAL_OUT/CHAIN_COMPLETE"
