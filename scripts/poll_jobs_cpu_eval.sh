#!/usr/bin/env bash
# FINAL-ONLY CPU eval poller for $OUT/jobs/*/run. Error 6b: CPU only.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:?OUT is required}"
PY="${PY:-/home/svcho/anaconda3/envs/offrl/bin/python}"
PARALLEL_EVAL="${PARALLEL_EVAL:-4}"
POLL_SEC="${POLL_SEC:-20}"
LOGDIR="$OUT/queue_logs"
mkdir -p "$LOGDIR" "$OUT/cpu_eval_logs"
LOG="$LOGDIR/cpu_final_eval_parallel_${PARALLEL_EVAL}_$(date +%Y%m%d_%H%M%S).log"

export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export D4RL_SUPPRESS_IMPORT_ERROR=1
export D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-/home/svcho/.d4rl/datasets}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-/home/svcho/.mujoco/mujoco210}"
export LD_LIBRARY_PATH="/home/svcho/.mujoco/mujoco210/bin:/usr/lib/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

exec > >(tee -a "$LOG") 2>&1
echo "=== cpu FINAL-ONLY PARALLEL=$PARALLEL_EVAL OUT=$OUT start $(date -Is) ==="

needs_final() {
  local cell="$1"
  local run="$cell/run"
  [[ -f "$run/checkpoint.npz" || -f "$run/checkpoints/step_1000000.npz" ]] || return 1
  "$PY" - "$run" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
ej = run / "eval.jsonl"
if not ej.exists():
    raise SystemExit(0)
for line in ej.read_text().splitlines():
    if not line.strip():
        continue
    o = json.loads(line)
    if int(o.get("step") or 0) >= 1_000_000 and (
        o.get("tag") == "posthoc_cpu_final" or int(o.get("episodes") or 0) >= 50
    ):
        raise SystemExit(1)
raise SystemExit(0)
PY
}

declare -A BUSY=()

reap_busy() {
  local tag pid
  for tag in "${!BUSY[@]}"; do
    pid="${BUSY[$tag]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      echo "[EVAL_DONE] $tag pid=$pid $(date -Is)"
      unset "BUSY[$tag]"
    fi
  done
}

start_eval() {
  local cell="$1"
  local tag
  tag="$(basename "$cell")"
  local elog="$OUT/cpu_eval_logs/${tag}_final.log"
  echo "[EVAL_START] $tag $(date -Is)"
  "$PY" "$ROOT/scripts/eval_checkpoints_cpu.py" \
    --runs-root "$cell" \
    --run run \
    --episodes 10 \
    --final-repeats 5 \
    --device cpu \
    --final-only \
    --keep-mid-ckpts \
    --max-per-run 1 \
    >"$elog" 2>&1 &
  BUSY["$tag"]=$!
}

queue_alive() {
  [[ -f "$OUT/QUEUE_COMPLETE" ]] && return 1
  pgrep -f 'run_svcho_amo_wave\.sh' >/dev/null 2>&1 && return 0
  pgrep -f 'wait_then_dual_lel2_then_iql_ddpgbc_svcho\.sh' >/dev/null 2>&1 && return 0
  pgrep -f "${OUT}/jobs" >/dev/null 2>&1 && return 0
  return 1
}

while true; do
  reap_busy
  n_need=0
  n_have=0
  for cell in "$OUT"/jobs/*; do
    [[ -d "$cell" ]] || continue
    tag="$(basename "$cell")"
    if [[ ! -f "$cell/DONE" ]] && [[ ! -f "$cell/run/checkpoints/step_1000000.npz" ]]; then
      continue
    fi
    if ! needs_final "$cell"; then
      n_have=$((n_have + 1))
      continue
    fi
    n_need=$((n_need + 1))
    [[ -n "${BUSY[$tag]:-}" ]] && continue
    if [[ "${#BUSY[@]}" -ge "$PARALLEL_EVAL" ]]; then
      continue
    fi
    start_eval "$cell"
  done
  echo "{\"at\":\"$(date -Is)\",\"busy\":${#BUSY[@]},\"need_or_running\":$n_need,\"finals_have\":$n_have}"
  if [[ "${#BUSY[@]}" -eq 0 && "$n_need" -eq 0 ]]; then
    if queue_alive; then
      sleep "$POLL_SEC"
      continue
    fi
    echo "=== all finals done $(date -Is); exiting ==="
    exit 0
  fi
  sleep "$POLL_SEC"
done
