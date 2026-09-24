#!/usr/bin/env bash
# Offline CPU eval poller for RAPO tinit5 JAX grid (1M checkpoints only).
set -uo pipefail
trap '' HUP

ROOT=/home/shchoi/AMO_td3-amo-bootrms
OUT=$ROOT/results/td3_rapo_jax_tinit5_td3bc_hw_s03
PY=/home/shchoi/miniconda3/envs/offrl/bin/python
PID_FILE=$OUT/eval_cpu.pid
LOG=$OUT/eval_cpu.log

mkdir -p "$OUT/runs"
if [[ -f "$PID_FILE" ]]; then
  old="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    printf '[%s] already running pid=%s\n' "$(TZ=Asia/Seoul date '+%F %T %Z')" "$old" | tee -a "$LOG"
    exit 0
  fi
fi
echo $$ >"$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export D4RL_SUPPRESS_IMPORT_ERROR=1
export D4RL_DATASET_DIR="$HOME/.d4rl/datasets"
export MUJOCO_GL=egl
export WANDB_MODE=offline

printf '[%s] td3_rapo tinit5 cpu eval start\n' "$(TZ=Asia/Seoul date '+%F %T %Z')" | tee -a "$LOG"
exec "$PY" -u "$ROOT/scripts/eval_checkpoints_cpu.py" \
  --runs-root "$OUT/runs" \
  --poll \
  --poll-sec 120 \
  --final-only \
  --device cpu \
  >>"$LOG" 2>&1
