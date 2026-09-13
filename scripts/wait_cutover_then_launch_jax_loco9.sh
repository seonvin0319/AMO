#!/usr/bin/env bash
# After torch band95 cutover READY: GPU smoke → detach JAX loco9 queue → canvas loop.
set -uo pipefail
READY=/home/ext_csv/logs/amo_torch_cutover_READY
LOG=/home/ext_csv/logs/amo_jax_loco9_post_cutover_launch.log
ROOT=/home/ext_csv/AMO-main
PY=/home/ext_csv/miniconda3/envs/amo-jax/bin/python
LAUNCHER="$ROOT/scripts/launch_td3_amo_jax_loco9.py"
STORE=/raid/ext_csv/AMO_store/td3_amo_jax_loco9_default_seeds0to3
CANVAS_LOOP="$ROOT/scripts/refresh_td3_amo_jax_loco9_canvas_10m.sh"
PIDFILE=/home/ext_csv/logs/amo_jax_loco9_post_cutover_launch.pid

mkdir -p "$(dirname "$LOG")" "$STORE"
echo $$ >"$PIDFILE"
{
  echo "==== $(TZ=Asia/Seoul date -Is) post-cutover chain start pid=$$ ===="
  while [ ! -f "$READY" ]; do
    echo "$(TZ=Asia/Seoul date -Is) waiting READY"
    sleep 60
  done
  echo "$(TZ=Asia/Seoul date -Is) READY seen: $(cat "$READY")"

  # Finalize band95 canvas once after cutover
  if [ -x /home/ext_csv/AMO-lambda0-te-tb/scripts/gen_amo_loco9_tlr3e4_band95_seeds1to3_canvas.py ] \
    || [ -f /home/ext_csv/AMO-lambda0-te-tb/scripts/gen_amo_loco9_tlr3e4_band95_seeds1to3_canvas.py ]; then
    /home/ext_csv/miniconda3/envs/mujo/bin/python \
      /home/ext_csv/AMO-lambda0-te-tb/scripts/gen_amo_loco9_tlr3e4_band95_seeds1to3_canvas.py \
      || true
  fi

  echo "$(TZ=Asia/Seoul date -Is) nvidia-smi before smoke"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv || true

  # Short GPU smoke on first free-enough GPU (prefer lowest used)
  SMOKE_DIR="$STORE/smoke_hopper_medium_s0"
  rm -rf "$SMOKE_DIR"
  mkdir -p "$SMOKE_DIR"
  export MUJOCO_GL=egl
  export MUJOCO_PY_MUJOCO_PATH=/home/ext_csv/.mujoco/mujoco210
  export LD_LIBRARY_PATH="/home/ext_csv/.mujoco/mujoco210/bin:/home/ext_csv/miniconda3/envs/amo-jax/lib:/home/ext_csv/.local/osmesa:${LD_LIBRARY_PATH:-}"
  export D4RL_SUPPRESS_IMPORT_ERROR=1
  export D4RL_DATASET_DIR=/raid/ext_csv/datasets/d4rl
  export WANDB_MODE=offline
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export CUDA_VISIBLE_DEVICES=0
  echo "$(TZ=Asia/Seoul date -Is) GPU smoke start (max_steps=200)"
  if ! "$PY" -u "$ROOT/train.py" \
      --algorithm=td3_amo --backend=jax \
      --env=hopper-medium-v2 --seed=0 --device=cuda:0 \
      --config="$ROOT/configs/td3_amo.yaml" \
      --dataset=/raid/ext_csv/datasets/d4rl/hopper_medium-v2.hdf5 \
      --output="$SMOKE_DIR" \
      --steps=200 --log-every=50 --save-every=200 --no-eval 2>&1; then
    echo "$(TZ=Asia/Seoul date -Is) ERROR: GPU smoke failed; not launching queue"
    exit 1
  fi
  echo "$(TZ=Asia/Seoul date -Is) GPU smoke OK"
  rm -rf "$SMOKE_DIR"

  echo "$(TZ=Asia/Seoul date -Is) detach JAX launcher"
  "$PY" "$LAUNCHER" --detach --gpus 0,1 --max-parallel 6 --max-used-mib 80000 --out "$STORE"
  sleep 5
  "$PY" "$LAUNCHER" --status-only --out "$STORE" || true
  pgrep -af 'train.py --algorithm td3_amo --backend jax' | grep -v pgrep || true

  if [ -f "$CANVAS_LOOP" ]; then
    if [ -f /home/ext_csv/logs/td3_amo_jax_loco9_canvas_10m.pid ] \
      && kill -0 "$(cat /home/ext_csv/logs/td3_amo_jax_loco9_canvas_10m.pid)" 2>/dev/null; then
      echo "canvas loop already running"
    else
      nohup bash "$CANVAS_LOOP" >>/home/ext_csv/logs/td3_amo_jax_loco9_canvas_10m.log 2>&1 &
      echo "started canvas loop pid=$!"
    fi
  fi

  # CPU final eval reaper (train used --no-eval)
  EVAL_PIDFILE=/home/ext_csv/logs/td3_amo_jax_loco9_cpu_eval.pid
  if [ -f "$EVAL_PIDFILE" ] && kill -0 "$(cat "$EVAL_PIDFILE")" 2>/dev/null; then
    echo "cpu eval poller already running"
  else
    nohup "$PY" -u "$ROOT/scripts/eval_checkpoints_cpu.py" \
      --runs-root "$STORE/runs" --episodes 10 --final-repeats 5 --poll --poll-sec 120 \
      >>/home/ext_csv/logs/td3_amo_jax_loco9_cpu_eval.log 2>&1 &
    echo $! >"$EVAL_PIDFILE"
    echo "started cpu eval poller pid=$(cat "$EVAL_PIDFILE")"
  fi

  # One-shot amo_log ingest (auto_push loop will continue)
  if [ -f /home/ext_csv/amo_log/scripts/ingest_runs.py ]; then
    /home/ext_csv/miniconda3/bin/python /home/ext_csv/amo_log/scripts/ingest_runs.py --host=ext_csv \
      >>/home/ext_csv/logs/amo_jax_loco9_ingest_once.log 2>&1 || true
  fi

  echo "$(TZ=Asia/Seoul date -Is) post-cutover chain done"
} >>"$LOG" 2>&1
