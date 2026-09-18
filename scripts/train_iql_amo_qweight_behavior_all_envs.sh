#!/usr/bin/env bash
# Train per-env frozen μ_hat BC (50k) for iql_amo_qweight loco+antmaze grid.
set -uo pipefail
ROOT=/home/ext_csv/AMO-main
PY=/home/ext_csv/miniconda3/envs/amo-jax/bin/python
OUT_ROOT=/raid/ext_csv/AMO_store/iql_amo_qweight_behavior_shared
DS=/raid/ext_csv/datasets/d4rl
export PYTHONPATH="$ROOT"
export WANDB_MODE=offline D4RL_SUPPRESS_IMPORT_ERROR=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.15
export LD_LIBRARY_PATH=/home/ext_csv/.mujoco/mujoco210/bin:/home/ext_csv/miniconda3/envs/amo-jax/lib:${LD_LIBRARY_PATH:-}

ENVS=(
  hopper-medium-v2
  hopper-medium-replay-v2
  hopper-medium-expert-v2
  walker2d-medium-v2
  walker2d-medium-replay-v2
  walker2d-medium-expert-v2
  halfcheetah-medium-v2
  halfcheetah-medium-replay-v2
  halfcheetah-medium-expert-v2
  antmaze-umaze-v2
  antmaze-umaze-diverse-v2
  antmaze-medium-play-v2
  antmaze-medium-diverse-v2
  antmaze-large-play-v2
  antmaze-large-diverse-v2
)

short() {
  echo "$1" | sed -e 's/halfcheetah/hc/' -e 's/hopper/h/' -e 's/walker2d/w/' \
    -e 's/antmaze-/am-/' -e 's/-diverse/-div/' -e 's/-medium-replay/-mr/' \
    -e 's/-medium-expert/-me/' -e 's/-medium/-m/' -e 's/-large/-l/' \
    -e 's/-umaze/-u/' -e 's/-play/-p/' -e 's/-v2//'
}

dataset_for() {
  local e=$1
  case "$e" in
    antmaze-umaze-v2) echo "$DS/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5" ;;
    antmaze-umaze-diverse-v2) echo "$DS/Ant_maze_u-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5" ;;
    antmaze-medium-play-v2) echo "$DS/Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5" ;;
    antmaze-medium-diverse-v2) echo "$DS/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5" ;;
    antmaze-large-play-v2) echo "$DS/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5" ;;
    antmaze-large-diverse-v2) echo "$DS/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5" ;;
    *) echo "$DS/${e//-/_}.hdf5" | sed 's/_v2/-v2/' ;;
  esac
}

GPU="${1:-0}"
mkdir -p "$OUT_ROOT" /home/ext_csv/logs
LOG=/home/ext_csv/logs/iql_amo_qweight_behavior_bc_all.log
echo "==== $(TZ=Asia/Seoul date -Is) behavior BC all start gpu=$GPU ====" | tee -a "$LOG"
for env in "${ENVS[@]}"; do
  tag=$(short "$env")
  out="$OUT_ROOT/$tag"
  if [[ -f "$out/behavior.npz" ]]; then
    echo "skip $env ($tag)" | tee -a "$LOG"
    continue
  fi
  ds=$(dataset_for "$env")
  echo "==== BC $env -> $out ====" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/scripts/train_iql_amo_qweight_behavior.py" \
    --env "$env" --dataset "$ds" --backend jax --device cuda:0 \
    --steps 50000 --seed 0 --bc-seed 4242 --output "$out" \
    >>"$LOG" 2>&1 || { echo "FAIL $env" | tee -a "$LOG"; exit 1; }
done
echo "==== $(TZ=Asia/Seoul date -Is) behavior BC all done ====" | tee -a "$LOG"
