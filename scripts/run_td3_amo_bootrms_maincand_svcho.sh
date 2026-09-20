#!/usr/bin/env bash
# TD3-AMO bootrms (L2_RMS-only α_B) candidate-main sweep
# Nest: α_init 5→2→1 · seed 0→1→2→3 · env hop→walk→ant→hc · α_lr 3e-4→1e-3→2e-3
# 3×4×15×3 = 540 cells. Reuses design-ablation bootrms matches when present.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$ROOT/results/td3_amo_bootrms_maincand_seeds03}"
ABLATION_OUT="${ABLATION_OUT:-$ROOT/results/td3_amo_design_ablation_rwdnone_seeds03}"
PY="${PY:-/home/svcho/anaconda3/envs/offrl/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
PARALLEL="${PARALLEL:-3}"
NO_EVAL_NEW="${NO_EVAL_NEW:-1}"
LOGDIR="$OUT/queue_logs"
mkdir -p "$OUT/jobs" "$LOGDIR"

export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
SP="/home/svcho/anaconda3/envs/offrl/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$(echo "$SP"/nvidia/*/lib | tr ' ' ':'):/usr/local/cuda/lib64:/usr/lib/nvidia:/home/svcho/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-/home/svcho/.d4rl/datasets}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-/home/svcho/.mujoco/mujoco210}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

ALPHAS=(5 2 1)
SEEDS=(0 1 2 3)
ALRS=(3e-4 1e-3 2e-3)
ENVS=(
  hopper-medium-v2
  hopper-medium-replay-v2
  hopper-medium-expert-v2
  walker2d-medium-v2
  walker2d-medium-replay-v2
  walker2d-medium-expert-v2
  antmaze-umaze-v2
  antmaze-umaze-diverse-v2
  antmaze-medium-play-v2
  antmaze-medium-diverse-v2
  antmaze-large-play-v2
  antmaze-large-diverse-v2
  halfcheetah-medium-v2
  halfcheetah-medium-replay-v2
  halfcheetah-medium-expert-v2
)

short_env() {
  case "$1" in
    hopper-medium-v2) echo h-m ;;
    hopper-medium-replay-v2) echo h-mr ;;
    hopper-medium-expert-v2) echo h-me ;;
    walker2d-medium-v2) echo w-m ;;
    walker2d-medium-replay-v2) echo w-mr ;;
    walker2d-medium-expert-v2) echo w-me ;;
    antmaze-umaze-v2) echo am-u ;;
    antmaze-umaze-diverse-v2) echo am-ud ;;
    antmaze-medium-play-v2) echo am-mp ;;
    antmaze-medium-diverse-v2) echo am-md ;;
    antmaze-large-play-v2) echo am-lp ;;
    antmaze-large-diverse-v2) echo am-ld ;;
    halfcheetah-medium-v2) echo hc-m ;;
    halfcheetah-medium-replay-v2) echo hc-mr ;;
    halfcheetah-medium-expert-v2) echo hc-me ;;
    *) echo "${1%-v2}" | tr '-' '_' ;;
  esac
}

alr_tag() {
  case "$1" in
    3e-4|0.0003) echo r3e4 ;;
    1e-3|0.001) echo r1e3 ;;
    2e-3|0.002) echo r2e3 ;;
    *) echo "r${1}" | tr '.' 'p' ;;
  esac
}

cfg_for() {
  local a="$1" alr="$2"
  echo "$ROOT/configs/td3_amo_bootrms_a${a}_$(alr_tag "$alr").yaml"
}

make_tag() {
  local a="$1" env="$2" alr="$3" seed="$4"
  echo "a${a}_$(short_env "$env")_$(alr_tag "$alr")_s${seed}"
}

# Map (alpha_init, env, alr, seed) → ablation job tag if reusable.
ablation_src_tag() {
  local a="$1" env="$2" alr="$3" seed="$4"
  [[ "$a" == "5" ]] || return 1
  case "$env|$alr" in
    hopper-medium-replay-v2|2e-3) echo "h-mr_bootrms_rwdnone_s${seed}" ;;
    walker2d-medium-replay-v2|2e-3) echo "w-mr_bootrms_rwdnone_s${seed}" ;;
    antmaze-medium-diverse-v2|3e-4) echo "ammd_bootrms_rwdnone_s${seed}" ;;
    *) return 1 ;;
  esac
}

try_import_ablation() {
  local a="$1" env="$2" alr="$3" seed="$4"
  local src_tag dst_tag src dst
  src_tag="$(ablation_src_tag "$a" "$env" "$alr" "$seed")" || return 1
  dst_tag="$(make_tag "$a" "$env" "$alr" "$seed")"
  src="$ABLATION_OUT/jobs/$src_tag"
  dst="$OUT/jobs/$dst_tag"
  [[ -f "$src/DONE" ]] || return 1
  [[ -f "$src/run/checkpoint.npz" || -f "$src/run/checkpoints/step_1000000.npz" ]] || return 1
  if [[ -f "$dst/DONE" ]] && [[ -f "$dst/run/checkpoint.npz" || -f "$dst/run/checkpoints/step_1000000.npz" ]]; then
    return 0
  fi
  mkdir -p "$OUT/jobs"
  rm -rf "$dst"
  cp -a "$src" "$dst"
  # Keep eval.jsonl if present so poller skips.
  echo "imported_from_ablation:$src_tag" >"$dst/IMPORTED_ABLATION"
  date -Is >"$dst/DONE"
  echo "[IMPORT] $dst_tag <- $src_tag"
  return 0
}

live_pid_for_tag() {
  local tag="$1"
  local needle="/jobs/${tag}/run"
  local line pid
  while read -r line; do
    case "$line" in
      *train.py*td3_amo*"$needle"*)
        pid="${line%% *}"
        if kill -0 "$pid" 2>/dev/null; then
          echo "$pid"
          return 0
        fi
        ;;
    esac
  done < <(ps -eo pid=,args=)
  echo 0
}

QUEUE_LOG="$LOGDIR/queue_bootrms_p${PARALLEL}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$QUEUE_LOG") 2>&1
echo "=== TD3-AMO bootrms maincand nest α→seed→env→α_lr PARALLEL=$PARALLEL start $(date -Is) ==="
echo "OUT=$OUT ABLATION_OUT=$ABLATION_OUT"
echo "alphas=${ALPHAS[*]} seeds=${SEEDS[*]} alrs=${ALRS[*]} envs=${#ENVS[@]}"
echo "cells=$(( ${#ALPHAS[@]} * ${#SEEDS[@]} * ${#ENVS[@]} * ${#ALRS[@]} ))"
rm -f "$OUT/QUEUE_COMPLETE"

declare -A PID_OF=()

is_done() {
  local tag="$1"
  local cell="$OUT/jobs/$tag"
  [[ -f "$cell/DONE" ]] && [[ -f "$cell/run/checkpoint.npz" || -f "$cell/run/checkpoints/step_1000000.npz" ]]
}

# Keep final ckpt only. Intermediate step_*.npz filled the disk (Error 32).
prune_mid_ckpts() {
  local run_dir="$1"
  local d="$run_dir/checkpoints" f
  [[ -d "$d" ]] || return 0
  for f in "$d"/step_*.npz "$d"/step_*.npz.tmp "$run_dir"/*.tmp; do
    [[ -f "$f" ]] || continue
    [[ "$(basename "$f")" == "step_1000000.npz" ]] && continue
    rm -f "$f"
  done
}

start_job() {
  local a="$1" env="$2" alr="$3" seed="$4"
  local tag cfg cell run_dir
  tag="$(make_tag "$a" "$env" "$alr" "$seed")"
  cfg="$(cfg_for "$a" "$alr")"
  cell="$OUT/jobs/$tag"
  run_dir="$cell/run"
  mkdir -p "$cell"

  if try_import_ablation "$a" "$env" "$alr" "$seed"; then
    return 1
  fi

  if [[ ! -f "$cfg" ]]; then
    echo "[FAIL] missing config $cfg"
    echo "{\"rc\":1,\"err\":\"missing_cfg\"}" >"$cell/FAILED"
    return 1
  fi

  if is_done "$tag"; then
    echo "[SKIP] $tag"
    return 1
  fi

  if [[ -f "$run_dir/metrics.jsonl" ]] && [[ -f "$run_dir/checkpoint.npz" ]] && [[ ! -f "$cell/DONE" ]]; then
    local last_step
    last_step="$("$PY" - <<PY
import json
from pathlib import Path
step=0
p=Path("$run_dir/metrics.jsonl")
for line in p.read_text().splitlines():
    if line.strip():
        step=max(step, int(json.loads(line).get("step",0)))
print(step)
PY
)"
    if [[ "$last_step" -ge 1000000 ]]; then
      date -Is >"$cell/DONE"
      rm -f "$cell/FAILED"
      prune_mid_ckpts "$run_dir"
      echo "[DONE] $tag recovered step=$last_step"
      return 1
    fi
  fi

  local resume_args=()
  if [[ -f "$run_dir/checkpoint.npz" ]] && [[ ! -f "$cell/DONE" ]]; then
    resume_args=(--resume "$run_dir/checkpoint.npz")
    echo "[RESUME] $tag"
  else
    rm -rf "$run_dir"
    mkdir -p "$run_dir"
  fi

  echo "[RUN] $tag a=$a env=$env alr=$alr seed=$seed cfg=$(basename "$cfg") $(date -Is)"
  set +e
  setsid "$PY" "$ROOT/train.py" \
    --algorithm td3_amo --backend jax \
    --env "$env" --config "$cfg" --seed "$seed" \
    --device "$DEVICE" --output "$run_dir" \
    --log-every 5000 --save-every 20000 --no-eval \
    "${resume_args[@]}" \
    >"$cell/stdout_stderr.log" 2>&1 &
  local pid=$!
  set +e
  PID_OF["$tag"]="$pid"
  echo "$pid" >"$cell/PID"
  echo "[SPAWN] $tag pid=$pid"
  return 0
}

reap_finished() {
  local tag pid cell run_dir rc steps
  for tag in "${!PID_OF[@]}"; do
    pid="${PID_OF[$tag]}"
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    cell="$OUT/jobs/$tag"
    run_dir="$cell/run"
    wait "$pid" 2>/dev/null
    rc=$?
    steps=0
    if [[ -f "$run_dir/checkpoint.npz" ]]; then
      steps="$("$PY" -c "import json,numpy as np; z=np.load('$run_dir/checkpoint.npz',allow_pickle=False); print(int(json.loads(str(z['__metadata__']))['steps']))" 2>/dev/null || echo 0)"
    fi
    if [[ -f "$run_dir/checkpoint.npz" ]] && [[ -f "$run_dir/run_meta.json" ]]; then
      if [[ $rc -eq 0 ]] || [[ "$steps" -ge 1000000 ]] || grep -q 'Completed .* steps' "$cell/stdout_stderr.log" 2>/dev/null; then
        date -Is >"$cell/DONE"
        rm -f "$cell/FAILED" "$cell/PID"
        prune_mid_ckpts "$run_dir"
        echo "[DONE] $tag rc=$rc steps=$steps"
      else
        echo "{\"rc\":$rc,\"at\":\"$(date -Is)\"}" >"$cell/FAILED"
        echo "[FAIL] $tag rc=$rc"
        tail -8 "$cell/stdout_stderr.log" || true
      fi
    else
      echo "{\"rc\":$rc,\"at\":\"$(date -Is)\"}" >"$cell/FAILED"
      echo "[FAIL] $tag no ckpt"
      tail -8 "$cell/stdout_stderr.log" || true
    fi
    unset "PID_OF[$tag]"
  done
}

active_count() {
  local n=0 tag
  for tag in "${!PID_OF[@]}"; do
    if kill -0 "${PID_OF[$tag]}" 2>/dev/null; then
      n=$((n + 1))
    fi
  done
  echo "$n"
}

queue_left() {
  local a seed env alr tag left=0
  for a in "${ALPHAS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for env in "${ENVS[@]}"; do
        for alr in "${ALRS[@]}"; do
          tag="$(make_tag "$a" "$env" "$alr" "$seed")"
          is_done "$tag" && continue
          if try_import_ablation "$a" "$env" "$alr" "$seed"; then
            continue
          fi
          left=$((left + 1))
        done
      done
    done
  done
  echo "$left"
}

# Next nest-order cell that is not DONE / live. Crosses env/seed/alpha so P stays 2.
start_next_cell() {
  local a seed env alr tag live
  for a in "${ALPHAS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for env in "${ENVS[@]}"; do
        for alr in "${ALRS[@]}"; do
          tag="$(make_tag "$a" "$env" "$alr" "$seed")"
          is_done "$tag" && continue
          [[ -f "$OUT/jobs/$tag/FAILED" ]] && continue
          if [[ -n "${PID_OF[$tag]:-}" ]] && kill -0 "${PID_OF[$tag]}" 2>/dev/null; then
            continue
          fi
          live="$(live_pid_for_tag "$tag")"
          if [[ "$live" != 0 ]]; then
            PID_OF["$tag"]="$live"
            echo "[ADOPT] $tag pid=$live"
            continue
          fi
          if start_job "$a" "$env" "$alr" "$seed"; then
            return 0
          fi
        done
      done
    done
  done
  return 1
}

# Prefetch imports for known ablation overlaps
for seed in "${SEEDS[@]}"; do
  try_import_ablation 5 hopper-medium-replay-v2 2e-3 "$seed" || true
  try_import_ablation 5 walker2d-medium-replay-v2 2e-3 "$seed" || true
  try_import_ablation 5 antmaze-medium-diverse-v2 3e-4 "$seed" || true
done

for a in "${ALPHAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for env in "${ENVS[@]}"; do
      for alr in "${ALRS[@]}"; do
        tag="$(make_tag "$a" "$env" "$alr" "$seed")"
        pid="$(live_pid_for_tag "$tag")"
        if [[ "$pid" != 0 ]]; then
          PID_OF["$tag"]="$pid"
          echo "[ADOPT] $tag pid=$pid"
        fi
      done
    done
  done
done

echo "PARALLEL=$PARALLEL (fill across env/seed/alpha; nest order preserved)"
while [[ "$(queue_left)" -gt 0 ]]; do
  reap_finished
  for f in "$OUT"/jobs/*/FAILED; do
    [[ -f "$f" ]] || continue
    tag="$(basename "$(dirname "$f")")"
    if ! is_done "$tag"; then
      rm -f "$f"
      echo "[RETRY] $tag"
    fi
  done
  while [[ "$(active_count)" -lt "$PARALLEL" ]]; do
    reap_finished
    if start_next_cell; then
      sleep 2
      continue
    fi
    break
  done
  echo "[STATUS] active=$(active_count) left=$(queue_left) $(date -Is)"
  [[ "$(queue_left)" -eq 0 ]] && break
  sleep 30
done

reap_finished
fail=0
total=0
for a in "${ALPHAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for env in "${ENVS[@]}"; do
      for alr in "${ALRS[@]}"; do
        total=$((total + 1))
        tag="$(make_tag "$a" "$env" "$alr" "$seed")"
        if [[ -f "$OUT/jobs/$tag/FAILED" ]] || ! is_done "$tag"; then
          fail=$((fail + 1))
        fi
      done
    done
  done
done
echo "=== BOOTRMS MAINCAND QUEUE COMPLETE $(date -Is) total=$total fail=$fail ==="
echo QUEUE_COMPLETE >"$OUT/QUEUE_COMPLETE"
[[ "$fail" -eq 0 ]]
