#!/usr/bin/env bash
# Combined queue. Live BootRMS trainers are adopted, not killed (Error 1).
# α=1 mix in P=3: remaining BootRMS + TD3 π_E-only + IQL π_E, round-robin.
# Then TD3/IQL 5, then both 2.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TD3_OUT="${TD3_OUT:-$ROOT/results/td3_amo_execonly_main_seeds03}"
IQL_OUT="${IQL_OUT:-$ROOT/results/iql_amo_lel2_seeds03}"
BOOT_OUT="${BOOT_OUT:-$ROOT/results/td3_amo_bootrms_maincand_seeds03}"
PY="${PY:-/home/svcho/anaconda3/envs/offrl/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
PARALLEL="${PARALLEL:-3}"
LOGDIR="$TD3_OUT/queue_logs"
mkdir -p "$TD3_OUT/jobs" "$IQL_OUT/jobs" "$BOOT_OUT/jobs" "$LOGDIR" "$IQL_OUT/queue_logs" "$BOOT_OUT/queue_logs"

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

# Start from 1 now. Mix remaining BootRMS α=1 with new α/β=1 in the same P=3.
A1_MIX=(td3 iql bootrms)
LATER_WAVES=(
  "td3 5"
  "iql 5"
  "td3 2"
  "iql 2"
)
A1_RR=0
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

out_for() {
  case "$1" in
    td3) echo "$TD3_OUT" ;;
    iql) echo "$IQL_OUT" ;;
    bootrms) echo "$BOOT_OUT" ;;
    *) return 1 ;;
  esac
}

algo_for() {
  case "$1" in
    td3|bootrms) echo td3_amo ;;
    iql) echo iql_amo ;;
    *) return 1 ;;
  esac
}

cfg_for() {
  local kind="$1" init="$2" alr="$3"
  local rt
  rt="$(alr_tag "$alr")"
  case "$kind" in
    td3) echo "$ROOT/configs/td3_amo_execonly_a${init}_${rt}.yaml" ;;
    iql) echo "$ROOT/configs/iql_amo_lel2_b${init}_${rt}.yaml" ;;
    bootrms) echo "$ROOT/configs/td3_amo_bootrms_a${init}_${rt}.yaml" ;;
    *) return 1 ;;
  esac
}

make_tag() {
  local kind="$1" init="$2" env="$3" alr="$4" seed="$5"
  case "$kind" in
    td3|bootrms) echo "a${init}_$(short_env "$env")_$(alr_tag "$alr")_s${seed}" ;;
    iql) echo "b${init}_$(short_env "$env")_$(alr_tag "$alr")_s${seed}" ;;
    *) return 1 ;;
  esac
}

cell_key() {
  echo "$1:$2"
}

live_pid_for() {
  local kind="$1" tag="$2"
  local algo out needle line pid
  algo="$(algo_for "$kind")"
  out="$(out_for "$kind")"
  needle="${out}/jobs/${tag}/run"
  while read -r line; do
    case "$line" in
      *train.py*"$algo"*"$needle"*)
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

QUEUE_LOG="$LOGDIR/queue_pel2_p${PARALLEL}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$QUEUE_LOG") 2>&1
date -Is >"$TD3_OUT/EXECONLY_MAIN_LAUNCHED"
date -Is >"$TD3_OUT/PEL2_LAUNCHED"
date -Is >"$IQL_OUT/PEL2_LAUNCHED"
date -Is >"$BOOT_OUT/PEL2_MIXED"
echo "=== PEL2 start-from-1 mix BootRMS α=1 + TD3/IQL 1, then 5, then 2  PARALLEL=$PARALLEL $(date -Is) ==="
echo "TD3_OUT=$TD3_OUT"
echo "IQL_OUT=$IQL_OUT"
echo "BOOT_OUT=$BOOT_OUT"
echo "a1_mix=${A1_MIX[*]} later=${LATER_WAVES[*]} seeds=${SEEDS[*]}"
rm -f "$TD3_OUT/QUEUE_COMPLETE" "$IQL_OUT/QUEUE_COMPLETE"

declare -A PID_OF=()
declare -A KIND_OF=()

is_done() {
  local kind="$1" tag="$2"
  local cell
  cell="$(out_for "$kind")/jobs/$tag"
  [[ -f "$cell/DONE" ]] && [[ -f "$cell/run/checkpoint.npz" || -f "$cell/run/checkpoints/step_1000000.npz" ]]
}

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
  local kind="$1" init="$2" env="$3" alr="$4" seed="$5"
  local tag cfg cell run_dir algo key
  tag="$(make_tag "$kind" "$init" "$env" "$alr" "$seed")"
  cfg="$(cfg_for "$kind" "$init" "$alr")"
  cell="$(out_for "$kind")/jobs/$tag"
  run_dir="$cell/run"
  algo="$(algo_for "$kind")"
  key="$(cell_key "$kind" "$tag")"
  mkdir -p "$cell"

  if [[ ! -f "$cfg" ]]; then
    echo "[FAIL] missing config $cfg"
    echo "{\"rc\":1,\"err\":\"missing_cfg\"}" >"$cell/FAILED"
    return 1
  fi
  if is_done "$kind" "$tag"; then
    echo "[SKIP] $kind $tag"
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
      echo "[DONE] $kind $tag recovered step=$last_step"
      return 1
    fi
  fi

  local resume_args=()
  if [[ -f "$run_dir/checkpoint.npz" ]] && [[ ! -f "$cell/DONE" ]]; then
    resume_args=(--resume "$run_dir/checkpoint.npz")
    echo "[RESUME] $kind $tag"
  else
    rm -rf "$run_dir"
    mkdir -p "$run_dir"
  fi

  echo "[RUN] $kind $tag init=$init env=$env alr=$alr seed=$seed cfg=$(basename "$cfg") $(date -Is)"
  set +e
  setsid "$PY" "$ROOT/train.py" \
    --algorithm "$algo" --backend jax \
    --env "$env" --config "$cfg" --seed "$seed" \
    --device "$DEVICE" --output "$run_dir" \
    --log-every 5000 --save-every 20000 --no-eval \
    "${resume_args[@]}" \
    >"$cell/stdout_stderr.log" 2>&1 &
  local pid=$!
  set +e
  PID_OF["$key"]="$pid"
  KIND_OF["$key"]="$kind"
  echo "$pid" >"$cell/PID"
  echo "[SPAWN] $kind $tag pid=$pid"
  return 0
}

reap_finished() {
  local key pid kind tag cell run_dir rc steps
  for key in "${!PID_OF[@]}"; do
    pid="${PID_OF[$key]}"
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    kind="${KIND_OF[$key]}"
    tag="${key#*:}"
    cell="$(out_for "$kind")/jobs/$tag"
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
        echo "[DONE] $kind $tag rc=$rc steps=$steps"
      else
        echo "{\"rc\":$rc,\"at\":\"$(date -Is)\"}" >"$cell/FAILED"
        echo "[FAIL] $kind $tag rc=$rc"
        tail -8 "$cell/stdout_stderr.log" || true
      fi
    else
      echo "{\"rc\":$rc,\"at\":\"$(date -Is)\"}" >"$cell/FAILED"
      echo "[FAIL] $kind $tag no ckpt"
      tail -8 "$cell/stdout_stderr.log" || true
    fi
    unset "PID_OF[$key]"
    unset "KIND_OF[$key]"
  done
}

active_count() {
  local n=0 key
  for key in "${!PID_OF[@]}"; do
    if kill -0 "${PID_OF[$key]}" 2>/dev/null; then
      n=$((n + 1))
    fi
  done
  echo "$n"
}

queue_left() {
  local kind init seed env alr tag left=0 wave
  for kind in "${A1_MIX[@]}"; do
    init=1
    for seed in "${SEEDS[@]}"; do
      for env in "${ENVS[@]}"; do
        for alr in "${ALRS[@]}"; do
          tag="$(make_tag "$kind" "$init" "$env" "$alr" "$seed")"
          is_done "$kind" "$tag" && continue
          left=$((left + 1))
        done
      done
    done
  done
  for wave in "${LATER_WAVES[@]}"; do
    kind="${wave%% *}"
    init="${wave##* }"
    for seed in "${SEEDS[@]}"; do
      for env in "${ENVS[@]}"; do
        for alr in "${ALRS[@]}"; do
          tag="$(make_tag "$kind" "$init" "$env" "$alr" "$seed")"
          is_done "$kind" "$tag" && continue
          left=$((left + 1))
        done
      done
    done
  done
  echo "$left"
}

try_start_kind_init() {
  local kind="$1" init="$2"
  local seed env alr tag live key
  for seed in "${SEEDS[@]}"; do
    for env in "${ENVS[@]}"; do
      for alr in "${ALRS[@]}"; do
        tag="$(make_tag "$kind" "$init" "$env" "$alr" "$seed")"
        key="$(cell_key "$kind" "$tag")"
        is_done "$kind" "$tag" && continue
        [[ -f "$(out_for "$kind")/jobs/$tag/FAILED" ]] && continue
        if [[ -n "${PID_OF[$key]:-}" ]] && kill -0 "${PID_OF[$key]}" 2>/dev/null; then
          continue
        fi
        live="$(live_pid_for "$kind" "$tag")"
        if [[ "$live" != 0 ]]; then
          PID_OF["$key"]="$live"
          KIND_OF["$key"]="$kind"
          echo "[ADOPT] $kind $tag pid=$live"
          continue
        fi
        if start_job "$kind" "$init" "$env" "$alr" "$seed"; then
          return 0
        fi
      done
    done
  done
  return 1
}

start_next_cell() {
  local i kind wave init
  for i in 0 1 2; do
    kind="${A1_MIX[$(( (A1_RR + i) % 3 ))]}"
    if try_start_kind_init "$kind" 1; then
      A1_RR=$(( (A1_RR + i + 1) % 3 ))
      return 0
    fi
  done
  for wave in "${LATER_WAVES[@]}"; do
    kind="${wave%% *}"
    init="${wave##* }"
    if try_start_kind_init "$kind" "$init"; then
      return 0
    fi
  done
  return 1
}

adopt_kind_init() {
  local kind="$1" init="$2"
  local seed env alr tag key pid
  for seed in "${SEEDS[@]}"; do
    for env in "${ENVS[@]}"; do
      for alr in "${ALRS[@]}"; do
        tag="$(make_tag "$kind" "$init" "$env" "$alr" "$seed")"
        key="$(cell_key "$kind" "$tag")"
        pid="$(live_pid_for "$kind" "$tag")"
        if [[ "$pid" != 0 ]]; then
          PID_OF["$key"]="$pid"
          KIND_OF["$key"]="$kind"
          echo "[ADOPT] $kind $tag pid=$pid"
        fi
      done
    done
  done
}

for kind in "${A1_MIX[@]}"; do
  adopt_kind_init "$kind" 1
done
for wave in "${LATER_WAVES[@]}"; do
  adopt_kind_init "${wave%% *}" "${wave##* }"
done

echo "PARALLEL=$PARALLEL (α=1 mix td3/iql/bootrms, then 5, then 2)"
while [[ "$(queue_left)" -gt 0 ]]; do
  reap_finished
  for out_kind in td3 iql bootrms; do
    local_out="$(out_for "$out_kind")"
    for f in "$local_out"/jobs/*/FAILED; do
      [[ -f "$f" ]] || continue
      tag="$(basename "$(dirname "$f")")"
      if ! is_done "$out_kind" "$tag"; then
        rm -f "$f"
        echo "[RETRY] $out_kind $tag"
      fi
    done
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
count_wave() {
  local kind="$1" init="$2" seed env alr tag
  for seed in "${SEEDS[@]}"; do
    for env in "${ENVS[@]}"; do
      for alr in "${ALRS[@]}"; do
        total=$((total + 1))
        tag="$(make_tag "$kind" "$init" "$env" "$alr" "$seed")"
        if [[ -f "$(out_for "$kind")/jobs/$tag/FAILED" ]] || ! is_done "$kind" "$tag"; then
          fail=$((fail + 1))
        fi
      done
    done
  done
}
for kind in "${A1_MIX[@]}"; do
  count_wave "$kind" 1
done
for wave in "${LATER_WAVES[@]}"; do
  count_wave "${wave%% *}" "${wave##* }"
done
echo "=== PEL2 QUEUE COMPLETE $(date -Is) total=$total fail=$fail ==="
echo QUEUE_COMPLETE >"$TD3_OUT/QUEUE_COMPLETE"
echo QUEUE_COMPLETE >"$IQL_OUT/QUEUE_COMPLETE"
echo QUEUE_COMPLETE >"$BOOT_OUT/QUEUE_COMPLETE"
[[ "$fail" -eq 0 ]]
