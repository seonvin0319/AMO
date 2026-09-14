#!/usr/bin/env bash
# Parallel final50_singlepass_v1 over beta1 manifest. Never deletes checkpoints.
set -u
WT="${WT:-/home/svcho/amo_iql_sweep_log_audit_20260914}"
PY="${PY:-/home/svcho/anaconda3/envs/fql_gpu310/bin/python}"
MANIFEST="${MANIFEST:-/home/svcho/amo_iql_sweep/results/iql_amo_lr1e3_beta_sweep_seed0/audit_20260914/manifest_beta1_final50.json}"
PARALLEL="${PARALLEL:-2}"
LOGDIR="${LOGDIR:-/home/svcho/amo_iql_sweep/results/iql_amo_lr1e3_beta_sweep_seed0/audit_20260914/final50_logs}"
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/final50_parallel_${PARALLEL}_${STAMP}.log"

export D4RL_SUPPRESS_IMPORT_ERROR=1
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=
export MUJOCO_GL=egl
export D4RL_DATASET_DIR=/home/svcho/.d4rl/datasets
export MUJOCO_PY_MUJOCO_PATH=/home/svcho/.mujoco/mujoco210
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

echo "[final50] start $(date --iso-8601=seconds) PARALLEL=$PARALLEL" | tee -a "$LOG"
echo "[final50] WT=$WT MANIFEST=$MANIFEST" | tee -a "$LOG"

# Build index of run dirs
mapfile -t RUNS < <("$PY" - <<PY
import json
from pathlib import Path
m=json.loads(Path("$MANIFEST").read_text())
for r in m["runs"]:
    print(r["run_dir"])
PY
)

N=${#RUNS[@]}
echo "[final50] n_runs=$N" | tee -a "$LOG"

running=0
fail=0
done_n=0
skip_n=0
i=0
for run in "${RUNS[@]}"; do
  while [ "$running" -ge "$PARALLEL" ]; do
    if wait -n; then
      :
    else
      fail=$((fail+1))
    fi
    running=$((running-1))
  done
  (
    "$PY" -u "$WT/scripts/eval_final50_singlepass_v1.py" --run-dir "$run" >>"$LOG" 2>&1
  ) &
  running=$((running+1))
  i=$((i+1))
  echo "[final50] spawned $i/$N $(basename "$(dirname "$run")") running=$running" | tee -a "$LOG"
done
while [ "$running" -gt 0 ]; do
  if wait -n; then :; else fail=$((fail+1)); fi
  running=$((running-1))
done

# Summarize markers
"$PY" - <<PY | tee -a "$LOG"
import json
from pathlib import Path
m=json.loads(Path("$MANIFEST").read_text())
done=skip=fail=0
missing=[]
rows=[]
for r in m["runs"]:
    run=Path(r["run_dir"])
    mk=run/"FINAL50_SINGLEPASS_V1_DONE.json"
    if mk.exists():
        payload=json.loads(mk.read_text())
        if payload.get("ok"):
            done+=1
            rows.append((r["env"], r["seed"], payload["row"]["normalized_score"]))
        else:
            fail+=1
            missing.append(r["job_name"])
    elif (run/"FINAL50_SINGLEPASS_V1_DONE.json.FAILED.json").exists():
        fail+=1
        missing.append(r["job_name"])
    else:
        missing.append(r["job_name"])
print(json.dumps({"done":done,"fail_or_missing":len(missing),"missing":missing[:20], "n_scores":len(rows)}, indent=2))
PY

echo "[final50] COMPLETE $(date --iso-8601=seconds) log=$LOG" | tee -a "$LOG"
