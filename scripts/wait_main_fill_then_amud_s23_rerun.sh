#!/usr/bin/env bash
# After pytorch amud 2e-3 s2/s3 finish AND the main JAX 32-cell fill drains,
# launch JAX re-runs for umaze-diverse seed2/3:
#   - always: T_lr 3e-4, 1e-3
#   - also T_lr 2e-3 if torch final D4RL < 70
#
# Does not kill/replace the existing handoff waiter.
set -uo pipefail

PY=/home/choi/miniconda3/envs/amo/bin/python
AMO=/home/choi/amo
PT_OUT=/home/choi/AMO_store/td3_amo_antmaze_te1_tlr_main_fill_adaptive_critic
JAX_OUT=/home/choi/AMO_store/td3_amo_jax_antmaze_te1_tlr_fill
HANDOFF_DIR=/home/choi/AMO_store/td3_amo_jax_antmaze_te1_tlr_fill_handoff
RERUN_HANDOFF="$HANDOFF_DIR/pending_amud_s23_rerun.json"
PLAN="$HANDOFF_DIR/amud_s23_rerun_plan.json"
LOG=/home/choi/logs/td3_amo_jax_antmaze_amud_s23_rerun.log
THRESHOLD=70

mkdir -p /home/choi/logs "$HANDOFF_DIR"
exec >>"$LOG" 2>&1
log() { printf '[%s] %s\n' "$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S %Z')" "$*"; }

log "amud s2/s3 jax-rerun companion start"

# --- Phase A: wait for torch 2e-3 s2/s3 COMPLETED ---
log "waiting for pytorch amud 2e-3 s2/s3 COMPLETED"
while true; do
  ok=1
  for seed in 2 3; do
    if [[ ! -f "$PT_OUT/jobs/td3amo_te1_tb1_tlr2e-3_amud_s${seed}/COMPLETED.json" ]]; then
      ok=0
      break
    fi
  done
  [[ "$ok" -eq 1 ]] && break
  sleep 30
done
log "pytorch amud 2e-3 s2/s3 completed"

# --- Phase B: build rerun handoff ---
"$PY" - <<'PY'
import json
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

pt = Path("/home/choi/AMO_store/td3_amo_antmaze_te1_tlr_main_fill_adaptive_critic")
handoff_dir = Path("/home/choi/AMO_store/td3_amo_jax_antmaze_te1_tlr_fill_handoff")
rerun_path = handoff_dir / "pending_amud_s23_rerun.json"
plan_path = handoff_dir / "amud_s23_rerun_plan.json"
threshold = 70.0
env = "antmaze-umaze-diverse-v2"

def final_score(rid: str):
    if not (pt / "jobs" / rid / "COMPLETED.json").exists():
        return None
    runs = list((pt / "runs").glob(f"{rid}-*"))
    if not runs:
        return None
    path = runs[0] / "eval.jsonl"
    if not path.exists():
        return None
    repeats, agg, best = [], None, None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "d4rl_normalized_score" not in row:
            continue
        score = float(row["d4rl_normalized_score"])
        if row.get("final_eval_aggregate"):
            agg = score
        if row.get("final_eval_repeat") is not None:
            repeats.append(score)
        step = int(row.get("step") or row.get("t") or 0)
        if step >= 999_000:
            best = score
    if agg is not None:
        return agg
    if repeats:
        return float(st.mean(repeats))
    return best

pending = [
    {"env": env, "seed": 2, "T_lr": 3e-4, "tlr_tag": "3e-4", "reason": "torch_low_rerun"},
    {"env": env, "seed": 3, "T_lr": 3e-4, "tlr_tag": "3e-4", "reason": "torch_low_rerun"},
    {"env": env, "seed": 2, "T_lr": 1e-3, "tlr_tag": "1e-3", "reason": "torch_low_rerun"},
    {"env": env, "seed": 3, "T_lr": 1e-3, "tlr_tag": "1e-3", "reason": "torch_low_rerun"},
]
scores = {}
conditional = []
for seed in (2, 3):
    rid = f"td3amo_te1_tb1_tlr2e-3_amud_s{seed}"
    score = final_score(rid)
    scores[rid] = score
    if score is None or score < threshold:
        row = {
            "env": env,
            "seed": seed,
            "T_lr": 2e-3,
            "tlr_tag": "2e-3",
            "reason": f"torch_2e-3_below_{threshold:g}" if score is not None else "torch_2e-3_score_missing",
            "torch_score": score,
        }
        pending.append(row)
        conditional.append(row)

payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "kind": "amud_s23_jax_rerun",
    "threshold": threshold,
    "torch_2e3_scores": scores,
    "pending": pending,
}
rerun_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
plan_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps({"scores": scores, "pending_n": len(pending), "conditional_n": len(conditional)}))
PY
log "wrote rerun handoff: $RERUN_HANDOFF"
[[ -f "$PLAN" ]] && log "plan: $(cat "$PLAN" | "$PY" -c 'import sys,json; d=json.load(sys.stdin); print(d.get("torch_2e3_scores"), "n=", len(d.get("pending") or []))')"

# --- Phase C: wait until main JAX fill has started ---
log "waiting for main JAX fill to start (launcher_process.json)"
while true; do
  if [[ -f "$JAX_OUT/launcher_process.json" ]]; then
    log "main jax launcher seen"
    break
  fi
  # if handoff waiter already gone and no pytorch left, keep waiting a bit for launcher
  sleep 30
done

# --- Phase D: wait until main fill drained ---
log "waiting for main JAX fill to drain"
while true; do
  alive_launcher=0
  if [[ -f "$JAX_OUT/launcher_process.json" ]]; then
    lp=$("$PY" -c "import json; print(json.load(open('$JAX_OUT/launcher_process.json')).get('pid',''))" 2>/dev/null || true)
    if [[ -n "$lp" ]] && kill -0 "$lp" 2>/dev/null; then
      alive_launcher=1
    fi
  fi
  alive_jobs=0
  for f in "$JAX_OUT"/jobs/*/RUNNING.json; do
    [[ -f "$f" ]] || continue
    p=$("$PY" -c "import json; print(json.load(open('$f')).get('pid',''))" 2>/dev/null || true)
    if [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; then
      alive_jobs=1
      break
    fi
  done
  if [[ "$alive_launcher" -eq 0 && "$alive_jobs" -eq 0 ]]; then
    # confirm no pending left in main manifest (if present)
    pending_n=$("$PY" - <<'PY'
import json
from pathlib import Path
p = Path("/home/choi/AMO_store/td3_amo_jax_antmaze_te1_tlr_fill/status_summary.json")
if not p.exists():
    print(0)
else:
    d = json.loads(p.read_text())
    print(int((d.get("counts") or {}).get("pending") or 0))
PY
)
    if [[ "${pending_n:-0}" -eq 0 ]]; then
      break
    fi
  fi
  sleep 60
done
log "main JAX fill drained; launching amud s2/s3 reruns"

cd "$AMO" || { log "ERROR cd $AMO"; exit 1; }
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-/home/choi/.mujoco/mujoco210}"
export D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-/home/choi/.d4rl/datasets}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if ! pgrep -af "eval_checkpoints_cpu.py --runs-root $JAX_OUT" | rg -qv pgrep; then
  nohup env -u CUDA_VISIBLE_DEVICES JAX_PLATFORMS=cpu \
    MUJOCO_PY_MUJOCO_PATH="$MUJOCO_PY_MUJOCO_PATH" \
    D4RL_DATASET_DIR="$D4RL_DATASET_DIR" \
    MUJOCO_GL="$MUJOCO_GL" \
    "$PY" -u "$AMO/scripts/eval_checkpoints_cpu.py" \
    --runs-root "$JAX_OUT/runs" \
    --episodes 10 \
    --final-repeats 5 \
    --device cpu \
    --poll --poll-sec 120 \
    >>/home/choi/logs/td3_amo_jax_antmaze_cpu_eval.log 2>&1 &
  log "started jax cpu eval pid=$!"
fi

"$PY" "$AMO/scripts/launch_td3_amo_jax_antmaze_te1_tlr_fill.py" \
  --detach --gpus 0 --max-parallel 2 --max-used-mib 11000 \
  --out "$JAX_OUT" \
  --handoff "$RERUN_HANDOFF"
log "amud s2/s3 rerun launcher detached"
"$PY" /home/choi/AMO_EXP/scripts/refresh_td3_amo_antmaze_te1_tlr_canvas.py \
  >>/home/choi/logs/td3_amo_antmaze_te1_tlr_canvas.log 2>&1 || true
log "amud s2/s3 jax-rerun companion done"
