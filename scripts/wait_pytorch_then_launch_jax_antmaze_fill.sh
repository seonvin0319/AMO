#!/usr/bin/env bash
# Wait for the last 2 pytorch fill trainers, then start JAX fill + CPU eval + canvas.
#
# After pytorch amud T_lr=2e-3 s2/s3 finish, also enqueue JAX re-runs for
# umaze-diverse seed2/3 @ 3e-4 and 1e-3 (always), and @ 2e-3 if final < 70.
set -uo pipefail

PY=/home/choi/miniconda3/envs/amo/bin/python
AMO=/home/choi/amo
PT_OUT=/home/choi/AMO_store/td3_amo_antmaze_te1_tlr_main_fill_adaptive_critic
JAX_OUT=/home/choi/AMO_store/td3_amo_jax_antmaze_te1_tlr_fill
HANDOFF_DIR=/home/choi/AMO_store/td3_amo_jax_antmaze_te1_tlr_fill_handoff
HANDOFF="$HANDOFF_DIR/pending_cells.json"
RERUN_PLAN="$HANDOFF_DIR/amud_s23_rerun_plan.json"
LOG=/home/choi/logs/td3_amo_jax_antmaze_te1_tlr_handoff.log
PIDS_FILE="$HANDOFF_DIR/wait_pids.txt"
SCORE_THRESHOLD=70

mkdir -p /home/choi/logs "$HANDOFF_DIR"
exec >>"$LOG" 2>&1

log() { printf '[%s] %s\n' "$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S %Z')" "$*"; }

collect_pids() {
  local pids=()
  if [[ -f "$PIDS_FILE" ]]; then
    mapfile -t pids <"$PIDS_FILE"
  fi
  if [[ ${#pids[@]} -eq 0 ]]; then
    for f in "$PT_OUT"/jobs/*/RUNNING.json; do
      [[ -f "$f" ]] || continue
      p=$("$PY" -c "import json; print(json.load(open('$f')).get('pid',''))")
      [[ -n "$p" ]] && pids+=("$p")
    done
  fi
  printf '%s\n' "${pids[@]}"
}

enqueue_amud_reruns() {
  "$PY" - <<'PY'
import json
import statistics as st
from pathlib import Path

pt = Path("/home/choi/AMO_store/td3_amo_antmaze_te1_tlr_main_fill_adaptive_critic")
handoff_path = Path(
    "/home/choi/AMO_store/td3_amo_jax_antmaze_te1_tlr_fill_handoff/pending_cells.json"
)
plan_path = Path(
    "/home/choi/AMO_store/td3_amo_jax_antmaze_te1_tlr_fill_handoff/amud_s23_rerun_plan.json"
)
threshold = 70.0
env = "antmaze-umaze-diverse-v2"

def tlr_tag(v: float) -> str:
    for tag, target in (("3e-4", 3e-4), ("1e-3", 1e-3), ("2e-3", 2e-3)):
        if abs(float(v) - target) < 1e-12:
            return tag
    return f"{v:g}"

def final_score(rid: str) -> float | None:
    marker = pt / "jobs" / rid / "COMPLETED.json"
    if not marker.exists():
        return None
    runs = list((pt / "runs").glob(f"{rid}-*"))
    if not runs:
        return None
    eval_path = runs[0] / "eval.jsonl"
    if not eval_path.exists():
        return None
    repeats = []
    agg = None
    for line in eval_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("final_eval_aggregate") and "d4rl_normalized_score" in row:
            agg = float(row["d4rl_normalized_score"])
        if row.get("final_eval_repeat") is not None and "d4rl_normalized_score" in row:
            repeats.append(float(row["d4rl_normalized_score"]))
    if agg is not None:
        return agg
    if repeats:
        return float(st.mean(repeats))
    # last high-step eval fallback
    best = None
    for line in eval_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "d4rl_normalized_score" not in row:
            continue
        step = int(row.get("step") or row.get("t") or 0)
        if step < 999_000:
            continue
        best = float(row["d4rl_normalized_score"])
    return best

always = [
    {"env": env, "seed": 2, "T_lr": 3e-4, "tlr_tag": "3e-4", "reason": "torch_amud_s2_s3_low"},
    {"env": env, "seed": 3, "T_lr": 3e-4, "tlr_tag": "3e-4", "reason": "torch_amud_s2_s3_low"},
    {"env": env, "seed": 2, "T_lr": 1e-3, "tlr_tag": "1e-3", "reason": "torch_amud_s2_s3_low"},
    {"env": env, "seed": 3, "T_lr": 1e-3, "tlr_tag": "1e-3", "reason": "torch_amud_s2_s3_low"},
]
conditional = []
scores = {}
for seed in (2, 3):
    rid = f"td3amo_te1_tb1_tlr2e-3_amud_s{seed}"
    score = final_score(rid)
    scores[rid] = score
    if score is None:
        conditional.append(
            {
                "env": env,
                "seed": seed,
                "T_lr": 2e-3,
                "tlr_tag": "2e-3",
                "reason": "torch_2e-3_score_missing_rerun",
                "torch_score": None,
            }
        )
    elif score < threshold:
        conditional.append(
            {
                "env": env,
                "seed": seed,
                "T_lr": 2e-3,
                "tlr_tag": "2e-3",
                "reason": f"torch_2e-3_below_{threshold:g}",
                "torch_score": score,
            }
        )

rerun = always + conditional
payload = json.loads(handoff_path.read_text()) if handoff_path.exists() else {"pending": []}
pending = list(payload.get("pending") or [])

def key(row):
    return (row["env"], row.get("tlr_tag") or tlr_tag(row["T_lr"]), int(row["seed"]))

existing = {key(r) for r in pending}
added = []
for row in rerun:
    k = key(row)
    if k in existing:
        continue
    pending.append(
        {
            "env": row["env"],
            "seed": row["seed"],
            "T_lr": row["T_lr"],
            "tlr_tag": row["tlr_tag"],
            "rerun": True,
            "reason": row["reason"],
            **({"torch_score": row["torch_score"]} if "torch_score" in row else {}),
        }
    )
    existing.add(k)
    added.append(row)

payload["pending"] = pending
payload["amud_s23_rerun"] = {
    "threshold": threshold,
    "torch_2e3_scores": scores,
    "always": always,
    "conditional": conditional,
    "added_now": added,
}
handoff_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
plan_path.write_text(
    json.dumps(
        {
            "threshold": threshold,
            "torch_2e3_scores": scores,
            "rerun_cells": rerun,
            "pending_total": len(pending),
            "added": added,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
print(
    json.dumps(
        {
            "torch_2e3_scores": scores,
            "rerun_n": len(rerun),
            "added_n": len(added),
            "pending_total": len(pending),
        }
    )
)
PY
}

log "handoff waiter start (with amud s2/s3 jax rerun enqueue)"
PIDS=($(collect_pids))
log "watching pids: ${PIDS[*]:-none}"
printf '%s\n' "${PIDS[@]}" >"$PIDS_FILE"

while true; do
  alive=0
  for p in "${PIDS[@]}"; do
    if [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; then
      alive=1
      break
    fi
  done
  if [[ "$alive" -eq 0 ]]; then
    fresh=()
    for f in "$PT_OUT"/jobs/*/RUNNING.json; do
      [[ -f "$f" ]] || continue
      p=$("$PY" -c "import json; print(json.load(open('$f')).get('pid',''))" 2>/dev/null || true)
      if [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; then
        fresh+=("$p")
      fi
    done
    if [[ ${#fresh[@]} -gt 0 ]]; then
      PIDS=("${fresh[@]}")
      printf '%s\n' "${PIDS[@]}" >"$PIDS_FILE"
      log "updated watch pids: ${PIDS[*]}"
      sleep 30
      continue
    fi
    break
  fi
  sleep 30
done

log "pytorch trainers finished; waiting briefly for COMPLETED/final eval markers"
for _ in $(seq 1 60); do
  ok=1
  for seed in 2 3; do
    if [[ ! -f "$PT_OUT/jobs/td3amo_te1_tb1_tlr2e-3_amud_s${seed}/COMPLETED.json" ]]; then
      ok=0
      break
    fi
  done
  [[ "$ok" -eq 1 ]] && break
  sleep 10
done

log "enqueue amud s2/s3 jax reruns (3e-4/1e-3 always; 2e-3 if <${SCORE_THRESHOLD})"
enqueue_amud_reruns | while read -r line; do log "rerun plan: $line"; done
[[ -f "$RERUN_PLAN" ]] && log "wrote $RERUN_PLAN"

log "starting JAX fill"
cd "$AMO" || { log "ERROR cd $AMO"; exit 1; }

export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-/home/choi/.mujoco/mujoco210}"
export D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-/home/choi/.d4rl/datasets}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

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
unset JAX_PLATFORMS CUDA_VISIBLE_DEVICES

"$PY" "$AMO/scripts/launch_td3_amo_jax_antmaze_te1_tlr_fill.py" \
  --detach --gpus 0 --max-parallel 2 --max-used-mib 11000 --out "$JAX_OUT"
log "JAX launcher detached; refreshing canvas"
"$PY" /home/choi/AMO_EXP/scripts/refresh_td3_amo_antmaze_te1_tlr_canvas.py \
  >>/home/choi/logs/td3_amo_antmaze_te1_tlr_canvas.log 2>&1 || true
log "handoff waiter done"
