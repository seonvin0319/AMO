#!/usr/bin/env bash
set -euo pipefail
PY=/home/choi/miniconda3/envs/amo/bin/python
AMO=/home/choi/amo
PT_OUT=/home/choi/AMO_store/td3_amo_antmaze_te1_tlr_main_fill_adaptive_critic
JAX_OUT=/home/choi/AMO_store/td3_amo_jax_antmaze_te1_tlr_fill
HANDOFF=/home/choi/AMO_store/td3_amo_jax_antmaze_te1_tlr_fill_handoff/pending_cells.json
LOG=/home/choi/logs/td3_amo_jax_antmaze_te1_tlr_handoff.log

pkill -f 'wait_pytorch_then_launch_jax_antmaze_fill.sh' 2>/dev/null || true
sleep 1

# discover live torch amud 2e-3 pids from RUNNING markers
PIDS=()
for f in "$PT_OUT"/jobs/td3amo_te1_tb1_tlr2e-3_amud_s{2,3}/RUNNING.json; do
  [[ -f "$f" ]] || continue
  p=$("$PY" -c "import json; print(json.load(open('$f')).get('pid',''))")
  [[ -n "$p" ]] && PIDS+=("$p")
done
echo "stopping pids: ${PIDS[*]:-none}"
for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null || true; done
sleep 3
for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && kill -KILL "$p" 2>/dev/null || true; done

"$PY" - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
pt = Path("/home/choi/AMO_store/td3_amo_antmaze_te1_tlr_main_fill_adaptive_critic")
now = datetime.now(timezone.utc).isoformat()
for rid in ("td3amo_te1_tb1_tlr2e-3_amud_s2", "td3amo_te1_tb1_tlr2e-3_amud_s3"):
    job = pt / "jobs" / rid
    run_marker = job / "RUNNING.json"
    rec = json.loads(run_marker.read_text()) if run_marker.exists() else {}
    if run_marker.exists():
        run_marker.unlink()
    step = None
    runs = list((pt / "runs").glob(f"{rid}-*"))
    if runs:
        m = runs[0] / "metrics.jsonl"
        if m.exists() and m.stat().st_size:
            step = int(json.loads(m.read_text().strip().splitlines()[-1]).get("step", 0))
    (job / "STOPPED.json").write_text(json.dumps({
        "run_id": rid, "stopped_at": now, "reason": "user_switch_to_jax",
        "last_pid": rec.get("pid"), "last_step": step,
    }, indent=2) + "\n")
    print(rid, "stopped", step)
PY

cd "$AMO"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-/home/choi/.mujoco/mujoco210}"
export D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-/home/choi/.d4rl/datasets}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
if ! pgrep -af "eval_checkpoints_cpu.py --runs-root $JAX_OUT" | rg -qv pgrep; then
  nohup env -u CUDA_VISIBLE_DEVICES JAX_PLATFORMS=cpu \
    MUJOCO_PY_MUJOCO_PATH="$MUJOCO_PY_MUJOCO_PATH" \
    D4RL_DATASET_DIR="$D4RL_DATASET_DIR" MUJOCO_GL="$MUJOCO_GL" \
    "$PY" -u "$AMO/scripts/eval_checkpoints_cpu.py" \
    --runs-root "$JAX_OUT/runs" --episodes 10 --final-repeats 5 --device cpu \
    --poll --poll-sec 120 >>/home/choi/logs/td3_amo_jax_antmaze_cpu_eval.log 2>&1 &
  echo "cpu_eval_pid=$!"
fi
"$PY" "$AMO/scripts/launch_td3_amo_jax_antmaze_te1_tlr_fill.py" \
  --detach --gpus 0 --max-parallel 2 --max-used-mib 11000 \
  --out "$JAX_OUT" --handoff "$HANDOFF" | tee -a "$LOG"
"$PY" /home/choi/AMO_EXP/scripts/refresh_td3_amo_antmaze_te1_tlr_canvas.py \
  >>/home/choi/logs/td3_amo_antmaze_te1_tlr_canvas.log 2>&1 || true
echo DONE
