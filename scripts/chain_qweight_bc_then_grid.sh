#!/usr/bin/env bash
# After per-env behavior BC finishes, detach the qweight loco+antmaze grid.
set -uo pipefail
ROOT=/home/ext_csv/AMO-main
PY=/home/ext_csv/miniconda3/envs/amo-jax/bin/python
BEH_ROOT=/raid/ext_csv/AMO_store/iql_amo_qweight_behavior_shared
BC_PIDFILE=/home/ext_csv/logs/iql_amo_qweight_behavior_bc_all.pid
LOG=/home/ext_csv/logs/iql_amo_qweight_chain_bc_to_grid.log
mkdir -p /home/ext_csv/logs

need=(h-m h-mr h-me w-m w-mr w-me hc-m hc-mr hc-me am-u am-u-div am-m-p am-m-div am-l-p am-l-div)

echo "==== $(TZ=Asia/Seoul date -Is) chain start ====" | tee -a "$LOG"
# wait for BC pid if present
if [[ -f "$BC_PIDFILE" ]]; then
  bcpid=$(cat "$BC_PIDFILE")
  while kill -0 "$bcpid" 2>/dev/null; do
    echo "$(TZ=Asia/Seoul date -Is) waiting BC pid=$bcpid" | tee -a "$LOG"
    sleep 60
  done
fi
# also wait until all behaviors exist
while true; do
  missing=0
  for t in "${need[@]}"; do
    [[ -f "$BEH_ROOT/$t/behavior.npz" ]] || missing=1
  done
  if [[ $missing -eq 0 ]]; then
    break
  fi
  echo "$(TZ=Asia/Seoul date -Is) waiting behavior artifacts missing" | tee -a "$LOG"
  sleep 60
done
echo "$(TZ=Asia/Seoul date -Is) all behaviors ready; launching grid" | tee -a "$LOG"
cd "$ROOT"
export PYTHONPATH="$ROOT"
"$PY" -u scripts/launch_iql_amo_qweight_jax_beta125_rho_loco_antmaze.py --detach --gpus 0,1 --max-parallel 6 \
  | tee -a "$LOG"
echo "==== $(TZ=Asia/Seoul date -Is) chain done ====" | tee -a "$LOG"
