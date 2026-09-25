#!/usr/bin/env bash
set -uo pipefail
cd /home/ext_csh/AMO_fixed1 || exit 1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 EIGEN_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_gpu_force_compilation_parallelism=1 --xla_gpu_autotune_level=0"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export D4RL_DATASET_DIR=/home/ext_csh/.d4rl/datasets
export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH=/home/ext_csh/.mujoco/mujoco210
export LD_LIBRARY_PATH="/home/ext_csh/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYOPENGL_PLATFORM=egl
unset JAX_PLATFORMS
unset JAX_PLATFORM_NAME
export FIXED1_MODE=gpu
echo "[queue-gpu] td3 rapo fixed1 $(date -Is)"
exec /home/ext_csh/miniconda3/envs/capo_jax/bin/python -u scripts/td3_amo_fixed1_ext_csh_queue.py
