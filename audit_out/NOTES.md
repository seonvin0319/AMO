# IQL+AMO β=1 final50 recovery notes (ext_csh)

- Date: 2026-09-14
- Code worktree: `/home/ext_csh/AMO_release_wt_log_audit_20260914`
- Branch: `codex/log-audit-20260914-ext-csh` @ `b9338d9815525482d2cf34d6fc6315ea4d2f93a6` (matches live `AMO_release` main checkout; no pull/reset on live tree)
- Eval env: `capo_jax` + `AMO_release/.py_overlay` (mujoco_py) + mujoco210 `LD_LIBRARY_PATH`
- Protocol: `final50_singlepass_v1` → `eval_final50_v1.jsonl` + `posthoc_eval_final50_v1/`
- Session logs: `/home/ext_csh/logs/iql_amo_final50/` (see `LATEST_SESSION`)

## Hyperparams (verified from config.yaml / ckpt)

- algorithm: `iql_amo` (JAX), backend: `jax`
- actor_lr = critic_lr = value_lr = **3e-4** (distinct from svcho default 1e-3)
- beta_initial = **1.0**
- rho_lr ∈ {3e-4, 2e-3}

## Dual-actor semantics

Shared critic uses IQL **V-based** Bellman target on dataset actions. Bootstrap actor
is used for AMO meta / L2_RMS proxy only and does **not** enter the critic target
(unlike TD3 actor–critic coupling). Eval policy = π_E via `agent.act` (main actor).

## Seed0 JAX compatibility

- Found under `iql_amo_bpi_jax_remainder/cells/rlr3e-4_b1/`: **6 AntMaze** only (β=1).
- **missing_seed0**: all 9 locomotion × both ρ; all 6 AntMaze for ρ=2e-3; (and no β=1 under rlr2e-3_* for jax rem).
- Do **not** fill with Torch/CORL seed0.

## Legacy eval

Prior `eval.jsonl` often has episodes=50 with repeats=5 (10×5 style). Preserved; **not** counted as final50_singlepass_v1.

## Update 2026-09-14: merge CORL seed0 into 4-seed tables

Per operator request, code/backend mismatch is accepted for seed0.
- Prefer JAX final50 seed0 when present (AntMaze ρ=3e-4).
- Else fill from CORL Torch `adaptive_beta` `mean_normalized` @ 1M.
