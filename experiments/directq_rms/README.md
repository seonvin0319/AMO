# Direct-Q with the current RMS-only bootstrap objective

Prepared only; no training has been launched.

Question: with the current dual-policy/RMS-only method held fixed, does replacing the execution outer objective Bpi with direct target-Q improvement change final performance?

| Environment | alpha LR | Seeds | Existing Bpi mean ± sample SD |
|---|---:|---|---:|
| hopper-medium-replay-v2 | .002 | 0,1,2,3 | 101.52 ± 0.59 |
| walker2d-medium-replay-v2 | .002 | 0,1,2,3 | 98.03 ± 1.38 |
| antmaze-medium-diverse-v2 | .0003 | 0,1,2,3 | 59.00 ± 17.17 |

12 new runs, each 1M updates. Both coefficients start at 5. The three YAML files freeze all resolved algorithm settings. Their sole difference from the corresponding archived bootrms control configs on shared settings is `execution_score: direct_q`. In particular, `bootstrap_loss: l2_rms`, `execution_meta_loss: le`, `execution_only: false`, and `critic_target: bootstrap` stay fixed. No LR search and no baseline retraining are included.

The existing implementation minimizes `-mean(Q_target,1(s,a_plus) - stop(Q_target,1(s,a_old))) / stop_scale`. The old-action term and scale are stopped, so its coefficient gradient is that of normalized negative candidate Q. The target critic parameters are frozen while the action derivative passes through the virtual execution actor update. The real actor inner objective and the bootstrap meta objective are unchanged. Historical direct-Q runs used L1+RMS bootstrap and are not the control for this experiment.

## Run from the AMO repository root

Use the existing AMO JAX/D4RL/MuJoCo environment and dataset cache. The default command is read-only: it verifies runtime hashes and the config differences, then prints all commands without creating output directories.

```bash
python scripts/run_directq_rms.py
python scripts/run_directq_rms.py --phase eval
```

When starting training is intended:

```bash
python scripts/run_directq_rms.py --execute --device cuda:0
```

This executes the 12 runs sequentially and stops on failure. To distribute work across devices, select distinct manifest IDs with `--run`, e.g.:

```bash
python scripts/run_directq_rms.py --execute --device cuda:0 --run h-mr_directq_rms_a5_s0
```

Existing run directories are never overwritten or silently resumed. After interruption, inspect the full checkpoint and use train.py's explicit `--resume` interface if resuming is appropriate. The launcher records config hashes, source hashes, and commands alongside each run.

After training, evaluate on CPU using the same final-evaluation protocol as the archived controls:

```bash
python scripts/run_directq_rms.py --phase eval --execute
```

Final checkpoint at 1M, 10 episodes × 5 repeats, evaluation seed equal to training seed, repeat seed stride 1, normalized score. The frozen AntMaze YAML uses `eval_episodes: 10`, matching the archived control configs (the generic repository preset currently says 100). Training disables in-process evaluation; the explicit CPU command uses 10×5 for all three tasks. No checkpoint selection by score. Checkpoints are saved every 200k to limit disk usage; this changes checkpoint frequency only, not updates or LR schedules.

## Analysis and interpretation

Use `bpi_reference.csv` for the 12 pinned controls. Report all four seed scores, mean ± sample SD, and seed-paired direct-Q minus Bpi differences per environment. Do not pick another LR after observing results. If direct-Q is comparable, state that these tasks do not establish a performance benefit from the Bpi surrogate. If it underperforms, the result supports the choice of outer objective under this protocol; it does not validate a global smoothness bound or prove a causal mechanism involving critic error.

## Preparation checks

- All 12 commands pass read-only validation against pinned runtime blob hashes.
- All 12 resolved configs agree with the original logged Bpi controls on shared algorithm settings except execution_score.
- Runtime source inspected: existing direct_q branch and frozen-target/input-gradient path; no algorithm source changed.
- No training, GPU/JAX execution, or environment rollout was performed in the preparation workspace (JAX and MuJoCo are unavailable here). Run previews validate configuration and command construction, not numerical runtime behavior.

Source snapshot: `c5bffd388f4cc5c504dc533b72df312d9a18c6b7`. A changed runtime fails preflight instead of silently changing the comparison. See `runtime_hashes.json` and immutable control log refs in `bpi_reference.csv`.
