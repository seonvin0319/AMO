# Numerical validation scope

**Status: amo_log performance reproduction is unverified.** This is a release
candidate, not a benchmark-reproduced release. No D4RL/MuJoCo learning curve or
one-million-step final score has been reproduced. The small regression suite
passes, but the stricter full-width numerical comparisons below do not all pass.

## Independent reference comparisons

The tests transplant identical initial weights and control target-action/VAE noise. Upstream update equations come from pinned reference trainers, not from the new implementation. This avoids treating agreement between two ports of the same equations as independent proof of correctness.

| Method | External oracle | Checked |
| --- | --- | --- |
| TD3+AMO | AMO adaptive_multiscale trainer | 22 updates including the scale update, actor/critic/target parameters, T_E/T_B parameters |
| ASPC | ASPC Adaptive_TD3_BC | 22 updates including alpha update, actor/critic/target parameters |
| wPC(RC) | ASPC wPC trainer | 22 updates, value mask timing, actor/value/critic/targets |
| TD3+BC(RC) | ASPC TD3_BC | 22 updates, actor/critic/targets |
| IQL | ASPC/CORL IQL | 8 updates, actor mean/log-std, value, twin Q, target Q |
| A2PR | A2PR trainer | 6 updates, actor, twin Q, value and all VAE parameters |
| ReBRAC | Native JAX ReBRAC update_actor/update_critic | 6 updates, actor/critic/target parameters |
| IQL+AMO | Uploaded AMO IQLAdaptiveBetaTrainer | Initial beta 1 and 5, 9 updates each, warm-up boundary, both actors, Q/V, target Q, log temperatures and outer loss terms |

The table's tests now run both PyTorch and JAX directly against the original
trainers, except ReBRAC, whose direct oracle comparison is native JAX. They use
small network widths with the reference layer counts and update equations.
Floating-point tolerances are explicit in the tests. ASPC's missing reward_mean
initialization is assigned zero explicitly, as documented in SOURCE_PROVENANCE.md.

The initial validation had a narrower scope: most original-trainer comparisons
ran only on PyTorch, and cross-backend tests compared losses without comparing
all training-state tensors. It did not establish JAX equivalence to amo_log.

## Full-width AMO comparison, 2026-09-12

The reproducible comparison uses CPU, hidden width 256, batch 256, observation
dimension 17 and action dimension 6, with the source layer counts. It runs 60
updates and three meta events for each AMO method. Batches change at every step;
both implementations receive identical initial parameters, batches and TD3
target noise. **Transitions are synthetic.** IQL's warm-up is explicitly changed
from 100,000 to zero to exercise meta updates; its cosine horizon remains one
million. This does not test a real post-warm-up IQL checkpoint.

Every actor, critic, value, used target network, Adam moment and counter is
compared with the original trainer. Scale hypergradients are recovered from
moment changes and compared with the original trainer's gradient logs. The
outer loss terms are also compared. Network parameters remain float32; IQL log
temperatures and their moments remain float64.

| JAX comparison | Maximum network parameter error | Outcome at the declared tolerances |
| --- | ---: | --- |
| IQL+AMO, 60 consecutive updates | 1.45e-5 | Fail; first parameter failure at step 51 |
| TD3+AMO, 60 consecutive updates | 4.31e-4 | Fail; first Adam-moment failure at step 22 |
| IQL+AMO, reset to original state before each of 60 updates | 5.59e-7 | Pass, including optimizer, scales, hypergradients and outer losses |
| TD3+AMO, reset to original state before each of 60 updates | 1.21e-5 | Fail; critic parameter/moment discrepancy at step 57 |

The reset comparison copies **parameters, targets and optimizer states** from
the original before every step. It isolates a single update and must not be
reported as a matching 60-step training trajectory. Its maximum hypergradient
errors were 5.35e-9 for IQL+AMO and 1.87e-8 for TD3+AMO.

The step-57 TD3 discrepancy was inspected at the ReLU boundary in critic Q1's
third hidden layer: the same original state produced a preactivation of
`+2.2351741790771484e-8` in JAX and `-7.450580596923828e-8` in the original.
This changes one ReLU activation and its gradient. This diagnoses that local
failure; it does not prove that all long-run differences are harmless or that
returns will match. No tolerances were widened to turn these failures into passes.

Two numerical details were corrected during this audit: PyTorch uses native
linear/layer-normalization operations, and Adam bias corrections avoid premature
float32 cancellation. PyTorch uses double scalar corrections, as torch.optim
does. JAX uses `-expm1(t * log(beta))` with the scalar log computed in Python,
so TD3 does not require an earlier IQL run to enable global JAX x64. A fresh
TD3-only process with x64 disabled was tested separately; its consecutive-update
comparison still fails the strict gate. PyTorch's corresponding consecutive
comparison also fails (maximum errors: IQL 1.45e-5, TD3 8.19e-5).

Raw results include all declared tolerances, failures and per-category maxima:
[JAX consecutive](validation/jax_free.json),
[JAX from original state](validation/jax_from_reference_state.json),
[JAX TD3 with x64 disabled](validation/jax_td3_x64_disabled.json), and
[PyTorch consecutive](validation/torch_free.json).

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m tests.release.compare_amo_native --output jax_free.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m tests.release.compare_amo_native --resync --output jax_from_reference_state.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m tests.release.compare_amo_native --algorithm td3_amo --output jax_td3_x64_disabled.json
```

These commands return exit code 1 when a numerical gate fails. They are separate
from the small pytest suite so a green pytest result does not conceal the
full-width failures.

### Interpreting the discrepancies

The strict tolerances above are diagnostic numerical thresholds, **not return
degradation thresholds**. Exceeding them does not establish an implementation
bug or lower benchmark performance. The comparison initializes the release from
NumPy seed 17, transplants those exact weights into the original trainer, and
supplies identical batches and TD3 target noise. It is not a rerun of an amo_log
training seed. Different libraries' independent initializers need not produce
the same weights from the same integer seed.

To measure function-space differences, the consecutive JAX comparison was
repeated and the resulting policies evaluated on 8,192 held-out synthetic
states (probe seed 170017). These are forward passes, not environment rollouts.
Both reported maximum network-weight errors occurred in critic parameters.

| Method | Execution-action mean absolute error | Execution-action maximum absolute error | Bootstrap-action maximum absolute error |
| --- | ---: | ---: | ---: |
| TD3+AMO | 1.24e-5 | 8.70e-5 | 2.36e-4 |
| IQL+AMO | 1.79e-8 | 1.94e-7 | 1.71e-7 |

Errors are per action coordinate, on an action range of [-1, 1]. TD3's maximum
execution-action discrepancy is 0.00435% of that full range. The measured
policy-output differences are small at this early checkpoint on these synthetic
inputs. This does not bound action discrepancies on actual D4RL/rollout states
or long-run returns. Critic output errors and all probe statistics are included
in [the raw output-probe report](validation/jax_policy_probe.json).

XLA may change floating-point results through algebraic rearrangement even when
the underlying expressions are mathematically equivalent; see the
[JAX numerical FAQ](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs).
The observed ReLU-boundary discrepancy is evidence of such local numerical
sensitivity, not a proof that every discrepancy is benign. Release acceptance
requires source-equation/update-order fidelity and benchmark-level comparisons
over multiple seeds with an explicit practically acceptable score difference;
it does not require every weight to remain within the same absolute tolerance
throughout one million updates.

## Backend and runtime checks

The current small regression suite passes **51 tests** on CPU. This count
excludes the separate full-width comparison commands and their recorded failures.

- Eight algorithms: matched initial arrays and batches, four updates on each backend, comparing losses and every parameter, target and optimizer-state tensor, including adaptive meta updates where applicable. Counters are checked exactly.
- Complete save/load followed by two further updates: exact equality of every saved-state tensor within each backend, including optimizer moments/counters and RNG continuation.
- Zero RMS: finite, exactly zero value and subgradient.
- Missing independently sampled AMO outer batch: fail before modifying the training step.
- Raw D4RL timeout filtering: preserve next-action alignment before filtering; require explicit next_actions for preprocessed ReBRAC data.
- Invalid configuration keys: reject rather than silently ignore.
- IQL+AMO: require a separate outer batch after warm-up; verify float64 log temperatures and float32 network parameters after updates. Preserve projection and the next cosine learning rate in the beta0 virtual step. Q/V remain identical when changing the actors' initial temperatures on identical batches; changing only beta0 also leaves the execution actor and beta_E unchanged.
- Float32 Adam: compare five controlled updates directly with native torch.optim.Adam on both backends, with JAX x64 explicitly disabled, to catch early bias-correction cancellation.

CLI checks use synthetic transitions with local NPZ loading, normalization, training, checkpoint writing and resumption. Synthetic data checks software behavior only.

The runtime checks additionally cover deferred metric collection on all eight JAX algorithms, the first unlogged NaN/Inf surviving later finite metrics, checkpoint rejection after that failure, actor JIT using new parameters after updates and loads, identical host/resident replay samples and RNG continuation on both backends, and CLI resume from resident replay to CPU replay.

## JAX runtime optimization

The baseline is AMO [`e6c1fc5`](https://github.com/seonvin0319/AMO/commit/e6c1fc52a6b341de377c45e52d6c5ee4d3693ea4), which already separates training from offline CPU evaluation. The optimized runtime adds actor JIT, bulk placement/retrieval of array trees and metric synchronization at logging boundaries. Its device-side failure indicator checks every update. Algorithm equations, gradient stops, optimizer precision, NumPy random streams and checkpoint format are retained.

Measurements below use Python 3.12.14, JAX 0.11.1 and NumPy 2.3.5 on an AMD EPYC 9V74 CPU in a virtualized environment. Both checkouts use hidden width 256, batch size 256, 8,192 synthetic transitions and CPU replay. Each trial resets the complete learning state, including targets, optimizer moments/counters and adaptive scales, plus all RNG states. Update timings cover 200 steps, including meta updates, repeated five times; actor timings cover 400 single-observation calls per trial. Values are medians. [Raw trials, state comparisons and runtime source hashes](validation/jax_runtime_cpu.json) are the numerical record.

| Operation | Baseline | Optimized | Time reduction |
| --- | ---: | ---: | ---: |
| TD3+AMO update | 9.43 ms | 7.00 ms | 25.8% |
| IQL+AMO update | 4.56 ms | 3.92 ms | 14.2% |
| TD3+AMO actor call | 0.546 ms | 0.108 ms | 80.2% |
| IQL+AMO actor call | 0.518 ms | 0.098 ms | 81.1% |

Replay sampling, target-noise generation, device placement, updates, periodic metric checks and final synchronization are timed. Initialization, compilation, file IO and environment rollouts are excluded. The before/after final learning states and probe actions match exactly in this 200-step comparison, including dtypes and all three RNG states. IQL starts with initial parameters and its step counter set to 100,000 to exercise the first meta event at 100,020; this is not a trained post-warm-up checkpoint. CPU results do not establish GPU speedups, device-replay speedups, long-run training equivalence or amo_log performance reproduction.

Run each checkout in a fresh process with the same installed dependencies and device. From the optimized checkout:

```bash
git worktree add --detach ../amo-runtime-baseline e6c1fc52a6b341de377c45e52d6c5ee4d3693ea4
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/benchmark_jax_runtime.py --repository ../amo-runtime-baseline --algorithm td3_amo --output results/before_td3.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/benchmark_jax_runtime.py --algorithm td3_amo --output results/after_td3.json
```

Repeat with `--algorithm iql_amo` and different output filenames for IQL+AMO. Each JSON report includes individual timings, runtime settings and final RNG states; the matching `.npz` beside it contains every final state tensor and probe actions. For example, `results/after_td3.json` is paired with `results/after_td3.npz`. Use `--device cuda:0` on a CUDA-enabled installation to measure GPU execution; add `--replay-device training` to the optimized checkout's command to measure resident replay separately.

## Limits

These checks provide bounded evidence about individual updates and software
behavior. The full-width failures prevent an unqualified numerical-equivalence
claim. They do not establish one-million-step learning-curve equivalence, CUDA
throughput, reproducibility of legacy random bitstreams, or new MuJoCo normalized
scores. No MuJoCo rollout benchmark was run. The validation environment has no
D4RL datasets or installed Gym/D4RL/MuJoCo and no available CUDA device. AntMaze
evaluation reset variability is not fixed by this refactor.

Before calling this benchmark-reproduced, run the original and each port on the
same D4RL dataset fingerprint, recorded configs and training/evaluation seeds,
for the full schedule. Compare learning curves, final normalized scores and
T/beta trajectories with amo_log across seeds. Identical integer seeds alone
do not produce identical runs because the release initializer and replay
sampler use a different NumPy random stream from the original framework code.

The B_pi implementation remains the source's empirical endpoint-gradient proxy. Its role and detached terms were checked against the source; no global smoothness or performance-improvement theorem is claimed by these numerical tests.

The config audit compares release choices with recorded fields for the 15 supported tasks. ReBRAC entries point directly to the original repository. IQL+AMO's trainer and meta config are available in AMO 7068c765. TD3's additional band95 profile is extracted from the uploaded launcher's first-priority values; it is not substituted for the completed shared profile.

The runner reports execution-policy evaluations. It does not reproduce the experiment's paired AntMaze comparison between IQL actors. IQL final repeat calls reuse the source's seed; they are not additional independent training runs.
