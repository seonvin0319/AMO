# AMO offline RL release candidate

PyTorch and JAX implementations with one algorithm module per backend, shared minimal configuration files, and a common offline training entry point. Each backend uses its native arrays and automatic differentiation. JAX compiles the training step with JIT and does not import PyTorch.

All eight algorithms run on both backends. AMO implementations follow the uploaded TD3 and IQL experiment branches; ReBRAC follows its original repository. [Source revisions](SOURCE_PROVENANCE.md) identify the exact implementations. This release validates numerical updates and runtime behavior; it does not report new MuJoCo benchmark results.

| Algorithm | Module under `algorithms/{torch,jax}/` | Configuration |
| --- | --- | --- |
| ASPC | `aspc.py` | `configs/aspc.yaml` |
| wPC(RC) | `wpc.py` | `configs/wpc.yaml` |
| TD3+BC(RC) | `td3_bc.py` | `configs/td3_bc.yaml` |
| A2PR | `a2pr.py` | `configs/a2pr.yaml` |
| ReBRAC | `rebrac.py` | `configs/rebrac.yaml` |
| IQL | `iql.py` | `configs/iql.yaml` |
| TD3+AMO | `td3_amo.py` | `configs/td3_amo.yaml` |
| IQL+AMO | `iql_amo.py` | `configs/iql_amo.yaml` |

Here RC means robust critic. It uses three hidden layers of width 256, ReLU followed by LayerNorm, and the ASPC initialization. A2PR, ReBRAC and IQL retain their reference architectures. The `--algorithm` key is the module filename without `.py`, for example `td3_bc` or `td3_amo`.

## Installation and training

Run commands from this directory. Install one backend:

```bash
pip install -r requirements/torch.txt
# Or JAX on CPU:
pip install -r requirements/jax.txt
```

For CUDA, use the framework's CUDA installation for your machine. D4RL evaluation requires an existing working Gym/D4RL/MuJoCo environment. Array-based training with `--dataset ... --no-eval` does not import Gym or D4RL.

Numerical checks were run with Python 3.12.14, PyTorch 2.14.0 and JAX 0.11.1 on CPU. Other combinations and MuJoCo environment installation were not validated in this build. The evaluation runner uses the legacy Gym `seed/reset/step` API expected by D4RL.

To verify the installation on CPU using the included synthetic fixture:

```bash
python train.py --algorithm td3_amo --backend torch --env synthetic \
  --dataset examples/smoke_dataset.npz --config examples/smoke_config.yaml \
  --device cpu --no-eval --steps 20 --output runs/smoke
python train.py --algorithm td3_amo --backend torch --env synthetic \
  --dataset examples/smoke_dataset.npz --config examples/smoke_config.yaml \
  --device cpu --no-eval --steps 40 --output runs/smoke --resume runs/smoke/checkpoint.npz
```

This exercises a meta update and resumption; the fixture is not a benchmark. Replace `torch` with `jax` and choose a different output directory for a separate JAX run.

```bash
python train.py --algorithm td3_amo --backend torch \
  --env hopper-medium-v2 --seed 0 --device cuda:0 \
  --output runs/td3_amo_torch_hopper_s0
```

Change `--backend` to `jax` or select another `--algorithm` from the table. The algorithm YAML is loaded automatically; `--config path.yaml` selects a different file. `--print-config` prints the resolved configuration and exits.

For a local dataset without MuJoCo evaluation:

```bash
python train.py --algorithm td3_amo --backend jax \
  --env hopper-medium-v2 --dataset /path/to/dataset.npz \
  --device cpu --no-eval --steps 1000 \
  --output runs/td3_amo_jax_smoke
```

`--steps` stops at a total training step without changing the configured learning-rate schedules. Defaults train for one million updates. Outputs are `config.yaml`, `run_meta.json`, `metrics.jsonl`, `normalization.npz`, and `checkpoint.npz`; evaluation also creates `eval.jsonl`.

```bash
python train.py --algorithm td3_amo --backend torch \
  --env hopper-medium-v2 --seed 0 --device cuda:0 \
  --output runs/td3_amo_torch_hopper_s0 \
  --resume runs/td3_amo_torch_hopper_s0/checkpoint.npz
```

Resume with the same algorithm configuration, environment, seed, and dataset. Checkpoints include online/target parameters, Adam moments and counters, adaptive scales, normalization statistics, and training/outer-batch RNG states. They contain arrays and JSON with no pickle payload. Numerical identity is tested within each backend; changing frameworks, devices, versions or hardware can change results. Existing nonempty output directories require `--resume`.

## Final TD3+AMO contract

- Two actors share twin critics. The target bootstrap actor supplies Bellman actions; the execution actor supplies evaluation actions. Both use dataset actions as their BC anchor.
- Actor loss is `-mean(Q1)/stop(mean(abs(Q1))) + MSE(pi,a_D)/(2*T)`.
- The execution scale minimizes normalized `-B_pi` through a virtual Adam update.
- The bootstrap scale minimizes `L1_B + L2_RMS_B` through a virtual SGD update. The `2*T_B` common factor and Q normalization are detached. RMS is exact with a finite zero subgradient.
- The outer batch is sampled independently. There is no fixed ratio or ordering constraint between the scales.
- Original timing is retained: the execution actor uses its pre-meta-update scale; the bootstrap actor uses its updated scale.

The default is the recorded completed multiseed profile `T_E=1`, `T_B=1`, `T_lr=0.001`. This does not claim that the profile is best for every environment. Four-evaluation, scheduled-horizon, lambda and behavior-critic experiments are outside the release modules. B_pi remains an empirical local proxy, not a certified performance lower bound.

The uploaded TD3 branch also contains a locomotion launcher with `T_lr=0.0003` and environment-specific initial scales. Its first-priority values are available through `--config configs/td3_amo_band95.yaml`; this is a separate recorded experiment profile. The default critic remains three hidden layers with LayerNorm. The branch's two-layer critic without LayerNorm is an ablation, selectable through `critic_depth` and `critic_layernorm` in a custom config.

See [the exact equations and detach conventions](docs/TD3_AMO.md) for normalization, virtual updates and zero handling.

## IQL+AMO contract

IQL+AMO keeps the standard expectile V update and `r + gamma*(1-done)*V(s')` critic target. Two Gaussian AWR actors learn with separate inverse temperatures. The execution temperature minimizes normalized `-B_pi`; the second temperature minimizes `L1_B+L2_RMS_B`. That RMS measures target-Q displacement at policy actions. In the uploaded implementation the second actor affects neither Q/V nor the execution actor; it is a parallel auxiliary policy. The release preserves this dependency structure.

Both temperatures use `beta=exp(rho)`, projected to `[0.05,100]`. Log temperatures and their optimizer moments use float64; networks use float32. JAX enables float64 when constructing this algorithm. The default starts both at 1, uses Adam `(0,0.999)` with `rho_lr=0.002`, and updates every 20 steps after 100,000 warm-up steps. The first meta update is step 100,020.

The execution virtual update is Adam. The second actor takes its real Adam update first, then its virtual SGD step uses the next cosine learning rate. Both outer losses use the target twin-Q minimum. See [the implementation contract](docs/IQL_AMO.md).

## Datasets and configs

Raw D4RL HDF5 and transition NPZ files are supported. A transition NPZ must contain `observations`, `actions`, `rewards`, `next_observations`, and binary `terminals`. ReBRAC also requires aligned `next_actions`. Raw-sequence loading omits timeout transitions and aligns next actions before filtering. The release targets continuous D4RL actions in `[-1,1]`.

| Array | Shape | Meaning |
| --- | --- | --- |
| observations / next_observations | `(N, observation_dim)` | Current / next state |
| actions / next_actions | `(N, action_dim)` | Current / next dataset action |
| rewards | `(N,)` or `(N,1)` | Untransformed reward |
| terminals | `(N,)` or `(N,1)` | True termination, 0 or 1 |

Numeric arrays are converted to float32 and must be finite. Transition NPZ files must already resolve timeouts; nonzero `timeouts` alongside `next_observations` is rejected. Do not normalize observations or transform rewards before loading: the runner applies the selected config's transformation. With `--no-eval`, `--env` selects task configuration and labels the run; it does not construct an environment.

For a custom transition export, omit timeout transitions and keep genuine terminal transitions with `terminals=1`. Take `next_observations` and ReBRAC `next_actions` from the next row of the raw sequence before filtering. They are masked out of Bellman targets on genuine terminal rows; do not shift the already-filtered array.

The YAML files hold experiment choices; shared architecture and optimizer defaults live in `common/config.py`. Profiles cover locomotion medium/medium-replay/medium-expert and AntMaze. [Config provenance](docs/config_sources.json) links to pinned source settings. ReBRAC uses the original repository's configs for all 15 tasks, including AntMaze reward multiplication by 100, discount 0.999, and critic learning rate 0.00005. Its actor updates start on the first critic step; evaluation uses seed 42 and the original epoch-based cadence.

The runner evaluates the execution policy. TD3's optional final repeats increment the evaluation seed; IQL's uploaded source repeats the same seed, which is retained and recorded in evaluation output. Repeated rounds are not treated as independent training seeds. The experiment's additional paired comparison between IQL's two actors is outside this runner.

## Validation and attribution

```bash
pip install -r requirements/test.txt
python -m pytest tests/release -q
```

The full test requirements install both backends because the suite compares them.
Tests compare both ports directly with pinned upstream trainers, compare complete
backend training states, verify checkpoint/resume equality, and check data boundaries.
**Benchmark reproduction is unverified.** The separate width-256, batch-256 AMO
comparisons include numerical-gate failures; a green small-test suite does not
establish agreement with amo_log. Treat this as a release candidate. See
[measured errors, raw results and reproduction commands](docs/VALIDATION.md).

The source checkout retains the earlier experimental `amo/` package. The release entry point uses `algorithms/{torch,jax}/`, `common/`, and the shared configs. See [NOTICE](NOTICE), [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES), and [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md) for attribution.

## Contributors

- [seonvin0319](https://github.com/seonvin0319)
- [Schoish](https://github.com/Schoish)
