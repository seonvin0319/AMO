# AMO: Adaptive Multiscale Optimization for Offline RL

AMO is an offline reinforcement learning implementation that learns proximal
scales with cross-fitted outer objectives. It supports the original
single-scale actor update, sequential actor steps, and an adaptive multiscale
mode with separate execution and bootstrap actors.

## Adaptive Multiscale AMO

Enable the method with `--adaptive_multiscale=True`. By default both scales
start from `--T_E`. Pass `--T_B` to initialize the bootstrap scale
independently. The two scales are then optimized independently:

- `T_E` controls the execution actor and minimizes `L_T_E = -B_PI_E`.
- `T_B` controls the bootstrap actor and minimizes `L_T_B = L1_B + L2_RMS_B`.

There is no fixed ratio or ordering between the scales. Their trajectories are
determined only by their role-specific losses and optimizers.

For a dataset action `a_D` and policy action `pi(s)`, each actor uses

```text
L_inner = -mean(Q(s, pi(s))) / mean(abs(Q(s, pi(s))))
          + mean(||pi(s) - a_D||^2) / (2 T).
```

The bootstrap outer objective is

```text
q_scale_B = stop_gradient(mean(abs(min_j Q_target_j(s, pi_B+(s))))) + eps
L1_B_Q = -2 stop_gradient(T_B)
         * mean(min_j Q_target_j(s, pi_B+(s))) / q_scale_B
L1_B = L1_B_Q + mean(||pi_B+(s) - a_D||^2)

delta_y_B = gamma (1-done) [
    min_j Q_target_j(s', pi_B+(s'))
    - stop_gradient(min_j Q_target_j(s', pi_B(s')))
]
L2_RMS_B = 2 stop_gradient(T_B) * RMS(delta_y_B / q_scale_B)
L_T_B = L1_B + L2_RMS_B
```

Target actions are deterministic and target-critic parameters are frozen. The
new-action branch retains the `T_B` hypergradient. `T_E` and `T_B` have
separate softplus parameters, Adam optimizers, and learning-rate schedulers.

## Installation

```bash
conda env create -f environment.yml
conda activate amo
```

MuJoCo 2.1 and a valid D4RL setup are required for the locomotion tasks.

## Training

Run one adaptive multiscale experiment from the repository root:

```bash
python -m amo \
  --env=hopper-medium-v2 \
  --seed=0 \
  --adaptive_multiscale=True \
  --max_timesteps=1000000
```

Run the seed-0 locomotion-9 sweep over HalfCheetah, Hopper, and Walker2d with
the `medium`, `medium-replay`, and `medium-expert` datasets:

```bash
python scripts/launch_amo_adaptive_multiscale_locomotion9.py \
  --gpus=0,1 \
  --detach
```

The launcher requires `validation_gate.json` in its output directory before a
full sweep starts. This keeps long runs behind an explicit test and smoke-run
check.

## Sequential Actor Steps

The separate `--proximal_n_steps` option retains the existing actor-chain
experiment. For `N` steps, AMO uses `tau = T / N`. This option is independent
of adaptive multiscale mode, and the two modes cannot be enabled together.

```bash
python -m amo \
  --env=hopper-medium-v2 \
  --T_E=1.2 \
  --proximal_n_steps=4
```

## Repository layout

- `amo/algorithm.py`: self-contained reference implementation and training loop.
- `amo/__main__.py`: command-line entry point for `python -m amo`.
- `scripts/`: restartable benchmark launchers with explicit validation gates.
- `tests/`: regression tests for equations, gradient routing, checkpoints, and launch manifests.

## Tests

```bash
python -m pytest -q tests/test_amo.py
```

The tests cover scale initialization and gradient isolation, the detached TQ
and exact RMS formulation, actor roles, checkpoint compatibility, numerical
diagnostics, and the locomotion-9 launch manifest. D4RL is loaded only when
training starts, so the unit tests do not require a configured MuJoCo runtime.

See `CONTRIBUTING.md` for formatting and contribution guidelines.

## License

AMO is released under the Apache License 2.0. See `NOTICE` and
`THIRD_PARTY_LICENSES` for upstream attributions.
