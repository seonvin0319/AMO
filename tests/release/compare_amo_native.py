"""Controlled CPU comparison against pinned original AMO trainers.

Run with ``python -m tests.release.compare_amo_native --output report.json``.
Synthetic transitions, identical initial weights and noise, width/batch 256.
This is NOT a D4RL return-reproduction experiment. IQL warm-up is shortened
explicitly to exercise several meta events in a short numerical comparison.
"""

import argparse
import copy
import json
from pathlib import Path

import jax
import numpy as np
import pytest
import torch
from torch import nn

from common.agent import make_agent
from common.config import load_config
from common.tree import flatten, map_tree, unflatten
from tests.release.test_iql_amo_reference import reference as reference_iql
from tests.release.test_upstream import numpy, reference_td3

KEYS = ("observations", "actions", "rewards", "next_observations", "terminals")


def batch(seed, size=256):
    rng = np.random.default_rng(seed)
    return {
        "observations": rng.normal(size=(size, 17)).astype(np.float32),
        "actions": np.tanh(rng.normal(size=(size, 6))).astype(np.float32),
        "rewards": rng.normal(size=(size, 1)).astype(np.float32),
        "next_observations": rng.normal(size=(size, 17)).astype(np.float32),
        "terminals": (rng.uniform(size=(size, 1)) < 0.1).astype(np.float32),
    }


def reference_arrays(ref, algorithm):
    """Map original modules and both Adam implementations to release keys."""
    out = {}

    def parameter(prefix, param, optimizer=None, transpose=False, opt_states=None):
        def transform(x):
            return numpy(x).T.copy() if transpose else numpy(x).copy()

        out[prefix] = transform(param)
        if optimizer is None and opt_states is None:
            return
        _, name, subpath = prefix.split("/", 2)
        state = (
            optimizer.state.get(param, {}) if opt_states is None else opt_states[param]
        )
        for ours, theirs in (("m", "exp_avg"), ("v", "exp_avg_sq")):
            out[f"opt/{name}/{ours}/{subpath}"] = transform(
                state.get(theirs, torch.zeros_like(param))
            )
        count_key = f"opt/{name}/count"
        count = numpy(state.get("step", 0)).copy()
        if count_key in out:
            np.testing.assert_array_equal(out[count_key], count)
        out[count_key] = count

    def mlp(prefix, module, optimizer=None, torchopt_state=None):
        opt_states = None
        if torchopt_state is not None:
            s = torchopt_state[0]
            opt_states = {
                p: {"exp_avg": m, "exp_avg_sq": v, "step": t}
                for p, m, v, t in zip(module.parameters(), s.mu, s.nu, s.count)
            }
        norms = iter(x for x in module.modules() if isinstance(x, nn.LayerNorm))
        linears = [x for x in module.modules() if isinstance(x, nn.Linear)]
        for i, layer in enumerate(linears):
            p = f"{prefix}/layer{i}"
            parameter(f"{p}/w", layer.weight, optimizer, True, opt_states)
            parameter(f"{p}/b", layer.bias, optimizer, False, opt_states)
            if i < len(linears) - 1 and any(
                isinstance(x, nn.LayerNorm) for x in module.modules()
            ):
                norm = next(norms)
                parameter(f"{p}/scale", norm.weight, optimizer, False, opt_states)
                parameter(f"{p}/offset", norm.bias, optimizer, False, opt_states)

    if algorithm == "td3_amo":
        for name, index in (("actor", 1), ("bootstrap", 0)):
            mlp(
                f"p/{name}/net",
                ref.actor_bank[index],
                torchopt_state=ref.actor_bank_states[index],
            )
        for i in (1, 2):
            mlp(
                f"p/critic/q{i}",
                getattr(ref, f"critic_{i}"),
                getattr(ref, f"critic_{i}_optimizer"),
            )
            mlp(f"target/critic/q{i}", getattr(ref, f"critic_{i}_target"))
        mlp("target/actor/net", ref.actor_target)
        for role, param, opt in (
            ("E", ref.log_T, ref.T_optimizer),
            ("B", ref.log_T_B, ref.T_B_optimizer),
        ):
            parameter(f"p/scale_{role}/rho", param, opt)
    else:
        for name, module, opt in (
            ("actor", ref.actor_adaptive, ref.actor_adaptive_optimizer),
            ("bootstrap", ref.actor_fixed, ref.actor_fixed_optimizer),
        ):
            mlp(f"p/{name}/net", module, opt)
            parameter(f"p/{name}/log_std", module.log_std, opt)
        mlp("p/value", ref.vf, ref.v_optimizer)
        for i in (1, 2):
            mlp(f"p/critic/q{i}", getattr(ref.qf, f"q{i}"), ref.q_optimizer)
            mlp(f"target/critic/q{i}", getattr(ref.q_target, f"q{i}"))
        parameter("p/scale_E/rho", ref.rho, ref.rho_optimizer)
        parameter("p/scale_B/rho", ref.rho_B, ref.rho_B_optimizer)
    return out


class Comparison:
    def __init__(self):
        self.groups = {}
        self.first_failure = None

    def add(self, key, actual, expected, step, atol=5e-6, rtol=3e-5):
        actual, expected = np.asarray(actual), np.asarray(expected)
        assert actual.shape == expected.shape, key
        if key.endswith("/count"):
            atol = rtol = 0.0
        group = key.split("/")[0]
        if group == "opt":
            group = "adam_" + key.split("/")[2]
        if key.startswith("p/scale"):
            group = "rho"
        diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        largest = float(diff.max())
        tolerance = atol + rtol * np.abs(expected.astype(np.float64))
        failed = bool(np.any(diff > tolerance) or not np.isfinite(diff).all())
        stats = self.groups.setdefault(
            group,
            {
                "max_abs_error": -1.0,
                "key": "",
                "step": 0,
                "compared_arrays": 0,
                "failing_arrays": 0,
                "atol": atol,
                "rtol": rtol,
            },
        )
        stats["compared_arrays"] += 1
        stats["failing_arrays"] += int(failed)
        if largest > stats["max_abs_error"]:
            stats.update(max_abs_error=largest, key=key, step=step)
        if failed and self.first_failure is None:
            self.first_failure = dict(
                key=key, step=step, max_abs_error=largest, atol=atol, rtol=rtol
            )


def policy_output_probe(agent, ref, algorithm):
    """Measure function outputs on held-out synthetic states, without rollouts."""
    data = batch(170017, size=8192)
    observations = data["observations"]
    states = torch.from_numpy(observations)
    actions = torch.from_numpy(data["actions"])
    out = {"seed": 170017, "states": len(observations), "data": "synthetic; not D4RL"}

    def difference(actual, expected):
        actual, expected = np.asarray(actual, np.float64), np.asarray(
            expected, np.float64
        )
        assert actual.shape == expected.shape
        delta = actual - expected
        return {
            "max_abs_error": float(np.abs(delta).max()),
            "mean_abs_error": float(np.abs(delta).mean()),
            "rmse": float(np.sqrt(np.mean(delta**2))),
            "p95_abs_error": float(np.quantile(np.abs(delta), 0.95)),
            "reference_rms": float(np.sqrt(np.mean(expected**2))),
        }

    with torch.no_grad():
        if algorithm == "iql_amo":
            execution = ref.actor_adaptive(states).mean.numpy()
            bootstrap = ref.actor_fixed(states).mean.numpy()
            qs = ref.qf.both(states, actions)
        else:
            execution = ref.execution_actor()(states).numpy()
            bootstrap = ref.critic_bootstrap_actor()(states).numpy()
            qs = (ref.critic_1(states, actions), ref.critic_2(states, actions))
    out["execution_action"] = difference(agent.act(observations), execution)
    out["bootstrap_action"] = difference(
        agent.ops.numpy(
            agent.actor(agent.state["p"]["bootstrap"], agent.ops.array(observations))
        ),
        bootstrap,
    )
    ours = agent.q(
        agent.state["p"]["critic"],
        agent.ops.array(observations),
        agent.ops.array(data["actions"]),
    )
    for i, (actual, expected) in enumerate(zip(ours, qs), 1):
        out[f"critic_q{i}_at_data_actions"] = difference(
            agent.ops.numpy(actual).ravel(), expected.numpy().ravel()
        )
    return out


def run(algorithm, backend, steps, resync):
    config = load_config(algorithm, "halfcheetah-medium-v2")
    if algorithm == "iql_amo":
        config["meta_warmup_steps"] = 0
    agent = make_agent(algorithm, backend, 17, 6, config, seed=17)
    check = Comparison()
    with pytest.MonkeyPatch.context() as patch:
        ref = (
            reference_iql(agent, patch)
            if algorithm == "iql_amo"
            else reference_td3(agent, algorithm)
        )
        meta_events = 0
        for step in range(1, steps + 1):
            if resync:
                # Oracle state at each step isolates one-update error from drift.
                arrays = flatten(map_tree(agent.ops.numpy, agent.state))
                arrays.update(reference_arrays(ref, algorithm))
                # Preserve the release state's explicit dtypes, including counts.
                original = flatten(agent.state)
                agent.state = map_tree(
                    agent.ops.array,
                    unflatten(
                        {
                            k: np.asarray(v, dtype=agent.ops.numpy(original[k]).dtype)
                            for k, v in arrays.items()
                        }
                    ),
                )
            before = {
                role: float(
                    agent.ops.numpy(agent.state["opt"][f"scale_{role}"]["m"]["rho"])
                )
                for role in ("E", "B")
            }
            inner, outer = batch(step), batch(step + 1000)
            rng = np.random.default_rng()
            rng.bit_generator.state = copy.deepcopy(agent.rng.bit_generator.state)
            noise = torch.from_numpy(rng.standard_normal((256, 6)).astype(np.float32))
            patch.setattr(torch, "randn_like", lambda x: noise.to(x))
            actual_metrics = agent.update(inner, outer)
            tb = [torch.from_numpy(inner[k]) for k in KEYS]
            to = [torch.from_numpy(outer[k]) for k in KEYS]
            due = agent.meta_due(step)
            if algorithm == "iql_amo":
                _, row = ref.train_step(tb, to, do_meta=due)
            else:
                row = ref.train(tb, to)
            expected = reference_arrays(ref, algorithm)
            actual = flatten(map_tree(agent.ops.numpy, agent.state))
            for key, value in expected.items():
                check.add(key, actual[key], value, step)
                if key.startswith("p/"):
                    dtype = (
                        np.float64
                        if algorithm == "iql_amo" and "/scale_" in key
                        else np.float32
                    )
                    assert actual[key].dtype == dtype, key
            if due:
                meta_events += 1
                beta1 = 0.0 if algorithm == "iql_amo" else 0.9
                for role in ("E", "B"):
                    m = float(actual[f"opt/scale_{role}/m/rho"])
                    gradient = (m - beta1 * before[role]) / (1 - beta1)
                    ref_key = (
                        ("g_rho" if role == "E" else "g_rho_B")
                        if algorithm == "iql_amo"
                        else ("amo/grad_T_E" if role == "E" else "amo/grad_T_B_total")
                    )
                    check.add(
                        f"hypergradient/{role}",
                        gradient,
                        row[ref_key],
                        step,
                        atol=2e-7,
                        rtol=2e-3,
                    )
                for key in (
                    "L_E" if algorithm == "iql_amo" else "L_T_E",
                    "L_T_B",
                    "L1_B",
                    "L2_RMS_B",
                ):
                    ref_key = key if algorithm == "iql_amo" else f"amo/{key}"
                    check.add(
                        f"outer_loss/{key}",
                        actual_metrics[key],
                        row[ref_key],
                        step,
                        atol=2e-6,
                        rtol=2e-3,
                    )
            if step % 20 == 0:
                print(
                    f"{algorithm}/{backend}: {step}/{steps}; first_failure={check.first_failure}",
                    flush=True,
                )
        output_probe = policy_output_probe(agent, ref, algorithm)
    return dict(
        algorithm=algorithm,
        backend=backend,
        jax_enable_x64=bool(jax.config.jax_enable_x64) if backend == "jax" else None,
        steps=steps,
        meta_events=meta_events,
        config=config,
        groups=check.groups,
        first_failure=check.first_failure,
        passed=check.first_failure is None,
        policy_output_probe=output_probe,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--resync", action="store_true")
    parser.add_argument("--backend", choices=("torch", "jax"), default="jax")
    parser.add_argument(
        "--algorithm", choices=("both", "td3_amo", "iql_amo"), default="both"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = {
        "data": "synthetic; not D4RL",
        "device": "cpu",
        "observation_dim": 17,
        "action_dim": 6,
        "initialization_seed": 17,
        "original_trainer_commits": {
            "iql_amo": "7068c765918917fc8ac5724120ffff2583baf693",
            "td3_amo": "eee3d486fac0c9f5ecbbea411424e6691b715265",
        },
        "torch_version": torch.__version__,
        "jax_version": jax.__version__,
        "iql_warmup_override": 0,
        "state_resynchronized_before_each_step": args.resync,
        "tolerances_set_before_run": True,
        "runs": [],
    }
    # IQL enables x64 globally. Following with TD3 also checks mixed-dtype safety.
    algorithms = (
        ("iql_amo", "td3_amo") if args.algorithm == "both" else (args.algorithm,)
    )
    for algorithm in algorithms:
        result["runs"].append(run(algorithm, args.backend, args.steps, args.resync))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if all(r["passed"] for r in result["runs"]) else 1)


if __name__ == "__main__":
    main()
