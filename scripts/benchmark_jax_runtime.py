#!/usr/bin/env python3
"""Measure warmed JAX execution on synthetic replay, including batch preparation.

Use --repository to compare another checkout in a fresh process. This measures
CPU/GPU execution time, not MuJoCo returns or amo_log reproduction.
"""

import argparse
import copy
import inspect
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repository", type=Path, default=Path(__file__).resolve().parents[1]
    )
    p.add_argument("--algorithm", choices=("td3_amo", "iql_amo"), required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--replay-device", choices=("cpu", "training"), default="cpu")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if min(args.steps, args.repeats, args.log_every) <= 0:
        p.error("steps, repeats and log-every must be positive")
    if args.output.suffix.lower() != ".json":
        p.error("output must end in .json; the final state is saved alongside as .npz")
    root = args.repository.resolve()
    sys.path.insert(0, str(root))

    import jax
    import numpy as np

    from common.agent import make_agent
    from common.config import load_config
    from common.data import ReplayBuffer
    from common.tree import flatten

    c = load_config(args.algorithm, "halfcheetah-medium-v2")
    agent = make_agent(args.algorithm, "jax", 17, 6, c, seed=17, device=args.device)
    deferred = "return_metrics" in inspect.signature(agent.update).parameters
    rng = np.random.default_rng(170017)
    n = 8192
    data = {
        "observations": rng.normal(size=(n, 17)).astype(np.float32),
        "actions": rng.uniform(-1, 1, (n, 6)).astype(np.float32),
        "next_observations": rng.normal(size=(n, 17)).astype(np.float32),
        "rewards": rng.normal(0, 0.1, (n, 1)).astype(np.float32),
        "terminals": (rng.random((n, 1)) < 0.05).astype(np.float32),
    }
    sampler = None
    if args.replay_device == "training":
        if not hasattr(agent.ops, "sample"):
            p.error("the selected repository does not support resident replay")
        data, sampler = agent.ops.batch(data), agent.ops.sample
    buffers = [ReplayBuffer(data, seed) for seed in (1026, 100020)]
    for buffer in buffers:
        if sampler is not None:
            buffer.sampler = sampler
    initial_state = agent.state
    initial_rng = copy.deepcopy(agent.rng.bit_generator.state)
    initial_buffer_rng = [copy.deepcopy(b.rng.bit_generator.state) for b in buffers]
    step0 = c.get("meta_warmup_steps", 0)
    # Activate the meta branch with initial parameters; not a post-warmup model.

    def reset():
        agent.state, agent.steps = initial_state, step0
        agent.rng.bit_generator.state = copy.deepcopy(initial_rng)
        for b, state in zip(buffers, initial_buffer_rng):
            b.rng.bit_generator.state = copy.deepcopy(state)

    def updates(count):
        for _ in range(count):
            step = agent.steps + 1
            batch = buffers[0].sample(c["batch_size"])
            outer = (
                buffers[1].sample(c.get("outer_batch_size", c["batch_size"]))
                if agent.meta_due(step)
                else None
            )
            if deferred:
                agent.update(batch, outer, return_metrics=False)
                if agent.steps % args.log_every == 0:
                    agent.get_metrics()
            else:
                agent.update(batch, outer)
        if deferred:
            agent.get_metrics()
        jax.block_until_ready(agent.state)

    reset()
    updates(c["meta_interval"] * 2)
    reset()
    observations = rng.normal(size=(64, 17)).astype(np.float32)
    for obs in observations[:20]:
        agent.act(obs)
    inference_times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        for i in range(400):
            agent.act(observations[i % len(observations)])
        inference_times.append(1000 * (time.perf_counter() - start) / 400)
    update_times = []
    for _ in range(args.repeats):
        reset()
        start = time.perf_counter()
        updates(args.steps)
        update_times.append(1000 * (time.perf_counter() - start) / args.steps)

    arrays = jax.device_get(flatten(agent.state))
    arrays["probe_actions"] = agent.act(observations)
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("Non-finite benchmark final state or probe actions")
    state_path = args.output.with_suffix(".npz")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(state_path, **arrays)
    result = {
        "algorithm": args.algorithm,
        "repository": str(root),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        ),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "device": str(agent.ops.device),
        "replay_device": args.replay_device,
        "hidden_dim": c["hidden_dim"],
        "batch_size": c["batch_size"],
        "steps": args.steps,
        "repeats": args.repeats,
        "log_every": args.log_every,
        "deferred_metrics": deferred,
        "initial_step_counter": step0,
        "model_seed": 17,
        "data_seed": 170017,
        "replay_seeds": [1026, 100020],
        "inference_ms": {
            "median": statistics.median(inference_times),
            "trials": inference_times,
        },
        "update_ms": {
            "median": statistics.median(update_times),
            "trials": update_times,
        },
        "jit_variants": agent._compiled_step._cache_size(),
        "final_agent_rng": agent.rng.bit_generator.state,
        "final_replay_rng": [b.rng.bit_generator.state for b in buffers],
        "scope": "Synthetic replay; timed sampling, noise, device placement, updates and metric checks. Compilation, initialization, checkpoint IO and environment evaluation excluded. IQL step counter starts at warmup to exercise meta; all parameters are initial parameters.",
        "final_state": state_path.name,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
