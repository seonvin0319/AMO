#!/usr/bin/env python3
"""Train a frozen Gaussian behavior density μ_hat via data BC for iql_amo_qweight.

Uses the same tanh-mean + shared log_std Gaussian as IQL actors (see
``Agent.log_prob``). Saves a checkpoint whose ``p/behavior`` (and ``p/actor``
alias) can be loaded with ``train.py --behavior-path``.

BC wall time / steps are recorded separately from RL.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from common.agent import make_agent
from common.config import load_config
from common.data import ReplayBuffer, fingerprint, load_dataset, preprocess
from common.tree import flatten, map_tree


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="hopper-medium-expert-v2")
    p.add_argument("--dataset")
    p.add_argument("--config", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bc-seed", type=int, default=None, help="RNG for BC only")
    p.add_argument("--device", default="cpu")
    p.add_argument("--backend", default="jax", choices=("jax", "torch"))
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--output", required=True)
    p.add_argument("--val-frac", type=float, default=0.05)
    args = p.parse_args()
    bc_seed = args.seed + 7919 if args.bc_seed is None else args.bc_seed

    c = load_config("iql_amo_qweight", args.env, args.config)
    raw = load_dataset(args.dataset, args.env, rebrac=False)
    data_hash = fingerprint(raw)
    data, mean, std = preprocess(raw, c)
    n = len(data["actions"])
    rng = np.random.default_rng(bc_seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train = {k: v[train_idx] for k, v in data.items()}
    val = {k: v[val_idx] for k, v in data.items()}
    buffer = ReplayBuffer(train, bc_seed + 1009)

    agent = make_agent(
        "iql_amo_qweight",
        args.backend,
        data["observations"].shape[1],
        data["actions"].shape[1],
        c,
        seed=bc_seed,
        device=args.device,
    )

    def bc_step(state, batch):
        obs, act = batch["observations"], batch["actions"]

        def loss_fn(p):
            return agent.behavior_bc_loss(p, obs, act)

        return agent.update_network(state, "behavior", loss_fn, args.lr)

    if args.backend == "jax":
        import jax

        bc_step = jax.jit(bc_step)

    # Train only behavior via BC; never touch π_E/π_B/Q/V.
    started = time.monotonic()
    losses = []
    for step in range(1, args.steps + 1):
        batch = agent.ops.batch(buffer.sample(args.batch_size))
        agent.state, loss = bc_step(agent.state, batch)
        if step % 2000 == 0 or step == args.steps:
            loss_f = float(agent.ops.numpy(loss))
            losses.append(loss_f)
            print(json.dumps({"bc_step": step, "bc_loss": loss_f}), flush=True)

    # Validation log-prob stats
    v_obs = agent.ops.array(val["observations"])
    v_act = agent.ops.array(val["actions"])
    lp = agent.ops.numpy(agent.log_prob(agent.state["p"]["behavior"], v_obs, v_act))
    finite = np.isfinite(lp).all()
    val_stats = {
        "n_val": int(n_val),
        "log_prob_finite": bool(finite),
        "log_prob_mean": float(np.mean(lp)),
        "log_prob_std": float(np.std(lp)),
        "log_prob_min": float(np.min(lp)),
        "log_prob_max": float(np.max(lp)),
    }
    if not finite:
        raise SystemExit(f"Non-finite behavior log_prob on val: {val_stats}")

    elapsed = time.monotonic() - started
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # Save full agent state but document that only behavior is meaningful.
    # Also duplicate under actor for loaders that expect actor-shaped trees.
    state = agent.state
    state = {
        **state,
        "p": {**state["p"], "actor": map_tree(lambda x: x, state["p"]["behavior"])},
    }
    agent.state = state
    agent.save(
        out / "behavior.npz",
        extra={
            "kind": "iql_amo_qweight_behavior_bc",
            "env": args.env,
            "dataset_sha256": data_hash,
            "bc_steps": args.steps,
            "bc_seed": bc_seed,
            "bc_lr": args.lr,
            "bc_wall_sec": elapsed,
            "val": val_stats,
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
    )
    (out / "behavior_meta.json").write_text(
        json.dumps(
            {
                "env": args.env,
                "dataset_sha256": data_hash,
                "bc_steps": args.steps,
                "bc_seed": bc_seed,
                "bc_lr": args.lr,
                "bc_wall_sec": elapsed,
                "bc_loss_last": losses[-1],
                "val": val_stats,
                "note": "Untuned diagnostic BC budget (50k default); not a tuned μ_hat.",
            },
            indent=2,
        )
        + "\n"
    )
    (out / "config.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
    print(
        json.dumps(
            {
                "status": "ok",
                "path": str(out / "behavior.npz"),
                "bc_wall_sec": elapsed,
                "val": val_stats,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
