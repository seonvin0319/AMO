"""Run a release algorithm with either native backend."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import yaml

from common.agent import make_agent
from common.config import ALGORITHMS, load_config
from common.data import ReplayBuffer, fingerprint, load_dataset, preprocess


def evaluate(agent, env_name, mean, std, seed, episodes, repeats=1, seed_stride=1):
    import d4rl  # noqa: F401
    import gym

    env = gym.make(env_name)
    scores = []
    try:
        repeat_returns = []
        for repeat in range(repeats):
            env.seed(seed + repeat * seed_stride)
            env.action_space.seed(seed + repeat * seed_stride)
            round_scores = []
            for _ in range(episodes):
                obs, done, total = env.reset(), False, 0.0
                while not done:
                    obs, reward, done, _ = env.step(agent.act((obs - mean) / std))
                    total += reward
                scores.append(total)
                round_scores.append(total)
            repeat_returns.append(float(np.mean(round_scores)))
        return {
            "return_mean": float(np.mean(scores)),
            "normalized_score": float(100 * env.get_normalized_score(np.mean(scores))),
            "episodes": episodes * repeats,
            "episodes_per_repeat": episodes,
            "repeats": repeats,
            "repeat_returns": repeat_returns,
            "eval_seed": seed,
            "seed_stride": seed_stride,
        }
    finally:
        env.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    p.add_argument("--backend", choices=("torch", "jax"), required=True)
    p.add_argument("--env", default="halfcheetah-medium-v2")
    p.add_argument(
        "--dataset", help="Raw D4RL HDF5 or transition NPZ; omit to load D4RL"
    )
    p.add_argument("--config", help="Override the algorithm YAML path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--resume", help="A full .npz checkpoint, including optimizer and RNG state"
    )
    p.add_argument(
        "--steps",
        type=int,
        help="Stop after this many total steps without changing schedules",
    )
    p.add_argument(
        "--no-eval",
        action="store_true",
        help="Train from arrays without a MuJoCo installation",
    )
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--print-config", action="store_true")
    args = p.parse_args(argv)
    c = load_config(args.algorithm, args.env, args.config)
    if args.print_config:
        print(yaml.safe_dump(c, sort_keys=False))
        return
    steps = c["max_steps"] if args.steps is None else args.steps
    if not 0 < steps <= c["max_steps"] or args.log_every <= 0 or args.save_every <= 0:
        p.error("steps must be in (0, max_steps]; log/save intervals must be positive")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        p.error("Output directory is not empty; use a new directory or --resume")
    raw = load_dataset(args.dataset, args.env, rebrac=args.algorithm == "rebrac")
    data_hash = fingerprint(raw)
    data, mean, std = preprocess(raw, c)
    del raw
    buffer = ReplayBuffer(data, args.seed + 1009)
    outer_buffer = ReplayBuffer(data, args.seed + 100003)
    agent = make_agent(
        args.algorithm,
        args.backend,
        data["observations"].shape[1],
        data["actions"].shape[1],
        c,
        seed=args.seed,
        device=args.device,
    )
    if args.resume:
        extra = agent.load(args.resume)
        if (
            extra.get("dataset_sha256") != data_hash
            or extra.get("env") != args.env
            or extra.get("seed") != args.seed
        ):
            raise ValueError(
                "Resume dataset, environment or seed does not match checkpoint"
            )
        buffer.rng.bit_generator.state = extra["buffer_rng"]
        outer_buffer.rng.bit_generator.state = extra["outer_rng"]
        for name in ("metrics.jsonl", "eval.jsonl"):
            log = output / name
            if log.exists():
                rows = [
                    json.loads(line)
                    for line in log.read_text().splitlines()
                    if line.strip()
                ]
                if rows and max(r["step"] for r in rows) > agent.steps:
                    raise ValueError(
                        "Output contains steps after this checkpoint; resume into a new output directory"
                    )
        if agent.steps >= steps:
            raise ValueError("Requested stop step must be after the checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
    (output / "run_meta.json").write_text(
        json.dumps(
            {
                "algorithm": args.algorithm,
                "backend": args.backend,
                "env": args.env,
                "seed": args.seed,
                "dataset_sha256": data_hash,
                "python": platform.python_version(),
                "device": args.device,
            },
            indent=2,
        )
    )
    np.savez(output / "normalization.npz", mean=mean, std=std)

    def save():
        extra = {
            "env": args.env,
            "seed": args.seed,
            "dataset_sha256": data_hash,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "buffer_rng": buffer.rng.bit_generator.state,
            "outer_rng": outer_buffer.rng.bit_generator.state,
        }
        agent.save(output / "checkpoint.npz", extra)

    started = time.monotonic()
    try:
        while agent.steps < steps:
            batch = buffer.sample(c["batch_size"])
            is_meta = agent.meta_due(agent.steps + 1)
            outer = (
                outer_buffer.sample(c.get("outer_batch_size", c["batch_size"]))
                if args.algorithm in ("td3_amo", "iql_amo") and is_meta
                else None
            )
            metrics = agent.update(batch, outer)
            if agent.steps % args.log_every == 0 or agent.steps == steps:
                row = {"step": agent.steps, **metrics}
                with (output / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row), flush=True)
            eval_due = (
                agent.steps >= c["eval_first_step"]
                and (agent.steps - c["eval_first_step"]) % c["eval_freq"] == 0
            )
            final = agent.steps == c["max_steps"]
            if not args.no_eval and (eval_due or final):
                row = {
                    "step": agent.steps,
                    **evaluate(
                        agent,
                        args.env,
                        mean,
                        std,
                        args.seed if c["eval_seed"] is None else c["eval_seed"],
                        c["eval_episodes"],
                        c["final_eval_repeats"] if final else 1,
                        seed_stride=0 if args.algorithm == "iql_amo" else 1,
                    ),
                }
                with (output / "eval.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if agent.steps % args.save_every == 0:
                save()
    except KeyboardInterrupt:
        save()
        raise
    save()
    print(
        f"Completed {agent.steps} steps in {time.monotonic()-started:.1f}s; checkpoint: {output / 'checkpoint.npz'}"
    )


if __name__ == "__main__":
    main()
