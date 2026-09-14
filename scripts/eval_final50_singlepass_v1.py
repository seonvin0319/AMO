#!/usr/bin/env python3
"""Final-only evaluator for protocol final50_singlepass_v1.

- 50 episodes in one continuous pass (repeats=1, episodes_per_repeat=50)
- Seed Python/NumPy/env/action_space once at start; do not re-seed per episode
- Write eval_final50_v1.jsonl + posthoc_final50_v1/ markers (never touch eval.jsonl
  completion logic of the stock reaper; never delete checkpoints)
- Skip when (checkpoint_sha256, protocol, eval_seed, episodes, eval_code_rev) match
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.agent import make_agent  # noqa: E402

PROTOCOL = "final50_singlepass_v1"
TAG = "final50_singlepass_v1"
EPISODES = 50
REPEATS = 1
EPISODES_PER_REPEAT = 50
SEED_STRIDE = 1  # recorded; unused when repeats==1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_host_env() -> None:
    # Force host paths (setdefault in stock script points at another machine).
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
    os.environ["D4RL_DATASET_DIR"] = "/home/shchoi/.d4rl/datasets"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_PY_MUJOCO_PATH"] = str(
        Path.home() / ".mujoco" / "mujoco210"
    )
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"


def code_revision() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            )
            .strip()
        )
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_run_meta(output: Path) -> dict:
    meta_path = output / "run_meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def resolve_final_ckpt(output: Path, budget: int = 1_000_000) -> tuple[int, Path] | None:
    step_path = output / "checkpoints" / f"step_{budget}.npz"
    if step_path.exists():
        return budget, step_path
    # Prefer highest step_*.npz >= budget
    ckpt_dir = output / "checkpoints"
    if ckpt_dir.is_dir():
        found = []
        for path in ckpt_dir.iterdir():
            if path.name.startswith("step_") and path.name.endswith(".npz"):
                try:
                    step = int(path.name[len("step_") : -len(".npz")])
                except ValueError:
                    continue
                if step >= budget:
                    found.append((step, path))
        if found:
            found.sort()
            return found[-1]
    ck = output / "checkpoint.npz"
    if ck.exists():
        with np.load(ck, allow_pickle=False) as data:
            meta = json.loads(str(data["__metadata__"]))
        step = int(meta.get("steps") or 0)
        if step >= budget:
            return step, ck
    return None


def already_done(output: Path, key: dict) -> bool:
    marker = output / "posthoc_final50_v1" / "final.json"
    if marker.exists():
        try:
            payload = json.loads(marker.read_text())
            if payload.get("ok") and payload.get("dedupe_key") == key:
                return True
            row = payload.get("row") or {}
            if (
                payload.get("ok")
                and row.get("protocol") == PROTOCOL
                and row.get("checkpoint_sha256") == key.get("checkpoint_sha256")
                and int(row.get("episodes") or 0) == EPISODES
                and int(row.get("repeats") or 0) == REPEATS
                and int(row.get("eval_seed") or -1) == int(key.get("eval_seed", -2))
                and row.get("eval_code_revision") == key.get("eval_code_revision")
            ):
                return True
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    ej = output / "eval_final50_v1.jsonl"
    if not ej.exists():
        return False
    for line in ej.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            row.get("protocol") == PROTOCOL
            and row.get("checkpoint_sha256") == key.get("checkpoint_sha256")
            and int(row.get("episodes") or 0) == EPISODES
            and int(row.get("repeats") or 0) == REPEATS
            and int(row.get("eval_seed") or -1) == int(key.get("eval_seed", -2))
            and row.get("eval_code_revision") == key.get("eval_code_revision")
            and row.get("status") == "done"
        ):
            return True
    return False


def evaluate_final50(
    agent,
    env_name: str,
    mean: np.ndarray,
    std: np.ndarray,
    eval_seed: int,
    episodes: int = EPISODES,
) -> dict:
    import d4rl  # noqa: F401
    import gym

    random.seed(eval_seed)
    np.random.seed(eval_seed)

    env = gym.make(env_name)
    try:
        env.seed(eval_seed)
        env.action_space.seed(eval_seed)

        episode_returns = []
        episode_lengths = []
        episode_success = []
        episode_terminals = []
        action_abs_max = []
        action_nan_count = 0
        action_clip_hits = 0
        action_count = 0
        low = np.asarray(env.action_space.low, dtype=np.float64)
        high = np.asarray(env.action_space.high, dtype=np.float64)

        for _ in range(episodes):
            obs = env.reset()
            done = False
            total = 0.0
            length = 0
            term_reason = "unknown"
            info = {}
            while not done:
                action = agent.act((np.asarray(obs, np.float32) - mean) / std)
                action = np.asarray(action, dtype=np.float64)
                if not np.all(np.isfinite(action)):
                    action_nan_count += 1
                action_abs_max.append(float(np.max(np.abs(action))))
                action_count += 1
                # saturation vs env bounds (epsilon)
                eps = 1e-6
                if np.any(action <= low + eps) or np.any(action >= high - eps):
                    action_clip_hits += 1
                obs, reward, done, info = env.step(action)
                total += float(reward)
                length += 1
                if done:
                    if bool(info.get("TimeLimit.truncated")):
                        term_reason = "timeout"
                    elif length >= getattr(env, "_max_episode_steps", 10**9):
                        term_reason = "timeout_inferred"
                    else:
                        term_reason = "terminal"
            success = float(total) > 0.0  # AntMaze sparse success proxy
            if "goal_achieved" in info:
                success = bool(info["goal_achieved"])
            elif "is_success" in info:
                success = bool(info["is_success"])
            episode_returns.append(float(total))
            episode_lengths.append(int(length))
            episode_success.append(bool(success))
            episode_terminals.append(term_reason)

        ret_mean = float(np.mean(episode_returns))
        normalized = float(100.0 * env.get_normalized_score(ret_mean))
        # Verify normalized score against mean of per-episode norms when defined.
        per_ep_norm = [
            float(100.0 * env.get_normalized_score(r)) for r in episode_returns
        ]
        return {
            "return_mean": ret_mean,
            "return_std": float(np.std(episode_returns, ddof=1))
            if len(episode_returns) > 1
            else 0.0,
            "normalized_score": normalized,
            "normalized_score_from_per_episode_mean": float(np.mean(per_ep_norm)),
            "episodes": episodes * REPEATS,
            "episodes_per_repeat": EPISODES_PER_REPEAT,
            "repeats": REPEATS,
            "repeat_returns": [ret_mean],
            "episode_returns": episode_returns,
            "episode_lengths": episode_lengths,
            "episode_success": episode_success,
            "episode_terminals": episode_terminals,
            "success_rate": float(np.mean(episode_success)),
            "eval_seed": int(eval_seed),
            "seed_stride": SEED_STRIDE,
            "action_abs_max_mean": float(np.mean(action_abs_max)) if action_abs_max else None,
            "action_abs_max_p99": float(np.quantile(action_abs_max, 0.99))
            if action_abs_max
            else None,
            "action_nonfinite_count": int(action_nan_count),
            "action_near_bound_steps": int(action_clip_hits),
            "action_steps": int(action_count),
        }
    finally:
        env.close()


def lock_path(output: Path) -> Path:
    dest = output / "posthoc_final50_v1"
    dest.mkdir(parents=True, exist_ok=True)
    return dest / ".eval.lock"


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict) -> None:
    """Append one JSONL row. Caller must already hold the run eval lock."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def eval_one_run(
    output: Path,
    device: str,
    force: bool,
    diag_fixed_obs: int,
    smoke_out: Path | None,
) -> dict:
    meta = load_run_meta(output)
    resolved = resolve_final_ckpt(output)
    if resolved is None:
        return {"status": "no_final_ckpt", "run": output.name}
    step, ckpt = resolved

    with np.load(ckpt, allow_pickle=False) as data:
        ck_meta = json.loads(str(data["__metadata__"]))
    c = ck_meta["config"]
    extra = ck_meta.get("extra") or {}
    environment = meta.get("env") or extra.get("env")
    seed = meta.get("seed")
    if seed is None:
        seed = extra.get("seed")
    algorithm = meta.get("algorithm") or c.get("algorithm")
    backend = meta.get("backend") or ck_meta.get("backend") or "jax"
    if environment is None or seed is None or algorithm is None:
        return {"status": "missing_meta", "run": output.name}

    seed = int(seed)
    eval_seed = seed if c.get("eval_seed") is None else int(c["eval_seed"])
    ckpt_sha = sha256_file(ckpt)
    rev = code_revision()
    dedupe_key = {
        "checkpoint_sha256": ckpt_sha,
        "protocol": PROTOCOL,
        "eval_seed": eval_seed,
        "episodes": EPISODES,
        "eval_code_revision": rev,
    }
    if not force and already_done(output, dedupe_key):
        return {
            "status": "skipped",
            "run": output.name,
            "reason": "dedupe",
            "dedupe_key": dedupe_key,
        }

    # Claim the run briefly, then release before the long rollout so nested
    # writers (and other workers) cannot deadlock on a non-reentrant flock.
    lock_file = lock_path(output)
    with lock_file.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            if not force and already_done(output, dedupe_key):
                return {
                    "status": "skipped",
                    "run": output.name,
                    "reason": "dedupe_locked",
                }
            claim = {
                "ok": False,
                "status": "running",
                "run": output.name,
                "step": step,
                "checkpoint_sha256": ckpt_sha,
                "protocol": PROTOCOL,
                "at": utc_now(),
            }
            atomic_write_json(output / "posthoc_final50_v1" / "RUNNING.json", claim)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    try:
        norm_path = output / "normalization.npz"
        if not norm_path.exists():
            raise FileNotFoundError(f"missing {norm_path}")
        norm = np.load(norm_path)
        mean = np.asarray(norm["mean"], dtype=np.float32)
        std = np.asarray(norm["std"], dtype=np.float32)
        if "mean" in extra and "std" in extra:
            em = np.asarray(extra["mean"], dtype=np.float32)
            es = np.asarray(extra["std"], dtype=np.float32)
            if em.shape == mean.shape and not np.allclose(em, mean):
                raise ValueError("normalization.npz mean != checkpoint extra mean")
            if es.shape == std.shape and not np.allclose(es, std):
                raise ValueError("normalization.npz std != checkpoint extra std")

        agent = make_agent(
            algorithm,
            backend,
            int(ck_meta["observation_dim"]),
            int(ck_meta["action_dim"]),
            c,
            seed=seed,
            device=device,
        )
        agent.load(str(ckpt))
        if int(agent.steps) != int(ck_meta.get("steps") or step):
            raise ValueError(
                f"checkpoint step mismatch: agent={agent.steps} meta={ck_meta.get('steps')} file={step}"
            )

        fixed_obs_actions = None
        if diag_fixed_obs > 0:
            rng = np.random.default_rng(eval_seed)
            obs_dim = int(ck_meta["observation_dim"])
            raw = rng.normal(size=(diag_fixed_obs, obs_dim)).astype(np.float32)
            normed = (raw - mean) / std
            acts = np.asarray(agent.act(normed), dtype=np.float64)
            fixed_obs_actions = {
                "obs_seed": eval_seed,
                "n": diag_fixed_obs,
                "action_mean": acts.mean(axis=0).tolist(),
                "action_std": acts.std(axis=0).tolist(),
                "action_abs_max": float(np.max(np.abs(acts))),
                "actions_head": acts[: min(5, len(acts))].tolist(),
            }

        t0 = time.time()
        result = evaluate_final50(agent, environment, mean, std, eval_seed, EPISODES)
        elapsed = time.time() - t0

        row = {
            "status": "done",
            "protocol": PROTOCOL,
            "tag": TAG,
            "step": int(agent.steps),
            "device": device,
            "backend": backend,
            "algorithm": algorithm,
            "env": environment,
            "seed": seed,
            "checkpoint": str(ckpt),
            "checkpoint_sha256": ckpt_sha,
            "eval_code_revision": rev,
            "evaluated_at": utc_now(),
            "elapsed_sec": elapsed,
            "config_T_E": c.get("T_E"),
            "config_T_B": c.get("T_B"),
            "config_T_lr": c.get("T_lr"),
            "config_reward_transform": c.get("reward_transform"),
            "norm_mean_l2": float(np.linalg.norm(mean)),
            "norm_std_mean": float(np.mean(std)),
            **result,
        }
        if fixed_obs_actions is not None:
            row["fixed_obs_actions"] = fixed_obs_actions

        if abs(row["normalized_score"] - 100.0 * row["return_mean"]) > 1e-6:
            row["normalized_score_check"] = "mismatch_vs_100x_return_mean"
        else:
            row["normalized_score_check"] = "ok_100x_return_mean"

        with lock_file.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                if not force and already_done(output, dedupe_key):
                    return {
                        "status": "skipped",
                        "run": output.name,
                        "reason": "dedupe_after_eval",
                    }
                target_jsonl = (
                    smoke_out / f"{output.name}.jsonl"
                    if smoke_out is not None
                    else output / "eval_final50_v1.jsonl"
                )
                if smoke_out is not None:
                    smoke_out.mkdir(parents=True, exist_ok=True)
                    target_jsonl.write_text(json.dumps(row, allow_nan=False) + "\n")
                    atomic_write_json(
                        smoke_out / f"{output.name}.marker.json",
                        {"ok": True, "dedupe_key": dedupe_key, "row": row},
                    )
                else:
                    append_jsonl(target_jsonl, row)
                    dest = output / "posthoc_final50_v1"
                    atomic_write_json(
                        dest / f"step_{int(agent.steps)}.json",
                        {"ok": True, "dedupe_key": dedupe_key, "row": row},
                    )
                    atomic_write_json(
                        dest / "final.json",
                        {"ok": True, "dedupe_key": dedupe_key, "row": row},
                    )
                running = output / "posthoc_final50_v1" / "RUNNING.json"
                if running.exists():
                    running.unlink()
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

        print(
            json.dumps(
                {
                    "status": "done",
                    "run": output.name,
                    "normalized_score": row["normalized_score"],
                    "success_rate": row["success_rate"],
                    "episodes": row["episodes"],
                    "repeats": row["repeats"],
                    "elapsed_sec": elapsed,
                    "smoke": smoke_out is not None,
                }
            ),
            flush=True,
        )
        return {"status": "done", "run": output.name, "row": row}
    except Exception as exc:
        fail = {
            "ok": False,
            "status": "failed",
            "run": output.name,
            "step": step,
            "checkpoint": str(ckpt),
            "error": f"{type(exc).__name__}: {exc}",
            "at": utc_now(),
            "protocol": PROTOCOL,
        }
        dest = output / "posthoc_final50_v1"
        dest.mkdir(parents=True, exist_ok=True)
        atomic_write_json(dest / f"step_{step}.FAILED.json", fail)
        running = dest / "RUNNING.json"
        if running.exists():
            running.unlink()
        print(json.dumps(fail), flush=True)
        return fail


def discover_run_dirs(paths: list[Path]) -> list[Path]:
    out = []
    seen = set()
    for p in paths:
        p = p.resolve()
        if p.is_dir() and (
            (p / "checkpoints").is_dir() or (p / "checkpoint.npz").exists()
        ):
            if p not in seen:
                out.append(p)
                seen.add(p)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=Path,
        default=[],
        help="Absolute run directory (repeatable)",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="If set, evaluate all final-ckpt runs under this root (1 level)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--diag-fixed-obs",
        type=int,
        default=8,
        help="Sample N fixed random obs and record actions (0 to disable)",
    )
    parser.add_argument(
        "--smoke-out",
        type=Path,
        default=None,
        help="Write results here instead of run dir (smoke / diagnosis)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max runs to evaluate this invocation (0=all)",
    )
    args = parser.parse_args(argv)

    configure_host_env()
    runs: list[Path] = list(args.run)
    if args.runs_root is not None:
        root = args.runs_root.resolve()
        for child in sorted(root.iterdir()):
            if child.is_dir() and ".bak" not in child.name:
                runs.append(child)
    runs = discover_run_dirs(runs)
    if args.limit > 0:
        runs = runs[: args.limit]

    summary = {"at": utc_now(), "protocol": PROTOCOL, "runs": []}
    for output in runs:
        rec = eval_one_run(
            output,
            args.device,
            args.force,
            args.diag_fixed_obs,
            args.smoke_out,
        )
        summary["runs"].append(
            {k: rec.get(k) for k in ("status", "run", "reason", "error") if k in rec}
            | (
                {
                    "normalized_score": (rec.get("row") or {}).get("normalized_score"),
                    "success_rate": (rec.get("row") or {}).get("success_rate"),
                }
                if rec.get("status") == "done"
                else {}
            )
        )
    print(json.dumps(summary, indent=2))
    failed = sum(1 for r in summary["runs"] if r.get("status") == "failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
