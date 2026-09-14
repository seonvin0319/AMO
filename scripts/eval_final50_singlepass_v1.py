#!/usr/bin/env python3
"""final50_singlepass_v1 CPU evaluator for finished IQL+AMO (and compatible) runs.

Protocol:
  - Seed Python random, NumPy, env, and action_space ONCE at start.
  - Run 50 episodes continuously without reseeding between episodes.
  - Record episodes=50, episodes_per_repeat=50, repeats=1, seed_stride=1.
  - Never delete checkpoints.
  - Write run/eval_final50_v1.jsonl + FINAL50_SINGLEPASS_V1_DONE.json under lock.
  - Skip when the same (ckpt_sha256, protocol, eval_seed, episodes, code_rev) exists.
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.agent import make_agent  # noqa: E402

PROTOCOL = "final50_singlepass_v1"
MARKER_NAME = "FINAL50_SINGLEPASS_V1_DONE.json"
EVAL_LOG_NAME = "eval_final50_v1.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_cpu_env() -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    os.environ.setdefault("D4RL_DATASET_DIR", "/home/svcho/.d4rl/datasets")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", "/home/svcho/.mujoco/mujoco210")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_revision() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip()
        )
    except Exception:
        return "unknown"


def load_ckpt_meta(ckpt: Path) -> Dict[str, Any]:
    with np.load(ckpt, allow_pickle=False) as data:
        return json.loads(str(data["__metadata__"]))


def pick_ckpt(run: Path, step: Optional[int] = None) -> Tuple[Path, int]:
    if step is not None:
        path = run / "checkpoints" / f"step_{int(step)}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        return path, int(step)
    preferred = run / "checkpoints" / "step_1000000.npz"
    if preferred.exists():
        return preferred, 1_000_000
    alias = run / "checkpoint.npz"
    if alias.exists():
        meta = load_ckpt_meta(alias)
        st = int(meta.get("steps") or 0)
        if st >= 1_000_000:
            return alias, st
        raise FileNotFoundError(f"checkpoint.npz step={st} is not final")
    raise FileNotFoundError(f"no final checkpoint under {run}")


def eval_key(
    ckpt_sha: str, protocol: str, eval_seed: int, episodes: int, code_rev: str
) -> Dict[str, Any]:
    return {
        "checkpoint_sha256": ckpt_sha,
        "protocol": protocol,
        "eval_seed": int(eval_seed),
        "episodes": int(episodes),
        "code_revision": code_rev,
    }


def keys_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    for k in ("checkpoint_sha256", "protocol", "eval_seed", "episodes", "code_revision"):
        if a.get(k) != b.get(k):
            return False
    return True


def find_existing(run: Path, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    marker = run / MARKER_NAME
    if marker.exists():
        try:
            payload = json.loads(marker.read_text())
            if payload.get("ok") and keys_equal(payload.get("key") or {}, key):
                return payload
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    log = run / EVAL_LOG_NAME
    if not log.exists():
        return None
    last = None
    for line in log.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if keys_equal(row.get("key") or {}, key) and row.get("ok") is True:
            last = row
    return last


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def append_jsonl_locked(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(row, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def evaluate_final50_singlepass(
    agent,
    env_name: str,
    mean: np.ndarray,
    std: np.ndarray,
    eval_seed: int,
    episodes: int = 50,
) -> Dict[str, Any]:
    """Seed once; 50 continuous episodes; return per-episode returns (+ success)."""
    import d4rl  # noqa: F401
    import gym

    random.seed(eval_seed)
    np.random.seed(eval_seed)

    env = gym.make(env_name)
    try:
        env.seed(eval_seed)
        env.action_space.seed(eval_seed)

        episode_returns: List[float] = []
        episode_lengths: List[int] = []
        episode_success: List[Optional[bool]] = []
        action_abs_max = 0.0
        action_sat_frac_sum = 0.0
        n_actions = 0
        finite_ok = True

        for _ in range(episodes):
            obs = env.reset()
            done = False
            total = 0.0
            length = 0
            while not done:
                obs_n = (np.asarray(obs, np.float32) - mean) / std
                action = np.asarray(agent.act(obs_n), dtype=np.float64)
                if not np.all(np.isfinite(action)):
                    finite_ok = False
                abs_a = np.abs(action)
                action_abs_max = max(action_abs_max, float(abs_a.max(initial=0.0)))
                action_sat_frac_sum += float(np.mean(abs_a >= 0.999))
                n_actions += 1
                # Execution policy is pi_E via agent.act → state["p"]["actor"].
                obs, reward, done, info = env.step(action)
                if not np.isfinite(reward):
                    finite_ok = False
                total += float(reward)
                length += 1
            episode_returns.append(float(total))
            episode_lengths.append(int(length))
            success = None
            if isinstance(info, dict):
                if "success" in info:
                    success = bool(info["success"])
                elif env_name.startswith("antmaze"):
                    # D4RL antmaze: sparse reward 1.0 on success terminal.
                    success = bool(total > 0.0)
            episode_success.append(success)

        mean_return = float(np.mean(episode_returns))
        normalized = float(100.0 * env.get_normalized_score(mean_return))
        # Recompute normalized from mean return for audit trail.
        normalized_check = float(100.0 * env.get_normalized_score(np.mean(episode_returns)))
        if abs(normalized - normalized_check) > 1e-9:
            raise ValueError("normalized_score recomputation mismatch")

        return {
            "return_mean": mean_return,
            "normalized_score": normalized,
            "episodes": int(episodes),
            "episodes_per_repeat": int(episodes),
            "repeats": 1,
            "repeat_returns": [mean_return],
            "episode_returns": episode_returns,
            "episode_lengths": episode_lengths,
            "episode_success": episode_success,
            "success_rate": (
                float(np.mean([1.0 if s else 0.0 for s in episode_success]))
                if all(s is not None for s in episode_success)
                else None
            ),
            "eval_seed": int(eval_seed),
            "seed_stride": 1,
            "policy": "pi_E",
            "finite_ok": bool(finite_ok),
            "action_abs_max": float(action_abs_max),
            "action_sat_frac": float(action_sat_frac_sum / max(n_actions, 1)),
            "mean_episode_length": float(np.mean(episode_lengths)),
        }
    finally:
        env.close()


def resolve_run_identity(run: Path, ck_meta: Dict[str, Any]) -> Tuple[str, int, str, str]:
    meta: Dict[str, Any] = {}
    mp = run / "run_meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    env = meta.get("env") or (ck_meta.get("extra") or {}).get("env") or cfg.get("env")
    seed = meta.get("seed")
    if seed is None:
        seed = (ck_meta.get("extra") or {}).get("seed", cfg.get("seed"))
    algorithm = meta.get("algorithm") or ck_meta.get("config", {}).get("algorithm")
    backend = meta.get("backend") or ck_meta.get("backend") or "jax"
    if env is None or seed is None or algorithm is None:
        raise ValueError(f"missing env/seed/algorithm for {run}")
    return str(env), int(seed), str(algorithm), str(backend)


def evaluate_run(
    run: Path,
    *,
    episodes: int,
    device: str,
    force: bool,
    step: Optional[int],
    out_subdir: Optional[str],
) -> Dict[str, Any]:
    run = run.resolve()
    ckpt, ckpt_step = pick_ckpt(run, step=step)
    ck_meta = load_ckpt_meta(ckpt)
    env, seed, algorithm, backend = resolve_run_identity(run, ck_meta)
    cfg = ck_meta["config"]
    eval_seed = seed if cfg.get("eval_seed") is None else int(cfg["eval_seed"])
    ckpt_sha = sha256_file(ckpt)
    rev = code_revision()
    key = eval_key(ckpt_sha, PROTOCOL, eval_seed, episodes, rev)

    out_run = run / out_subdir if out_subdir else run
    out_run.mkdir(parents=True, exist_ok=True)

    # For diagnosis subdir, still key off same protocol name + step in row.
    existing = None if force else find_existing(out_run if out_subdir else run, key)
    # When writing to subdir, look there; for standard path look in run.
    if out_subdir:
        # reuse skip only if marker in subdir matches
        existing = None if force else find_existing(out_run, key)
    else:
        existing = None if force else find_existing(run, key)

    if existing is not None:
        return {
            "status": "skipped",
            "run": str(run),
            "normalized_score": (existing.get("row") or existing).get("normalized_score"),
            "key": key,
        }

    # Lock around claim+write so two workers cannot evaluate the same run.
    lock_path = (out_run if out_subdir else run) / ".final50_singlepass_v1.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        # Re-check under lock.
        existing2 = None if force else find_existing(out_run if out_subdir else run, key)
        if existing2 is not None:
            return {
                "status": "skipped",
                "run": str(run),
                "normalized_score": (existing2.get("row") or existing2).get(
                    "normalized_score"
                ),
                "key": key,
            }

        norm_path = run / "normalization.npz"
        if not norm_path.exists():
            raise FileNotFoundError(norm_path)
        with np.load(norm_path) as norm:
            mean = np.asarray(norm["mean"], dtype=np.float32)
            std = np.asarray(norm["std"], dtype=np.float32)

        # Optional consistency: if checkpoint embeds mean/std arrays, compare.
        with np.load(ckpt, allow_pickle=False) as raw:
            names = set(raw.files)
            if "normalization_mean" in names and "normalization_std" in names:
                cm = np.asarray(raw["normalization_mean"], dtype=np.float32)
                cs = np.asarray(raw["normalization_std"], dtype=np.float32)
                if cm.shape == mean.shape and cs.shape == std.shape:
                    if not (np.allclose(cm, mean) and np.allclose(cs, std)):
                        raise ValueError("normalization.npz disagrees with checkpoint arrays")

        agent = make_agent(
            algorithm,
            backend,
            int(ck_meta["observation_dim"]),
            int(ck_meta["action_dim"]),
            cfg,
            seed=seed,
            device=device,
        )
        agent.load(str(ckpt))
        if int(agent.steps) != int(ckpt_step) and step is None:
            # Allow checkpoint.npz alias where filename step differs but metadata matches.
            if int(agent.steps) < 1_000_000:
                raise ValueError(
                    f"checkpoint step mismatch: agent={agent.steps} expected={ckpt_step}"
                )

        t0 = time.time()
        try:
            result = evaluate_final50_singlepass(
                agent, env, mean, std, eval_seed, episodes=episodes
            )
        except Exception as exc:  # noqa: BLE001
            fail = {
                "ok": False,
                "protocol": PROTOCOL,
                "run_dir": str(run),
                "checkpoint": str(ckpt),
                "error": f"{type(exc).__name__}: {exc}",
                "at": utc_now(),
                "key": key,
            }
            append_jsonl_locked((out_run if out_subdir else run) / EVAL_LOG_NAME, fail)
            atomic_write_json(
                (out_run if out_subdir else run) / f"{MARKER_NAME}.FAILED.json", fail
            )
            return {"status": "failed", "run": str(run), "error": fail["error"]}

        if not result.get("finite_ok", True):
            fail = {
                "ok": False,
                "protocol": PROTOCOL,
                "run_dir": str(run),
                "checkpoint": str(ckpt),
                "error": "non_finite_action_or_reward",
                "at": utc_now(),
                "key": key,
                "partial": result,
            }
            append_jsonl_locked((out_run if out_subdir else run) / EVAL_LOG_NAME, fail)
            atomic_write_json(
                (out_run if out_subdir else run) / f"{MARKER_NAME}.FAILED.json", fail
            )
            return {"status": "failed", "run": str(run), "error": fail["error"]}

        row = {
            "ok": True,
            "protocol": PROTOCOL,
            "step": int(agent.steps),
            "tag": PROTOCOL,
            "device": device,
            "backend": backend,
            "algorithm": algorithm,
            "env": env,
            "seed": seed,
            "checkpoint": str(ckpt),
            "checkpoint_sha256": ckpt_sha,
            "normalization": str(norm_path),
            "normalization_sha256": sha256_file(norm_path),
            "evaluated_at": utc_now(),
            "elapsed_sec": round(time.time() - t0, 3),
            "code_revision": rev,
            "code_root": str(ROOT),
            "key": key,
            **result,
        }
        target = out_run if out_subdir else run
        append_jsonl_locked(target / EVAL_LOG_NAME, row)
        atomic_write_json(
            target / MARKER_NAME,
            {"ok": True, "protocol": PROTOCOL, "key": key, "row": row},
        )
        return {
            "status": "done",
            "run": str(run),
            "env": env,
            "seed": seed,
            "normalized_score": row["normalized_score"],
            "elapsed_sec": row["elapsed_sec"],
        }


def load_manifest_runs(manifest: Path) -> List[Path]:
    data = json.loads(manifest.read_text())
    return [Path(r["run_dir"]) for r in data.get("runs", [])]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Single run directory")
    parser.add_argument("--manifest", type=Path, help="Manifest JSON from build_* script")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--step", type=int, default=None, help="Evaluate mid checkpoint step")
    parser.add_argument(
        "--out-subdir",
        default=None,
        help="Write markers under run/<subdir> (for diagnostics; keeps finals separate)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Evaluate at most N runs")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--only-env", default=None)
    parser.add_argument("--only-seed", type=int, default=None)
    args = parser.parse_args(argv)

    configure_cpu_env()

    runs: List[Path] = []
    if args.run_dir:
        runs = [args.run_dir]
    elif args.manifest:
        runs = load_manifest_runs(args.manifest)
    else:
        parser.error("provide --run-dir or --manifest")

    if args.only_env or args.only_seed is not None:
        filtered = []
        for run in runs:
            meta = {}
            mp = run / "run_meta.json"
            if mp.exists():
                meta = json.loads(mp.read_text())
            env = meta.get("env")
            seed = meta.get("seed")
            if args.only_env and env != args.only_env:
                continue
            if args.only_seed is not None and int(seed) != int(args.only_seed):
                continue
            filtered.append(run)
        runs = filtered

    runs = runs[args.offset :]
    if args.limit > 0:
        runs = runs[: args.limit]

    summary = {"done": 0, "skipped": 0, "failed": 0, "results": []}
    for run in runs:
        try:
            rec = evaluate_run(
                run,
                episodes=args.episodes,
                device=args.device,
                force=args.force,
                step=args.step,
                out_subdir=args.out_subdir,
            )
        except Exception as exc:  # noqa: BLE001
            rec = {"status": "failed", "run": str(run), "error": f"{type(exc).__name__}: {exc}"}
        summary["results"].append(rec)
        summary[rec.get("status", "failed")] = summary.get(rec.get("status", "failed"), 0) + 1
        print(json.dumps(rec), flush=True)

    print(json.dumps({k: summary[k] for k in ("done", "skipped", "failed")}, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
