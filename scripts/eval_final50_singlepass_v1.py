#!/usr/bin/env python3
"""final50_singlepass_v1: one continuous 50-episode eval per 1M checkpoint.

Protocol (this workstream):
  - episodes=50, episodes_per_repeat=50, repeats=1, seed_stride=1
  - Seed Python/NumPy/env/action_space once at start; do NOT re-seed per episode
  - Write eval_final50_v1.jsonl + eval_final50_v1.DONE.json (atomic)
  - Never delete checkpoints; never touch eval.jsonl
  - Skip when DONE marker matches (ckpt sha256, protocol, eval_seed, episodes, code rev)

Uses the same make_agent / act path as train.py. Does not call the mid-ckpt
deletion poller in eval_checkpoints_cpu.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.agent import make_agent  # noqa: E402

PROTOCOL = "final50_singlepass_v1"
TAG = "final50_singlepass_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def configure_cpu_env() -> None:
    home = Path.home()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    for candidate in (
        Path(os.environ["D4RL_DATASET_DIR"]) if os.environ.get("D4RL_DATASET_DIR") else None,
        home / ".d4rl" / "datasets",
    ):
        if candidate and candidate.exists():
            os.environ["D4RL_DATASET_DIR"] = str(candidate)
            break
    else:
        os.environ.setdefault("D4RL_DATASET_DIR", str(home / ".d4rl" / "datasets"))
    mujoco = home / ".mujoco" / "mujoco210"
    if mujoco.exists():
        os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", str(mujoco))
        bin_dir = str(mujoco / "bin")
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        if bin_dir not in ld.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{bin_dir}:{ld}" if ld else bin_dir


def load_run_meta(run: Path) -> dict:
    path = run / "run_meta.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def pick_final_ckpt(run: Path) -> tuple[Path, int] | None:
    step_path = run / "checkpoints" / "step_1000000.npz"
    if step_path.exists():
        return step_path, 1_000_000
    ckpt = run / "checkpoint.npz"
    if not ckpt.exists():
        return None
    try:
        with np.load(ckpt, allow_pickle=False) as data:
            meta = json.loads(str(data["__metadata__"]))
        steps = int(meta.get("steps", 0))
        if steps >= 1_000_000:
            return ckpt, steps
    except Exception:
        return None
    return None


def marker_path(run: Path) -> Path:
    return run / "eval_final50_v1.DONE.json"


def already_done(run: Path, ckpt: Path, eval_seed: int, episodes: int, rev: str) -> bool:
    path = marker_path(run)
    if not path.exists():
        return False
    try:
        marker = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    return (
        marker.get("ok") is True
        and marker.get("protocol") == PROTOCOL
        and marker.get("checkpoint_sha256") == sha256_file(ckpt)
        and int(marker.get("eval_seed", -1)) == int(eval_seed)
        and int(marker.get("episodes", -1)) == int(episodes)
        and marker.get("code_revision") == rev
        and int(marker.get("repeats", -1)) == 1
        and int(marker.get("episodes_per_repeat", -1)) == int(episodes)
    )


def evaluate_final50(
    agent,
    env_name: str,
    mean: np.ndarray,
    std: np.ndarray,
    eval_seed: int,
    episodes: int = 50,
) -> dict:
    import d4rl  # noqa: F401
    import gym

    random.seed(eval_seed)
    np.random.seed(eval_seed)

    env = gym.make(env_name)
    try:
        env.seed(eval_seed)
        env.action_space.seed(eval_seed)
        episode_returns: list[float] = []
        episode_success: list[float] = []
        for _ in range(episodes):
            obs, done, total = env.reset(), False, 0.0
            while not done:
                obs, reward, done, info = env.step(agent.act((obs - mean) / std))
                total += float(reward)
            episode_returns.append(total)
            # Adroit/locomotion: info may lack success; AntMaze often has it.
            if isinstance(info, dict) and "success" in info:
                episode_success.append(float(info["success"]))
            elif total > 0 and "antmaze" in env_name:
                # sparse antmaze: positive return implies success in many wrappers
                episode_success.append(1.0 if total > 0 else 0.0)

        mean_ret = float(np.mean(episode_returns))
        norm = float(100.0 * env.get_normalized_score(mean_ret))
        # Recompute from per-episode normalized scores for audit.
        per_ep_norm = [float(100.0 * env.get_normalized_score(r)) for r in episode_returns]
        row = {
            "return_mean": mean_ret,
            "normalized_score": norm,
            "normalized_score_from_episode_mean": float(np.mean(per_ep_norm)),
            "episodes": episodes,
            "episodes_per_repeat": episodes,
            "repeats": 1,
            "seed_stride": 1,
            "eval_seed": eval_seed,
            "episode_returns": episode_returns,
            "episode_normalized_scores": per_ep_norm,
            "protocol": PROTOCOL,
        }
        if episode_success:
            row["episode_success"] = episode_success
            row["success_rate"] = float(np.mean(episode_success))
        return row
    finally:
        env.close()


def eval_run(run: Path, device: str = "cpu", force: bool = False) -> dict:
    meta = load_run_meta(run)
    picked = pick_final_ckpt(run)
    if picked is None:
        return {"status": "skip", "run": run.name, "reason": "no_1m_checkpoint"}
    ckpt, step = picked
    with np.load(ckpt, allow_pickle=False) as data:
        ckpt_meta = json.loads(str(data["__metadata__"]))
    cfg = ckpt_meta["config"]
    environment = str(meta.get("env") or cfg.get("env") or "")
    seed = int(meta.get("seed") if meta.get("seed") is not None else cfg.get("seed", 0))
    algorithm = str(meta.get("algorithm") or ckpt_meta.get("algorithm") or "td3_amo")
    backend = str(meta.get("backend") or ckpt_meta.get("backend") or "jax")
    if not environment:
        return {"status": "failed", "run": run.name, "error": "missing env"}

    eval_seed = seed if cfg.get("eval_seed") is None else int(cfg["eval_seed"])
    rev = code_revision()
    if not force and already_done(run, ckpt, eval_seed, 50, rev):
        return {"status": "skip", "run": run.name, "reason": "already_done"}

    # Lock to avoid duplicate workers.
    lock = run / ".eval_final50_v1.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {utc_now()}\n".encode())
        os.close(fd)
    except FileExistsError:
        return {"status": "skip", "run": run.name, "reason": "locked"}

    try:
        norm_path = run / "normalization.npz"
        if not norm_path.exists():
            raise FileNotFoundError("normalization.npz missing")
        norm = np.load(norm_path)
        mean = np.asarray(norm["mean"], dtype=np.float32)
        std = np.asarray(norm["std"], dtype=np.float32)
        # Cross-check checkpoint extras if present.
        extra_mean = ckpt_meta.get("obs_mean")
        extra_std = ckpt_meta.get("obs_std")
        if extra_mean is not None and extra_std is not None:
            if not (
                np.allclose(mean, np.asarray(extra_mean), atol=1e-5)
                and np.allclose(std, np.asarray(extra_std), atol=1e-5)
            ):
                raise ValueError("normalization.npz disagrees with checkpoint extra")

        agent = make_agent(
            algorithm,
            backend,
            int(ckpt_meta["observation_dim"]),
            int(ckpt_meta["action_dim"]),
            cfg,
            seed=seed,
            device=device,
        )
        agent.load(str(ckpt))
        if int(agent.steps) < 1_000_000:
            raise ValueError(f"checkpoint steps {agent.steps} < 1M")

        result = evaluate_final50(agent, environment, mean, std, eval_seed, episodes=50)
        row = {
            "step": int(agent.steps),
            "tag": TAG,
            "device": device,
            "backend": backend,
            "algorithm": algorithm,
            "env": environment,
            "seed": seed,
            "checkpoint": str(ckpt),
            "checkpoint_sha256": sha256_file(ckpt),
            "evaluated_at": utc_now(),
            "code_revision": rev,
            **result,
        }
        out = run / "eval_final50_v1.jsonl"
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        marker = {
            "ok": True,
            "protocol": PROTOCOL,
            "checkpoint": str(ckpt),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "eval_seed": eval_seed,
            "episodes": 50,
            "episodes_per_repeat": 50,
            "repeats": 1,
            "seed_stride": 1,
            "code_revision": rev,
            "normalized_score": row["normalized_score"],
            "at": utc_now(),
            "row": row,
        }
        tmp = marker_path(run).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(marker, indent=2) + "\n")
        tmp.replace(marker_path(run))
        print(json.dumps({"status": "done", "run": run.name, "score": row["normalized_score"]}), flush=True)
        return {"status": "done", "run": run.name, "score": row["normalized_score"]}
    except Exception as exc:
        fail = {
            "ok": False,
            "protocol": PROTOCOL,
            "run": run.name,
            "checkpoint": str(ckpt),
            "error": f"{type(exc).__name__}: {exc}",
            "at": utc_now(),
        }
        (run / "eval_final50_v1.FAILED.json").write_text(json.dumps(fail, indent=2) + "\n")
        print(json.dumps({"status": "failed", **fail}), flush=True)
        return {"status": "failed", "run": run.name, "error": fail["error"]}
    finally:
        try:
            lock.unlink(missing_ok=True)
        except OSError:
            pass


def discover_runs(root: Path) -> list[Path]:
    runs = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "config.yaml").exists():
            runs.append(path)
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run", type=str, default=None, help="Single run directory name")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Refuse: use separate smoke path")
    args = parser.parse_args(argv)
    if args.smoke:
        raise SystemExit("Refuse --smoke here; keep final50 outputs clean.")

    configure_cpu_env()
    root = args.runs_root
    if args.run:
        targets = [root / args.run]
    else:
        targets = discover_runs(root)

    summary = {"done": 0, "skip": 0, "failed": 0, "seen": 0}
    for run in targets:
        if not run.exists():
            continue
        summary["seen"] += 1
        rec = eval_run(run, device=args.device, force=args.force)
        summary[rec["status"]] = summary.get(rec["status"], 0) + 1
    print(json.dumps({"summary": summary}, indent=2), flush=True)
    return 0 if summary.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
