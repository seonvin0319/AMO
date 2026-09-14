#!/usr/bin/env python3
"""Final-only CPU eval: final50_singlepass_v1 (50 episodes, one pass, no ckpt delete).

Evaluates pi_E via agent.act (main actor params). Does not call mid-ckpt deletion.
Writes eval_final50_v1.jsonl + posthoc_eval_final50_v1 markers with flock.
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.agent import make_agent  # noqa: E402

PROTOCOL = "final50_singlepass_v1"
TAG = "final50_singlepass_v1"
MARKER_DIR = "posthoc_eval_final50_v1"
EVAL_FILE = "eval_final50_v1.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
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
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def configure_cpu_env() -> None:
    home = Path.home()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    os.environ.setdefault("D4RL_DATASET_DIR", str(home / ".d4rl" / "datasets"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    mujoco = home / ".mujoco" / "mujoco210"
    os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", str(mujoco))
    ld = [
        "/usr/lib/x86_64-linux-gnu",
        str(mujoco / "bin"),
        "/usr/lib/nvidia",
        os.environ.get("LD_LIBRARY_PATH", ""),
    ]
    os.environ["LD_LIBRARY_PATH"] = ":".join(p for p in ld if p)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    # Prefer prebuilt mujoco_py from capo env when present.
    overlay = Path("/home/ext_csh/AMO_release/.py_overlay")
    if (overlay / "mujoco_py").exists():
        os.environ["PYTHONPATH"] = (
            f"{overlay}:{os.environ.get('PYTHONPATH', '')}".rstrip(":")
        )


def resolve_final_ckpt(run: Path) -> tuple[Path, int]:
    step = run / "checkpoints" / "step_1000000.npz"
    if step.is_file():
        return step, 1_000_000
    ck = run / "checkpoint.npz"
    if not ck.is_file():
        raise FileNotFoundError(f"no final checkpoint in {run}")
    with np.load(ck, allow_pickle=False) as data:
        meta = json.loads(str(data["__metadata__"]))
    st = int(meta.get("steps") or 0)
    if st < 1_000_000:
        raise ValueError(f"checkpoint.npz steps={st} < 1M in {run}")
    return ck, st


def already_done(run: Path, ckpt_sha: str, eval_seed: int, code_rev: str) -> bool:
    marker = run / MARKER_DIR / "final.json"
    if marker.is_file():
        try:
            payload = json.loads(marker.read_text())
            if (
                payload.get("ok")
                and payload.get("protocol") == PROTOCOL
                and payload.get("checkpoint_sha256") == ckpt_sha
                and int(payload.get("eval_seed", -1)) == int(eval_seed)
                and payload.get("code_revision") == code_rev
                and int(payload.get("episodes", 0)) == 50
            ):
                return True
        except Exception:
            pass
    path = run / EVAL_FILE
    if not path.is_file():
        return False
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            row.get("protocol") == PROTOCOL
            and row.get("checkpoint_sha256") == ckpt_sha
            and int(row.get("eval_seed", -1)) == int(eval_seed)
            and row.get("code_revision") == code_rev
            and int(row.get("episodes", 0)) == 50
            and int(row.get("repeats", 0)) == 1
        ):
            try:
                float(row["normalized_score"])
                return True
            except Exception:
                continue
    return False


def evaluate_final50(agent, env_name: str, mean, std, eval_seed: int) -> dict:
    """50 episodes, single pass; seed RNGs once; do not re-seed per episode."""
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
        is_ant = "antmaze" in env_name
        for _ in range(50):
            obs = env.reset()
            done = False
            total = 0.0
            while not done:
                action = agent.act((np.asarray(obs, dtype=np.float32) - mean) / std)
                obs, reward, done, info = env.step(action)
                total += float(reward)
            episode_returns.append(float(total))
            if is_ant:
                # D4RL antmaze: sparse success often equals return (0/1).
                succ = info.get("success") if isinstance(info, dict) else None
                if succ is None:
                    succ = 1.0 if total > 0 else 0.0
                episode_success.append(float(succ))
        ret_mean = float(np.mean(episode_returns))
        norm = float(100.0 * env.get_normalized_score(ret_mean))
        # Verify with per-episode mean of normalized if API allows episode-wise
        out = {
            "return_mean": ret_mean,
            "normalized_score": norm,
            "d4rl_normalized_score": norm,
            "episodes": 50,
            "episodes_per_repeat": 50,
            "repeats": 1,
            "repeat_returns": [ret_mean],
            "episode_returns": episode_returns,
            "eval_seed": int(eval_seed),
            "seed_stride": 1,
            "protocol": PROTOCOL,
        }
        if is_ant:
            out["episode_success"] = episode_success
            out["success_mean"] = float(np.mean(episode_success))
        return out
    finally:
        env.close()


def load_mean_std(run: Path, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    extra = meta.get("extra") or {}
    norm_path = run / "normalization.npz"
    if norm_path.is_file():
        with np.load(norm_path) as norm:
            mean = np.asarray(norm["mean"], dtype=np.float32)
            std = np.asarray(norm["std"], dtype=np.float32)
        if "mean" in extra and "std" in extra:
            em = np.asarray(extra["mean"], dtype=np.float32)
            es = np.asarray(extra["std"], dtype=np.float32)
            if em.shape == mean.shape and es.shape == std.shape:
                if not (np.allclose(em, mean, atol=1e-5) and np.allclose(es, std, atol=1e-5)):
                    raise ValueError(
                        f"normalization.npz != checkpoint extra mean/std in {run}"
                    )
        return mean, std
    if "mean" in extra and "std" in extra:
        return (
            np.asarray(extra["mean"], dtype=np.float32),
            np.asarray(extra["std"], dtype=np.float32),
        )
    raise FileNotFoundError(f"no normalization for {run}")


def eval_one(run: Path, device: str, force: bool, code_rev: str) -> dict:
    configure_cpu_env()
    lock_path = run / MARKER_DIR / ".eval.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            ckpt, step = resolve_final_ckpt(run)
            with np.load(ckpt, allow_pickle=False) as data:
                meta = json.loads(str(data["__metadata__"]))
            cfg = meta["config"]
            extra = meta.get("extra") or {}
            run_meta = {}
            if (run / "run_meta.json").is_file():
                run_meta = json.loads((run / "run_meta.json").read_text())
            env_name = run_meta.get("env") or extra.get("env")
            seed = run_meta.get("seed")
            if seed is None:
                seed = extra.get("seed")
            if env_name is None or seed is None:
                raise ValueError(f"missing env/seed for {run}")
            seed = int(seed)
            if run_meta.get("env") and extra.get("env") and run_meta["env"] != extra["env"]:
                raise ValueError(
                    f"env mismatch run_meta={run_meta['env']} ckpt={extra['env']}"
                )
            algorithm = cfg.get("algorithm") or run_meta.get("algorithm") or "iql_amo"
            backend = meta.get("backend") or run_meta.get("backend") or "jax"
            eval_seed = seed if cfg.get("eval_seed") is None else int(cfg["eval_seed"])
            ckpt_sha = sha256_file(ckpt)
            if not force and already_done(run, ckpt_sha, eval_seed, code_rev):
                return {
                    "status": "skip",
                    "run": str(run),
                    "reason": "already_done",
                }

            mean, std = load_mean_std(run, meta)
            agent = make_agent(
                algorithm,
                backend,
                int(meta["observation_dim"]),
                int(meta["action_dim"]),
                cfg,
                seed=seed,
                device=device,
            )
            agent.load(str(ckpt))
            if int(agent.steps) < 1_000_000:
                raise ValueError(f"loaded steps={agent.steps} < 1M")

            result = evaluate_final50(agent, env_name, mean, std, eval_seed)
            row = {
                "step": int(agent.steps),
                "tag": TAG,
                "protocol": PROTOCOL,
                "device": device,
                "backend": backend,
                "algorithm": algorithm,
                "checkpoint": str(ckpt),
                "checkpoint_sha256": ckpt_sha,
                "code_revision": code_rev,
                "env": env_name,
                "seed": seed,
                "evaluated_at": utc_now(),
                **result,
            }
            # atomic append eval file
            eval_path = run / EVAL_FILE
            tmp = eval_path.with_suffix(".jsonl.tmp")
            existing = eval_path.read_text() if eval_path.exists() else ""
            tmp.write_text(existing + json.dumps(row, allow_nan=False) + "\n")
            tmp.replace(eval_path)

            marker = {
                "ok": True,
                "protocol": PROTOCOL,
                "checkpoint_sha256": ckpt_sha,
                "code_revision": code_rev,
                "eval_seed": eval_seed,
                "episodes": 50,
                "row": row,
                "at": utc_now(),
            }
            mpath = run / MARKER_DIR / "final.json"
            mtmp = mpath.with_suffix(".json.tmp")
            mtmp.write_text(json.dumps(marker, indent=2) + "\n")
            mtmp.replace(mpath)
            print(
                json.dumps(
                    {
                        "status": "done",
                        "run": str(run),
                        "normalized_score": row["normalized_score"],
                        "env": env_name,
                        "seed": seed,
                    }
                ),
                flush=True,
            )
            return {"status": "done", **row}
        except Exception as exc:
            fail = {
                "ok": False,
                "protocol": PROTOCOL,
                "error": f"{type(exc).__name__}: {exc}",
                "at": utc_now(),
                "run": str(run),
            }
            fpath = run / MARKER_DIR / "final.FAILED.json"
            fpath.write_text(json.dumps(fail, indent=2) + "\n")
            print(json.dumps({"status": "failed", **fail}), flush=True)
            return {"status": "failed", **fail}
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def load_pending(manifest: Path) -> list[Path]:
    data = json.loads(manifest.read_text())
    out = []
    for rec in data.get("pending_final50") or []:
        out.append(Path(rec["path"]))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Single run directory")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="classification.json with pending_final50",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--shard", default=None, help="i/n partition of pending list")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    configure_cpu_env()
    # Ensure overlay on sys.path for this process too
    overlay = Path("/home/ext_csh/AMO_release/.py_overlay")
    if (overlay / "mujoco_py").exists():
        sys.path.insert(0, str(overlay))

    code_rev = git_revision()
    runs: list[Path] = []
    if args.run:
        runs = [args.run]
    elif args.manifest:
        runs = load_pending(args.manifest)
    else:
        raise SystemExit("need --run or --manifest")

    if args.shard:
        i, n = map(int, args.shard.split("/"))
        runs = [r for idx, r in enumerate(sorted(runs)) if idx % n == i]
    if args.limit > 0:
        runs = runs[: args.limit]

    print(
        json.dumps(
            {"event": "start", "n_runs": len(runs), "code_revision": code_rev, "at": utc_now()}
        ),
        flush=True,
    )
    ok = fail = skip = 0
    for run in runs:
        rec = eval_one(run, args.device, args.force, code_rev)
        st = rec.get("status")
        if st == "done":
            ok += 1
        elif st == "skip":
            skip += 1
        else:
            fail += 1
    print(
        json.dumps(
            {"event": "complete", "done": ok, "skip": skip, "fail": fail, "at": utc_now()}
        ),
        flush=True,
    )
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
