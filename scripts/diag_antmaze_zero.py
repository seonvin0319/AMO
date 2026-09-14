#!/usr/bin/env python3
"""AntMaze TD3+AMO zero-score diagnosis helpers (read-only + short numeric probes)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from common.agent import make_agent  # noqa: E402
from eval_checkpoints_cpu import configure_cpu_env  # noqa: E402
from train import evaluate as stock_evaluate  # noqa: E402


def host_env() -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
    os.environ["D4RL_DATASET_DIR"] = "/home/shchoi/.d4rl/datasets"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_PY_MUJOCO_PATH"] = str(Path.home() / ".mujoco" / "mujoco210")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"


def load_agent(run: Path, ckpt: Path, device: str = "cpu"):
    with np.load(ckpt, allow_pickle=False) as data:
        meta = json.loads(str(data["__metadata__"]))
    c = meta["config"]
    extra = meta.get("extra") or {}
    rm = {}
    if (run / "run_meta.json").exists():
        rm = json.loads((run / "run_meta.json").read_text())
    env = rm.get("env") or extra.get("env")
    seed = int(rm.get("seed") if rm.get("seed") is not None else extra.get("seed"))
    algo = rm.get("algorithm") or c.get("algorithm")
    backend = rm.get("backend") or meta.get("backend") or "jax"
    norm = np.load(run / "normalization.npz")
    mean = np.asarray(norm["mean"], np.float32)
    std = np.asarray(norm["std"], np.float32)
    agent = make_agent(
        algo, backend, int(meta["observation_dim"]), int(meta["action_dim"]), c, seed=seed, device=device
    )
    agent.load(str(ckpt))
    return agent, mean, std, env, seed, c, meta


def metric_summary(run: Path) -> dict:
    path = run / "metrics.jsonl"
    lines = path.read_text().strip().splitlines()
    first = json.loads(lines[0])
    last = json.loads(lines[-1])
    keys = [
        "T_E",
        "T_B",
        "L1_B",
        "L2_RMS_B",
        "L_T_E",
        "L_T_B",
        "actor_loss",
        "critic_loss",
        "q_mean",
        "q_scale",
        "B_pi",
    ]
    out = {"n": len(lines), "first_step": first.get("step"), "last_step": last.get("step")}
    for k in keys:
        if k in last or k in first:
            out[k] = {"first": first.get(k), "last": last.get(k)}
    # midpoints
    mid = json.loads(lines[len(lines) // 2])
    out["mid_step"] = mid.get("step")
    for k in ("T_E", "T_B", "actor_loss", "critic_loss"):
        if k in mid:
            out[k]["mid"] = mid.get(k)
    return out


def fixed_obs_compare(run: Path, ckpt: Path, n: int = 32, device: str = "cpu") -> dict:
    agent, mean, std, env, seed, c, meta = load_agent(run, ckpt, device)
    rng = np.random.default_rng(0)
    obs_dim = int(meta["observation_dim"])
    raw = rng.normal(size=(n, obs_dim)).astype(np.float32)
    normed = (raw - mean) / std
    a1 = np.asarray(agent.act(normed), dtype=np.float64)
    # Reload fresh agent — same ckpt — second pass must match exactly
    agent2, *_ = load_agent(run, ckpt, device)
    a2 = np.asarray(agent2.act(normed), dtype=np.float64)
    # Stock evaluate path uses same act; compare against bootstrap actor if exposed
    boot = None
    try:
        boot_params = agent.state["p"]["bootstrap"]
        boot_act = agent._compiled_actor(boot_params, agent.ops.array(normed))
        boot = agent.ops.numpy(boot_act)
    except Exception as exc:
        boot = f"error:{type(exc).__name__}:{exc}"
    # target actor
    try:
        tgt = agent.state["target"]["actor"]
        tgt_act = agent._compiled_actor(tgt, agent.ops.array(normed))
        tgt_np = agent.ops.numpy(tgt_act)
    except Exception as exc:
        tgt_np = f"error:{type(exc).__name__}:{exc}"

    def diff(a, b):
        if isinstance(a, str) or isinstance(b, str):
            return {"error": True, "a": str(type(a)), "b": str(type(b))}
        d = np.asarray(a) - np.asarray(b)
        return {
            "max_abs": float(np.max(np.abs(d))),
            "mean_abs": float(np.mean(np.abs(d))),
            "rmse": float(np.sqrt(np.mean(d**2))),
        }

    return {
        "env": env,
        "seed": seed,
        "steps": int(agent.steps),
        "T_E_config": c.get("T_E"),
        "T_B_config": c.get("T_B"),
        "reward_transform": c.get("reward_transform"),
        "reload_same_ckpt": diff(a1, a2),
        "pi_E_vs_bootstrap": diff(a1, boot) if not isinstance(boot, str) else {"error": boot},
        "pi_E_vs_target_actor": diff(a1, tgt_np) if not isinstance(tgt_np, str) else {"error": tgt_np},
        "pi_E_action_abs_max": float(np.max(np.abs(a1))),
        "pi_E_action_mean": a1.mean(axis=0).tolist(),
        "ckpt_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
    }


def stock_vs_final50_proto_note() -> dict:
    return {
        "stock_posthoc_default": {
            "episodes": 10,
            "final_repeats": 5,
            "seed_stride_td3": 1,
            "note": "5 repeats of 10 eps; reseeds env each repeat with seed+repeat*stride",
        },
        "final50_singlepass_v1": {
            "episodes": 50,
            "repeats": 1,
            "seed_once": True,
            "note": "single continuous 50-episode pass",
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    host_env()
    run = args.run.resolve()
    ckpt = args.ckpt
    if ckpt is None:
        cand = run / "checkpoints" / "step_1000000.npz"
        ckpt = cand if cand.exists() else run / "checkpoint.npz"
    report = {
        "run": str(run),
        "ckpt": str(ckpt),
        "metrics": metric_summary(run),
        "fixed_obs": fixed_obs_compare(run, ckpt, device=args.device),
        "protocol_note": stock_vs_final50_proto_note(),
    }
    # optional: tiny stock eval 2eps for sanity only into out (not final50)
    agent, mean, std, env, seed, c, meta = load_agent(run, ckpt, args.device)
    tiny = stock_evaluate(agent, env, mean, std, seed, episodes=2, repeats=1, seed_stride=1)
    report["stock_evaluate_2ep_smoke"] = tiny
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"wrote": str(args.out), "normalized_2ep": tiny.get("normalized_score")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
