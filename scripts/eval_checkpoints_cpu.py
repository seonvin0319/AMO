#!/usr/bin/env python3
"""Default AMO-main offline CPU eval reaper.

Training defaults (train.py): no in-process MuJoCo eval, save every 20k to
`checkpoints/step_{N}.npz`. This script evaluates those checkpoints on CPU,
appends `eval.jsonl`, and deletes mid-step ckpts only after verified success.
Final (max_steps) checkpoints are always kept.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.agent import make_agent  # noqa: E402
from train import evaluate  # noqa: E402

STEP_RE = re.compile(r"^step_(\d+)\.npz$")
TAG_MID = "posthoc_cpu"
TAG_FINAL = "posthoc_cpu_final"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def eval_row_ok(row: dict, expected_episodes: int | None = None) -> bool:
    if not isinstance(row, dict):
        return False
    if row.get("tag") not in (TAG_MID, TAG_FINAL):
        return False
    try:
        step = int(row["step"])
        episodes = int(row["episodes"])
        return_mean = float(row["return_mean"])
        normalized = float(row["normalized_score"])
    except (KeyError, TypeError, ValueError):
        return False
    if step <= 0 or episodes <= 0:
        return False
    if expected_episodes is not None and episodes != int(expected_episodes):
        return False
    if not np.isfinite(return_mean) or not np.isfinite(normalized):
        return False
    repeats = int(row.get("repeats") or 1)
    per = int(row.get("episodes_per_repeat") or 0)
    if per > 0 and episodes != per * repeats:
        return False
    repeat_returns = row.get("repeat_returns")
    if isinstance(repeat_returns, list):
        if len(repeat_returns) != repeats:
            return False
        if not all(np.isfinite(float(x)) for x in repeat_returns):
            return False
    return True


def evaluated_steps(output: Path) -> set[int]:
    out: set[int] = set()
    ej = output / "eval.jsonl"
    if not ej.exists():
        return out
    for line in ej.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("tag") not in (TAG_MID, TAG_FINAL):
            continue
        if "step" not in row:
            continue
        step = int(row["step"])
        expected = int(row.get("episodes") or 0)
        if eval_row_ok(row, expected_episodes=expected if expected > 0 else None):
            out.add(step)
    return out


def marker_ok(output: Path, step: int) -> bool:
    path = output / "posthoc_eval_cpu" / f"step_{step}.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    if not payload.get("ok"):
        return False
    row = payload.get("row")
    if not isinstance(row, dict) or int(row.get("step", -1)) != int(step):
        return False
    expected = int(row.get("episodes") or 0)
    return eval_row_ok(row, expected_episodes=expected if expected > 0 else None)


def verified_ready_to_delete(output: Path, step: int, budget: int) -> bool:
    if step >= budget:
        return False
    if not marker_ok(output, step):
        return False
    return step in evaluated_steps(output)


def list_step_ckpts(output: Path) -> list[tuple[int, Path]]:
    d = output / "checkpoints"
    if not d.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for path in d.iterdir():
        match = STEP_RE.match(path.name)
        if not match:
            continue
        found.append((int(match.group(1)), path))
    found.sort(key=lambda item: item[0])
    return found


def load_run_meta(output: Path) -> dict:
    meta_path = output / "run_meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def discover_runs(runs_root: Path) -> list[Path]:
    if not runs_root.is_dir():
        return []
    runs = []
    for path in sorted(runs_root.iterdir()):
        if path.is_dir() and (path / "checkpoints").is_dir():
            runs.append(path)
        elif path.is_dir() and (path / "checkpoint.npz").exists():
            runs.append(path)
    return runs


def eval_ckpt(
    output: Path,
    environment: str,
    seed: int,
    algorithm: str,
    backend: str,
    ckpt: Path,
    step: int,
    episodes: int,
    final_repeats: int,
    device: str,
    delete_non_final: bool,
) -> dict:
    with np.load(ckpt, allow_pickle=False) as data:
        meta = json.loads(str(data["__metadata__"]))
    c = meta["config"]
    budget = int(c.get("max_steps", 1_000_000))
    is_final = step >= budget
    tag = TAG_FINAL if is_final else TAG_MID
    expected_episodes = episodes * (final_repeats if is_final else 1)

    try:
        norm = np.load(output / "normalization.npz")
        mean = np.asarray(norm["mean"], dtype=np.float32)
        std = np.asarray(norm["std"], dtype=np.float32)
        agent = make_agent(
            algorithm,
            backend,
            int(meta["observation_dim"]),
            int(meta["action_dim"]),
            c,
            seed=seed,
            device=device,
        )
        agent.load(str(ckpt))
        if int(agent.steps) != int(step):
            raise ValueError(f"checkpoint step mismatch: agent={agent.steps} file={step}")
        eval_seed = seed if c.get("eval_seed") is None else int(c["eval_seed"])
        result = evaluate(
            agent,
            environment,
            mean,
            std,
            eval_seed,
            episodes,
            final_repeats if is_final else 1,
            seed_stride=0 if algorithm in ("iql_amo", "iql_ddpgbc_amo") else 1,
        )
        row = {
            "step": int(agent.steps),
            "tag": tag,
            "device": device,
            "backend": backend,
            "algorithm": algorithm,
            "checkpoint": str(ckpt),
            "evaluated_at": utc_now(),
            **result,
        }
        if not eval_row_ok(row, expected_episodes=expected_episodes):
            raise ValueError(f"eval_row_ok failed: {row}")
    except Exception as exc:
        dest = output / "posthoc_eval_cpu"
        dest.mkdir(parents=True, exist_ok=True)
        fail = {
            "ok": False,
            "step": step,
            "checkpoint": str(ckpt),
            "error": f"{type(exc).__name__}: {exc}",
            "at": utc_now(),
        }
        (dest / f"step_{step}.FAILED.json").write_text(json.dumps(fail, indent=2) + "\n")
        print(json.dumps({"status": "failed", "run": output.name, "step": step, **fail}), flush=True)
        return {"status": "failed", "step": step, "deleted_ckpt": False}

    with (output / "eval.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    dest = output / "posthoc_eval_cpu"
    dest.mkdir(parents=True, exist_ok=True)
    marker_payload = {"ok": True, "row": row}
    marker_path = dest / f"step_{step}.json"
    tmp = marker_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(marker_payload, indent=2) + "\n")
    tmp.replace(marker_path)
    if is_final:
        final_path = dest / "final.json"
        tmp_f = final_path.with_suffix(".json.tmp")
        tmp_f.write_text(json.dumps(marker_payload, indent=2) + "\n")
        tmp_f.replace(final_path)

    deleted = False
    if delete_non_final and not is_final:
        if verified_ready_to_delete(output, step, budget=budget):
            try:
                ckpt.unlink()
                deleted = True
            except OSError as exc:
                print(json.dumps({"warn": "delete_failed", "path": str(ckpt), "error": str(exc)}), flush=True)
        else:
            print(
                json.dumps(
                    {
                        "warn": "skip_delete_unverified",
                        "run": output.name,
                        "step": step,
                        "path": str(ckpt),
                    }
                ),
                flush=True,
            )

    print(
        json.dumps(
            {
                "status": "done",
                "run": output.name,
                "step": row["step"],
                "tag": tag,
                "normalized_score": row["normalized_score"],
                "episodes": row["episodes"],
                "deleted_ckpt": deleted,
            }
        ),
        flush=True,
    )
    return {"status": "done", "step": step, "is_final": is_final, "deleted_ckpt": deleted}


def sweep_output(
    output: Path,
    episodes: int,
    final_repeats: int,
    device: str,
    force: bool,
    delete_non_final: bool,
    max_per_run: int,
    final_only: bool = False,
) -> dict:
    meta = load_run_meta(output)
    with_context = None
    ckpts = list_step_ckpts(output)
    if ckpts:
        with np.load(ckpts[0][1], allow_pickle=False) as data:
            ck_meta = json.loads(str(data["__metadata__"]))
        with_context = ck_meta
    elif (output / "checkpoint.npz").exists():
        with np.load(output / "checkpoint.npz", allow_pickle=False) as data:
            with_context = json.loads(str(data["__metadata__"]))

    environment = meta.get("env") or (with_context or {}).get("extra", {}).get("env")
    seed = meta.get("seed")
    if seed is None and with_context:
        seed = (with_context.get("extra") or {}).get("seed")
    algorithm = meta.get("algorithm") or (with_context or {}).get("config", {}).get("algorithm")
    backend = meta.get("backend") or (with_context or {}).get("backend") or "jax"
    if environment is None or seed is None or algorithm is None:
        return {
            "status": "missing_meta",
            "done": 0,
            "pending": 0,
            "final_done": False,
            "run": output.name,
        }
    seed = int(seed)

    done_steps = evaluated_steps(output)
    budget = int(((with_context or {}).get("config") or {}).get("max_steps", 1_000_000))
    if delete_non_final and not force:
        for step, path in list_step_ckpts(output):
            if verified_ready_to_delete(output, step, budget=budget):
                path.unlink(missing_ok=True)

    ckpts = list_step_ckpts(output)
    pending = [(s, p) for s, p in ckpts if force or s not in done_steps]
    if final_only:
        pending = [(s, p) for s, p in pending if s >= budget]
    else:
        # Prefer final (max_steps) before mid; among mids keep ascending step order.
        pending.sort(key=lambda item: (0 if item[0] >= budget else 1, item[0]))
    finished = 0
    if final_only:
        to_run = pending[: max(1, max_per_run)]
    elif pending and pending[0][0] >= budget:
        # If a final ckpt is ready, evaluate it first and don't spend this tick on mids.
        to_run = pending[:1]
    else:
        to_run = pending[:max_per_run]
    for step, path in to_run:
        rec = eval_ckpt(
            output,
            environment,
            seed,
            algorithm,
            backend,
            path,
            step,
            episodes,
            final_repeats,
            device,
            delete_non_final,
        )
        if rec.get("status") == "done":
            finished += 1
            done_steps.add(step)

    final_done = budget in done_steps
    return {
        "status": "ok",
        "done": finished,
        "pending": max(0, len(pending) - finished),
        "final_done": final_done,
        "final_pending": any(s >= budget for s, _ in pending),
        "evaluated": len(done_steps),
        "run": output.name,
    }


def configure_cpu_env() -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    os.environ.setdefault("D4RL_DATASET_DIR", "/raid/ext_csv/datasets/d4rl")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", "/home/ext_csv/.mujoco/mujoco210")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        required=True,
        help="Directory containing run folders with checkpoints/",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--final-repeats", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-mid-ckpts", action="store_true")
    parser.add_argument("--poll", action="store_true")
    parser.add_argument("--poll-sec", type=int, default=120)
    parser.add_argument("--max-per-run", type=int, default=2)
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Evaluate only max_steps / final checkpoints; skip mid ckpts",
    )
    parser.add_argument("--run", default=None, help="Only this run directory name")
    args = parser.parse_args(argv)

    configure_cpu_env()
    runs_root = args.runs_root.resolve()
    runs = discover_runs(runs_root)
    if args.run:
        runs = [r for r in runs if r.name == args.run]
    delete_non_final = not args.keep_mid_ckpts
    # final-only mode: never delete mid ckpts from this process
    if args.final_only:
        delete_non_final = False

    def run_has_final_pending(output: Path) -> bool:
        ckpts = list_step_ckpts(output)
        if not ckpts:
            return False
        budget = 1_000_000
        try:
            with np.load(ckpts[-1][1], allow_pickle=False) as data:
                ck_meta = json.loads(str(data["__metadata__"]))
            budget = int((ck_meta.get("config") or {}).get("max_steps", budget))
        except Exception:
            pass
        done = evaluated_steps(output)
        return any(s >= budget and s not in done for s, _ in ckpts)

    def tick() -> dict:
        evaluated = pending = finals = 0
        live = discover_runs(runs_root)
        if args.run:
            live = [r for r in live if r.name == args.run]
        if args.final_only:
            live = [r for r in live if run_has_final_pending(r)]
        else:
            # Runs with an unevaluated final checkpoint jump the queue.
            live = sorted(
                live, key=lambda o: (0 if run_has_final_pending(o) else 1, o.name)
            )
        for output in live:
            rec = sweep_output(
                output,
                args.episodes,
                args.final_repeats,
                args.device,
                args.force,
                delete_non_final,
                args.max_per_run,
                final_only=args.final_only,
            )
            evaluated += int(rec.get("done", 0))
            pending += int(rec.get("pending", 0))
            if rec.get("final_done"):
                finals += 1
            if rec.get("final_pending") and rec.get("done", 0):
                print(
                    json.dumps(
                        {
                            "status": "final_priority",
                            "run": rec.get("run"),
                            "at": utc_now(),
                        }
                    ),
                    flush=True,
                )
        return {
            "at": utc_now(),
            "ckpts_evaluated_this_tick": evaluated,
            "ckpts_pending_known": pending,
            "runs_final_done": finals,
            "runs_seen": len(live),
            "final_only": bool(args.final_only),
        }

    if not args.poll:
        print(json.dumps(tick(), indent=2))
        return 0

    while True:
        summary = tick()
        print(json.dumps(summary), flush=True)
        if (
            summary["runs_seen"] > 0
            and summary["runs_final_done"] >= summary["runs_seen"]
            and summary["ckpts_pending_known"] == 0
        ):
            print("all discovered runs final-evaluated", flush=True)
            return 0
        time.sleep(max(30, args.poll_sec))


if __name__ == "__main__":
    raise SystemExit(main())
