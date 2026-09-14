#!/usr/bin/env python3
"""Classify IQL+AMO β=1 runs and emit audit inventory for final50 recovery."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import yaml

KST = timezone(timedelta(hours=9))
PROTOCOL = "final50_singlepass_v1"
LOCO = [
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-medium-expert-v2",
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-medium-expert-v2",
]
ANT = [
    "antmaze-umaze-v2",
    "antmaze-umaze-diverse-v2",
    "antmaze-medium-play-v2",
    "antmaze-medium-diverse-v2",
    "antmaze-large-play-v2",
    "antmaze-large-diverse-v2",
]
ALL_ENVS = LOCO + ANT


def metrics_step(run: Path) -> int:
    m = run / "metrics.jsonl"
    if not m.exists():
        return 0
    with m.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 12000))
        chunk = f.read().decode(errors="ignore")
    for line in reversed(chunk.splitlines()):
        if not line.strip():
            continue
        try:
            return int(json.loads(line).get("step") or 0)
        except Exception:
            pass
    return 0


def resolve_final_ckpt(run: Path) -> tuple[Path | None, int | None]:
    step = run / "checkpoints" / "step_1000000.npz"
    if step.is_file():
        return step, 1_000_000
    ck = run / "checkpoint.npz"
    if not ck.is_file():
        return None, None
    try:
        with np.load(ck, allow_pickle=False) as data:
            meta = json.loads(str(data["__metadata__"]))
        st = int(meta.get("steps") or 0)
        return ck, st if st > 0 else None
    except Exception:
        return ck, None


def final50_valid(run: Path) -> dict[str, Any] | None:
    path = run / "eval_final50_v1.jsonl"
    if not path.is_file():
        return None
    last = None
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("protocol") != PROTOCOL:
            continue
        if int(row.get("step") or 0) < 1_000_000:
            continue
        if int(row.get("episodes") or 0) != 50:
            continue
        if int(row.get("repeats") or 0) != 1:
            continue
        if int(row.get("episodes_per_repeat") or 0) != 50:
            continue
        try:
            float(row["normalized_score"])
        except Exception:
            continue
        last = row
    return last


def legacy_eval_info(run: Path) -> dict[str, Any] | None:
    path = run / "eval.jsonl"
    if not path.is_file():
        return None
    last = None
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(row.get("step") or 0) >= 1_000_000:
            last = row
    return last


def classify_run(run: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if (run / "run_meta.json").is_file():
        try:
            meta = json.loads((run / "run_meta.json").read_text())
        except Exception:
            meta = {}
    cfg: dict[str, Any] = {}
    if (run / "config.yaml").is_file():
        try:
            cfg = yaml.safe_load((run / "config.yaml").read_text()) or {}
        except Exception:
            cfg = {}
    env = meta.get("env") or cfg.get("env") or run.name
    seed = meta.get("seed")
    if seed is None:
        seed = cfg.get("seed")
    try:
        seed = int(seed) if seed is not None else None
    except Exception:
        seed = None
    beta = cfg.get("beta_initial")
    rho = cfg.get("rho_lr")
    algo = cfg.get("algorithm") or meta.get("algorithm")
    backend = meta.get("backend") or "unknown"
    ms = metrics_step(run)
    ckpt, ckpt_step = resolve_final_ckpt(run)
    f50 = final50_valid(run)
    legacy = legacy_eval_info(run)

    status = "failed"
    if ckpt is None:
        status = "checkpoint_missing"
    elif ms < 1_000_000 and (ckpt_step is None or ckpt_step < 1_000_000):
        status = "training_incomplete"
    elif ms >= 1_000_000 or (ckpt_step is not None and ckpt_step >= 1_000_000):
        if f50 is not None:
            status = "evaluation_valid"
        else:
            status = "evaluation_missing"
        # training finished gate
        if status in ("evaluation_missing", "evaluation_valid"):
            pass
    else:
        status = "training_incomplete"

    # Prefer explicit training_finished label when train done but we still use above
    training_finished = (ms >= 1_000_000) or (
        ckpt_step is not None and ckpt_step >= 1_000_000
    )

    return {
        "path": str(run),
        "env": env,
        "seed": seed,
        "beta_initial": beta,
        "rho_lr": rho,
        "algorithm": algo,
        "backend": backend,
        "metrics_step": ms,
        "ckpt": str(ckpt) if ckpt else None,
        "ckpt_step": ckpt_step,
        "training_finished": training_finished,
        "status": status,
        "final50": f50 is not None,
        "final50_score": (
            float(f50["normalized_score"])
            if f50 is not None and f50.get("normalized_score") is not None
            else None
        ),
        "final50_eval_seed": (f50 or {}).get("eval_seed") if f50 else None,
        "legacy_eval": legacy is not None,
        "legacy_episodes": (legacy or {}).get("episodes"),
        "legacy_repeats": (legacy or {}).get("repeats"),
        "legacy_normalized_score": (legacy or {}).get("normalized_score")
        or (legacy or {}).get("d4rl_normalized_score"),
    }


def discover_runs(root: Path) -> list[Path]:
    runs: list[Path] = []
    if not root.is_dir():
        return runs
    for p in sorted(root.rglob("run_meta.json")):
        runs.append(p.parent)
    # also config-only
    for p in sorted(root.rglob("config.yaml")):
        if p.parent not in runs and (
            (p.parent / "checkpoints").is_dir() or (p.parent / "checkpoint.npz").exists()
        ):
            runs.append(p.parent)
    return sorted(set(runs))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--beta1-root",
        type=Path,
        default=Path(
            "/home/ext_csh/AMO_release/results/iql_amo_bpi_beta1_rlr_seeds0to3/cells"
        ),
    )
    parser.add_argument(
        "--jax-remainder-root",
        type=Path,
        default=Path(
            "/home/ext_csh/AMO_release/results/iql_amo_bpi_jax_remainder/cells"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/home/ext_csh/AMO_release_wt_log_audit_20260914/audit_out/classification.json"
        ),
    )
    args = parser.parse_args()

    rows = []
    for label, root in (
        ("beta1_seeds123", args.beta1_root),
        ("jax_remainder", args.jax_remainder_root),
    ):
        for run in discover_runs(root):
            rec = classify_run(run)
            rec["tree"] = label
            rows.append(rec)

    # Focus filters
    beta1 = [
        r
        for r in rows
        if r.get("beta_initial") in (1, 1.0)
        and r.get("algorithm") in (None, "iql_amo")
    ]
    new90 = [
        r
        for r in beta1
        if r["tree"] == "beta1_seeds123" and r.get("seed") in (1, 2, 3)
    ]
    seed0 = [
        r
        for r in beta1
        if r["tree"] == "jax_remainder" and r.get("seed") == 0 and r.get("backend") == "jax"
    ]

    by_status: dict[str, int] = {}
    for r in new90:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    pending = [
        r
        for r in new90 + seed0
        if r["status"] == "evaluation_missing" and r["training_finished"]
    ]

    payload = {
        "at": datetime.now(KST).isoformat(),
        "protocol_target": PROTOCOL,
        "counts": {
            "discovered_beta1_all": len(beta1),
            "new_seed123": len(new90),
            "seed0_jax_beta1": len(seed0),
            "new90_by_status": by_status,
            "pending_final50_train_done": len(pending),
        },
        "seed0_jax_beta1": seed0,
        "new90": new90,
        "pending_final50": pending,
        "all_beta1": beta1,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["counts"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
