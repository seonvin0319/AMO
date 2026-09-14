#!/usr/bin/env python3
"""Aggregate final50_singlepass_v1 scores for beta1 IQL+AMO (do not mix other LRs)."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

LOCO9 = [
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
ANT6 = [
    "antmaze-umaze-v2",
    "antmaze-umaze-diverse-v2",
    "antmaze-medium-play-v2",
    "antmaze-medium-diverse-v2",
    "antmaze-large-play-v2",
    "antmaze-large-diverse-v2",
]


def mean_std(xs: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if len(xs) < 1:
        return None, None
    arr = np.asarray(xs, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(xs) >= 2 else None
    return mean, std


def load_final50(run: Path) -> Optional[Dict[str, Any]]:
    mk = run / "FINAL50_SINGLEPASS_V1_DONE.json"
    if not mk.exists():
        return None
    payload = json.loads(mk.read_text())
    if not payload.get("ok"):
        return None
    return payload.get("row")


def legacy_best(run: Path) -> Optional[Dict[str, Any]]:
    fj = run / "posthoc_eval_cpu" / "final.json"
    if fj.exists():
        row = json.load(open(fj)).get("row")
        if isinstance(row, dict):
            return row
    ej = run / "eval.jsonl"
    if not ej.exists():
        return None
    last = None
    for line in ej.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if int(o.get("step") or 0) >= 1_000_000 and int(o.get("episodes") or 0) >= 50:
            last = o
    return last


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/home/svcho/amo_iql_sweep/results/iql_amo_lr1e3_beta_sweep_seed0/"
            "audit_20260914/manifest_beta1_final50.json"
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/home/svcho/amo_iql_sweep/results/iql_amo_lr1e3_beta_sweep_seed0/"
            "audit_20260914/tables_final50_singlepass_v1.json"
        ),
    )
    args = p.parse_args()
    man = json.loads(args.manifest.read_text())

    cells = {}
    incomplete = []
    for r in man["runs"]:
        run = Path(r["run_dir"])
        row = load_final50(run)
        leg = legacy_best(run)
        key = (r["env"], r["seed"])
        if row is None:
            incomplete.append({"env": r["env"], "seed": r["seed"], "job": r["job_name"]})
            cells[key] = {
                "final50": None,
                "legacy_ns": None if leg is None else leg.get("normalized_score"),
                "legacy_protocol_guess": None,
            }
        else:
            cells[key] = {
                "final50": float(row["normalized_score"]),
                "success_rate": row.get("success_rate"),
                "legacy_ns": None if leg is None else leg.get("normalized_score"),
                "action_sat_frac": row.get("action_sat_frac"),
                "mean_episode_length": row.get("mean_episode_length"),
                "finite_ok": row.get("finite_ok"),
            }

    def table_for(envs: List[str], antmaze: bool = False) -> List[Dict[str, Any]]:
        rows = []
        for env in envs:
            scores = []
            seed_vals = {}
            missing_seeds = []
            for s in range(4):
                cell = cells.get((env, s))
                if cell is None or cell["final50"] is None:
                    missing_seeds.append(s)
                    seed_vals[f"s{s}"] = None
                else:
                    seed_vals[f"s{s}"] = cell["final50"]
                    scores.append(cell["final50"])
            mean, std = mean_std(scores)
            row = {
                "env": env,
                **seed_vals,
                "n": len(scores),
                "missing_seeds": missing_seeds,
                "mean": mean if len(scores) == 4 else None,
                "std": std if len(scores) == 4 else None,
                "mean_std_str": (
                    f"{mean:.2f}±{std:.2f}" if mean is not None and std is not None and len(scores) == 4 else "--"
                ),
            }
            if antmaze:
                row["median"] = float(np.median(scores)) if len(scores) == 4 else None
                row["paper_value"] = row["median"]
            rows.append(row)
        return rows

    before_after = []
    for r in man["runs"]:
        key = (r["env"], r["seed"])
        cell = cells[key]
        before_after.append(
            {
                "env": r["env"],
                "seed": r["seed"],
                "legacy_ns": cell.get("legacy_ns"),
                "final50_ns": cell.get("final50"),
                "delta": (
                    None
                    if cell.get("legacy_ns") is None or cell.get("final50") is None
                    else float(cell["final50"]) - float(cell["legacy_ns"])
                ),
            }
        )

    payload = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "protocol": "final50_singlepass_v1",
        "base_lr": 1e-3,
        "rho_lr": 2e-3,
        "beta_initial": 1.0,
        "n_manifest": len(man["runs"]),
        "n_final50_done": sum(1 for v in cells.values() if v.get("final50") is not None),
        "incomplete": incomplete,
        "locomotion_table": table_for(LOCO9, antmaze=False),
        "antmaze_table": table_for(ANT6, antmaze=True),
        "before_after": before_after,
        "note": "Do not pool with ext_csh base_lr=3e-4. AntMaze paper value = seed median.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    # Also markdown
    md = args.out.with_suffix(".md")
    lines = [
        "# IQL+AMO β=1 · final50_singlepass_v1 (svcho, lr=1e-3)",
        "",
        f"Generated: {payload['created_at']}",
        f"Done: {payload['n_final50_done']}/{payload['n_manifest']}",
        "",
        "## Locomotion (mean±sample std, ddof=1)",
        "",
        "| env | s0 | s1 | s2 | s3 | mean±std |",
        "|-----|----|----|----|----|----------|",
    ]
    for row in payload["locomotion_table"]:
        def fmt(v):
            return "--" if v is None else f"{v:.2f}"
        lines.append(
            f"| {row['env']} | {fmt(row['s0'])} | {fmt(row['s1'])} | {fmt(row['s2'])} | {fmt(row['s3'])} | {row['mean_std_str']} |"
        )
    lines += [
        "",
        "## AntMaze (mean±std and median)",
        "",
        "| env | s0 | s1 | s2 | s3 | mean±std | median |",
        "|-----|----|----|----|----|----------|--------|",
    ]
    for row in payload["antmaze_table"]:
        def fmt(v):
            return "--" if v is None else f"{v:.2f}"
        med = "--" if row.get("median") is None else f"{row['median']:.2f}"
        lines.append(
            f"| {row['env']} | {fmt(row['s0'])} | {fmt(row['s1'])} | {fmt(row['s2'])} | {fmt(row['s3'])} | {row['mean_std_str']} | {med} |"
        )
    md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(args.out), "md": str(md), "done": payload["n_final50_done"]}, indent=2))
    return 0 if not incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
