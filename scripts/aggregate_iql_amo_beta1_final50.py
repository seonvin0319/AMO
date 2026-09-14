#!/usr/bin/env python3
"""Aggregate final50_singlepass_v1 scores for IQL+AMO β=1 (actor_lr=3e-4)."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

KST = timezone(timedelta(hours=9))
PROTOCOL = "final50_singlepass_v1"
SEEDS = (0, 1, 2, 3)
RHO = (3e-4, 2e-3)
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


def fmt_rho(x: float) -> str:
    return f"{x:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def mean_std(xs: list[float]) -> tuple[float | None, float | None]:
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, None
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def median(xs: list[float]) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    if n % 2:
        return ys[n // 2]
    return 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def load_audit(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def index_runs(audit: dict[str, Any]) -> dict[tuple, dict]:
    """Key: (rho_lr rounded, seed, env) -> run record (prefer jax beta1 trees)."""
    out: dict[tuple, dict] = {}
    for r in audit.get("all_beta1") or []:
        if r.get("beta_initial") not in (1, 1.0):
            continue
        if r.get("algorithm") not in (None, "iql_amo"):
            continue
        rho = float(r.get("rho_lr") or 0)
        seed = int(r.get("seed"))
        env = r["env"]
        key = (round(rho, 10), seed, env)
        prev = out.get(key)
        # Prefer evaluation_valid / newer final50
        if prev is None:
            out[key] = r
            continue
        rank = lambda x: (
            1 if x.get("status") == "evaluation_valid" else 0,
            1 if x.get("final50") else 0,
            1 if x.get("tree") == "beta1_seeds123" else 0,
        )
        if rank(r) >= rank(prev):
            out[key] = r
    return out


def score_of(r: dict[str, Any]) -> float | None:
    f50 = r.get("final50")
    if isinstance(f50, dict) and f50.get("normalized_score") is not None:
        try:
            return float(f50["normalized_score"])
        except (TypeError, ValueError):
            pass
    # Prefer explicit score fields written by audit; else read eval file.
    for key in ("final50_score", "normalized_score"):
        if r.get(key) is not None:
            try:
                return float(r[key])
            except (TypeError, ValueError):
                pass
    path = r.get("path")
    if path:
        p = Path(path) / "eval_final50_v1.jsonl"
        if p.is_file():
            last = None
            for line in p.read_text(errors="ignore").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("protocol") != PROTOCOL:
                    continue
                if int(row.get("episodes") or 0) != 50:
                    continue
                if int(row.get("repeats") or 0) != 1:
                    continue
                if int(row.get("episodes_per_repeat") or 0) != 50:
                    continue
                try:
                    last = float(row["normalized_score"])
                except Exception:
                    continue
            return last
    return None


def cell_report(
    idx: dict[tuple, dict], rho: float, env: str
) -> dict[str, Any]:
    seeds_present = {}
    missing = []
    scores = []
    for s in SEEDS:
        r = idx.get((round(rho, 10), s, env))
        if r is None:
            missing.append(s)
            continue
        sc = score_of(r)
        if sc is None or r.get("status") != "evaluation_valid":
            missing.append(s)
            seeds_present[str(s)] = {
                "status": r.get("status"),
                "training_finished": r.get("training_finished"),
                "path": r.get("path"),
                "score": sc,
            }
            continue
        seeds_present[str(s)] = {
            "status": "evaluation_valid",
            "score": sc,
            "path": r.get("path"),
            "backend": r.get("backend"),
            "tree": r.get("tree"),
        }
        scores.append(sc)
    m, sd = mean_std(scores)
    med = median(scores)
    complete = len(scores) == 4 and not missing
    return {
        "env": env,
        "rho_lr": rho,
        "n": len(scores),
        "missing_seeds": missing,
        "scores_by_seed": seeds_present,
        "mean": m,
        "std_sample": sd,
        "median": med,
        "four_seed_complete": complete,
        "mean_pm_std": (
            f"{m:.2f} ± {sd:.2f}" if complete and sd is not None else "--"
        ),
        "median_str": f"{med:.2f}" if complete and med is not None else "--",
    }


def md_table(cells: list[dict[str, Any]], antmaze: bool) -> str:
    lines = []
    if antmaze:
        lines.append("| env | n | missing | mean±std | median (paper) |")
        lines.append("|-----|---|---------|----------|----------------|")
        for c in cells:
            miss = ",".join(map(str, c["missing_seeds"])) or "—"
            lines.append(
                f"| {c['env']} | {c['n']} | {miss} | {c['mean_pm_std']} | {c['median_str']} |"
            )
    else:
        lines.append("| env | n | missing | mean±std (ddof=1) |")
        lines.append("|-----|---|---------|-------------------|")
        for c in cells:
            miss = ",".join(map(str, c["missing_seeds"])) or "—"
            lines.append(
                f"| {c['env']} | {c['n']} | {miss} | {c['mean_pm_std']} |"
            )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--audit",
        type=Path,
        default=Path(
            "/home/ext_csh/AMO_release_wt_log_audit_20260914/audit_out/classification.json"
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/home/ext_csh/AMO_release_wt_log_audit_20260914/audit_out"
        ),
    )
    ap.add_argument(
        "--amo-log-dir",
        type=Path,
        default=Path("/home/ext_csh/amo_log"),
    )
    args = ap.parse_args()

    audit = load_audit(args.audit)
    idx = index_runs(audit)

    missing_seed0 = []
    for rho in RHO:
        for env in LOCO + ANT:
            if (round(rho, 10), 0, env) not in idx:
                missing_seed0.append({"rho_lr": rho, "env": env, "reason": "no_jax_beta1_seed0_run"})

    tables: dict[str, Any] = {
        "at": datetime.now(KST).isoformat(),
        "protocol": PROTOCOL,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "value_lr": 3e-4,
        "beta_initial": 1.0,
        "note": (
            "Distinct from svcho default actor_lr=1e-3. "
            "Torch/CORL seed0 is NOT merged into these JAX cells."
        ),
        "missing_seed0": missing_seed0,
        "by_rho": {},
    }

    md_parts = [
        f"# IQL+AMO β=1 final50_singlepass_v1 (ext_csh)",
        "",
        f"- Generated: {tables['at']}",
        f"- Protocol: `{PROTOCOL}`",
        "- actor/critic/value LR: **3e-4** (not svcho 1e-3)",
        "- Eval policy: π_E (`agent.act` → main actor)",
        "- Seeds: 0–3 where JAX seed0 exists; else `missing_seed0`",
        "",
        "## Dual-actor IQL semantics (reference)",
        "",
        "Critic target is V-based IQL Bellman (`target_q` from target critic on dataset actions; "
        "value expectile). Bootstrap actor enters AMO meta/proxy terms only — "
        "**not** the critic Bellman target (unlike TD3 critic coupling).",
        "",
        "## missing_seed0 (JAX β=1 compatible)",
        "",
    ]
    if missing_seed0:
        md_parts.append("| rho_lr | env |")
        md_parts.append("|--------|-----|")
        for m in missing_seed0:
            md_parts.append(f"| {m['rho_lr']} | {m['env']} |")
    else:
        md_parts.append("_none_")
    md_parts.append("")

    for rho in RHO:
        loco_cells = [cell_report(idx, rho, e) for e in LOCO]
        ant_cells = [cell_report(idx, rho, e) for e in ANT]
        tables["by_rho"][fmt_rho(rho)] = {
            "rho_lr": rho,
            "locomotion": loco_cells,
            "antmaze": ant_cells,
            "n_four_seed_loco": sum(1 for c in loco_cells if c["four_seed_complete"]),
            "n_four_seed_ant": sum(1 for c in ant_cells if c["four_seed_complete"]),
        }
        md_parts.append(f"## rho_lr = {rho:g}")
        md_parts.append("")
        md_parts.append("### Locomotion (mean ± sample std)")
        md_parts.append("")
        md_parts.append(md_table(loco_cells, antmaze=False))
        md_parts.append("")
        md_parts.append("### AntMaze (mean±std and median; paper uses median)")
        md_parts.append("")
        md_parts.append(md_table(ant_cells, antmaze=True))
        md_parts.append("")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "aggregate_final50.json").write_text(
        json.dumps(tables, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "aggregate_final50.md").write_text("\n".join(md_parts) + "\n")

    # Mirror into amo_log host branch working tree for push.
    log_dir = args.amo_log_dir / "ext_csh" / "iql_amo_beta1_final50_v1"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "aggregate_final50.json").write_text(
        json.dumps(tables, indent=2, sort_keys=True) + "\n"
    )
    (log_dir / "aggregate_final50.md").write_text("\n".join(md_parts) + "\n")
    # classification snapshot
    if args.audit.is_file():
        (log_dir / "classification.json").write_text(args.audit.read_text())

    print(
        json.dumps(
            {
                "wrote": [
                    str(args.out_dir / "aggregate_final50.json"),
                    str(log_dir / "aggregate_final50.md"),
                ],
                "missing_seed0": len(missing_seed0),
                "four_seed": {
                    k: {
                        "loco": v["n_four_seed_loco"],
                        "ant": v["n_four_seed_ant"],
                    }
                    for k, v in tables["by_rho"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
