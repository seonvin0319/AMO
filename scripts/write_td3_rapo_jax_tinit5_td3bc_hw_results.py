#!/usr/bin/env python3
"""Write sweep_results/td3_rapo_jax_tinit5_td3bc_hw from local RAPO 1M evals."""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "results" / "td3_rapo_jax_tinit5_td3bc_hw_s03"
RUNS = SWEEP / "runs"
OUT = ROOT / "sweep_results" / "td3_rapo_jax_tinit5_td3bc_hw"
KST = timezone(timedelta(hours=9))

ENVS = (
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-medium-expert-v2",
)
ENV_SHORT = {
    "hopper-medium-v2": "h-m",
    "hopper-medium-replay-v2": "h-mr",
    "hopper-medium-expert-v2": "h-me",
    "walker2d-medium-v2": "w-m",
    "walker2d-medium-replay-v2": "w-mr",
    "walker2d-medium-expert-v2": "w-me",
}
RHOS = ("2em3", "1em3", "3em4")
RHO_LABEL = {"2em3": "2e-3", "1em3": "1e-3", "3em4": "3e-4"}
SEEDS = (0, 1, 2, 3)
TOTAL = len(ENVS) * len(RHOS) * len(SEEDS)
RUN_RE = re.compile(
    r"^td3rapo_jax_t5_tlr(?P<rho>.+)_(?P<short>.+)_s(?P<seed>\d+)$"
)
SHORT_TO_ENV = {v: k for k, v in ENV_SHORT.items()}


def last_metric_step(run_dir: Path) -> int:
    metrics = run_dir / "metrics.jsonl"
    if not metrics.is_file() or metrics.stat().st_size == 0:
        return 0
    last = None
    for line in metrics.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            last = json.loads(line)
        except json.JSONDecodeError:
            continue
    if not last:
        return 0
    return int(last.get("step") or last.get("steps") or 0)


def final_score(run_dir: Path) -> tuple[float | None, int | None]:
    path = run_dir / "eval.jsonl"
    if not path.is_file() or path.stat().st_size == 0:
        return None, None
    best = None
    episodes = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(row.get("step") or 0) < 999_000:
            continue
        score = row.get("normalized_score")
        if score is None:
            score = row.get("mean_normalized")
        if score is None:
            continue
        best = float(score)
        episodes = int(row.get("episodes") or 0)
    return best, episodes


def trained_1m(run_dir: Path) -> bool:
    ckpt = run_dir / "checkpoints" / "step_1000000.npz"
    if ckpt.is_file():
        return True
    return last_metric_step(run_dir) >= 1_000_000


def collect() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    scored = trained = 0
    for rho in RHOS:
        for env in ENVS:
            for seed in SEEDS:
                short = ENV_SHORT[env]
                name = f"td3rapo_jax_t5_tlr{rho}_{short}_s{seed}"
                run_dir = RUNS / name
                score, episodes = (None, None)
                step = 0
                done = False
                if run_dir.is_dir():
                    score, episodes = final_score(run_dir)
                    step = last_metric_step(run_dir)
                    done = trained_1m(run_dir)
                if score is not None:
                    scored += 1
                if done:
                    trained += 1
                rows.append(
                    {
                        "env": env,
                        "short": short,
                        "alpha_lr": RHO_LABEL[rho],
                        "rho_tag": rho,
                        "seed": seed,
                        "step": step,
                        "trained_1m": int(done),
                        "normalized_score": "" if score is None else f"{score:.6f}",
                        "episodes": "" if episodes is None else episodes,
                        "run": name,
                    }
                )
    summary = SWEEP / "status_summary.json"
    counts = {}
    if summary.is_file():
        try:
            counts = json.loads(summary.read_text()).get("counts") or {}
        except (OSError, json.JSONDecodeError, TypeError):
            counts = {}
    status = {
        "updated_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "algorithm": "td3_amo",
        "variant": "rapo",
        "backend": "jax",
        "alpha_E": 5.0,
        "alpha_B": 5.0,
        "critic_depth": 2,
        "critic_layernorm": False,
        "critic_target": "actor",
        "bootstrap_loss": "l2_rms",
        "alpha_lrs": [RHO_LABEL[r] for r in RHOS],
        "queue_order": "alpha_lr 2e-3 then 1e-3 then 3e-4, seed 0-3, hopper then walker",
        "envs": list(ENVS),
        "seeds": list(SEEDS),
        "total": TOTAL,
        "trained_1m": trained,
        "scored_1m": scored,
        "counts": counts,
        "source": str(RUNS),
        "eval": "CPU posthoc 10x5 at 1M",
    }
    return rows, status


AM_HC_ENVS = (
    "antmaze-umaze-v2",
    "antmaze-umaze-diverse-v2",
    "antmaze-medium-play-v2",
    "antmaze-medium-diverse-v2",
    "antmaze-large-play-v2",
    "antmaze-large-diverse-v2",
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-medium-expert-v2",
)


def load_am_hc(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def am_hc_section(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    by: dict[tuple[str, int], str] = {}
    trained = scored = 0
    for row in rows:
        seed = int(row["seed"])
        by[(row["env"], seed)] = row.get("normalized_score") or ""
        trained += int(row.get("trained_1m") or 0)
        if row.get("normalized_score"):
            scored += 1
    lines = [
        "",
        "## antmaze / halfcheetah",
        "",
        "Same critic (2-layer, no LayerNorm, target π_E), α_E=α_B=5, α_lr 3e-4 only. "
        "1e-3 and 2e-3 were not trained. CPU eval 10×5 at 1M. "
        "A blank cell has no final eval. 0.0 is a recorded score. "
        f"Trained {trained}/{len(rows)}, scored {scored}/{len(rows)}.",
        "",
        "| env | s0 | s1 | s2 | s3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for env in AM_HC_ENVS:
        cells = [env]
        for seed in SEEDS:
            raw = by.get((env, seed), "")
            cells.append("—" if raw == "" else f"{float(raw):.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def write_summary_md(rows: list[dict], status: dict) -> str:
    by: dict[tuple[str, str, int], float] = {}
    for row in rows:
        if row["normalized_score"] == "":
            continue
        by[(row["rho_tag"], row["short"], int(row["seed"]))] = float(
            row["normalized_score"]
        )
    lines = [
        "# TD3 RAPO hopper-walker",
        "",
        f"Updated {status['updated_at']}.",
        "",
        "π_E −B_π, π_B L2 RMS, critic 2-layer no LayerNorm, target π_E, "
        "α_E=α_B=5, seeds 0–3, CPU eval 10×5 at 1M.",
        "",
        f"Trained {status['trained_1m']}/{status['total']}, "
        f"scored {status['scored_1m']}/{status['total']}.",
        "",
        "| env | 2e-3 μ (seeds) | 1e-3 μ (seeds) | 3e-4 μ (seeds) |",
        "|---|---|---|---|",
    ]
    for env in ENVS:
        short = ENV_SHORT[env]
        cells = [env]
        for rho in RHOS:
            vals = [
                (seed, by[(rho, short, seed)])
                for seed in SEEDS
                if (rho, short, seed) in by
            ]
            if not vals:
                cells.append("—")
                continue
            mu = sum(v for _, v in vals) / len(vals)
            parts = "/".join(f"{v:.1f}" for _, v in vals)
            cells.append(f"{mu:.1f} ({parts})")
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(am_hc_section(load_am_hc(OUT / "am_hc_3e4.csv")))
    return "\n".join(lines)


def write() -> dict:
    rows, status = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "scores_long.csv"
    fields = [
        "env",
        "short",
        "alpha_lr",
        "rho_tag",
        "seed",
        "step",
        "trained_1m",
        "normalized_score",
        "episodes",
        "run",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        for extra in load_am_hc(OUT / "am_hc_3e4.csv"):
            writer.writerow({key: extra.get(key, "") for key in fields})
    (OUT / "STATUS.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "SUMMARY.md").write_text(write_summary_md(rows, status), encoding="utf-8")
    return status


if __name__ == "__main__":
    payload = write()
    print(
        f"[td3_rapo hw] trained={payload['trained_1m']}/{payload['total']} "
        f"scored={payload['scored_1m']} wrote {OUT}"
    )
