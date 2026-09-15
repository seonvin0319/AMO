#!/usr/bin/env python3
"""Regenerate td3-amo-jax-loco9-default.canvas.tsx from live JAX store status."""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

STORE = Path("/raid/ext_csv/AMO_store/td3_amo_jax_loco9_default_seeds0to3")
OUT = Path(
    "/home/ext_csv/.cursor/projects/home-ext-csv/canvases/"
    "td3-amo-jax-loco9-default.canvas.tsx"
)
KST = timezone(timedelta(hours=9))

ENVIRONMENTS = tuple(
    f"{domain}-{dataset}-v2"
    for domain in ("halfcheetah", "hopper", "walker2d")
    for dataset in ("medium", "medium-replay", "medium-expert")
)
ENV_SHORT = {
    environment: environment.replace("halfcheetah", "hc")
    .replace("hopper", "h")
    .replace("walker2d", "w")
    .replace("medium-replay", "mr")
    .replace("medium-expert", "me")
    .replace("medium", "m")
    .replace("-v2", "")
    for environment in ENVIRONMENTS
}
SEEDS = (0, 1, 2, 3)


def run_id(environment: str, seed: int) -> str:
    return f"td3amo_jax_{ENV_SHORT[environment]}_s{seed}"


def run_dir(environment: str, seed: int) -> Path:
    return STORE / "runs" / run_id(environment, seed)


def last_metric_step(path: Path) -> int | None:
    mj = path / "metrics.jsonl"
    if not mj.exists():
        return None
    with mj.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - 8000))
        chunk = handle.read().decode("utf-8", "replace")
    for line in reversed(chunk.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = row.get("step") or row.get("steps")
        if step is not None:
            return int(step)
    return None


def last_eval_score(path: Path) -> float | None:
    ej = path / "eval.jsonl"
    if not ej.exists():
        return None
    best = None
    with ej.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - 12000))
        chunk = handle.read().decode("utf-8", "replace")
    for line in reversed(chunk.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key in ("normalized_score", "d4rl_normalized_score", "return_mean"):
            if key in row and row[key] is not None:
                if row.get("tag") == "posthoc_cpu_final":
                    return float(row[key])
                if best is None:
                    best = float(row[key])
                    break
    return best


def fmt_step(step: int | None) -> str:
    if step is None:
        return "—"
    if step >= 1000:
        if step % 1000 == 0:
            return f"{step // 1000}k"
        return f"{step / 1000:.1f}k".rstrip("0").rstrip(".")
    return str(step)


def mean_std(vals: list[float]) -> str:
    if not vals:
        return "—"
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return f"{mean:.1f}"
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return f"{mean:.1f}±{math.sqrt(var):.1f}"


def status_for(environment: str, seed: int, jobs: dict) -> str:
    key = f"{environment}|{seed}"
    if key in jobs:
        return str(jobs[key].get("status", "—"))
    d = run_dir(environment, seed)
    if (d / "COMPLETED.json").exists():
        return "completed"
    if (d / "checkpoint.npz").exists() and (last_metric_step(d) or 0) >= 1_000_000:
        return "completed"
    if last_metric_step(d) is not None:
        return "running"
    return "pending"


def load_jobs() -> dict:
    path = STORE / "status_summary.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out = {}
    for job in payload.get("jobs", []):
        out[f"{job.get('environment')}|{job.get('seed')}"] = job
    return out


def js_str(s: str) -> str:
    return json.dumps(s)


def row_tone(status: str) -> str:
    if status == "completed":
        return "success"
    if status == "running":
        return "info"
    if status in ("failed", "interrupted", "cancelled"):
        return "danger"
    return "neutral"


def main() -> None:
    jobs = load_jobs()
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
    counts: dict[str, int] = {}
    table_rows: list[list[str]] = []
    tones: list[str] = []
    for environment in ENVIRONMENTS:
        scores: list[float] = []
        for seed in SEEDS:
            d = run_dir(environment, seed)
            st = status_for(environment, seed, jobs)
            counts[st] = counts.get(st, 0) + 1
            step = last_metric_step(d)
            score = last_eval_score(d)
            if score is not None and st == "completed":
                scores.append(score)
            table_rows.append(
                [
                    ENV_SHORT[environment],
                    f"s{seed}",
                    st,
                    fmt_step(step),
                    "—" if score is None else f"{score:.1f}",
                ]
            )
            tones.append(row_tone(st))
        # mean row omitted — keep one row per seed for scanning

    pinned = ""
    pin_path = Path("/home/ext_csv/AMO-main/.pinned_sha")
    if pin_path.exists():
        pinned = pin_path.read_text().strip()[:12]

    # Serialize rows/tones as JS literals
    rows_js = ",\n    ".join(
        "[" + ", ".join(js_str(c) for c in row) + "]" for row in table_rows
    )
    tones_js = ", ".join(js_str(t) for t in tones)
    stats = []
    for key in ("completed", "running", "pending", "failed", "cancelled", "interrupted"):
        if key in counts:
            tone = {
                "completed": "success",
                "running": "info",
                "failed": "danger",
                "cancelled": "warning",
                "interrupted": "warning",
            }.get(key)
            tone_attr = f' tone="{tone}"' if tone else ""
            stats.append(
                f'      <Stat value="{counts[key]}" label="{key}"{tone_attr} />'
            )

    tsx = f'''import {{ H1, H2, Row, Stack, Stat, Table, Text }} from "cursor/canvas";

const ROWS = [
    {rows_js}
];
const ROW_TONE = [{tones_js}] as const;

export default function TD3AMOJaxLoco9() {{
  return (
    <Stack gap={{16}}>
      <H1>TD3+AMO JAX loco9</H1>
      <Text tone="secondary" size="small">
        alpha_E=alpha_B=2 · alpha_lr=1e-3 · seeds 0–3 · updated {now} · sha {pinned or "—"}
      </Text>
      <Text tone="tertiary" size="small">
        Source: /raid/ext_csv/AMO_store/td3_amo_jax_loco9_default_seeds0to3 · AMO-main train.py --backend jax
      </Text>
      <Row gap={{16}}>
{chr(10).join(stats) if stats else '      <Stat value="0" label="cells" />'}
      </Row>
      <H2>Per-seed status</H2>
      <Table
        headers={{["env", "seed", "status", "step", "last eval (norm)"]}}
        rows={{ROWS}}
        rowTone={{[...ROW_TONE]}}
        striped
        stickyHeader
        columnAlign={{["left", "left", "left", "right", "right"]}}
      />
    </Stack>
  );
}}
'''
    OUT.write_text(tsx)
    print(f"wrote {OUT} counts={counts}")


if __name__ == "__main__":
    main()
