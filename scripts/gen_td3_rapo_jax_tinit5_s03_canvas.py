#!/usr/bin/env python3
"""Rebuild td3-rapo-jax-tinit5-td3bc-hw.canvas.tsx from the live JAX grid."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
SWEEP = Path("/home/shchoi/AMO_td3-amo-bootrms/results/td3_rapo_jax_tinit5_td3bc_hw_s03")
CANVAS = Path(
    "/home/shchoi/.cursor/projects/home-shchoi/canvases/"
    "td3-rapo-jax-tinit5-td3bc-hw.canvas.tsx"
)

LOCO_ORDER = (
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-medium-expert-v2",
)
ENV_ORDER = LOCO_ORDER
ENV_ID = {
    "hopper-medium-v2": "h-m",
    "hopper-medium-replay-v2": "h-mr",
    "hopper-medium-expert-v2": "h-me",
    "walker2d-medium-v2": "w-m",
    "walker2d-medium-replay-v2": "w-mr",
    "walker2d-medium-expert-v2": "w-me",
}
SHORT_TO_ENV = {v: k for k, v in ENV_ID.items()}
RHOS = ("3em4", "1em3", "2em3")
RHO_LABEL = {"3em4": "3e-4", "1em3": "1e-3", "2em3": "2e-3"}
SEEDS = (0, 1, 2, 3)
TOTAL = 72
RUN_RE = re.compile(
    r"^td3rapo_jax_t5_tlr(?P<rho>.+)_(?P<short>.+)_s(?P<seed>\d+)$"
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def metrics_step(run_dir: Path) -> int:
    lines = load_jsonl(run_dir / "metrics.jsonl")
    if not lines:
        return 0
    return int(lines[-1].get("step") or lines[-1].get("steps") or 0)


def final_score(run_dir: Path) -> float | None:
    best = None
    for row in load_jsonl(run_dir / "eval.jsonl"):
        step = int(row.get("step") or 0)
        if step < 999_000:
            continue
        if row.get("tag") == "posthoc_cpu":
            continue
        score = row.get("normalized_score")
        if score is None:
            score = row.get("mean_normalized")
        if score is None:
            continue
        best = float(score)
    return best


def parse_run(name: str) -> tuple[str, str, int] | None:
    match = RUN_RE.match(name)
    if not match:
        return None
    env = SHORT_TO_ENV.get(match.group("short"))
    if not env:
        return None
    return env, match.group("rho"), int(match.group("seed"))


def js_string_matrix(rows: list[list[str]]) -> str:
    lines = []
    for row in rows:
        cells = ", ".join(json.dumps(c) for c in row)
        lines.append(f"  [{cells}]")
    return ",\n".join(lines)


def cell_label(cell: dict | None) -> str:
    if not cell:
        return "-"
    if cell.get("score") is not None:
        return f"{float(cell['score']):.1f}"
    status = cell.get("status")
    step = int(cell.get("step") or 0)
    if status == "running":
        return f"{step // 1000}k"
    if status == "completed" or step >= 1_000_000:
        return "1M"
    if status == "failed":
        return "fail"
    return "-"


def collect() -> tuple[dict, list[dict], dict]:
    grid: dict[tuple, dict] = {}
    running: list[dict] = []
    counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
    summary = SWEEP / "status_summary.json"
    if summary.exists():
        try:
            payload = json.loads(summary.read_text())
            counts.update(payload.get("counts") or {})
            jobs = payload.get("jobs") or []
        except (OSError, ValueError, json.JSONDecodeError):
            jobs = []
    else:
        jobs = []

    for job in jobs:
        env = job.get("environment")
        rho = job.get("rho_tag")
        seed = int(job["seed"]) if job.get("seed") is not None else -1
        if env not in ENV_ID or rho not in RHOS or seed not in SEEDS:
            continue
        run_dir = Path(job.get("output") or "")
        status = str(job.get("status") or "pending")
        pid = job.get("pid")
        if status == "running" and pid and not pid_alive(int(pid)):
            status = "pending"
        step = metrics_step(run_dir) if run_dir.is_dir() else 0
        score = final_score(run_dir) if run_dir.is_dir() else None
        rec = {
            "env": env,
            "rho": rho,
            "seed": seed,
            "status": status,
            "step": step,
            "score": score,
            "run_id": job.get("run_id"),
        }
        grid[(env, rho, seed)] = rec
        if status == "running":
            running.append(rec)

    runs = SWEEP / "runs"
    if runs.is_dir():
        for run_dir in runs.iterdir():
            parsed = parse_run(run_dir.name)
            if not parsed:
                continue
            env, rho, seed = parsed
            key = (env, rho, seed)
            score = final_score(run_dir)
            step = metrics_step(run_dir)
            rec = grid.get(key) or {
                "env": env,
                "rho": rho,
                "seed": seed,
                "status": "pending",
                "step": 0,
                "score": None,
                "run_id": run_dir.name,
            }
            rec["step"] = max(int(rec.get("step") or 0), step)
            if score is not None:
                rec["score"] = score
            if rec["status"] == "pending" and rec["step"] >= 1_000_000:
                rec["status"] = "completed"
            grid[key] = rec

    return grid, running, counts


def table_for(envs: tuple[str, ...], rho: str, grid: dict) -> str | None:
    rows = []
    for env in envs:
        recs = [grid.get((env, rho, seed)) for seed in SEEDS]
        if not any(
            rec
            and (
                rec.get("score") is not None
                or rec.get("status") in {"running", "completed", "failed"}
                or int(rec.get("step") or 0) > 0
            )
            for rec in recs
        ):
            continue
        cells = [ENV_ID[env]]
        vals = []
        for rec in recs:
            cells.append(cell_label(rec))
            if rec and rec.get("score") is not None:
                vals.append(float(rec["score"]))
        cells.append(f"{sum(vals) / len(vals):.1f}" if vals else "-")
        rows.append(cells)
    if not rows:
        return None
    return f"""
        <Table
          headers={{["env", "s0", "s1", "s2", "s3", "mean"]}}
          rows={{[
{js_string_matrix(rows)}
          ]}}
        />"""


def render(grid: dict, running: list[dict], counts: dict) -> str:
    snap = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
    scored = sum(1 for rec in grid.values() if rec.get("score") is not None)
    trained = sum(
        1
        for rec in grid.values()
        if rec.get("status") == "completed" or int(rec.get("step") or 0) >= 1_000_000
    )
    running_n = len(running) if running else int(counts.get("running") or 0)
    pending = int(counts.get("pending") or max(0, TOTAL - trained - running_n))
    failed = int(counts.get("failed") or 0)
    rest = max(0, TOTAL - trained - running_n)
    live = (
        " · ".join(
            f"{r['run_id']} {int(r.get('step') or 0) // 1000}k" for r in running
        )
        or "none"
    )

    usage_parts = [f'{{ id: "done", value: {trained}, color: "green" }}']
    if running_n:
        usage_parts.append(f'{{ id: "run", value: {running_n}, color: "blue" }}')
    if rest:
        usage_parts.append(f'{{ id: "rest", value: {rest}, color: "gray" }}')
    usage_segments = ",\n          ".join(usage_parts)

    table_blocks = []
    for rho in RHOS:
        table = table_for(LOCO_ORDER, rho, grid)
        if not table:
            continue
        table_blocks.append(
            f"""
        <Stack gap={{8}}>
          <Text tone="secondary">
            hopper then walker · α_E=α_B=5 · env × seeds 0–3 + mean of 1M D4RL normalized ·
            running shows step k · 1M = train done, eval pending
          </Text>
          <H2>hopper → walker · α_lr = {RHO_LABEL[rho]}</H2>{table}
        </Stack>"""
        )
    matrices = "\n\n      <Divider />\n".join(table_blocks)

    fail_pill = ""
    if failed:
        fail_pill = f"""
          <Pill tone="danger" active>
            failed {failed}
          </Pill>"""

    score_section = ""
    if matrices:
        score_section = f"""
      <Divider />

      {matrices}"""

    return f"""import {{
  Callout,
  Divider,
  Grid,
  H1,
  H2,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  UsageBar,
}} from "cursor/canvas";

export default function Td3RapoJaxTinit5Td3bcHw() {{
  return (
    <Stack gap={{24}}>
      <Stack gap={{8}}>
        <H1>TD3 RAPO · α=5 · critic 2-layer no LN</H1>
        <Text tone="secondary">
          JAX RAPO π_E −B_π · π_B L2RMS · critic target π_E · 2 hidden layers ·
          no LayerNorm · α_E=α_B=5 · hopper then walker ·
          α_lr 3e-4 / 1e-3 / 2e-3 · train-only / CPU eval at 1M · snapshot {snap}
        </Text>
        <Row gap={{8}} wrap>
          <Pill tone="success" active>
            1M scored {scored}
          </Pill>
          <Pill tone="info" active>
            running {running_n}
          </Pill>
          <Pill>
            pending {pending}
          </Pill>{fail_pill}
        </Row>
      </Stack>

      <Grid columns={{4}} gap={{16}}>
        <Stat value="{scored}/{TOTAL}" label="cells with 1M score" tone="success" />
        <Stat value="{trained}" label="1M trained" tone="info" />
        <Stat value="{running_n}" label="live trains" tone="info" />
        <Stat value="{pending}" label="queued" />
      </Grid>

      <UsageBar
        total={{{TOTAL}}}
        topLeftLabel="{trained} trained · {scored} scored · {rest} remaining"
        topRightLabel="live: {live}"
        segments={{[
          {usage_segments}
        ]}}
      />

      <Callout tone="info" title="Protocol">
        RAPO JAX · π_E −B_π · π_B L2RMS only · critic target = π_E ·
        critic 2 hidden layers, no LayerNorm · α_E=α_B=5 · hopper then walker ·
        no in-process eval · ckpt 20k · CPU posthoc 10×5 at 1M ·
        GPU0 max-parallel 2
      </Callout>

      <Callout tone="neutral" title="Live now">
        {live}
      </Callout>
      {score_section}
    </Stack>
  );
}}
"""


def main() -> int:
    grid, running, counts = collect()
    CANVAS.write_text(render(grid, running, counts))
    scored = sum(1 for rec in grid.values() if rec.get("score") is not None)
    print(
        f"wrote {CANVAS} scored={scored} running={len(running)} "
        f"pending={counts.get('pending', '?')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
