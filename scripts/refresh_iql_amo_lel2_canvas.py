#!/usr/bin/env python3
"""Collect IQL-AMO π_E L_E+L2_RMS 540 (β=5,1,2); rewrite Cursor canvas + PROGRESS.md."""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
OUT = Path("/home/svcho/amo/results/iql_amo_lel2_seeds03")
JOBS = OUT / "jobs"
CANVAS = Path(
    "/home/svcho/.cursor/projects/home-svcho/canvases/iql-amo-lel2.canvas.tsx"
)
SNAP = OUT / "canvas_snapshot.json"
PROGRESS = OUT / "PROGRESS.md"

ALPHAS = [5, 1, 2]
SEEDS = [0, 1, 2, 3]
ALRS = [("3e-4", "r3e4"), ("1e-3", "r1e3"), ("2e-3", "r2e3")]
ENVS = [
    ("hopper-medium-v2", "h-m"),
    ("hopper-medium-replay-v2", "h-mr"),
    ("hopper-medium-expert-v2", "h-me"),
    ("walker2d-medium-v2", "w-m"),
    ("walker2d-medium-replay-v2", "w-mr"),
    ("walker2d-medium-expert-v2", "w-me"),
    ("antmaze-umaze-v2", "am-u"),
    ("antmaze-umaze-diverse-v2", "am-ud"),
    ("antmaze-medium-play-v2", "am-mp"),
    ("antmaze-medium-diverse-v2", "am-md"),
    ("antmaze-large-play-v2", "am-lp"),
    ("antmaze-large-diverse-v2", "am-ld"),
    ("halfcheetah-medium-v2", "hc-m"),
    ("halfcheetah-medium-replay-v2", "hc-mr"),
    ("halfcheetah-medium-expert-v2", "hc-me"),
]
STEPS = 1_000_000
TOTAL = len(ALPHAS) * len(SEEDS) * len(ENVS) * len(ALRS)  # 540


def make_tag(a: int, short: str, rt: str, seed: int) -> str:
    return f"b{a}_{short}_{rt}_s{seed}"


def live_trainers() -> dict[str, int]:
    out: dict[str, int] = {}
    ps = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    needle = str(JOBS)
    for line in ps.splitlines():
        if "train.py" not in line or "td3_amo" not in line or needle not in line:
            continue
        pid = int(line.strip().split()[0])
        i = line.find("/jobs/")
        if i < 0:
            continue
        t = line[i + 6 :].split("/run")[0]
        out[t] = pid
    return out


def live_evals() -> set[str]:
    out: set[str] = set()
    ps = subprocess.check_output(["ps", "-eo", "args="], text=True)
    needle = str(JOBS)
    for line in ps.splitlines():
        if "eval_checkpoints_cpu.py" not in line or needle not in line:
            continue
        i = line.find("/jobs/")
        if i < 0:
            continue
        rest = line[i + 6 :]
        t = rest.split()[0].rstrip("/")
        if "/" in t:
            t = t.split("/")[0]
        out.add(t)
    return out


def last_metrics(path: Path) -> tuple[int, dict | None]:
    if not path.exists():
        return 0, None
    last = None
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                pass
    if not last:
        return 0, None
    return int(last.get("step") or 0), last


def final_score(run: Path) -> float | None:
    ej = run / "eval.jsonl"
    if not ej.exists():
        return None
    best = None
    for line in ej.read_text().splitlines():
        if not line.strip():
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(o.get("step") or 0) < STEPS:
            continue
        eps = int(o.get("episodes") or 0)
        tag_ = o.get("tag")
        if tag_ != "posthoc_cpu_final" and eps < 50:
            continue
        try:
            best = float(o["normalized_score"])
        except (KeyError, TypeError, ValueError):
            continue
    return best


def gpu_info() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except Exception:
        return "n/a"


def mean_std(xs: list[float]) -> str:
    if not xs:
        return "—"
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return f"{m:.1f}"
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return f"{m:.1f} ± {sd:.1f}"


def cell_label(score: float | None, status: str, step: int, evaluating: bool) -> str:
    if score is not None:
        return f"{score:.1f}"
    if evaluating:
        return "eval…"
    if status == "done":
        return "[T]"
    if status == "running":
        return f"run {max(1, step // 1000)}k"
    if status == "failed":
        return "FAIL"
    return "—"


def collect() -> dict:
    now = time.time()
    live = live_trainers()
    evaluating = live_evals()
    done = running = queued = failed = imported = 0
    eval_done = eval_pending = eval_running = 0
    live_sps: list[float] = []
    rem_steps = 0
    live_rows: list[dict] = []
    alpha_prog: list[dict] = []
    score_rows: list[dict] = []
    all_scores: list[float] = []

    for a in ALPHAS:
        a_done = a_run = a_eval = a_fail = 0
        a_total = len(SEEDS) * len(ENVS) * len(ALRS)
        for short_name, short in ENVS:
            for alr, rt in ALRS:
                seeds_lab: list[str] = []
                seeds_num: list[float] = []
                for seed in SEEDS:
                    t = make_tag(a, short, rt, seed)
                    cell = JOBS / t
                    run = cell / "run"
                    status = "queued"
                    step = 0
                    score = final_score(run) if run.exists() else None
                    if score is not None:
                        eval_done += 1
                        all_scores.append(score)
                        seeds_num.append(score)
                    if (cell / "DONE").exists():
                        status = "done"
                        done += 1
                        a_done += 1
                        step = STEPS
                        if (cell / "IMPORTED_ABLATION").exists():
                            imported += 1
                        if score is None:
                            if t in evaluating:
                                eval_running += 1
                                a_eval += 0
                            else:
                                eval_pending += 1
                        else:
                            a_eval += 1
                    elif (cell / "FAILED").exists():
                        status = "failed"
                        failed += 1
                        a_fail += 1
                        rem_steps += STEPS
                    elif t in live:
                        status = "running"
                        running += 1
                        a_run += 1
                        step, last = last_metrics(run / "metrics.jsonl")
                        rem_steps += max(0, STEPS - step)
                        sps = None
                        if last and step > 5000:
                            # rough from process age
                            try:
                                st = os.stat(f"/proc/{live[t]}")
                                age = max(1.0, now - st.st_ctime)
                                sps = round(step / age, 1)
                                live_sps.append(sps)
                            except Exception:
                                pass
                        live_rows.append(
                            {
                                "tag": t,
                                "step": step,
                                "pct": round(100.0 * step / STEPS, 1),
                                "sps": sps,
                                "alpha_E": None
                                if not last
                                else last.get("alpha_E") or last.get("train/alpha_E"),
                                "alpha_B": None
                                if not last
                                else last.get("alpha_B") or last.get("train/alpha_B"),
                            }
                        )
                    else:
                        queued += 1
                        rem_steps += STEPS
                        if cell.exists() and (run / "metrics.jsonl").exists():
                            step, _ = last_metrics(run / "metrics.jsonl")
                            if step > 0:
                                status = "paused"
                                rem_steps = rem_steps - STEPS + max(0, STEPS - step)

                    seeds_lab.append(
                        cell_label(score, status, step, t in evaluating)
                    )
                # only emit score rows when something interesting exists
                if any(x not in ("—",) for x in seeds_lab):
                    score_rows.append(
                        {
                            "a": a,
                            "env": short,
                            "alr": alr,
                            "s0": seeds_lab[0],
                            "s1": seeds_lab[1],
                            "s2": seeds_lab[2],
                            "s3": seeds_lab[3],
                            "mean": mean_std(seeds_num)
                            if seeds_num
                            else (" · ".join(seeds_lab) if any(x.startswith("run") or x in ("[T]", "eval…") for x in seeds_lab) else "—"),
                            "n": len(seeds_num),
                        }
                    )
        alpha_prog.append(
            {
                "a": a,
                "done": a_done,
                "running": a_run,
                "eval": a_eval,
                "failed": a_fail,
                "total": a_total,
            }
        )

    pool_sps = sum(live_sps) if live_sps else 0.0
    per_job = (pool_sps / len(live_sps)) if live_sps else 0.0
    eta_h = (rem_steps / pool_sps / 3600.0) if pool_sps > 0 else None

    active = next(
        (f"α={ap['a']}" for ap in alpha_prog if ap["running"] or (0 < ap["done"] < ap["total"])),
        "α=5",
    )
    if live_rows:
        active = live_rows[0]["tag"].rsplit("_s", 1)[0]

    score_summary = None
    if all_scores:
        score_summary = {
            "n": len(all_scores),
            "mean": round(sum(all_scores) / len(all_scores), 1),
            "min": round(min(all_scores), 1),
            "max": round(max(all_scores), 1),
        }

    # keep table manageable: prefer rows with scores or live/train
    hot_rows = [
        r
        for r in score_rows
        if r["n"] > 0
        or any(
            str(r[k]).startswith("run") or r[k] in ("[T]", "eval…", "FAIL")
            for k in ("s0", "s1", "s2", "s3")
        )
    ]

    return {
        "updated_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST"),
        "done": done,
        "running": running,
        "queued": queued,
        "failed": failed,
        "imported": imported,
        "total": TOTAL,
        "eval_done": eval_done,
        "eval_pending": eval_pending,
        "eval_running": eval_running,
        "pool_sps": round(pool_sps, 1),
        "per_job_sps": round(per_job, 1),
        "rem_steps_m": round(rem_steps / 1e6, 2),
        "eta_h": round(eta_h, 2) if eta_h is not None else None,
        "gpu": gpu_info(),
        "active": active,
        "alpha_prog": alpha_prog,
        "live": live_rows,
        "score_rows": hot_rows[:80],
        "score_summary": score_summary,
        "schedule": (
            "IQL π_E · L_E+L2_RMS · β 5→1 then 2 · seed0–3 · "
            "hop→walk→ant→hc · α_lr 3e-4→1e-3→2e-3 · 540 · P=3 · CPU final∥=6"
        ),
        "legend": "[T]=train DONE · number=norm final50 · eval…=CPU · run NNk=train · import counted in DONE",
    }


def render(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    return f"""import {{
  Callout,
  Code,
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

/** Auto-refreshed by scripts/refresh_td3_amo_execonly_main_canvas.py — do not hand-edit DATA. */
const DATA = {payload} as const;

function fmtEta(h: number | null) {{
  if (h == null) return "TBD";
  if (h < 1) return `${{Math.round(h * 60)}}m`;
  const hh = Math.floor(h);
  const mm = Math.round((h - hh) * 60);
  return `${{hh}}h ${{mm}}m`;
}}

export default function Td3AmoExeconlyMain() {{
  const pct = (100 * DATA.done) / DATA.total;
  const evalPct = DATA.done > 0 ? (100 * DATA.eval_done) / DATA.done : 0;

  return (
    <Stack gap={{24}} style={{{{ padding: 24, maxWidth: 1120 }}}}>
      <Stack gap={{8}}>
        <H1>TD3-AMO · π_E only · 540</H1>
        <Text tone="secondary">{{DATA.schedule}}</Text>
        <Row gap={{8}} wrap>
          <Pill tone="info">{{DATA.active}}</Pill>
          <Pill>{{DATA.updated_at}}</Pill>
          <Pill tone="neutral">import {{DATA.imported}}</Pill>
          <Pill tone={{DATA.failed ? "danger" : "success"}}>
            fail {{DATA.failed}}
          </Pill>
        </Row>
      </Stack>

      <Grid columns={{4}} gap={{12}}>
        <Stat
          value={{`${{DATA.done}}/${{DATA.total}}`}}
          label="Train DONE"
          tone="success"
        />
        <Stat
          value={{`${{DATA.eval_done}}/${{DATA.done || 0}}`}}
          label="Final eval (of train done)"
          tone="info"
        />
        <Stat value={{`${{DATA.pool_sps}}`}} label="Pool steps/s" />
        <Stat value={{fmtEta(DATA.eta_h)}} label="Train ETA" />
      </Grid>

      <UsageBar
        total={{DATA.total}}
        topLeftLabel={{`${{pct.toFixed(1)}}% train done`}}
        topRightLabel={{`${{DATA.done}} done · ${{DATA.running}} run · ${{DATA.queued}} queued`}}
        segments={{[
          {{ id: "done", value: DATA.done, color: "green" }},
          {{ id: "running", value: DATA.running, color: "blue" }},
          {{ id: "failed", value: DATA.failed, color: "orange" }},
        ]}}
      />

      <UsageBar
        total={{Math.max(DATA.done, 1)}}
        topLeftLabel={{`Final50 of trained: ${{evalPct.toFixed(0)}}%`}}
        topRightLabel={{`${{DATA.eval_done}} scored · ${{DATA.eval_running}} eval… · ${{DATA.eval_pending}} pending`}}
        segments={{[
          {{ id: "scored", value: DATA.eval_done, color: "green" }},
          {{ id: "evaling", value: DATA.eval_running, color: "blue" }},
          {{ id: "pending", value: DATA.eval_pending, color: "yellow" }},
        ]}}
      />

      <Callout tone="info" title="Throughput">
        <Text>
          Parallel=2 · pool ≈ {{DATA.pool_sps}} steps/s · per job ≈ {{DATA.per_job_sps}}{" "}
          · GPU {{DATA.gpu}} · remaining {{DATA.rem_steps_m}}M · ETA ≈ {{fmtEta(DATA.eta_h)}}.
        </Text>
        <Text tone="secondary" size="small">
          {{DATA.legend}}
        </Text>
      </Callout>

      {{DATA.score_summary ? (
        <Callout tone="success" title="Final50 scores so far">
          <Text>
            n={{DATA.score_summary.n}} · mean={{DATA.score_summary.mean}} ·
            min={{DATA.score_summary.min}} · max={{DATA.score_summary.max}}
          </Text>
        </Callout>
      ) : null}}

      <Stack gap={{12}}>
        <H2>α barrier (180 cells each)</H2>
        <Table
          headers={{["α_init", "train", "eval", "running", "fail"]}}
          columnAlign={{["left", "right", "right", "right", "right"]}}
          striped
          rows={{DATA.alpha_prog.map((e) => [
            String(e.a),
            `${{e.done}}/${{e.total}}`,
            `${{e.eval}}/${{e.total}}`,
            String(e.running),
            String(e.failed),
          ])}}
        />
      </Stack>

      {{DATA.live.length > 0 ? (
        <Stack gap={{12}}>
          <H2>Live trainers</H2>
          <Table
            headers={{["Tag", "Step", "%", "steps/s", "α_E", "α_B"]}}
            columnAlign={{["left", "right", "right", "right", "right", "right"]}}
            striped
            rows={{DATA.live.map((r) => [
              r.tag,
              r.step.toLocaleString(),
              `${{r.pct}}%`,
              r.sps == null ? "—" : String(r.sps),
              r.alpha_E == null ? "—" : String(r.alpha_E),
              r.alpha_B == null ? "—" : String(r.alpha_B),
            ])}}
          />
        </Stack>
      ) : null}}

      {{DATA.score_rows.length > 0 ? (
        <Stack gap={{12}}>
          <H2>Active / scored cells (α × env × α_lr)</H2>
          <Table
            headers={{["α", "Env", "α_lr", "s0", "s1", "s2", "s3", "mean"]}}
            columnAlign={{[
              "right",
              "left",
              "right",
              "right",
              "right",
              "right",
              "right",
              "right",
            ]}}
            striped
            stickyHeader
            rows={{DATA.score_rows.map((r) => [
              String(r.a),
              r.env,
              r.alr,
              r.s0,
              r.s1,
              r.s2,
              r.s3,
              r.mean,
            ])}}
          />
        </Stack>
      ) : null}}

      <Divider />
      <Text tone="secondary" size="small">
        OUT <Code>results/td3_amo_execonly_main_seeds03</Code> · refresh{" "}
        <Code>scripts/refresh_td3_amo_execonly_main_canvas.py</Code>
      </Text>
    </Stack>
  );
}}
"""


def write_progress(data: dict) -> None:
    lines = [
        "# IQL-AMO π_E L_E+L2_RMS",
        "",
        f"**Snapshot:** {data['updated_at']}",
        f"**Setting:** `execution_meta_loss=le_l2_rms` · IQL original π_E · nest after TD3 5→1, β=2 last · **{data['total']}** cells",
        f"**OUT:** `{OUT}`",
        "",
        "## Live",
        "",
        f"- Train DONE **{data['done']}/{data['total']}** (import {data['imported']}) · run {data['running']} · queued {data['queued']} · fail {data['failed']}",
        f"- Final eval **{data['eval_done']}** scored · {data['eval_running']} eval… · {data['eval_pending']} pending",
        f"- Pool ≈ {data['pool_sps']} steps/s · GPU {data['gpu']} · rem {data['rem_steps_m']}M · ETA {data['eta_h'] if data['eta_h'] is not None else 'TBD'}h",
        f"- Active: `{data['active']}`",
    ]
    if data["live"]:
        lines.append("- Live: " + ", ".join(f"`{r['tag']}`@{r['step']}" for r in data["live"]))
    if data.get("score_summary"):
        s = data["score_summary"]
        lines.append(f"- Scores so far: n={s['n']} mean={s['mean']} [{s['min']},{s['max']}]")
    lines.append("")
    PROGRESS.write_text("\n".join(lines) + "\n")


def main() -> None:
    data = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    SNAP.write_text(json.dumps(data, indent=2) + "\n")
    CANVAS.write_text(render(data))
    write_progress(data)
    brief = {
        k: data[k]
        for k in (
            "updated_at",
            "done",
            "running",
            "eval_done",
            "eval_pending",
            "pool_sps",
            "eta_h",
            "imported",
            "active",
            "live",
            "score_summary",
        )
    }
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
