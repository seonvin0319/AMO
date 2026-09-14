#!/usr/bin/env python3
"""Diagnose hopper-medium-expert IQL+AMO beta=1 seed3 vs seed0 (no retrain).

Writes under audit_20260914/hme_diag/ — separate from final50_singlepass_v1 selection.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
PY = Path("/home/svcho/anaconda3/envs/fql_gpu310/bin/python")
EVAL = ROOT / "scripts" / "eval_final50_singlepass_v1.py"
JOBS = Path("/home/svcho/amo_iql_sweep/results/iql_amo_lr1e3_beta_sweep_seed0/jobs")
OUT = Path(
    "/home/svcho/amo_iql_sweep/results/iql_amo_lr1e3_beta_sweep_seed0/audit_20260914/hme_diag"
)
STEPS = [200_000, 400_000, 600_000, 800_000, 1_000_000]


def load_metrics_at(run: Path, steps: List[int]) -> Dict[str, Any]:
    rows = []
    mp = run / "metrics.jsonl"
    if not mp.exists():
        return {"error": "missing_metrics"}
    by_step = {}
    for line in mp.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        by_step[int(obj["step"])] = obj
    keys = [
        "beta_E",
        "beta_B",
        "L_E",
        "L1_B",
        "L2_RMS_B",
        "actor_loss",
        "critic_loss",
        "value_loss",
    ]
    out = {}
    for st in steps:
        r = by_step.get(st)
        out[str(st)] = None if r is None else {k: r.get(k) for k in keys}
    # first time beta_B near floor
    floor_step = None
    for st in sorted(by_step):
        b = by_step[st].get("beta_B")
        if b is not None and float(b) <= 0.0501:
            floor_step = st
            break
    out["beta_B_floor_first_step"] = floor_step
    out["last"] = {k: by_step[max(by_step)].get(k) for k in keys} if by_step else None
    return out


def legacy_score(run: Path) -> Optional[float]:
    fj = run / "posthoc_eval_cpu" / "final.json"
    if fj.exists():
        return float(json.load(open(fj))["row"]["normalized_score"])
    ej = run / "eval.jsonl"
    if not ej.exists():
        return None
    best = None
    for line in ej.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if int(o.get("step") or 0) >= 1_000_000 and int(o.get("episodes") or 0) >= 50:
            best = float(o["normalized_score"])
    return best


def run_eval(run: Path, step: int, out_subdir: str) -> Dict[str, Any]:
    ckpt = run / "checkpoints" / f"step_{step}.npz"
    if not ckpt.exists():
        return {"status": "missing_ckpt", "step": step, "path": str(ckpt)}
    cmd = [
        str(PY),
        str(EVAL),
        "--run-dir",
        str(run),
        "--step",
        str(step),
        "--out-subdir",
        out_subdir,
        "--episodes",
        "50",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    rec = None
    for ln in proc.stdout.splitlines():
        ln = ln.strip()
        if not ln.startswith("{") or '"status"' not in ln:
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "status" in obj:
            rec = obj
    if rec is None:
        # Fallback: read marker written by evaluator
        marker = run / out_subdir / "FINAL50_SINGLEPASS_V1_DONE.json"
        if marker.exists():
            payload = json.loads(marker.read_text())
            row = payload.get("row") or {}
            rec = {
                "status": "done" if payload.get("ok") else "failed",
                "normalized_score": row.get("normalized_score"),
                "run": str(run),
                "from_marker": True,
            }
        else:
            rec = {
                "status": "failed",
                "error": "no_json",
                "stdout_tail": proc.stdout[-1500:],
                "stderr_tail": proc.stderr[-1500:],
            }
    rec["returncode"] = proc.returncode
    rec["step"] = step
    return rec


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {s: JOBS / f"b1_h-me_s{s}" / "run" for s in range(4)}
    summary: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "note": (
            "Diagnostics only. final50_singlepass_v1 selection scores are separate. "
            "act() uses state['p']['actor'] (pi_E). "
            "Terminal L2_RMS_B=0 may follow cosine actor LR → 0; not alone a bug."
        ),
        "legacy_scores": {},
        "metrics": {},
        "mid_ckpt_presence": {},
        "diag_evals": {},
        "code_root": str(ROOT),
    }

    for s, run in runs.items():
        summary["legacy_scores"][str(s)] = legacy_score(run)
        summary["metrics"][str(s)] = load_metrics_at(run, STEPS)
        presence = {}
        for st in STEPS:
            presence[str(st)] = (run / "checkpoints" / f"step_{st}.npz").exists()
        summary["mid_ckpt_presence"][str(s)] = presence

    # Mid-ckpt final50-style diag for seed3 and seed0 only.
    for s in (3, 0):
        run = runs[s]
        summary["diag_evals"][str(s)] = {}
        for st in STEPS:
            print(f"[hme_diag] seed={s} step={st}", flush=True)
            rec = run_eval(run, st, out_subdir=f"diag_final50_v1/step_{st}")
            summary["diag_evals"][str(s)][str(st)] = rec

    # Compact comparison table
    table = []
    for st in STEPS:
        row = {"step": st}
        for s in (0, 3):
            rec = summary["diag_evals"][str(s)].get(str(st), {})
            row[f"s{s}_status"] = rec.get("status")
            row[f"s{s}_ns"] = rec.get("normalized_score")
            row[f"s{s}_err"] = rec.get("error")
        table.append(row)
    summary["comparison_table"] = table

    out = OUT / "hme_s3_vs_s0_report.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"out": str(out), "table": table}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
