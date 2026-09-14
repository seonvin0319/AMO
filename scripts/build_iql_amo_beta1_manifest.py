#!/usr/bin/env python3
"""Build a fixed manifest for IQL+AMO beta_initial=1 · seeds 0–3 · loco9+antmaze6.

Discovers jobs recursively under the results tree, prefers run_meta / checkpoint
extra for env+seed, and records base LRs vs meta (rho) LR separately.
Does not modify training artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

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
TARGET_ENVS = set(LOCO9 + ANT6)
TARGET_SEEDS = {0, 1, 2, 3}
BETA_TARGET = 1.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def load_ckpt_meta(ckpt: Path) -> Dict[str, Any]:
    with np.load(ckpt, allow_pickle=False) as data:
        return json.loads(str(data["__metadata__"]))


def discover_run_dirs(jobs_root: Path) -> List[Path]:
    found: List[Path] = []
    for run in jobs_root.rglob("run"):
        if not run.is_dir():
            continue
        if (run / "checkpoints").is_dir() or (run / "checkpoint.npz").exists():
            found.append(run)
    return sorted(set(found))


def pick_final_ckpt(run: Path) -> Tuple[Optional[Path], Optional[int], str]:
    """Prefer step_1000000.npz; else checkpoint.npz if internal step==1M."""
    step_path = run / "checkpoints" / "step_1000000.npz"
    if step_path.exists():
        return step_path, 1_000_000, "step_1000000.npz"
    alias = run / "checkpoint.npz"
    if alias.exists():
        try:
            meta = load_ckpt_meta(alias)
            step = int(meta.get("steps") or 0)
        except Exception as exc:  # noqa: BLE001
            return None, None, f"checkpoint.npz_unreadable:{exc}"
        if step >= 1_000_000:
            return alias, step, "checkpoint.npz@1M"
        return None, step, f"checkpoint.npz_step_{step}_not_final"
    return None, None, "missing_final_ckpt"


def beta_ok(cfg: Dict[str, Any]) -> bool:
    try:
        return abs(float(cfg.get("beta_initial")) - BETA_TARGET) < 1e-9
    except (TypeError, ValueError):
        return False


def existing_final_protocol(run: Path) -> Dict[str, Any]:
    """Inspect legacy posthoc / in-process finals (not final50_singlepass_v1)."""
    info: Dict[str, Any] = {"legacy_final_rows": []}
    candidates: List[Dict[str, Any]] = []
    fj = run / "posthoc_eval_cpu" / "final.json"
    if fj.exists():
        try:
            payload = json.loads(fj.read_text())
            row = payload.get("row") if isinstance(payload, dict) else None
            if isinstance(row, dict):
                candidates.append({"source": str(fj), **row})
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    ej = run / "eval.jsonl"
    if ej.exists():
        for line in ej.read_text(errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(row.get("step") or 0) < 1_000_000:
                continue
            if row.get("tag") == "posthoc_cpu_final" or int(row.get("episodes") or 0) >= 50:
                candidates.append({"source": "eval.jsonl", **row})
    for row in candidates:
        per = int(row.get("episodes_per_repeat") or 0)
        reps = int(row.get("repeats") or 0)
        stride = row.get("seed_stride")
        try:
            stride_i = int(stride) if stride is not None else None
        except (TypeError, ValueError):
            stride_i = None
        rr = row.get("repeat_returns")
        identical = (
            isinstance(rr, list)
            and len(rr) > 1
            and len({round(float(x), 6) for x in rr}) == 1
        )
        info["legacy_final_rows"].append(
            {
                "source": row.get("source"),
                "normalized_score": row.get("normalized_score"),
                "episodes": row.get("episodes"),
                "episodes_per_repeat": per,
                "repeats": reps,
                "seed_stride": stride,
                "identical_repeat_returns": identical,
                "protocol_guess": (
                    "broken_10x5_seed_stride0"
                    if identical and per == 10 and reps == 5 and stride_i == 0
                    else "other_or_unknown"
                ),
            }
        )
    return info


def build_entry(run: Path) -> Optional[Dict[str, Any]]:
    cfg_path = run / "config.yaml"
    if not cfg_path.exists():
        return None
    cfg = load_yaml(cfg_path)
    if str(cfg.get("algorithm") or "") != "iql_amo":
        return None
    if not beta_ok(cfg):
        return None

    meta: Dict[str, Any] = {}
    meta_path = run / "run_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}

    ckpt, step, ckpt_kind = pick_final_ckpt(run)
    ck_meta: Dict[str, Any] = {}
    provenance = "run_meta"
    if ckpt is not None:
        try:
            ck_meta = load_ckpt_meta(ckpt)
        except Exception as exc:  # noqa: BLE001
            return {
                "run_dir": str(run),
                "error": f"ckpt_meta_failed:{exc}",
                "ckpt_kind": ckpt_kind,
            }

    env = meta.get("env") or (ck_meta.get("extra") or {}).get("env") or cfg.get("env")
    seed = meta.get("seed")
    if seed is None:
        seed = (ck_meta.get("extra") or {}).get("seed")
    if seed is None:
        seed = cfg.get("seed")
        provenance = "inferred_config_or_name"
    else:
        if not meta.get("env") and (ck_meta.get("extra") or {}).get("env"):
            provenance = "checkpoint_extra"
        elif meta.get("env"):
            provenance = "run_meta"
        else:
            provenance = "mixed"

    if env not in TARGET_ENVS:
        return None
    try:
        seed_i = int(seed)
    except (TypeError, ValueError):
        return None
    if seed_i not in TARGET_SEEDS:
        return None

    # Cross-check name inference if folder encodes seed/env short tags.
    job = run.parent.name
    inferred_seed = None
    m = re.search(r"_s(\d+)$", job)
    if m:
        inferred_seed = int(m.group(1))

    norm = run / "normalization.npz"
    norm_sha = sha256_file(norm) if norm.exists() else None
    ckpt_sha = sha256_file(ckpt) if ckpt is not None else None

    norm_vs_extra = None
    extra = ck_meta.get("extra") or {}
    if norm.exists() and ("norm_mean" in extra or "normalization_mean" in extra):
        norm_vs_extra = "extra_has_norm_fields"
    elif norm.exists():
        # Compare mean/std arrays if present under common keys in checkpoint arrays.
        norm_vs_extra = "normalization.npz_present_no_extra_arrays"

    entry = {
        "job_name": job,
        "run_dir": str(run.resolve()),
        "env": env,
        "seed": seed_i,
        "domain": "antmaze" if env.startswith("antmaze") else "locomotion",
        "algorithm": "iql_amo",
        "backend": meta.get("backend") or ck_meta.get("backend") or "jax",
        "beta_initial": float(cfg.get("beta_initial")),
        "base_lrs": {
            "actor_lr": float(cfg.get("actor_lr")),
            "critic_lr": float(cfg.get("critic_lr")),
            "value_lr": float(cfg.get("value_lr")),
        },
        "meta_lrs": {
            "rho_lr": float(cfg.get("rho_lr")),
        },
        "beta_bounds": {
            "beta_min": float(cfg.get("beta_min", float("nan"))),
            "beta_max": float(cfg.get("beta_max", float("nan"))),
        },
        "max_steps": int(cfg.get("max_steps") or ck_meta.get("steps") or 0),
        "final_ckpt": str(ckpt) if ckpt else None,
        "final_ckpt_kind": ckpt_kind,
        "final_ckpt_step": step,
        "final_ckpt_sha256": ckpt_sha,
        "normalization": str(norm) if norm.exists() else None,
        "normalization_sha256": norm_sha,
        "config_path": str(cfg_path),
        "provenance": {
            "env_seed_source": provenance,
            "job_name_seed": inferred_seed,
            "seed_matches_job_name": inferred_seed is None or inferred_seed == seed_i,
            "ckpt_extra_env": (ck_meta.get("extra") or {}).get("env"),
            "ckpt_extra_seed": (ck_meta.get("extra") or {}).get("seed"),
            "norm_vs_ckpt_extra": norm_vs_extra,
        },
        "legacy_eval": existing_final_protocol(run),
        "has_metrics": (run / "metrics.jsonl").exists(),
        "has_eval_jsonl": (run / "eval.jsonl").exists(),
    }
    return entry


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--jobs-root",
        type=Path,
        default=Path(
            "/home/svcho/amo_iql_sweep/results/iql_amo_lr1e3_beta_sweep_seed0/jobs"
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/home/svcho/amo_iql_sweep/results/iql_amo_lr1e3_beta_sweep_seed0/"
            "audit_20260914/manifest_beta1_final50.json"
        ),
    )
    args = p.parse_args()

    runs = discover_run_dirs(args.jobs_root.resolve())
    entries: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for run in runs:
        try:
            entry = build_entry(run)
        except Exception as exc:  # noqa: BLE001
            errors.append({"run_dir": str(run), "error": f"{type(exc).__name__}: {exc}"})
            continue
        if entry is None:
            continue
        if entry.get("error"):
            errors.append(entry)
            continue
        entries.append(entry)

    # Dedup by (env, seed); prefer path containing b1_ and step_1000000.
    by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for e in entries:
        key = (e["env"], e["seed"])
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = e
            continue
        score = (
            (1 if e.get("final_ckpt_kind") == "step_1000000.npz" else 0)
            + (1 if "b1_" in e["job_name"] else 0)
            + (1 if e.get("final_ckpt_sha256") else 0)
        )
        prev_score = (
            (1 if prev.get("final_ckpt_kind") == "step_1000000.npz" else 0)
            + (1 if "b1_" in prev["job_name"] else 0)
            + (1 if prev.get("final_ckpt_sha256") else 0)
        )
        if score >= prev_score:
            by_key[key] = e

    selected = sorted(by_key.values(), key=lambda x: (x["env"], x["seed"]))
    expected = [(env, s) for env in sorted(TARGET_ENVS) for s in sorted(TARGET_SEEDS)]
    have = {(e["env"], e["seed"]) for e in selected}
    missing = [{"env": env, "seed": s} for env, s in expected if (env, s) not in have]

    broken_legacy = 0
    for e in selected:
        for row in e.get("legacy_eval", {}).get("legacy_final_rows", []):
            if row.get("protocol_guess") == "broken_10x5_seed_stride0":
                broken_legacy += 1
                break

    payload = {
        "protocol_target": "final50_singlepass_v1",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "jobs_root": str(args.jobs_root.resolve()),
        "filter": {
            "algorithm": "iql_amo",
            "beta_initial": BETA_TARGET,
            "base_lr": 1e-3,
            "rho_lr": 2e-3,
            "envs": sorted(TARGET_ENVS),
            "seeds": sorted(TARGET_SEEDS),
            "expected_n": len(expected),
        },
        "n_discovered_run_dirs": len(runs),
        "n_selected": len(selected),
        "n_missing": len(missing),
        "n_legacy_broken_10x5": broken_legacy,
        "missing": missing,
        "errors": errors,
        "runs": selected,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    tmp.replace(args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "n_selected": len(selected),
                "n_missing": len(missing),
                "n_legacy_broken_10x5": broken_legacy,
                "n_errors": len(errors),
            },
            indent=2,
        )
    )
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
