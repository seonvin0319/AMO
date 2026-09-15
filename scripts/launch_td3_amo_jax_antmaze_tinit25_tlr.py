#!/usr/bin/env python3
"""JAX queue for td3_amo antmaze alpha_E=alpha_B=5 × T_lr × seed0-3.

Reads pending cells from:
  /home/choi/AMO_store/td3_amo_jax_antmaze_tinit25_tlr_handoff/pending_cells.json
Uses /home/choi/amo train.py --backend jax (train-only + save-every 20k;
CPU final eval is a separate reaper).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/choi/miniconda3/envs/amo/bin/python")
OUT = Path("/home/choi/AMO_store/td3_amo_jax_antmaze_tinit25_tlr")
HANDOFF = Path(
    "/home/choi/AMO_store/td3_amo_jax_antmaze_tinit25_tlr_handoff/pending_cells.json"
)
GROUP = "td3-amo-jax-antmaze-tinit25-tlr"
TLR_CONFIG = {
    "3e-4": ROOT / "configs" / "td3_amo_tinit25_tlr3e-4.yaml",
    "1e-3": ROOT / "configs" / "td3_amo_tinit25_tlr1e-3.yaml",
    "2e-3": ROOT / "configs" / "td3_amo_tinit25_tlr2e-3.yaml",
}
ENV_SHORT = {
    "antmaze-umaze-v2": "amu",
    "antmaze-umaze-diverse-v2": "amud",
    "antmaze-medium-play-v2": "ammp",
    "antmaze-medium-diverse-v2": "ammd",
    "antmaze-large-play-v2": "amlp",
    "antmaze-large-diverse-v2": "amld",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git_metadata() -> dict:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).splitlines()
    return {"commit": commit, "dirty": bool(dirty), "dirty_status": dirty}


def tlr_tag(value: float) -> str:
    for tag, target in (("3e-4", 3e-4), ("1e-3", 1e-3), ("2e-3", 2e-3)):
        if abs(float(value) - target) < 1e-12:
            return tag
    raise ValueError(f"unsupported alpha_lr={value}")


def run_id(environment: str, seed: int, tag: str) -> str:
    return f"td3amo_jax_te25_tb25_tlr{tag}_{ENV_SHORT[environment]}_s{seed}"


def run_dir(rid: str) -> Path:
    return OUT / "runs" / rid


def completed(rid: str) -> bool:
    d = run_dir(rid)
    if (d / "COMPLETED.json").exists():
        return True
    meta = d / "run_meta.json"
    if meta.exists():
        try:
            if int(json.loads(meta.read_text()).get("steps", 0)) >= 1_000_000:
                return True
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    metrics = d / "metrics.jsonl"
    if metrics.exists() and metrics.stat().st_size:
        with metrics.open("rb") as handle:
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
            if step is not None and int(step) >= 1_000_000:
                return True
            break
    return False


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def gpu_memory_used() -> dict[str, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return {
        index.strip(): int(used.strip())
        for index, used in (line.split(",", 1) for line in output.splitlines())
    }


def live_count_from_markers() -> dict[str, int]:
    counts: dict[str, int] = {}
    jobs = OUT / "jobs"
    if not jobs.exists():
        return counts
    for marker in jobs.glob("*/RUNNING.json"):
        try:
            rec = json.loads(marker.read_text())
            pid = int(rec.get("pid", -1))
            gpu = str(rec.get("gpu", ""))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not process_alive(pid) or not gpu:
            continue
        counts[gpu] = counts.get(gpu, 0) + 1
    return counts


def live_train_count() -> int:
    try:
        output = subprocess.check_output(
            ["pgrep", "-af", r"train\.py --algorithm[= ]td3_amo"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return 0
    return sum(
        1
        for line in output.splitlines()
        if "train.py" in line
        and "td3_amo" in line
        and "--backend=jax" in line
        and "pgrep" not in line
    )


def command(environment: str, seed: int, tag: str, rid: str) -> list[str]:
    output = run_dir(rid)
    config = TLR_CONFIG[tag]
    cmd = [
        str(PYTHON),
        "-u",
        str(ROOT / "train.py"),
        "--algorithm=td3_amo",
        "--backend=jax",
        f"--env={environment}",
        f"--seed={seed}",
        "--device=cuda:0",
        f"--config={config}",
        f"--output={output}",
        "--no-eval",
        "--save-every=20000",
    ]
    ckpt = output / "checkpoint.npz"
    if ckpt.exists():
        cmd.append(f"--resume={ckpt}")
    return cmd


def worker_env(gpu: str) -> dict[str, str]:
    mujoco = os.environ.get("MUJOCO_PY_MUJOCO_PATH", "/home/choi/.mujoco/mujoco210")
    conda_prefix = "/home/choi/miniconda3/envs/amo"
    environment = os.environ.copy()
    ld = [
        f"{mujoco}/bin",
        f"{conda_prefix}/lib",
        "/usr/lib/x86_64-linux-gnu",
        environment.get("LD_LIBRARY_PATH", ""),
    ]
    py_path = [str(ROOT)]
    if environment.get("PYTHONPATH"):
        py_path.append(environment["PYTHONPATH"])
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "PYTHONPATH": os.pathsep.join(py_path),
            "WANDB_MODE": "offline",
            "WANDB_DIR": str(OUT / "wandb"),
            "WANDB_SILENT": "true",
            "D4RL_SUPPRESS_IMPORT_ERROR": "1",
            "MUJOCO_GL": "egl",
            "MUJOCO_PY_MUJOCO_PATH": mujoco,
            "LD_LIBRARY_PATH": ":".join(p for p in ld if p),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "EIGEN_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    return environment


def write_status(manifest: dict) -> None:
    counts: dict[str, int] = {}
    for job in manifest["jobs"]:
        status = job["status"]
        counts[status] = counts.get(status, 0) + 1
    atomic_json(
        OUT / "status_summary.json",
        {"updated_at": utc_now(), "counts": counts, "jobs": manifest["jobs"]},
    )
    atomic_json(OUT / "launch_manifest.json", manifest)


def load_cells(handoff: Path) -> list[dict]:
    if not handoff.exists():
        raise SystemExit(f"missing handoff list: {handoff}")
    payload = json.loads(handoff.read_text())
    cells = []
    for row in payload.get("pending") or []:
        environment = row["env"]
        seed = int(row["seed"])
        alr = row.get("alpha_lr", row.get("T_lr"))
        tag = row.get("tlr_tag") or tlr_tag(alr)
        rid = run_id(environment, seed, tag)
        cells.append(
            {
                "environment": environment,
                "seed": seed,
                "alpha_lr": float(row.get("alpha_lr", row.get("T_lr"))),
                "tlr_tag": tag,
                "run_id": rid,
            }
        )
    return cells


def main() -> int:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--max-used-mib", type=int, default=11000)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--handoff", type=Path, default=HANDOFF)
    args = parser.parse_args()
    OUT = args.out.resolve()
    handoff = args.handoff.resolve()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    max_parallel = max(1, int(args.max_parallel))
    if not PYTHON.exists():
        raise SystemExit(f"missing python: {PYTHON}")
    for path in TLR_CONFIG.values():
        if not path.exists():
            raise SystemExit(f"missing config: {path}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "jobs").mkdir(exist_ok=True)
    (OUT / "runs").mkdir(exist_ok=True)
    (OUT / "wandb").mkdir(exist_ok=True)

    if args.detach:
        if args.status_only:
            raise ValueError("--detach and --status-only cannot be combined")
        cmd = [
            str(PYTHON),
            str(Path(__file__).resolve()),
            "--gpus",
            args.gpus,
            "--max-used-mib",
            str(args.max_used_mib),
            "--max-parallel",
            str(max_parallel),
            "--out",
            str(OUT),
            "--handoff",
            str(handoff),
        ]
        if args.retry_failed:
            cmd.append("--retry-failed")
        log = (OUT / "launcher.log").open("a", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        atomic_json(
            OUT / "launcher_process.json",
            {
                "pid": process.pid,
                "launched_at": utc_now(),
                "command": cmd,
                "gpus": gpus,
                "max_parallel": max_parallel,
                "group": GROUP,
                "backend": "jax",
                "handoff": str(handoff),
            },
        )
        print(f"detached launcher pid={process.pid}", flush=True)
        return 0

    metadata = git_metadata()
    cells = load_cells(handoff)
    manifest = {
        "created_at": utc_now(),
        "root": str(ROOT),
        "out": str(OUT),
        "group": GROUP,
        "git": metadata,
        "handoff": str(handoff),
        "jobs": [],
    }
    pending: list[dict] = []
    for cell in cells:
        rid = cell["run_id"]
        job_dir = OUT / "jobs" / rid
        job_dir.mkdir(exist_ok=True)
        if (job_dir / "CANCELLED.json").exists():
            status = "cancelled"
        elif completed(rid) or (job_dir / "COMPLETED.json").exists():
            status = "completed"
        elif (job_dir / "FAILED.json").exists() and not args.retry_failed:
            status = "failed"
        elif (job_dir / "RUNNING.json").exists():
            rec = json.loads((job_dir / "RUNNING.json").read_text())
            pid = int(rec.get("pid", -1))
            if process_alive(pid):
                status = "running"
            else:
                (job_dir / "RUNNING.json").unlink(missing_ok=True)
                status = "pending"
        else:
            status = "pending"
        job = {
            "run_id": rid,
            "environment": cell["environment"],
            "seed": cell["seed"],
            "alpha_lr": cell["alpha_lr"],
            "tlr_tag": cell["tlr_tag"],
            "backend": "jax",
            "status": status,
            "pid": None,
            "gpu": None,
            "log": str(job_dir / "stdout_stderr.log"),
            "job_dir": str(job_dir),
            "output": str(run_dir(rid)),
        }
        if status == "running":
            job["pid"] = int(rec.get("pid", -1))
            job["gpu"] = str(rec.get("gpu"))
        manifest["jobs"].append(job)
        if status == "pending":
            pending.append(job)
    write_status(manifest)
    if args.status_only:
        print(
            json.dumps(
                {
                    "pending": len(pending),
                    "counts": json.loads((OUT / "status_summary.json").read_text())[
                        "counts"
                    ],
                },
                indent=2,
            )
        )
        return 0

    print(
        f"jax antmaze tinit25 pending={len(pending)} max_parallel={max_parallel} out={OUT}",
        flush=True,
    )
    running: dict[int, tuple[subprocess.Popen | None, dict, object | None]] = {}
    for job in manifest["jobs"]:
        if job["status"] == "running" and job.get("pid") is not None:
            running[int(job["pid"])] = (None, job, None)

    queue = list(pending)
    while queue or running:
        used = gpu_memory_used()
        occupancy = live_count_from_markers()
        global_live = live_train_count()
        free_gpus = [
            g
            for g in sorted(gpus, key=lambda item: occupancy.get(item, 0))
            if used.get(g, 10**9) <= args.max_used_mib
        ]
        while queue and free_gpus and global_live < max_parallel:
            job = queue.pop(0)
            gpu = free_gpus.pop(0)
            job_dir = Path(job["job_dir"])
            failure = job_dir / "FAILED.json"
            if failure.exists() and args.retry_failed:
                failure.rename(job_dir / f"FAILED.{int(time.time())}.json")
            out_path = Path(job["output"])
            out_path.mkdir(parents=True, exist_ok=True)
            log_handle = (job_dir / "stdout_stderr.log").open("a", encoding="utf-8")
            launched_at = utc_now()
            process = subprocess.Popen(
                command(
                    job["environment"],
                    int(job["seed"]),
                    job["tlr_tag"],
                    job["run_id"],
                ),
                cwd=ROOT,
                env=worker_env(gpu),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            job.update(
                {
                    "status": "running",
                    "pid": process.pid,
                    "gpu": gpu,
                    "launched_at": launched_at,
                }
            )
            atomic_json(
                job_dir / "RUNNING.json",
                {
                    "run_id": job["run_id"],
                    "pid": process.pid,
                    "gpu": gpu,
                    "launched_at": launched_at,
                    "git": metadata,
                    "backend": "jax",
                },
            )
            running[process.pid] = (process, job, log_handle)
            occupancy[gpu] = occupancy.get(gpu, 0) + 1
            global_live += 1
            write_status(manifest)
            free_gpus = [
                g
                for g in sorted(gpus, key=lambda item: occupancy.get(item, 0))
                if used.get(g, 10**9) <= args.max_used_mib
            ]

        if queue or running:
            time.sleep(5)
        for pid, (process, job, log_handle) in list(running.items()):
            if process is not None:
                return_code = process.poll()
                if return_code is None:
                    continue
            else:
                if process_alive(pid):
                    continue
                return_code = -1
            job_dir = Path(job["job_dir"])
            (job_dir / "RUNNING.json").unlink(missing_ok=True)
            if log_handle is not None:
                log_handle.close()
            if return_code == 0:
                job["status"] = "completed"
                atomic_json(
                    job_dir / "COMPLETED.json",
                    {"completed_at": utc_now(), "return_code": return_code},
                )
            else:
                job["status"] = "failed"
                atomic_json(
                    job_dir / "FAILED.json",
                    {"failed_at": utc_now(), "return_code": return_code},
                )
            job["pid"] = None
            job["gpu"] = None
            del running[pid]
            write_status(manifest)

    print("queue drained", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
