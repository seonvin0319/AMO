#!/usr/bin/env python3
"""Detachable loco-9 × seeds 0–3 queue for AMO main TD3+AMO JAX (train.py).

Does not modify algorithms/jax. Uses configs/td3_amo.yaml defaults
(T_E=T_B=1, T_lr=1e-3). Output under /raid/ext_csv/AMO_store.
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
OUT = Path("/raid/ext_csv/AMO_store/td3_amo_jax_loco9_default_seeds0to3")
PYTHON = Path("/home/ext_csv/miniconda3/envs/amo-jax/bin/python")
ENVIRONMENTS = tuple(
    f"{domain}-{dataset}-v2"
    for domain in ("halfcheetah", "hopper", "walker2d")
    for dataset in ("medium", "medium-replay", "medium-expert")
)
SEEDS = (0, 1, 2, 3)
GROUP = "td3-amo-jax-loco9-default-seeds0-3"
CONFIG = ROOT / "configs" / "td3_amo.yaml"

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


def run_id(environment: str, seed: int) -> str:
    return f"td3amo_jax_{ENV_SHORT[environment]}_s{seed}"


def run_dir(environment: str, seed: int) -> Path:
    return OUT / "runs" / run_id(environment, seed)


def completed(environment: str, seed: int) -> bool:
    d = run_dir(environment, seed)
    ckpt = d / "checkpoint.npz"
    if not ckpt.exists():
        return False
    meta = d / "run_meta.json"
    if meta.exists():
        try:
            payload = json.loads(meta.read_text())
            if int(payload.get("steps", 0)) >= 1_000_000:
                return True
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    metrics = d / "metrics.jsonl"
    if metrics.exists():
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
    # train.py writes checkpoint at end; nonempty COMPLETED marker
    return (d / "COMPLETED.json").exists()


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
            ["pgrep", "-af", r"train\.py --algorithm td3_amo --backend jax"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return 0
    return sum(
        1
        for line in output.splitlines()
        if "train.py --algorithm td3_amo --backend jax" in line and "pgrep" not in line
    )


def dataset_path(environment: str) -> Path:
    # hopper-medium-v2 -> hopper_medium-v2.hdf5
    name = environment.replace("-", "_").replace("_v2", "-v2") + ".hdf5"
    return Path("/raid/ext_csv/datasets/d4rl") / name


def command(environment: str, seed: int) -> list[str]:
    output = run_dir(environment, seed)
    cmd = [
        str(PYTHON),
        "-u",
        str(ROOT / "train.py"),
        "--algorithm=td3_amo",
        "--backend=jax",
        f"--env={environment}",
        f"--seed={seed}",
        "--device=cuda:0",  # visible device remapped via CUDA_VISIBLE_DEVICES
        f"--config={CONFIG}",
        f"--dataset={dataset_path(environment)}",
        f"--output={output}",
        # Train only on GPU; MuJoCo eval is a separate CPU posthoc job.
        "--no-eval",
        "--save-every=20000",
    ]
    ckpt = output / "checkpoint.npz"
    if ckpt.exists():
        cmd.append(f"--resume={ckpt}")
    return cmd


def worker_env(gpu: str) -> dict[str, str]:
    mujoco = "/home/ext_csv/.mujoco/mujoco210"
    conda_prefix = "/home/ext_csv/miniconda3/envs/amo-jax"
    environment = os.environ.copy()
    ld = [
        f"{mujoco}/bin",
        f"{conda_prefix}/lib",
        "/home/ext_csv/.local/osmesa",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/nvidia",
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
            "D4RL_DATASET_DIR": "/raid/ext_csv/datasets/d4rl",
            "MUJOCO_GL": "egl",
            "MUJOCO_PY_MUJOCO_PATH": mujoco,
            "LD_LIBRARY_PATH": ":".join(p for p in ld if p),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "EIGEN_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONNOUSERSITE": "1",
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


def job_cells() -> list[tuple[str, int]]:
    cells: list[tuple[str, int]] = []
    for environment in ENVIRONMENTS:
        for seed in SEEDS:
            cells.append((environment, seed))
    return cells


def main() -> int:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--max-used-mib", type=int, default=80000)
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out.resolve()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    max_parallel = max(1, int(args.max_parallel))
    if not PYTHON.exists():
        raise SystemExit(f"missing python: {PYTHON}")
    if not CONFIG.exists():
        raise SystemExit(f"missing config: {CONFIG}")

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
                "config": str(CONFIG),
                "backend": "jax",
                "algorithm": "td3_amo",
                "train_no_eval": True,
                "save_every": 20000,
                "cpu_eval": {
                    "script": "scripts/eval_checkpoints_cpu.py",
                    "every_steps": 20000,
                    "episodes": 10,
                    "final_repeats": 5,
                    "delete_non_final_ckpt": True,
                },
                "seeds": list(SEEDS),
                "pinned_sha": (ROOT / ".pinned_sha").read_text().strip()
                if (ROOT / ".pinned_sha").exists()
                else None,
            },
        )
        print(f"detached launcher pid={process.pid}")
        return 0

    metadata = git_metadata()
    manifest = {
        "created_at": utc_now(),
        "root": str(ROOT),
        "out": str(OUT),
        "group": GROUP,
        "git": metadata,
        "config": str(CONFIG),
        "jobs": [],
    }
    pending: list[dict] = []
    for environment, seed in job_cells():
        rid = run_id(environment, seed)
        job_dir = OUT / "jobs" / rid
        job_dir.mkdir(exist_ok=True)
        if (job_dir / "CANCELLED.json").exists():
            status = "cancelled"
        elif completed(environment, seed) or (job_dir / "COMPLETED.json").exists():
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
            "environment": environment,
            "seed": seed,
            "status": status,
            "pid": None,
            "gpu": None,
            "log": str(job_dir / "stdout_stderr.log"),
            "job_dir": str(job_dir),
            "output": str(run_dir(environment, seed)),
        }
        if status == "running":
            job["pid"] = int(rec.get("pid", -1))
            job["gpu"] = str(rec.get("gpu"))
        manifest["jobs"].append(job)
        if status == "pending":
            pending.append(job)
    write_status(manifest)
    if args.status_only:
        print(json.dumps({"pending": len(pending), "counts": json.loads((OUT / "status_summary.json").read_text())["counts"]}, indent=2))
        return 0

    print(
        f"jax loco9 queue pending={len(pending)} max_parallel={max_parallel} out={OUT}",
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
                command(job["environment"], int(job["seed"])),
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
            if return_code == 0 and completed(job["environment"], int(job["seed"])):
                job["status"] = "completed"
                atomic_json(
                    job_dir / "COMPLETED.json",
                    {"completed_at": utc_now(), "return_code": return_code},
                )
            elif return_code == 0:
                # train finished but completion heuristic missed — still mark done
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
