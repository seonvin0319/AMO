#!/usr/bin/env python3
"""Restartable seed-0 adaptive-multiscale AMO locomotion-9 launcher."""

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
OUT = ROOT / "results/amo_adaptive_multiscale_locomotion9_seed0"
ENVIRONMENTS = tuple(
    f"{domain}-{dataset}-v2"
    for domain in ("halfcheetah", "hopper", "walker2d")
    for dataset in ("medium", "medium-replay", "medium-expert")
)
LAUNCH_SCALES = {"T_E": None, "T_B": None, "T_lr": None}

ENV_SHORT = {
    environment: environment.replace("halfcheetah", "hc")
    .replace("hopper", "h")
    .replace("walker2d", "w")
    .replace("medium-replay", "mr")
    .replace("medium-expert", "me")
    .replace("medium", "m")
    .replace("expert", "e")
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


def run_id(environment: str) -> str:
    return f"amo_adaptive_multiscale_{ENV_SHORT[environment]}_s0"


def result_dirs(environment: str) -> list[Path]:
    tag = run_id(environment)
    return sorted((OUT / "runs").glob(f"{tag}-{environment}-*"))


def completed(environment: str) -> bool:
    for directory in result_dirs(environment):
        if (directory / "checkpoint_999999.pt").exists():
            return True
        evaluation = directory / "eval.jsonl"
        if evaluation.exists():
            for line in evaluation.read_text().splitlines():
                if line.strip() and int(json.loads(line).get("step", 0)) >= 1_000_000:
                    return True
    return False


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
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


def live_amo_per_gpu() -> dict[str, int]:
    counts: dict[str, int] = {}
    try:
        output = subprocess.check_output(
            ["pgrep", "-f", r"python -u -m amo"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return counts
    for pid in output.split():
        env_path = Path("/proc") / pid / "environ"
        try:
            raw = env_path.read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError):
            continue
        gpu = ""
        for item in raw:
            if item.startswith(b"CUDA_VISIBLE_DEVICES="):
                gpu = item.split(b"=", 1)[1].decode("ascii", "ignore")
                break
        if gpu:
            counts[gpu] = counts.get(gpu, 0) + 1
    return counts


def resolved_config(environment: str) -> dict:
    te = LAUNCH_SCALES["T_E"]
    tb = LAUNCH_SCALES["T_B"]
    tlr = LAUNCH_SCALES["T_lr"]
    initial_t_e = "TrainConfig.T_E default" if te is None else te
    if tb is not None:
        initial_t_b = tb
    elif te is not None:
        initial_t_b = te
    else:
        initial_t_b = "TrainConfig.T_E default"
    return {
        "algorithm": "amo_adaptive_multiscale",
        "env": environment,
        "seed": 0,
        "adaptive_multiscale": True,
        "initial_T_E": initial_t_e,
        "initial_T_B": initial_t_b,
        "T_lr": "TrainConfig.T_lr default" if tlr is None else tlr,
        "execution_scale_loss": "-B_PI_E",
        "bootstrap_scale_loss": "L1_B+L2_RMS_B",
        "bootstrap_outer_loss_version": "tq_detached_rms_target_v1",
        "bootstrap_scale_loss_formula": (
            "-2*stopgrad(T_B)*mean(Q_B+)/S_B + C_B+ "
            "+ 2*stopgrad(T_B)*RMS(delta_y_B/S_B)"
        ),
        "max_timesteps": 1_000_000,
        "eval_freq": 5_000,
        "n_episodes": 10,
        "save_freq": 5_000,
        "all_other_hyperparameters": "TrainConfig defaults",
    }


def command(environment: str) -> list[str]:
    config = resolved_config(environment)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "amo",
        f"--env={environment}",
        "--seed=0",
        "--adaptive_multiscale=True",
        "--max_timesteps=1000000",
        f"--eval_freq={config['eval_freq']}",
        "--n_episodes=10",
        f"--save_freq={config['save_freq']}",
        "--metrics_log_freq=200",
        f"--checkpoints_path={OUT / 'runs'}",
        f"--name={run_id(environment)}",
        "--project=AMO-adaptive-multiscale",
        "--group=amo-locomotion9-seed0",
    ]
    if LAUNCH_SCALES["T_E"] is not None:
        cmd.append(f"--T_E={LAUNCH_SCALES['T_E']}")
    if LAUNCH_SCALES["T_B"] is not None:
        cmd.append(f"--T_B={LAUNCH_SCALES['T_B']}")
    if LAUNCH_SCALES["T_lr"] is not None:
        cmd.append(f"--T_lr={LAUNCH_SCALES['T_lr']}")
    if result_dirs(environment):
        cmd.append(f"--resume_tag={run_id(environment)}")
    return cmd


def worker_env(gpu: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "WANDB_MODE": "offline",
            "WANDB_DIR": str(OUT / "wandb"),
            "WANDB_SILENT": "true",
            "D4RL_SUPPRESS_IMPORT_ERROR": "1",
            "MUJOCO_GL": "egl",
            "MUJOCO_PY_MUJOCO_PATH": "/home/ext_csh/.mujoco/mujoco210",
            "LD_LIBRARY_PATH": (
                "/usr/lib/x86_64-linux-gnu:/home/ext_csh/.mujoco/mujoco210/bin:"
                "/usr/lib/nvidia:" + environment.get("LD_LIBRARY_PATH", "")
            ),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
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


def main() -> int:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--max-used-mib", type=int, default=1024)
    parser.add_argument("--slots-per-gpu", type=int, default=1)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--T_E", type=float, default=None)
    parser.add_argument("--T_B", type=float, default=None)
    parser.add_argument("--T_lr", type=float, default=None)
    args = parser.parse_args()
    LAUNCH_SCALES["T_E"] = args.T_E
    LAUNCH_SCALES["T_B"] = args.T_B
    LAUNCH_SCALES["T_lr"] = args.T_lr
    OUT = Path(args.out)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")

    for directory in (OUT, OUT / "runs", OUT / "jobs", OUT / "wandb"):
        directory.mkdir(parents=True, exist_ok=True)
    if args.detach:
        if args.status_only:
            raise ValueError("--detach and --status-only cannot be combined")
        child_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--gpus",
            args.gpus,
            "--max-used-mib",
            str(args.max_used_mib),
            "--slots-per-gpu",
            str(args.slots_per_gpu),
            "--out",
            str(OUT),
        ]
        if args.retry_failed:
            child_command.append("--retry-failed")
        if args.T_E is not None:
            child_command.extend(["--T_E", str(args.T_E)])
        if args.T_B is not None:
            child_command.extend(["--T_B", str(args.T_B)])
        if args.T_lr is not None:
            child_command.extend(["--T_lr", str(args.T_lr)])
        log_handle = (OUT / "launcher.log").open("a", encoding="utf-8")
        process = subprocess.Popen(
            child_command,
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_handle.close()
        atomic_json(
            OUT / "launcher_process.json",
            {
                "pid": process.pid,
                "launched_at": utc_now(),
                "command": child_command,
                "gpus": gpus,
            },
        )
        print(f"detached launcher pid={process.pid}")
        return 0

    gate_path = OUT / "validation_gate.json"
    if not gate_path.exists():
        raise RuntimeError(f"validation gate is missing: {gate_path}")
    gate = json.loads(gate_path.read_text())
    if not (gate.get("tests_passed") and gate.get("smoke_passed")):
        raise RuntimeError(
            "unit/integration tests and smoke run must pass before launch"
        )

    memory = gpu_memory_used()
    unavailable = [gpu for gpu in gpus if memory.get(gpu, 10**9) > args.max_used_mib]
    if unavailable:
        raise RuntimeError(
            f"GPU(s) exceed --max-used-mib={args.max_used_mib}: "
            + ", ".join(f"{gpu}={memory.get(gpu)} MiB" for gpu in unavailable)
        )

    metadata = git_metadata()
    manifest = {
        "sweep": "amo_adaptive_multiscale_locomotion9_seed0",
        "created_at": utc_now(),
        "canonical_definition": (
            "AMO adaptive multiscale: D4RL v2 "
            "medium/medium-replay/medium-expert x "
            "halfcheetah/hopper/walker2d; independently learned T_E and T_B"
        ),
        "environments": list(ENVIRONMENTS),
        "git": metadata,
        "validation_gate": gate,
        "gpu_memory_at_launch_mib": memory,
        "jobs": [],
    }
    pending = []
    for environment in ENVIRONMENTS:
        job_dir = OUT / "jobs" / run_id(environment)
        job_dir.mkdir(parents=True, exist_ok=True)
        config = resolved_config(environment)
        atomic_json(job_dir / "resolved_config.json", config)
        failure = job_dir / "FAILED.json"
        status = "completed" if completed(environment) else "pending"
        running_marker = job_dir / "RUNNING.json"
        if running_marker.exists() and status != "completed":
            running_record = json.loads(running_marker.read_text())
            if process_alive(int(running_record.get("pid", -1))):
                status = "running_external"
        if failure.exists() and not args.retry_failed and status == "pending":
            status = "failed"
        job = {
            "run_id": run_id(environment),
            "environment": environment,
            "seed": 0,
            "status": status,
            "pid": None,
            "gpu": None,
            "log": str(job_dir / "stdout_stderr.log"),
            "job_dir": str(job_dir),
            "result_dirs": [str(path) for path in result_dirs(environment)],
        }
        manifest["jobs"].append(job)
        if status == "pending":
            pending.append(job)
    write_status(manifest)
    if args.status_only:
        return 0

    running: dict[int, tuple[subprocess.Popen, dict, object]] = {}
    queue = list(pending)
    slots = max(1, int(args.slots_per_gpu))
    while queue or running:
        occupancy = live_amo_per_gpu()
        free_gpus = [
            gpu
            for gpu in sorted(gpus, key=lambda item: occupancy.get(item, 0))
            if occupancy.get(gpu, 0) < slots
        ]
        while queue and free_gpus:
            job = queue.pop(0)
            gpu = free_gpus.pop(0)
            job_dir = Path(job["job_dir"])
            failure = job_dir / "FAILED.json"
            if failure.exists() and args.retry_failed:
                failure.rename(job_dir / f"FAILED.{int(time.time())}.json")
            log_handle = (job_dir / "stdout_stderr.log").open("a", encoding="utf-8")
            launched_at = utc_now()
            process = subprocess.Popen(
                command(job["environment"]),
                cwd=ROOT,
                env=worker_env(gpu),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
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
            write_status(manifest)
            if occupancy.get(gpu, 0) < slots:
                free_gpus.append(gpu)
            free_gpus.sort(key=lambda item: occupancy.get(item, 0))

        if queue or running:
            time.sleep(5)
        for pid, (process, job, log_handle) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            running.pop(pid)
            job_dir = Path(job["job_dir"])
            (job_dir / "RUNNING.json").unlink(missing_ok=True)
            job["finished_at"] = utc_now()
            job["return_code"] = return_code
            job["result_dirs"] = [str(path) for path in result_dirs(job["environment"])]
            if return_code == 0 and completed(job["environment"]):
                job["status"] = "completed"
                atomic_json(job_dir / "COMPLETED.json", job)
            else:
                job["status"] = "failed"
                atomic_json(job_dir / "FAILED.json", job)
            write_status(manifest)
    return 0 if all(job["status"] == "completed" for job in manifest["jobs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
