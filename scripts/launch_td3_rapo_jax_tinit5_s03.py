#!/usr/bin/env python3
"""td3 RAPO JAX: π_E −B_π, π_B L2RMS, critic=TD3+BC (2-layer, no LN).

α_E=α_B=5, seeds 0–3, hopper then walker2d only.
Grid: 6 envs × alpha_lr ∈ {3e-4,1e-3,2e-3} × seeds 0–3 = 72.
Queue: α_lr-major (3e-4→1e-3→2e-3), then seed 0→3, then hopper then walker.
Train with --no-eval; checkpoints every 20k for offline CPU eval.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "td3_rapo_jax_tinit5_td3bc_hw_s03"
PYTHON = Path("/home/shchoi/miniconda3/envs/offrl/bin/python")
GIT = Path("/home/shchoi/miniconda3/bin/git")
DATASET_ROOT = Path.home() / ".d4rl" / "datasets"
GROUP = "td3-rapo-jax-tinit5-td3bc-hw-s03"
SEEDS = (0, 1, 2, 3)

HOPPER_WALKER = (
    ("hopper-medium-v2", "hopper_medium-v2.hdf5"),
    ("hopper-medium-replay-v2", "hopper_medium_replay-v2.hdf5"),
    ("hopper-medium-expert-v2", "hopper_medium_expert-v2.hdf5"),
    ("walker2d-medium-v2", "walker2d_medium-v2.hdf5"),
    ("walker2d-medium-replay-v2", "walker2d_medium_replay-v2.hdf5"),
    ("walker2d-medium-expert-v2", "walker2d_medium_expert-v2.hdf5"),
)
ENVIRONMENTS = tuple(env for env, _ in HOPPER_WALKER)
DATASET_FILES = {env: name for env, name in HOPPER_WALKER}

ALPHA_LRS = (
    (3e-4, "3em4", ROOT / "configs" / "td3_rapo_tinit5_tlr3e-4.yaml"),
    (1e-3, "1em3", ROOT / "configs" / "td3_rapo_tinit5_tlr1e-3.yaml"),
    (2e-3, "2em3", ROOT / "configs" / "td3_rapo_tinit5_tlr2e-3.yaml"),
)


def short_env(env_id: str) -> str:
    return (
        env_id.replace("halfcheetah", "hc")
        .replace("hopper", "h")
        .replace("walker2d", "w")
        .replace("antmaze-", "am-")
        .replace("-diverse", "-div")
        .replace("-medium-replay", "-mr")
        .replace("-medium-expert", "-me")
        .replace("-medium", "-m")
        .replace("-large", "-l")
        .replace("-umaze", "-u")
        .replace("-play", "-p")
        .replace("-v2", "")
    )


ENV_SHORT = {env: short_env(env) for env in ENVIRONMENTS}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git_metadata() -> dict:
    git = str(GIT if GIT.exists() else "git")
    commit = subprocess.check_output(
        [git, "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        [git, "status", "--short"], cwd=ROOT, text=True
    ).splitlines()
    return {"commit": commit, "dirty": bool(dirty), "dirty_status": dirty}


def run_id(environment: str, rho_tag: str, seed: int) -> str:
    return f"td3rapo_jax_t5_tlr{rho_tag}_{ENV_SHORT[environment]}_s{seed}"


def run_dir(environment: str, rho_tag: str, seed: int) -> Path:
    return OUT / "runs" / run_id(environment, rho_tag, seed)


def dataset_path(environment: str) -> Path:
    path = DATASET_ROOT / DATASET_FILES[environment]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def completed(environment: str, rho_tag: str, seed: int) -> bool:
    d = run_dir(environment, rho_tag, seed)
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
    return (d / "COMPLETED.json").exists()


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def hold_file() -> Path:
    return OUT / "HOLD.json"


def mpi_sweep_occupying_gpu() -> bool:
    try:
        output = subprocess.check_output(
            ["pgrep", "-af", r"MPI_sweep/(train_iql_mpi\.py|launch_mpi_sweep\.py)"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return False
    return any("pgrep" not in line for line in output.splitlines())


def launch_blocked() -> str | None:
    if hold_file().exists():
        return f"held: {hold_file()}"
    if mpi_sweep_occupying_gpu():
        return "MPI_sweep live; not claiming GPU 0"
    return None


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
            ["pgrep", "-af", r"train\.py --algorithm[= ]td3_amo --backend[= ]jax"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return 0
    return sum(
        1
        for line in output.splitlines()
        if "train.py" in line
        and "td3_amo" in line
        and "jax" in line
        and "pgrep" not in line
        and str(OUT) in line
    )


def checkpoint_steps(ckpt: Path) -> int | None:
    try:
        with np.load(ckpt, allow_pickle=False) as data:
            return int(json.loads(str(data["__metadata__"]))["steps"])
    except (OSError, KeyError, ValueError, json.JSONDecodeError, TypeError):
        return None


def truncate_logs_to_checkpoint(output: Path, ckpt: Path) -> None:
    steps = checkpoint_steps(ckpt)
    if steps is None:
        return
    for name in ("metrics.jsonl", "eval.jsonl"):
        path = output / name
        if not path.exists():
            continue
        kept: list[str] = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = row.get("step")
            if step is None or int(step) <= steps:
                kept.append(json.dumps(row, allow_nan=False))
        path.write_text(("\n".join(kept) + ("\n" if kept else "")))


def command(environment: str, rho_tag: str, seed: int, config: Path) -> list[str]:
    output = run_dir(environment, rho_tag, seed)
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
        f"--dataset={dataset_path(environment)}",
        f"--output={output}",
        "--no-eval",
        "--save-every=20000",
    ]
    ckpt = output / "checkpoint.npz"
    if ckpt.exists():
        truncate_logs_to_checkpoint(output, ckpt)
        cmd.append(f"--resume={ckpt}")
    return cmd


def worker_env(gpu: str) -> dict[str, str]:
    mujoco = str(Path.home() / ".mujoco" / "mujoco210")
    conda_prefix = str(PYTHON.parent.parent)
    environment = os.environ.copy()
    ld = [
        f"{mujoco}/bin",
        f"{conda_prefix}/lib",
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
            "JAX_PLATFORMS": "cuda",
            "PYTHONPATH": os.pathsep.join(py_path),
            "WANDB_MODE": "offline",
            "WANDB_DIR": str(OUT / "wandb"),
            "WANDB_SILENT": "true",
            "D4RL_SUPPRESS_IMPORT_ERROR": "1",
            "D4RL_DATASET_DIR": str(DATASET_ROOT),
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
    environment.pop("CUDA_DEVICE_ORDER", None)
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


def job_cells() -> list[tuple[str, str, int, Path, float]]:
    """α_lr-major 3e-4→1e-3→2e-3, then seed 0→3, then hopper then walker."""
    cells: list[tuple[str, str, int, Path, float]] = []
    for rho_lr, rho_tag, config in ALPHA_LRS:
        for seed in SEEDS:
            for environment in ENVIRONMENTS:
                cells.append((environment, rho_tag, seed, config, rho_lr))
    return cells


def main() -> int:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--max-used-mib", type=int, default=30000)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out.resolve()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    max_parallel = max(1, int(args.max_parallel))
    if not PYTHON.exists():
        raise SystemExit(f"missing python: {PYTHON}")
    for _, _, config in ALPHA_LRS:
        if not config.exists():
            raise SystemExit(f"missing config: {config}")
    for environment in ENVIRONMENTS:
        dataset_path(environment)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "jobs").mkdir(exist_ok=True)
    (OUT / "runs").mkdir(exist_ok=True)
    (OUT / "wandb").mkdir(exist_ok=True)

    blocked = launch_blocked()
    if blocked and not args.status_only:
        print(blocked, flush=True)
        return 0

    n_jobs = len(ENVIRONMENTS) * len(ALPHA_LRS) * len(SEEDS)
    if n_jobs != 72:
        raise SystemExit(f"expected 72 cells, got {n_jobs}")
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
                "algorithm": "td3_amo",
                "variant": "rapo",
                "backend": "jax",
                "alpha_E": 5.0,
                "alpha_B": 5.0,
                "critic_target": "actor",
                "bootstrap_loss": "l2_rms",
                "critic_depth": 2,
                "critic_layernorm": False,
                "alpha_lrs": [3e-4, 1e-3, 2e-3],
                "queue_order": "alpha_lr(3e-4>1e-3>2e-3) → seed → hopper then walker",
                "environments": list(ENVIRONMENTS),
                "env_order": "hopper then walker",
                "seeds": list(SEEDS),
                "n_jobs": n_jobs,
                "train_no_eval": True,
                "save_every": 20000,
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
        "jobs": [],
    }
    pending: list[dict] = []
    for environment, rho_tag, seed, config, rho_lr in job_cells():
        rid = run_id(environment, rho_tag, seed)
        job_dir = OUT / "jobs" / rid
        job_dir.mkdir(exist_ok=True)
        rec = {}
        if (job_dir / "CANCELLED.json").exists():
            status = "cancelled"
        elif completed(environment, rho_tag, seed) or (
            job_dir / "COMPLETED.json"
        ).exists():
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
            "alpha_E": 5.0,
            "alpha_B": 5.0,
            "alpha_lr": rho_lr,
            "rho_tag": rho_tag,
            "config": str(config),
            "seed": seed,
            "status": status,
            "pid": None,
            "gpu": None,
            "log": str(job_dir / "stdout_stderr.log"),
            "job_dir": str(job_dir),
            "output": str(run_dir(environment, rho_tag, seed)),
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
                    "first_pending": [
                        {
                            "env": j["environment"],
                            "alpha_lr": j["rho_tag"],
                            "seed": j["seed"],
                        }
                        for j in pending[:6]
                    ],
                },
                indent=2,
            )
        )
        return 0

    print(
        f"td3_rapo tinit5 queue pending={len(pending)} max_parallel={max_parallel} "
        f"n_jobs={n_jobs} out={OUT}",
        flush=True,
    )
    running: dict[int, tuple[subprocess.Popen | None, dict, object | None]] = {}
    for job in manifest["jobs"]:
        if job["status"] == "running" and job.get("pid") is not None:
            running[int(job["pid"])] = (None, job, None)

    queue = list(pending)
    while queue or running:
        blocked = launch_blocked()
        if blocked:
            print(blocked, flush=True)
            return 0
        used = gpu_memory_used()
        occupancy = live_count_from_markers()
        global_live = live_train_count()
        free_gpus = [
            g
            for g in sorted(gpus, key=lambda item: occupancy.get(item, 0))
            if used.get(g, 10**9) <= args.max_used_mib
        ]
        while (
            queue
            and free_gpus
            and len(running) < max_parallel
            and global_live < max_parallel
        ):
            if len(running) >= max_parallel or global_live >= max_parallel:
                break
            job = queue.pop(0)
            gpu = free_gpus.pop(0)
            job_dir = Path(job["job_dir"])
            failure = job_dir / "FAILED.json"
            if failure.exists() and args.retry_failed:
                failure.rename(job_dir / f"FAILED.{int(time.time())}.json")
            out_path = Path(job["output"])
            if (
                args.retry_failed
                and out_path.exists()
                and not (out_path / "checkpoint.npz").exists()
            ):
                for child in list(out_path.iterdir()):
                    if child.is_file():
                        child.unlink(missing_ok=True)
                    elif child.is_dir() and child.name in {
                        "checkpoints",
                        "posthoc_eval_cpu",
                        "wandb",
                    }:
                        shutil.rmtree(child, ignore_errors=True)
            out_path.mkdir(parents=True, exist_ok=True)
            log_handle = (job_dir / "stdout_stderr.log").open("a", encoding="utf-8")
            launched_at = utc_now()
            process = subprocess.Popen(
                command(
                    job["environment"],
                    job["rho_tag"],
                    int(job["seed"]),
                    Path(job["config"]),
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
                },
            )
            running[process.pid] = (process, job, log_handle)
            occupancy[gpu] = occupancy.get(gpu, 0) + 1
            global_live += 1
            write_status(manifest)
            break

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
                return_code = (
                    0
                    if completed(job["environment"], job["rho_tag"], int(job["seed"]))
                    else -1
                )
            job_dir = Path(job["job_dir"])
            (job_dir / "RUNNING.json").unlink(missing_ok=True)
            if log_handle is not None:
                log_handle.close()
            ok = completed(job["environment"], job["rho_tag"], int(job["seed"]))
            if return_code == 0 or ok:
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
            if (
                args.retry_failed
                and job["status"] == "failed"
                and not completed(job["environment"], job["rho_tag"], int(job["seed"]))
            ):
                queue.append(job)

    print("queue drained", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
