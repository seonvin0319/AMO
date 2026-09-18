#!/usr/bin/env python3
"""iql_amo_qweight JAX: hopper→walker→antmaze→halfcheetah.

π_B-weighted Q + β_B outer TD MSE (μ_hat frozen).
Grid: beta_initial ∈ {5,2,1} × seeds 0–3 × env × rho_lr ∈ {3e-4,1e-3,1e-4}.
Queue order: **β-major** (5→2→1), then seed 0→1–3, then
hopper→walker→antmaze→halfcheetah, then ρ_lr 3e-4→1e-3→1e-4.
So all β5 (180 cells) finish before any β2/β1.
Train with --no-eval; checkpoints every 20k for offline CPU eval.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(
    "/raid/ext_csv/AMO_store/iql_amo_qweight_jax_beta125_rho_loco_antmaze_seeds0to3"
)
PYTHON = Path("/home/ext_csv/miniconda3/envs/amo-jax/bin/python")
DATASET_ROOT = Path("/raid/ext_csv/datasets/d4rl")
BEHAVIOR_ROOT = Path("/raid/ext_csv/AMO_store/iql_amo_qweight_behavior_shared")
GROUP = "iql-amo-qweight-jax-beta125-rho-loco-antmaze-seeds0-3"
SEEDS = (0, 1, 2, 3)

LOCO_VARIANTS = ("medium-v2", "medium-replay-v2", "medium-expert-v2")
HOPPER = tuple(f"hopper-{v}" for v in LOCO_VARIANTS)
WALKER = tuple(f"walker2d-{v}" for v in LOCO_VARIANTS)
HALFCHEETAH = tuple(f"halfcheetah-{v}" for v in LOCO_VARIANTS)
ANTMAZE = (
    (
        "antmaze-umaze-v2",
        "Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5",
    ),
    (
        "antmaze-umaze-diverse-v2",
        "Ant_maze_u-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5",
    ),
    (
        "antmaze-medium-play-v2",
        "Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5",
    ),
    (
        "antmaze-medium-diverse-v2",
        "Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5",
    ),
    (
        "antmaze-large-play-v2",
        "Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5",
    ),
    (
        "antmaze-large-diverse-v2",
        "Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5",
    ),
)
ENVIRONMENTS = HOPPER + WALKER + tuple(env for env, _ in ANTMAZE) + HALFCHEETAH
DATASET_FILES = {env: name for env, name in ANTMAZE}

# Queue priority: beta 5 → 2 → 1 (then ρ_lr 3e-4/1e-3/1e-4).
BETA_RHOS = (
    (5, 3e-4, "3em4", ROOT / "configs" / "iql_amo_qweight_beta5_rho3em4.yaml"),
    (5, 1e-3, "1em3", ROOT / "configs" / "iql_amo_qweight_beta5_rho1em3.yaml"),
    (5, 2e-3, "2em3", ROOT / "configs" / "iql_amo_qweight_beta5_rho2em3.yaml"),
    (2, 3e-4, "3em4", ROOT / "configs" / "iql_amo_qweight_beta2_rho3em4.yaml"),
    (2, 1e-3, "1em3", ROOT / "configs" / "iql_amo_qweight_beta2_rho1em3.yaml"),
    (2, 2e-3, "2em3", ROOT / "configs" / "iql_amo_qweight_beta2_rho2em3.yaml"),
    (1, 3e-4, "3em4", ROOT / "configs" / "iql_amo_qweight_beta1_rho3em4.yaml"),
    (1, 1e-3, "1em3", ROOT / "configs" / "iql_amo_qweight_beta1_rho1em3.yaml"),
    (1, 2e-3, "2em3", ROOT / "configs" / "iql_amo_qweight_beta1_rho2em3.yaml"),
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
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).splitlines()
    return {"commit": commit, "dirty": bool(dirty), "dirty_status": dirty}


def cell_tag(beta: int, rho_tag: str) -> str:
    return f"b{beta}_rho{rho_tag}"


def run_id(environment: str, beta: int, rho_tag: str, seed: int) -> str:
    return f"iqlamo_qw_jax_{ENV_SHORT[environment]}_{cell_tag(beta, rho_tag)}_s{seed}"


def run_dir(environment: str, beta: int, rho_tag: str, seed: int) -> Path:
    return OUT / "runs" / run_id(environment, beta, rho_tag, seed)


def dataset_path(environment: str) -> Path:
    if environment in DATASET_FILES:
        path = DATASET_ROOT / DATASET_FILES[environment]
    else:
        name = environment.replace("-", "_").replace("_v2", "-v2") + ".hdf5"
        path = DATASET_ROOT / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def completed(environment: str, beta: int, rho_tag: str, seed: int) -> bool:
    d = run_dir(environment, beta, rho_tag, seed)
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
            ["pgrep", "-af", "AMO-main/train.py"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return 0
    return sum(
        1
        for line in output.splitlines()
        if "AMO-main/train.py" in line
        and "--algorithm=iql_amo_qweight" in line
        and "--backend=jax" in line
        and "pgrep" not in line
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


def command(
    environment: str, beta: int, rho_tag: str, seed: int, config: Path
) -> list[str]:
    output = run_dir(environment, beta, rho_tag, seed)
    behavior = BEHAVIOR_ROOT / ENV_SHORT[environment] / "behavior.npz"
    if not behavior.exists():
        raise FileNotFoundError(
            f"Missing behavior density for {environment}: {behavior}"
        )
    cmd = [
        str(PYTHON),
        "-u",
        str(ROOT / "train.py"),
        "--algorithm=iql_amo_qweight",
        "--backend=jax",
        f"--env={environment}",
        f"--seed={seed}",
        "--device=cuda:0",
        f"--config={config}",
        f"--dataset={dataset_path(environment)}",
        f"--output={output}",
        "--behavior-path",
        str(behavior),
        "--no-eval",
        "--save-every=20000",
    ]
    ckpt = output / "checkpoint.npz"
    if ckpt.exists():
        truncate_logs_to_checkpoint(output, ckpt)
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


def job_cells() -> list[tuple[str, int, str, int, Path, float]]:
    """Beta-major 5→2→1, then seed 0→1–3, then env, then ρ_lr.

    Example: all β5 (seed0…3 × all envs × 3 ρ) before any β2/β1.
    """
    by_beta: dict[int, list[tuple[float, str, Path]]] = {5: [], 2: [], 1: []}
    for beta, rho_lr, rho_tag, config in BETA_RHOS:
        by_beta[int(beta)].append((rho_lr, rho_tag, config))
    cells: list[tuple[str, int, str, int, Path, float]] = []
    for beta in (5, 2, 1):
        for seed in SEEDS:
            for environment in ENVIRONMENTS:
                for rho_lr, rho_tag, config in by_beta[beta]:
                    cells.append((environment, beta, rho_tag, seed, config, rho_lr))
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
    for _, _, _, config in BETA_RHOS:
        if not config.exists():
            raise SystemExit(f"missing config: {config}")
    for environment in ENVIRONMENTS:
        dataset_path(environment)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "jobs").mkdir(exist_ok=True)
    (OUT / "runs").mkdir(exist_ok=True)
    (OUT / "wandb").mkdir(exist_ok=True)

    n_jobs = len(ENVIRONMENTS) * len(BETA_RHOS) * len(SEEDS)
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
                "algorithm": "iql_amo_qweight",
                "backend": "jax",
                "rc_qv": False,
                "beta_initials": [5, 2, 1],
                "beta_priority": "5 then 2 then 1 (outermost)",
                "queue_order": "beta(5>2>1) → seed → env → rho_lr",
                "rho_lrs": [3e-4, 1e-3, 1e-4],
                "environments": list(ENVIRONMENTS),
                "env_order": "hopper→walker→antmaze→halfcheetah",
                "seeds": list(SEEDS),
                "seed_major": False,
                "beta_major": True,
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
    for environment, beta, rho_tag, seed, config, rho_lr in job_cells():
        rid = run_id(environment, beta, rho_tag, seed)
        job_dir = OUT / "jobs" / rid
        job_dir.mkdir(exist_ok=True)
        if (job_dir / "CANCELLED.json").exists():
            status = "cancelled"
        elif completed(environment, beta, rho_tag, seed) or (
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
            "beta_initial": beta,
            "rho_lr": rho_lr,
            "rho_tag": rho_tag,
            "config": str(config),
            "seed": seed,
            "status": status,
            "pid": None,
            "gpu": None,
            "log": str(job_dir / "stdout_stderr.log"),
            "job_dir": str(job_dir),
            "output": str(run_dir(environment, beta, rho_tag, seed)),
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
                            "beta": j["beta_initial"],
                            "rho": j["rho_tag"],
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
        f"iql_amo_qweight queue pending={len(pending)} max_parallel={max_parallel} "
        f"n_jobs={n_jobs} out={OUT}",
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
                    int(job["beta_initial"]),
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
                    if completed(
                        job["environment"],
                        int(job["beta_initial"]),
                        job["rho_tag"],
                        int(job["seed"]),
                    )
                    else -1
                )
            job_dir = Path(job["job_dir"])
            (job_dir / "RUNNING.json").unlink(missing_ok=True)
            if log_handle is not None:
                log_handle.close()
            ok = completed(
                job["environment"],
                int(job["beta_initial"]),
                job["rho_tag"],
                int(job["seed"]),
            )
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
                and not completed(
                    job["environment"],
                    int(job["beta_initial"]),
                    job["rho_tag"],
                    int(job["seed"]),
                )
            ):
                queue.append(job)

    print("queue drained", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
