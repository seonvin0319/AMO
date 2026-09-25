#!/usr/bin/env python3
"""TD3+RAPO Fixed-1 missing-lr grid. Does not touch the alpha=1 adaptive runs."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/ext_csh/AMO_fixed1")
OUT = Path("/raid/ext_csh/AMO_store/td3_amo_fixed1_rapo_ext_csh")
PY = Path("/home/ext_csh/miniconda3/envs/capo_jax/bin/python")
TEMPLATE = ROOT / "configs" / "td3_amo_fixed1_rapo.yaml"
OTHER = "/raid/ext_csh/AMO_store/td3_amo_bootrms_a1_fill14"

LOCO = (
    ("halfcheetah-medium-v2", "hc-m"),
    ("halfcheetah-medium-replay-v2", "hc-mr"),
    ("halfcheetah-medium-expert-v2", "hc-me"),
    ("hopper-medium-v2", "h-m"),
    ("hopper-medium-replay-v2", "h-mr"),
    ("hopper-medium-expert-v2", "h-me"),
    ("walker2d-medium-v2", "w-m"),
    ("walker2d-medium-replay-v2", "w-mr"),
    ("walker2d-medium-expert-v2", "w-me"),
)
ANT = (
    ("antmaze-umaze-v2", "am-u"),
    ("antmaze-umaze-diverse-v2", "am-ud"),
    ("antmaze-medium-play-v2", "am-mp"),
    ("antmaze-medium-diverse-v2", "am-md"),
    ("antmaze-large-play-v2", "am-lp"),
    ("antmaze-large-diverse-v2", "am-ld"),
)
LRS = {"0.0003": "r3e4", "0.001": "r1e3", "0.002": "r2e3"}


def jobs():
    out = []
    for env, short in LOCO:
        for lr in ("0.0003", "0.001"):
            for seed in (0, 1, 2, 3):
                out.append((env, seed, lr, f"f1_{short}_{LRS[lr]}_s{seed}"))
    for env, short in ANT:
        for lr in ("0.001", "0.002"):
            for seed in (0, 1, 2, 3):
                out.append((env, seed, lr, f"f1_{short}_{LRS[lr]}_s{seed}"))
    return out


def cell(tag: str) -> Path:
    return OUT / "jobs" / tag


def ckpt_steps(run: Path) -> int:
    path = run / "checkpoint.npz"
    if not path.is_file():
        return 0
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        return int(json.loads(str(data["__metadata__"]))["steps"])


def eval_done(run: Path) -> bool:
    path = run / "eval.jsonl"
    if not path.is_file():
        return False
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("step") or 0) >= 1_000_000 and int(row.get("episodes") or 0) >= 50:
            return True
    return False


def other_alive() -> bool:
    proc = Path("/proc")
    for pid in proc.iterdir():
        if not pid.name.isdigit():
            continue
        try:
            cmd = (pid / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if "train.py" in cmd and OTHER in cmd:
            return True
    return False


def write_cfg(dest: Path, lr: str) -> None:
    text = TEMPLATE.read_text()
    text = text.replace("alpha_lr: 0.0003", f"alpha_lr: {lr}", 1)
    dest.write_text(text)


def main() -> None:
    mode = os.environ.get("FIXED1_MODE", "gpu")
    gpus = ["0", "1"] if mode == "gpu" else ["cpu"]
    pending = []
    eval_only = []
    done = 0
    for env, seed, lr, tag in jobs():
        run = cell(tag) / "run"
        if ckpt_steps(run) >= 1_000_000 and eval_done(run):
            done += 1
            continue
        if ckpt_steps(run) >= 1_000_000:
            eval_only.append((env, seed, lr, tag))
        else:
            pending.append((env, seed, lr, tag))
    print(
        f"[fixed1] mode={mode} done={done} eval_only={len(eval_only)} train={len(pending)}",
        flush=True,
    )
    queue = eval_only + pending
    slots = {gpu: None for gpu in gpus}
    stop = False

    def handle(signum, _frame):
        nonlocal stop
        stop = True
        print(f"[fixed1] signal {signum}", flush=True)
        for proc in slots.values():
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)

    while queue or any(slots.values()):
        for gpu, proc in list(slots.items()):
            if proc is None or proc.poll() is None:
                continue
            tag = proc.tag
            kind = proc.kind
            rc = proc.returncode
            run = cell(tag) / "run"
            if kind == "train" and rc == 0 and ckpt_steps(run) >= 1_000_000:
                print(f"[TRAINED] {tag}", flush=True)
                queue.append((None, None, None, tag))
                queue[-1] = ("eval", 0, "", tag)
            elif kind == "eval" and eval_done(run):
                (cell(tag) / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
                print(f"[DONE] {tag}", flush=True)
            else:
                print(f"[FAIL] {tag} {kind} rc={rc}", flush=True)
            slots[gpu] = None
        if stop:
            time.sleep(2)
            continue
        if mode == "gpu" and other_alive():
            print("[fixed1] waiting for the alpha=1 adaptive runs", flush=True)
            time.sleep(30)
            continue
        for gpu in gpus:
            if slots[gpu] is not None or not queue:
                continue
            item = queue.pop(0)
            if item[0] == "eval":
                tag = item[3]
                kind = "eval"
                cmd = [
                    str(PY), "-u", str(ROOT / "scripts" / "eval_checkpoints_cpu.py"),
                    "--runs-root", str(cell(tag)), "--run", "run",
                    "--episodes", "10", "--final-repeats", "5",
                    "--device", "cpu", "--final-only", "--keep-mid-ckpts",
                ]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ""
                env["JAX_PLATFORMS"] = "cpu"
            else:
                env_name, seed, lr, tag = item
                kind = "train"
                run = cell(tag) / "run"
                run.mkdir(parents=True, exist_ok=True)
                cfg = cell(tag) / "config.yaml"
                write_cfg(cfg, lr)
                env = os.environ.copy()
                if gpu == "cpu":
                    env["CUDA_VISIBLE_DEVICES"] = ""
                    env["JAX_PLATFORMS"] = "cpu"
                    device = "cpu"
                else:
                    env["CUDA_VISIBLE_DEVICES"] = gpu
                    env.pop("JAX_PLATFORMS", None)
                    env.pop("JAX_PLATFORM_NAME", None)
                    device = "cuda:0"
                cmd = [
                    str(PY), "-u", str(ROOT / "train.py"),
                    "--algorithm", "td3_amo", "--backend", "jax",
                    "--env", env_name, "--config", str(cfg),
                    "--seed", str(seed), "--output", str(run),
                    "--device", device, "--log-every", "5000",
                    "--save-every", "20000", "--no-eval",
                ]
                if (run / "checkpoint.npz").is_file():
                    cmd.extend(["--resume", str(run / "checkpoint.npz")])
            log = open(cell(tag) / f"{kind}.log", "a")
            proc = subprocess.Popen(
                cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            proc.tag = tag
            proc.kind = kind
            slots[gpu] = proc
            print(f"[SPAWN] {tag} {kind} pid={proc.pid} gpu={gpu}", flush=True)
        time.sleep(5)
    print("[fixed1] exit", flush=True)


if __name__ == "__main__":
    main()
