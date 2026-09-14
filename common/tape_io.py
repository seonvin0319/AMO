"""Shared helpers for TD3+BC tape-driven backend experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
KEYS = (
    "observations",
    "actions",
    "rewards",
    "terminals",
    "next_observations",
    "next_actions",
)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_array(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr)
    return digest_bytes(arr.tobytes() + str(arr.dtype).encode() + str(arr.shape).encode())


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def read_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def block_hashes(indices: np.ndarray, noise: np.ndarray, block: int = 100) -> list:
    """SHA256 of each contiguous block of tape rows."""
    assert len(indices) == len(noise)
    out = []
    for start in range(0, len(indices), block):
        sl = slice(start, min(start + block, len(indices)))
        payload = indices[sl].tobytes() + noise[sl].tobytes()
        out.append(
            {
                "start": int(start),
                "end": int(min(start + block, len(indices))),
                "sha256": digest_bytes(payload),
            }
        )
    return out


def index_batch(data: dict, indices: np.ndarray) -> dict:
    idx = np.asarray(indices, np.int64)
    batch = {k: np.asarray(data[k][idx], np.float32) for k in KEYS if k in data}
    for k in ("rewards", "terminals"):
        if k in batch and batch[k].ndim == 1:
            batch[k] = batch[k].reshape(-1, 1)
    return batch


def runtime_fingerprint(backend: str) -> dict:
    import importlib.metadata
    import os

    info = {
        "python": platform.python_version(),
        "executable": None,
        "platform": platform.platform(),
        "packages": {},
        "environment": {
            k: os.environ.get(k)
            for k in (
                "JAX_PLATFORMS",
                "JAX_ENABLE_X64",
                "XLA_FLAGS",
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_TF32_OVERRIDE",
            )
        },
    }
    try:
        import sys

        info["executable"] = sys.executable
    except Exception:
        pass
    for package in (
        "numpy",
        "torch",
        "jax",
        "jaxlib",
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12",
    ):
        try:
            info["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if backend == "torch":
        try:
            import torch

            info.update(
                cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                cuda_available=torch.cuda.is_available(),
            )
            if torch.cuda.is_available():
                info["device_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:
            info["torch_error"] = str(exc)
    if backend == "jax":
        try:
            import jax

            info.update(
                jax_x64=bool(jax.config.jax_enable_x64),
                matmul_precision=str(jax.config.jax_default_matmul_precision),
                devices=[str(d) for d in jax.devices()],
                default_backend=jax.default_backend(),
            )
        except Exception as exc:
            info["jax_error"] = str(exc)
    try:
        info["nvidia_smi"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except Exception:
        info["nvidia_smi"] = None
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ).strip()
        info["git_dirty"] = bool(dirty)
    except Exception:
        info["git_commit"] = None
        info["git_dirty"] = None
    return info


def source_hashes() -> dict:
    paths = [
        ROOT / "common" / "agent.py",
        ROOT / "common" / "tape_io.py",
        ROOT / "common" / "optim.py",
        ROOT / "common" / "jax_backend.py",
        ROOT / "common" / "torch_backend.py",
        ROOT / "algorithms" / "torch" / "td3_bc.py",
        ROOT / "algorithms" / "jax" / "td3_bc.py",
    ]
    return {str(p.relative_to(ROOT)): digest_file(p) for p in paths if p.exists()}
