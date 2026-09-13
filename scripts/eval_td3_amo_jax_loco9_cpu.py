#!/usr/bin/env python3
"""Loco9 wrapper: default AMO-main CPU eval reaper on the JAX loco9 store."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = Path("/raid/ext_csv/AMO_store/td3_amo_jax_loco9_default_seeds0to3")
sys.path.insert(0, str(ROOT / "scripts"))

from eval_checkpoints_cpu import main  # noqa: E402


if __name__ == "__main__":
    argv = ["--runs-root", str(STORE / "runs"), *sys.argv[1:]]
    raise SystemExit(main(argv))
