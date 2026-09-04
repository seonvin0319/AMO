#!/usr/bin/env python3
"""Compatibility entry point for pre-package AMO launchers."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))
    runpy.run_module("amo", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
