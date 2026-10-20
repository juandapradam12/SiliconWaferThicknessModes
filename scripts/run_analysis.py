#!/usr/bin/env python3
"""Thin wrapper around ``python -m wafer_pca`` for script-style invocation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafer_pca.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["--figures", str(ROOT / "figures")]))
