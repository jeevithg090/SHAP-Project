#!/usr/bin/env python3
"""Backwards-compatible entry point for the objective-level summary.

Equivalent to ``hand-injury-ai summary [--run-pipeline]``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from hand_injury_ai.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["summary", *sys.argv[1:]]))
