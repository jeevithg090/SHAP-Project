#!/usr/bin/env python3
"""Backwards-compatible entry point: runs Task A and Task B end to end.

Equivalent to ``hand-injury-ai run``. See README.md for configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from hand_injury_ai.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
