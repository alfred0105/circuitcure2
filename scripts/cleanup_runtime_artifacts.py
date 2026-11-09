#!/usr/bin/env python3
"""Safely preview or remove old generated data/ui_runs directories."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--older-than-days", type=int, default=30)
    parser.add_argument("--apply", action="store_true", help="Actually remove eligible generated run folders")
    args = parser.parse_args()
    if args.older_than_days < 1:
        parser.error("--older-than-days must be at least 1")
    run_root = ROOT / "data/ui_runs"
    cutoff = time.time() - args.older_than_days * 86400
    if not run_root.is_dir():
        print("No generated run directory exists.")
        return 0
    candidates = []
    for child in run_root.iterdir():
        if (child.is_dir() and child.name != "" and child.stat().st_mtime < cutoff
                and not (child / ".circuitcare-active").exists()):
            candidates.append(child)
    for child in sorted(candidates):
        size = sum(file.stat().st_size for file in child.rglob("*") if file.is_file())
        action = "REMOVE" if args.apply else "WOULD_REMOVE"
        print(f"{action}\t{child.relative_to(ROOT)}\t{size} bytes")
        if args.apply:
            shutil.rmtree(child)
    if not candidates:
        print("No eligible generated run folders.")
    if not args.apply:
        print("Dry run only. Add --apply to remove these generated directories.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
