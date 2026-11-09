#!/usr/bin/env python3
"""Fetch the pinned PRISM sample data with sparse checkout, outside Git history."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "external" / "prism"
REPO = "https://github.com/KamMalinski/PRISM.git"
COMMIT = "9189c87bbec6995b745b809db99cef6d3269401f"


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEST)
    args = parser.parse_args()
    dest = args.destination.expanduser().resolve()
    try:
        if (dest / ".git").is_dir():
            current = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                                     check=True, capture_output=True, text=True).stdout.strip()
            if current == COMMIT:
                print(f"PRISM {COMMIT} already present; reusing {dest}")
            else:
                raise RuntimeError(f"{dest} contains another Git checkout ({current}); left it untouched. "
                                   "Choose a different --destination.")
        elif dest.exists() and any(dest.iterdir()):
            raise RuntimeError(f"{dest} is non-empty and is not the pinned PRISM checkout; left untouched. "
                               "Choose a different --destination.")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            run("git", "clone", "--no-checkout", "--filter=blob:none", REPO, str(dest))
            run("git", "-C", str(dest), "sparse-checkout", "init", "--cone")
            run("git", "-C", str(dest), "sparse-checkout", "set", "Dataset", "evaluation/reference")
            run("git", "-C", str(dest), "checkout", COMMIT)
        sys.path.insert(0, str(ROOT / "src"))
        from circuitcare.vision.reconstruction.prism_dataset import validate_inventory
        inventory = validate_inventory(dest)
        if inventory["ready_samples"] != 12:
            raise RuntimeError(f"PRISM inventory incomplete: {inventory['ready_samples']}/12 ready")
        print(f"Verified {inventory['ready_samples']} PRISM boards at {COMMIT}: {dest}")
        return 0
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"PRISM download/validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
