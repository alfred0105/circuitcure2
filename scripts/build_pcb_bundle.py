#!/usr/bin/env python3
"""Write provenance manifests for external PCB data without copying source files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circuitcare.vision.reconstruction.dataset_bundle import build_prism_bundle
from circuitcare.vision.reconstruction.prism_dataset import SAMPLES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/external/prism")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/benchmarks/phase21_1/bundles")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", choices=SAMPLES)
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    samples = SAMPLES if args.all else (args.sample,)
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in samples:
        manifest = build_prism_bundle(args.dataset_root, name)
        (out / f"{name}.json").write_text(manifest.to_json(), encoding="utf-8")
    print(f"Wrote {len(samples)} manifest(s) to {out}; source data were not copied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
