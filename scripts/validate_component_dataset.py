#!/usr/bin/env python3
"""Validate a CircuitCare component annotation manifest and board split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circuitcare.vision.detection_evaluation import validate_annotation_manifest
from circuitcare.vision.detection_postprocess import PCB_MODEL_CLASSES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"Could not read JSON manifest: {exc}")
    result = validate_annotation_manifest(manifest, class_names=list(PCB_MODEL_CLASSES))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "VALID" and result["scoreable_status"] == "READY_REVIEWED_LABELS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
