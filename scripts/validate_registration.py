#!/usr/bin/env python3
"""Validate registration report provenance, point splits, side and thresholds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circuitcare.vision.reconstruction.prism_dataset import PRISM_COMMIT, SAMPLES, load_sample


def validate_report(report: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    if report.get("prism_commit") != PRISM_COMMIT:
        issues.append({"code": "DATASET_COMMIT_MISMATCH", "detail": "PRISM commit differs from pinned revision"})
    records = {row.get("sample_id"): row for row in report.get("sample_results", [])}
    if len(records) != len(report.get("sample_results", [])):
        issues.append({"code": "DUPLICATE_SAMPLE", "detail": "Sample IDs are duplicated"})
    for name in report.get("sample_results", []):
        sample_id = name.get("sample_id")
        if sample_id not in SAMPLES:
            issues.append({"code": "UNKNOWN_SAMPLE", "detail": str(sample_id)})
            continue
        sample = load_sample(dataset_root, sample_id)
        if name.get("dataset_hash") != sample.dataset_hash:
            issues.append({"code": "INPUT_HASH_MISMATCH", "detail": sample_id})
        sides = name.get("sides", {})
        for side in ("TOP", "BOTTOM"):
            row = sides.get(side)
            if row is None:
                issues.append({"code": "MISSING_SIDE", "detail": f"{sample_id}/{side}"})
                continue
            transform = row.get("transform")
            if transform:
                calibration_ids = set(transform.get("calibration_point_ids", []))
                validation_ids = set((transform.get("validation_error_px") or {}).keys())
                if calibration_ids & validation_ids:
                    issues.append({"code": "CALIBRATION_VALIDATION_LEAK", "detail": f"{sample_id}/{side}"})
                if transform.get("board_side") != side:
                    issues.append({"code": "BOARD_SIDE_MISMATCH", "detail": f"{sample_id}/{side}"})
                if row.get("registration_status") == "VERIFIED":
                    threshold = report.get("thresholds", {}).get("acceptance_threshold_px")
                    errors = list((transform.get("validation_error_px") or {}).values())
                    metrics = row.get("metrics", {})
                    if (threshold is None or not errors or max(errors) > threshold
                            or metrics.get("validation_independently_reviewed") is not True):
                        issues.append({"code": "INVALID_VERIFIED_STATUS", "detail": f"{sample_id}/{side}"})
            elif row.get("registration_status") == "VERIFIED":
                issues.append({"code": "VERIFIED_WITHOUT_TRANSFORM", "detail": f"{sample_id}/{side}"})
    missing_samples = sorted(set(SAMPLES) - set(records))
    if report.get("samples_requested") == len(SAMPLES) and missing_samples:
        issues.append({"code": "SAMPLES_MISSING", "detail": ", ".join(missing_samples)})
    return {"schema": "phase21.1-registration-validation-v1",
            "status": "VALID" if not issues else "INVALID",
            "samples_checked": len(records), "views_checked": sum(len(row.get("sides", {})) for row in records.values()),
            "verified_view_count": sum(row.get("registration_status") == "VERIFIED"
                                        for sample in records.values() for row in sample.get("sides", {}).values()),
            "issues": issues}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/external/prism")
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/benchmarks/phase21_1_1/validation/summary.json")
    args = parser.parse_args()
    try:
        report = json.loads(args.registration.read_text(encoding="utf-8"))
        result = validate_report(report, args.dataset_root)
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Registration record {result['status']}: {result['views_checked']} views, "
              f"{result['verified_view_count']} verified; report {output}")
        return 0 if result["status"] == "VALID" else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Registration validation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
