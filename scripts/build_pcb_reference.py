#!/usr/bin/env python3
"""Build image-space PRISM references only for independently verified transforms."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circuitcare.vision.reconstruction.coordinate_frames import (
    CoordinateFrame, CoordinateTransform, TransformType, VerificationStatus,
)
from circuitcare.vision.reconstruction.ground_truth import (
    project_reference_geometry, rasterize_projected_geometry, summarize_reference_geometry,
)
from circuitcare.vision.reconstruction.prism_dataset import PRISM_COMMIT, SAMPLES, load_references, load_sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/external/prism")
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/benchmarks/phase21_1/projected_references")
    parser.add_argument("--sample", choices=SAMPLES)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all and not args.sample:
        parser.error("Choose --sample or --all")
    try:
        registration = json.loads(args.registration.read_text(encoding="utf-8"))
        if registration.get("prism_commit") != PRISM_COMMIT:
            parser.error("Registration record uses a different PRISM revision")
        samples = {row["sample_id"]: row for row in registration.get("sample_results", [])}
        output_root = args.output_dir.expanduser().resolve()
        report_rows = []
        for name in list(SAMPLES) if args.all else [args.sample]:
            sample = load_sample(args.dataset_root, name)
            reference = load_references(args.dataset_root, name)["expected_geometry"]
            side_report = samples.get(name, {}).get("sides", {})
            side_rows = {}
            for side, image in (("TOP", sample.top_image), ("BOTTOM", sample.bottom_image)):
                row = side_report.get(side, {})
                if row.get("registration_status") != VerificationStatus.VERIFIED.value or not row.get("transform"):
                    side_rows[side] = {
                        "status": "NOT_PROJECTED_REGISTRATION_NOT_VERIFIED",
                        "registration_status": row.get("registration_status", "NOT_AVAILABLE"),
                        "reference_inventory": summarize_reference_geometry(reference),
                        "visibility_mask": "NOT_AVAILABLE",
                    }
                    continue
                raw = row["transform"]
                transform = CoordinateTransform(
                    source_frame=CoordinateFrame(raw["source_frame"]),
                    target_frame=CoordinateFrame(raw["target_frame"]),
                    matrix=tuple(tuple(float(value) for value in line) for line in raw["matrix"]),
                    transform_type=TransformType(raw["transform_type"]),
                    board_side=raw["board_side"], source_evidence=raw["source_evidence"],
                    calibration_point_ids=tuple(raw["calibration_point_ids"]),
                    fit_error_px=raw["fit_error_px"], validation_error_px=raw["validation_error_px"],
                    verification_status=VerificationStatus(raw["verification_status"]),
                    inlier_count=raw["inlier_count"], inlier_ratio=raw["inlier_ratio"],
                    ransac_threshold_px=raw["ransac_threshold_px"],
                )
                from PIL import Image
                width, height = Image.open(image).size
                projected = project_reference_geometry(reference, transform, board_side=side,
                                                       image_size_px=(width, height))
                masks = rasterize_projected_geometry(projected)
                out = output_root / name / f"{side.lower()}_design_reference.npz"
                out.parent.mkdir(parents=True, exist_ok=True)
                import numpy as np
                np.savez_compressed(out, **masks)
                json_path = out.with_suffix(".json")
                json_path.write_text(json.dumps(projected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                side_rows[side] = {"status": "PROJECTED_NOT_VISIBILITY_GROUNDED",
                                   "registration_status": "VERIFIED", "json": str(json_path),
                                   "mask_archive": str(out), "visibility_mask": "NOT_AVAILABLE"}
            report_rows.append({"sample_id": name, "sides": side_rows})
        result = {
            "schema": "phase21.1-projected-reference-v1", "prism_commit": PRISM_COMMIT,
            "circuitcare_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                                   capture_output=True, text=True).stdout.strip(),
            "sample_results": report_rows,
            "note": "Projected design geometry is not equivalent to photo-visible copper. Occlusion requires independent annotation.",
        }
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved projected-reference summary to {output_root / 'summary.json'}")
        return 0
    except (OSError, subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"Reference build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
