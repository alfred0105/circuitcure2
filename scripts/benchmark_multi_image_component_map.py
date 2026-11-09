#!/usr/bin/env python3
"""Reproducible synthetic and pinned PRISM pipeline smoke; not accuracy evaluation."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from circuitcare.vision.detector_backend import model_file_info
from circuitcare.vision.reconstruction.multi_image_component_map import (
    CaptureRole, MultiImageComponentMapService, PhotoInput,
)

ROOT = Path(__file__).resolve().parents[1]


def _png(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def _count(values):
    result = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def run_synthetic() -> dict:
    rng = np.random.default_rng(216)
    pixels = rng.integers(0, 256, (420, 660, 3), dtype=np.uint8)
    source = np.asarray(Image.fromarray(pixels).filter(ImageFilter.GaussianBlur(0.8))).copy()
    crops = (source[:, :420], source[:, 120:540], source[:, 240:660])
    encoded = [_png(crop) for crop in crops]
    service = MultiImageComponentMapService()
    inputs = [PhotoInput(encoded[i], "TOP", role, f"view-{i}.png") for i, role in enumerate(
        (CaptureRole.REFERENCE, CaptureRole.OVERVIEW, CaptureRole.CLOSEUP))]
    session = service.ingest("synthetic-board", inputs)

    def make_detector(photo_session):
        # The same injected component appears at scene x=300 in every crop.
        local_x_by_image = {id(photo.image): 300 - offset
                            for photo, offset in zip(photo_session.photos, (0, 120, 240))}

        def detector(image):
            local_x = local_x_by_image[id(image)]
            return {"status": "INFERENCE_COMPLETE", "detections": [{
                "detection_id": "U1", "class_name": "integrated_circuit", "confidence": 0.9,
                "bbox_xyxy": [local_x - 12, 190, local_x + 12, 214]}]}
        return detector

    detector = make_detector(session)

    start = time.perf_counter()
    geometry_result = service.build_map(session)
    elapsed = time.perf_counter() - start
    tracemalloc.start()
    service.build_map(session)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result = service.build_map(session, detector=detector)
    side = result["sides"]["TOP"]
    registrations = side["registrations"]
    errors = []
    for photo, dx in zip(session.photos, (0, 120, 240)):
        matrix = np.asarray(next(r for r in registrations if r["photo_id"] == photo.photo_id)
                            ["transform_moving_to_canonical"])
        point = matrix @ np.array([0.0, 200.0, 1.0])
        errors.append(float(np.linalg.norm(point[:2] / point[2] - np.array([dx, 200.0]))))
    coverage = side["canonical_image"]["coverage_metrics"]
    components = side["components"]
    previous_session = service.ingest("synthetic-board", inputs[:2])
    previous = service.build_map(previous_session, detector=make_detector(previous_session))
    prior_id = previous["sides"]["TOP"]["components"][0]["component_id"] if previous["sides"]["TOP"]["components"] else None
    expanded = service.build_map(session, detector=detector, previous_component_map=previous)
    preserved_id = next((item["component_id"] for item in expanded["sides"]["TOP"]["components"]
                         if item["component_id"] == prior_id), None)
    baseline = json.loads((ROOT / "data/benchmarks/phase21_2/synthetic_summary.json").read_text(encoding="utf-8"))
    return {
        "schema_version": "circuitcare.phase21.2.1.synthetic-benchmark.v1",
        "track": "SYNTHETIC_GEOMETRY_AND_INJECTED_DETECTION_NOT_REAL_ACCURACY",
        "seed": 216, "input_views": 3, "input_size_px": [420, 420],
        "known_crop_offsets_px": [0, 120, 240],
        "registration_status_counts": _count(r["status"] for r in registrations),
        "known_translation_error_px": errors,
        "independent_landmark_error": "NOT_EVALUATED_NO_INDEPENDENT_LANDMARK_BUNDLE",
        "mosaic_size_px": side["canonical_image"]["image_size_px"],
        "valid_pixel_coverage_ratio": coverage["valid_pixel_coverage_ratio"],
        "canvas_coverage_ratio": coverage["canvas_coverage_ratio"],
        "unobserved_area_ratio": coverage["unobserved_area_ratio"],
        "overlap_union_ratio_to_canvas": coverage["overlap_union_ratio_to_canvas"],
        "source_contribution_ratio": coverage["source_contribution_ratio"],
        "overlap_photometric_residual_mean_abs_rgb": coverage["overlap_photometric_residual_mean_abs_rgb"],
        "ghosting_parallax": coverage["ghosting_diagnostics"],
        "injected_component_count": len(components),
        "injected_observation_count": sum(len(c["component_observation_ids"]) for c in components),
        "identity_stability_after_added_view": "PRESERVED" if prior_id and preserved_id else "NOT_PRESERVED",
        "identity_precision": "NOT_EVALUATED_NO_INDEPENDENT_COMPONENT_IDENTITY_LABELS",
        "fault_overlay": "UNIT_TESTED_WITH_INJECTED_DETECTIONS_ONLY",
        "runtime_seconds": elapsed,
        "python_tracemalloc_peak_bytes": peak_bytes,
        "memory_note": "Python allocation peak only; excludes native OpenCV/Pillow and process RSS.",
        "phase21_2_baseline_comparison": {
            "baseline_runtime_seconds": baseline["runtime_seconds"],
            "current_runtime_seconds_same_geometry_no_detector": elapsed,
            "baseline_mosaic_size_px": baseline["mosaic_size_px"],
            "current_mosaic_size_px": geometry_result["sides"]["TOP"]["canonical_image"]["image_size_px"],
            "baseline_scene_rectangle_coverage_estimate": baseline["expected_scene_coverage_ratio"],
            "current_valid_pixel_coverage_ratio": coverage["valid_pixel_coverage_ratio"],
            "comparison_note": "Same seeded crop fixture; one local timing per version. Legacy coverage used scene rectangle area, current coverage uses valid warped masks.",
        },
    }


def run_prism(model_path: Path) -> dict:
    dataset = ROOT / "data/external/prism/Dataset"
    model_info = model_file_info(model_path)
    if model_info.get("status") == "READY":
        model_info["path"] = str(model_path)
    records = []
    started = time.perf_counter()
    for board_dir in sorted(path for path in dataset.iterdir() if path.is_dir()):
        inputs = []
        hashes = {}
        for side in ("TOP", "BOTTOM"):
            path = board_dir / f"{side}.png"
            if not path.is_file():
                continue
            raw = path.read_bytes()
            hashes[path.name] = hashlib.sha256(raw).hexdigest()
            inputs.append(PhotoInput(raw, side, CaptureRole.REFERENCE, path.name))
        if not inputs:
            continue
        service = MultiImageComponentMapService()
        session = service.ingest(board_dir.name, inputs)
        result = service.build_map(session, model_path=model_path if model_info["status"] == "READY" else None)
        records.append({"board_id": board_dir.name, "input_sha256": hashes,
            "side_status": {side: {"registration_statuses": _count(r["status"] for r in value["registrations"]),
                "detection_status": value["detection_status"],
                "detection_observations": sum(1 for item in value["component_observations"] if item.get("bbox_xyxy")),
                "component_candidates": len(value["components"]),
                "valid_pixel_coverage_ratio": value["canonical_image"]["coverage_metrics"]["valid_pixel_coverage_ratio"]}
                for side, value in result["sides"].items()}})
    return {"schema_version": "circuitcare.phase21.2.1.prism-pipeline-smoke.v1",
        "track": "PINNED_PUBLIC_DATASET_IMAGES_SINGLE_VIEW_PER_SIDE_NOT_PHYSICAL_MULTIVIEW_VALIDATION",
        "dataset_commit": "9189c87bbec6995b745b809db99cef6d3269401f",
        "dataset_image_count": sum(len(item["input_sha256"]) for item in records),
        "boards": len(records), "detector": model_info,
        "model_accuracy": "NOT_EVALUATED_NO_INDEPENDENT_DETECTION_LABEL_SCORING_IN_THIS_RUN",
        "real_physical_multiview_evaluation": "BLOCKED_REAL_IMAGE_DATA_UNAVAILABLE",
        "runtime_seconds": time.perf_counter() - started, "results": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/benchmarks/phase21_2_1"))
    parser.add_argument("--model", type=Path, default=Path("models/pcb_component_detector.onnx"))
    args = parser.parse_args()
    reports = {"synthetic_summary.json": run_synthetic(),
               "prism_pipeline_smoke.json": run_prism(args.model)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, report in reports.items():
        output = args.output_dir / filename
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{output}:\n{json.dumps(report, indent=2)}")


if __name__ == "__main__":
    main()
