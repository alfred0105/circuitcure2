#!/usr/bin/env python3
"""Run CircuitCare's image-only candidate pipeline on pinned PRISM images.

Reference files are loaded only after each photo has been processed. Geometry
metrics remain NOT_VERIFIED unless a separately validated pixel/mm transform
is supplied by a future evaluation implementation.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circuitcare.vision.board_detector import detect_board_candidates
from circuitcare.vision.perspective import rectify_board
from circuitcare.vision.reconstruction.copper_segmentation import (
    detect_pad_via_candidates, segment_visible_copper, vectorize_copper,
)
from circuitcare.vision.reconstruction.prism_dataset import (
    PRISM_COMMIT, SAMPLES, hash_reference_files, load_references, load_sample, validate_inventory,
)
from circuitcare.vision.reconstruction.topology import build_observed_topology

PRIORITY = ("Simple", "Divider", "Crossover", "Groundplane", "Meander")


def image_pipeline(path: Path, sample: str, side: str, output: Path, *, improved: bool) -> dict:
    started = time.perf_counter()
    image = Image.open(path).convert("RGB")
    detection = detect_board_candidates(image)
    candidates = detection.get("candidates", [])
    corners = candidates[0].get("corners_px") if candidates else None
    frame_status = "UNRECTIFIED_IMAGE_FRAME"
    if corners:
        try:
            image = rectify_board(image, corners)["image"]
            frame_status = "BOARD_BOUNDARY_CANDIDATE_RECTIFIED"
        except (ValueError, cv2.error, KeyError, TypeError):
            frame_status = "BOARD_DETECTION_CANDIDATE_RECTIFICATION_FAILED"
    original_size = image.size
    image.thumbnail((500, 500), Image.Resampling.LANCZOS)
    arr = np.asarray(image)
    if improved:
        result = segment_adaptive_color(arr, board_side=side)
    else:
        result = segment_visible_copper(arr, board_side=side)
    mask = result.mask.astype(np.uint8)
    board_side = "front" if side == "TOP" else "back"
    photo_id = f"prism-{sample}-{side.lower()}"
    traces = vectorize_copper(mask, photo_id=photo_id, board_side=board_side)
    pads = detect_pad_via_candidates(arr, photo_id=photo_id, board_side=board_side, copper_mask=mask)
    graph = build_observed_topology(mask, photo_id=photo_id, board_side=board_side, pad_candidates=pads)
    overlay = arr.copy()
    overlay[mask.astype(bool)] = (0.55 * overlay[mask.astype(bool)] + 0.45 * np.array([20, 230, 90])).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output)
    return {
        "status": "PIPELINE_COMPLETED", "board_detection_candidate_count": len(candidates),
        "frame_status": frame_status, "original_image_size_px": list(original_size),
        "processed_image_size_px": list(image.size), "inference_scale_xy": [image.width/original_size[0], image.height/original_size[1]],
        "visible_copper_like_pixels": int(mask.sum()), "candidate_fraction": float(mask.mean()),
        "trace_component_count": len(traces), "pad_via_candidate_count": len(pads),
        "same_side_contact_candidate_count": len(graph.edges),
        "electrical_connectivity_metrics": {"status": "NOT_VERIFIED", "reason": "Observed pixel contact is not a validated pin/net correspondence."},
        "overlay": str(output), "runtime_seconds": time.perf_counter()-started,
        "segmentation_mode": "ADAPTIVE_BOARD_COLOR_EXPERIMENT" if improved else "PHASE20_1_HSV_BASELINE",
    }


def segment_adaptive_color(image: np.ndarray, *, board_side: str):
    """Lightweight color-cluster experiment for flat-rendered soldermask boards.

    The mask is still an image candidate: no net class or electrical meaning is
    assigned. This experiment is not activated in the product pipeline.
    """
    from circuitcare.vision.reconstruction.copper_segmentation import CopperSegmentation
    lab = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)
    # Use a board-color mode estimated from the central 70% of the image; other
    # regions are considered contrast candidates. Restrict candidate growth to
    # the central board field to avoid the canvas background.
    h, w = lab.shape[:2]
    x0, x1, y0, y1 = int(w*.05), int(w*.95), int(h*.05), int(h*.95)
    field = lab[y0:y1, x0:x1]
    # Quantized Lab histogram mode gives a local board substrate reference.
    quantized = (field // 8).reshape(-1, 3)
    values, counts = np.unique(quantized, axis=0, return_counts=True)
    mode = values[int(np.argmax(counts))].astype(np.float32) * 8 + 4
    distance = np.linalg.norm(lab.astype(np.float32)-mode, axis=2)
    # Candidate copper is brighter than board base but not near-white silkscreen.
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    board = (np.indices((h, w))[1] >= x0) & (np.indices((h, w))[1] < x1)
    board &= (np.indices((h, w))[0] >= y0) & (np.indices((h, w))[0] < y1)
    candidate = board & (distance >= 18) & (distance <= 105) & (gray >= np.percentile(gray[board], 55))
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(bool)
    overlay = np.asarray(image).copy()
    tint = np.zeros_like(overlay); tint[:, :, 0] = 30; tint[:, :, 1] = 220
    overlay[candidate] = (.55*overlay[candidate]+.45*tint[candidate]).astype(np.uint8)
    return CopperSegmentation(candidate, overlay, "INFERRED_COPPER_LIKE_CANDIDATES",
                              1-float(candidate.mean()), float(candidate.mean()),
                              ("Adaptive Lab candidate experiment; rendered PRISM images only.",
                               "No electrical connectivity is inferred.", f"board_side={board_side}"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/external/prism")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/benchmarks/prism")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    try:
        inventory = validate_inventory(args.dataset_root, parse_references=False)
    except (FileNotFoundError, OSError) as exc:
        parser.error(str(exc))
    if inventory["ready_samples"] != 12:
        parser.error(f"PRISM inventory invalid: {inventory['ready_samples']}/12 ready")
    samples = list(SAMPLES) if args.all else [args.sample]
    out_root = args.output_dir.expanduser().resolve()
    commit = "UNKNOWN"
    try:
        import subprocess
        commit = subprocess.run(["git", "-C", str(args.dataset_root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        pass
    if commit != PRISM_COMMIT:
        parser.error(f"Dataset commit mismatch: expected {PRISM_COMMIT}, got {commit}")
    results = []
    for sample_name in samples:
        sample = load_sample(args.dataset_root, sample_name)
        variants = [False] if args.baseline_only else [False, True]
        sample_result = {"sample_id": sample.sample_id, "dataset_hash": sample.dataset_hash,
                         "input_hashes": sample.input_hashes}
        for improved in variants:
            key = "improved" if improved else "baseline"
            sample_result[key] = {}
            folder = out_root / key / "sample_results"
            overlay_dir = out_root / key / "overlays"
            for side, path in (("TOP", sample.top_image), ("BOTTOM", sample.bottom_image)):
                try:
                    sample_result[key][side] = image_pipeline(
                        path, sample.sample_id, side, overlay_dir / f"{sample.sample_id}_{side}.png",
                        improved=improved)
                except Exception as exc:  # Preserve per-image failures and continue the inventory.
                    sample_result[key][side] = {"status": "PIPELINE_FAILED", "error": f"{type(exc).__name__}: {exc}"}
        # Parse Ground Truth only once image-only predictions are complete.
        references = load_references(args.dataset_root, sample.sample_id)
        sample_result["reference_schema_keys"] = sorted(references)
        sample_result["reference_hashes"] = hash_reference_files(args.dataset_root, sample.sample_id)
        sample_result["reference_counts"] = {
            "pads": len(references["expected_geometry"].get("pads", [])),
            "segments": len(references["expected_geometry"].get("segments", [])),
            "vias": len(references["expected_geometry"].get("vias", [])),
            "zones": len(references["expected_geometry"].get("zones", [])),
            "net_count": references["expected_stats"].get("net_count"),
        }
        results.append(sample_result)
        for variant in ("baseline", "improved"):
            if variant in sample_result:
                result_path = out_root / variant / "sample_results" / f"{sample.sample_id}.json"
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(
                    json.dumps(sample_result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
                failures = {side: payload for side, payload in sample_result[variant].items()
                            if payload.get("status") != "PIPELINE_COMPLETED"}
                error_path = out_root / variant / "error_cases" / f"{sample.sample_id}.json"
                if failures:
                    error_path.parent.mkdir(parents=True, exist_ok=True)
                    error_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False)+"\n",
                                          encoding="utf-8")
                elif error_path.exists():
                    error_path.unlink()
        print(f"{sample.sample_id}: baseline complete" + (" + adaptive experiment" if not args.baseline_only else ""), flush=True)
    summary = {
        "source": "https://github.com/KamMalinski/PRISM", "prism_commit": commit,
        "circuitcare_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                              capture_output=True, text=True).stdout.strip(),
        "python_version": platform.python_version(), "opencv_version": cv2.__version__,
        "dataset_hashes": {row["sample_id"]: row["dataset_hash"] for row in results},
        "experiment": "IMAGE_ONLY", "configuration": {"baseline": "Phase 20.1 visible copper HSV", "improved": "adaptive Lab contrast prototype", "rectification": "existing board detector candidate, if found", "max_processing_edge_px": 500},
        "alignment_status": "NOT_VERIFIED", "alignment_reason": "PRISM references are in KiCad mm; image-to-board pixel transform is not supplied or independently validated.",
        "metric_policy": "All pixel geometry and pad/via precision/recall are NOT_VERIFIED until image-to-design registration is validated.",
        "benchmark_schema": "prism-benchmark-v1", "samples_requested": len(samples), "results": results,
        "runtime_seconds": sum(side.get("runtime_seconds", 0) for row in results
                                for variant in ("baseline", "improved") for side in row.get(variant, {}).values()),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    for variant in ("baseline", "improved"):
        if any(variant in row for row in results):
            variant_summary = {**summary, "pipeline_variant": variant}
            variant_root = out_root / variant
            variant_root.mkdir(parents=True, exist_ok=True)
            (variant_root / "summary.json").write_text(
                json.dumps(variant_summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"Saved benchmark outputs to {out_root}")
    return 0 if all(any(row.get(v, {}).get(side, {}).get("status") == "PIPELINE_COMPLETED"
                        for v in ("baseline", "improved") for side in ("TOP", "BOTTOM")) for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
