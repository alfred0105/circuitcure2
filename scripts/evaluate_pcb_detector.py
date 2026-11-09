#!/usr/bin/env python3
"""Evaluate PCB detector predictions using a provenance manifest or legacy YOLO paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, __version__ as PILLOW_VERSION

from circuitcare.vision.component_detector import detect_components, detect_components_tiled
from circuitcare.vision.detection_evaluation import evaluate_detections, validate_annotation_manifest
from circuitcare.vision.detection_postprocess import PCB_MODEL_CLASSES
from circuitcare.vision.detector_backend import model_file_info


def load_manifest_inputs(manifest_path: Path, *, split: str | None = None
                         ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load images and GT only from explicit manifest records; never infer provenance from paths."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = validate_annotation_manifest(manifest, class_names=list(PCB_MODEL_CLASSES))
    images = manifest.get("images", [])
    available_splits = sorted({str(row.get("split")) for row in images if row.get("split")})
    requested_split = split or manifest.get("split_name")
    if requested_split not in available_splits:
        raise ValueError(f"Requested split {requested_split!r} is absent; available splits: {available_splits}")
    records = []
    class_to_id = {name: index for index, name in enumerate(PCB_MODEL_CLASSES)}
    for row in images:
        if row.get("split") != requested_split:
            continue
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = (manifest_path.parent / image_path).resolve()
        image_bytes = image_path.read_bytes()
        image_sha = hashlib.sha256(image_bytes).hexdigest()
        if row.get("image_sha256") and image_sha.lower() != str(row["image_sha256"]).lower():
            raise ValueError(f"Image hash mismatch for {row['image_id']}: {image_path}")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        if list(image.size) != list(row["image_size_px"]):
            raise ValueError(f"Image dimensions mismatch for {row['image_id']}: manifest {row['image_size_px']}, file {image.size}")
        annotations = []
        for index, annotation in enumerate(row.get("annotations", [])):
            class_name = annotation.get("class")
            if class_name not in class_to_id:
                raise ValueError(f"Unsupported class {class_name!r} in image {row['image_id']}")
            if annotation.get("annotation_status") == "UNRESOLVED":
                continue
            annotations.append({"class_id": class_to_id[class_name],
                                "bbox_xyxy": annotation["bbox_xyxy"],
                                "annotation_id": annotation.get("component_id", f"annotation:{index + 1:04d}"),
                                "annotation_status": annotation.get("annotation_status")})
        label_sha = row.get("label_sha256")
        label_path = row.get("label_path")
        if label_path:
            label_file = Path(label_path)
            if not label_file.is_absolute():
                label_file = (manifest_path.parent / label_file).resolve()
            if label_file.is_file():
                observed_label_sha = hashlib.sha256(label_file.read_bytes()).hexdigest()
                if label_sha and observed_label_sha.lower() != str(label_sha).lower():
                    raise ValueError(f"Label hash mismatch for {row['image_id']}: {label_file}")
                label_sha = observed_label_sha
            elif label_sha:
                raise ValueError(f"Declared label file is missing for {row['image_id']}: {label_file}")
        records.append({"image_id": str(row["image_id"]), "path": image_path, "image": image,
                        "ground_truth": annotations, "image_sha256": image_sha,
                        "label_sha256": label_sha, "image_size_px": list(image.size),
                        "board_id": row.get("board_id"), "board_side": row.get("board_side"),
                        "label_state": row.get("label_state")})
    if not records:
        raise ValueError(f"Split {requested_split!r} contains no scoreable image records")
    dataset = {key: manifest.get(key) for key in (
        "dataset_id", "dataset_name", "dataset_source", "dataset_source_url", "source_version",
        "dataset_license", "annotation_source", "annotation_review_status", "split_method",
        "board_grouping_policy", "split_policy")}
    dataset.update({"split": requested_split, "available_splits": available_splits,
                    "manifest_image_count": len([row for row in images if row.get("split") == requested_split]),
                    "unannotated_image_count": sum(row.get("split") == requested_split and row.get("label_state") == "UNANNOTATED" for row in images),
                    "annotation_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    "manifest_validation": validation})
    return records, dataset


def load_yolo_path_inputs(images_dir: Path, labels_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Legacy path inputs remain supported but are explicitly provenance-incomplete."""
    image_files = sorted(path for path in images_dir.iterdir()
                         if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not image_files:
        raise ValueError(f"No images found in {images_dir}")
    records = []
    for path in image_files:
        label_path = labels_dir / f"{path.stem}.txt"
        if not label_path.is_file():
            raise ValueError(f"Missing label file for {path.name}: {label_path}")
        image_bytes = path.read_bytes()
        label_bytes = label_path.read_bytes()
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        ground_truth = _read_yolo_labels(label_bytes.decode("utf-8"), width, height, source=label_path)
        records.append({"image_id": path.name, "path": path, "image": image,
                        "ground_truth": ground_truth,
                        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                        "label_sha256": hashlib.sha256(label_bytes).hexdigest(),
                        "image_size_px": [width, height], "board_id": "BOARD_ID_UNRESOLVED",
                        "board_side": "UNKNOWN", "label_state": "UNANNOTATED" if not ground_truth else "LABELED"})
    return records, {"dataset_id": None, "dataset_name": None, "dataset_source": None,
        "dataset_source_url": None, "source_version": None, "dataset_license": None,
        "annotation_source": None, "annotation_review_status": None, "split": "UNSPECIFIED",
        "split_method": None, "board_grouping_policy": None, "split_policy": None,
        "manifest_validation": None, "metadata_status": "PROVENANCE_INCOMPLETE"}


def _read_yolo_labels(text: str, width: int, height: int, *, source: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label at {source}:{line_number}")
        try:
            class_id, cx, cy, box_w, box_h = int(parts[0]), *map(float, parts[1:])
        except ValueError as exc:
            raise ValueError(f"Invalid numeric YOLO label at {source}:{line_number}") from exc
        if not 0 <= class_id < len(PCB_MODEL_CLASSES):
            raise ValueError(f"Class index outside model taxonomy at {source}:{line_number}")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < box_w <= 1 and 0 < box_h <= 1):
            raise ValueError(f"Normalized box outside image bounds at {source}:{line_number}")
        x1, y1, x2, y2 = cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"Bounding box corners outside image bounds at {source}:{line_number}")
        rows.append({"class_id": class_id, "bbox_xyxy": [x1 * width, y1 * height, x2 * width, y2 * height]})
    return rows


def _repository_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"commit_sha": commit, "working_tree_dirty": dirty}


def _runtime_versions() -> dict[str, Any]:
    import cv2
    import numpy
    try:
        import onnxruntime
        ort_version = onnxruntime.__version__
    except ImportError:
        ort_version = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "opencv": cv2.__version__, "numpy": numpy.__version__, "onnxruntime": ort_version,
            "pillow": PILLOW_VERSION}


def _run(records: list[dict[str, Any]], dataset: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    model = model_file_info(args.model)
    if model["status"] != "READY":
        raise ValueError(f"Model unavailable: {model['path']}")
    thresholds = args.confidence_sweep
    inference_threshold = min([0.001, args.confidence, *thresholds])
    infer = detect_components if args.mode == "single" else detect_components_tiled
    infer_options = {"confidence_threshold": inference_threshold, "iou_threshold": args.nms_iou}
    if args.mode == "tiled":
        infer_options.update(tile_size_px=args.tile_size, tile_overlap_px=args.tile_overlap)
    warmup_start = time.perf_counter()
    warmup = infer(records[0]["image"], args.model, **infer_options)
    warmup_ms = (time.perf_counter() - warmup_start) * 1000
    predictions, ground_truth, latencies, fingerprints = {}, {}, [], {}
    model_info = dict(warmup.get("model") or model)
    resolved_model = args.model.expanduser().resolve()
    model_info["repository_relative_path"] = (resolved_model.relative_to(ROOT).as_posix()
        if resolved_model.is_relative_to(ROOT) else None)
    model_info["path"] = args.model.name
    for record in records:
        name = record["image_id"]
        ground_truth[name] = record["ground_truth"]
        started = time.perf_counter()
        result = infer(record["image"], args.model, **infer_options)
        latencies.append((time.perf_counter() - started) * 1000)
        predictions[name] = result.get("detections", [])
        fingerprints[name] = {key: record[key] for key in (
            "image_sha256", "label_sha256", "image_size_px", "board_id", "board_side", "label_state")}
        fingerprints[name]["ground_truth_count"] = len(record["ground_truth"])
        fingerprints[name]["prediction_count"] = len(predictions[name])
    per_image = {record["image_id"]: evaluate_detections(
        {record["image_id"]: ground_truth[record["image_id"]]},
        {record["image_id"]: predictions[record["image_id"]]}, class_names=list(PCB_MODEL_CLASSES),
        confidence_threshold=args.confidence, confidence_thresholds=[args.confidence]) for record in records}
    metrics = evaluate_detections(ground_truth, predictions, class_names=list(PCB_MODEL_CLASSES),
        confidence_threshold=args.confidence, confidence_thresholds=thresholds)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)
    manifest_validation = dataset.get("manifest_validation")
    if not manifest_validation:
        evaluation_status = "PROVENANCE_INCOMPLETE"
    elif manifest_validation.get("errors"):
        evaluation_status = "BLOCKED_INVALID_MANIFEST"
    elif dataset.get("annotation_review_status") == "SOURCE_PROVIDED_UNADJUDICATED":
        evaluation_status = "SOURCE_LABEL_COMPARISON"
    elif manifest_validation.get("scoreable_status") == "READY_REVIEWED_LABELS":
        evaluation_status = "SEALED_BOARD_LEVEL_TEST" if dataset["split"] == "sealed_test" else "REVIEWED_VALIDATION"
    else:
        evaluation_status = manifest_validation.get("scoreable_status", "PROVENANCE_INCOMPLETE")
    return {"schema_version": "circuitcare.pcb-detector-evaluation.v3",
        "evaluation_status": evaluation_status, "dataset": {**dataset,
            "image_count": len(records), "labeled_object_count": sum(map(len, ground_truth.values())),
            "board_id_status": "RESOLVED" if all(record.get("board_id") not in {None, "BOARD_ID_UNRESOLVED"} for record in records) else "BOARD_ID_UNRESOLVED",
            "input_fingerprints": fingerprints},
        "model": model_info, "runtime": "onnxruntime-cpu", "runtime_versions": _runtime_versions(),
        "repository": _repository_state(), "inference_mode": args.mode,
        "inference_configuration": {"model_input_size_px": 960,
            "nms_iou_threshold": args.nms_iou, "metric_iou_threshold": 0.5,
            "tile_size_px": args.tile_size if args.mode == "tiled" else None,
            "tile_overlap_px": args.tile_overlap if args.mode == "tiled" else None,
            "raw_prediction_confidence_floor": inference_threshold,
            "reported_confidence_threshold": args.confidence,
            "confidence_threshold_sweep": thresholds},
        "first_call_model_load_and_inference_ms": warmup_ms,
        "steady_per_image_latency_ms": {"mean": statistics.mean(latencies),
            "median": statistics.median(latencies), "p95": float(__import__("numpy").percentile(latencies, 95)),
            "image_count": len(latencies)},
        "peak_process_rss_bytes": peak_rss_bytes,
        "peak_memory_note": "Process peak RSS includes Python and imported libraries; it is not isolated to ONNX inference.",
        "metrics": metrics, "per_image_metrics": per_image,
        "ground_truth": ground_truth, "predictions": predictions,
        "limits": ["Source-provided or unresolved annotations do not establish reviewed ground truth.",
            "Image-split metrics are not board-independent without resolved board grouping and a board-held-out split.",
            "Board side and physical component identity are not inferred from filenames or pixels."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--split", help="manifest split to score; defaults to manifest split_name")
    parser.add_argument("--images", type=Path, help="legacy YOLO image directory (requires --labels)")
    parser.add_argument("--labels", type=Path, help="legacy YOLO labels directory (requires --images)")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "pcb_component_detector.onnx")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--mode", choices=("single", "tiled"), default="single")
    parser.add_argument("--tile-size", type=int, default=1536)
    parser.add_argument("--tile-overlap", type=int, default=320)
    parser.add_argument("--confidence-sweep", type=float, nargs="*",
                        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60])
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1 or not 0 < args.nms_iou <= 1 or any(not 0 <= value <= 1 for value in args.confidence_sweep):
        parser.error("confidence and IoU thresholds must be in [0, 1], with NMS IoU > 0")
    if args.dataset_manifest:
        if args.images or args.labels:
            parser.error("Use --dataset-manifest or the legacy --images/--labels pair, not both")
        try:
            records, dataset = load_manifest_inputs(args.dataset_manifest, split=args.split)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f"Could not load evaluation manifest: {exc}")
    else:
        if args.images is None or args.labels is None or args.split:
            parser.error("Provide --dataset-manifest or both --images and --labels")
        try:
            records, dataset = load_yolo_path_inputs(args.images, args.labels)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    try:
        output = _run(records, dataset, args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items()
                      if key not in {"predictions", "per_image_metrics"}}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
