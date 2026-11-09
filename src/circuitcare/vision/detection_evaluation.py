"""Deterministic detection metrics and board-aware annotation manifest checks."""

from __future__ import annotations

from collections import defaultdict
import math
import re
from typing import Any

import numpy as np

ANNOTATION_STATUSES = {"REVIEWED_GROUND_TRUTH", "DRAFT_ANNOTATION", "MODEL_PSEUDO_LABEL", "UNRESOLVED"}
DATASET_SPLITS = {"development", "training", "validation", "sealed_test"}
MIN_CLASS_SUPPORT_FOR_REPORTING = 10
UNRESOLVED_BOARD_IDS = {"BOARD_ID_UNRESOLVED", ""}


def box_iou_xyxy(first: list[float] | np.ndarray, second: list[float] | np.ndarray) -> float:
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(intersection / max(area_a + area_b - intersection, 1e-12))


def validate_board_splits(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Reject a board or identical image hash assigned to multiple splits."""
    errors = []
    board_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    hash_images: dict[str, list[tuple[str, str]]] = defaultdict(list)
    image_ids: set[str] = set()
    unresolved_images = 0
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            errors.append({"index": index, "code": "INVALID_SPLIT_RECORD"})
            continue
        image_id, board_id, split = row.get("image_id"), row.get("board_id"), row.get("split")
        if (not isinstance(image_id, str) or not image_id.strip()
                or not isinstance(split, str) or split not in DATASET_SPLITS):
            errors.append({"index": index, "code": "MISSING_ID_OR_INVALID_SPLIT"})
            continue
        image_id = str(image_id)
        if image_id in image_ids:
            errors.append({"index": index, "code": "DUPLICATE_IMAGE_ID", "image_id": image_id})
        image_ids.add(image_id)
        if board_id is None or str(board_id) in UNRESOLVED_BOARD_IDS:
            unresolved_images += 1
        else:
            board_splits[str(board_id)].add(str(split))
        digest = row.get("image_sha256")
        if digest:
            hash_splits[str(digest)].add(str(split))
            hash_images[str(digest)].append((image_id, str(split)))
    for board_id, splits in board_splits.items():
        if len(splits) > 1:
            errors.append({"code": "BOARD_SPLIT_LEAKAGE", "board_id": board_id, "splits": sorted(splits)})
    for digest, splits in hash_splits.items():
        if len(splits) > 1:
            errors.append({"code": "IMAGE_HASH_SPLIT_LEAKAGE", "image_sha256": digest, "splits": sorted(splits)})
    for digest, copies in hash_images.items():
        if len(copies) > 1:
            errors.append({"code": "DUPLICATE_IMAGE_HASH", "image_sha256": digest,
                           "images": [{"image_id": image_id, "split": split} for image_id, split in copies]})
    return {"status": ("LEAKAGE_OR_MANIFEST_ERROR" if errors else
                        "BLOCKED_BOARD_ID_UNRESOLVED" if unresolved_images else "VALID"),
            "image_count": len(image_ids), "board_count": len(board_splits),
            "unresolved_board_image_count": unresolved_images, "errors": errors}


def validate_annotation_manifest(manifest: dict[str, Any], *, class_names: list[str]) -> dict[str, Any]:
    """Validate annotation provenance, coordinate bounds, and board grouping."""
    errors = []
    images = manifest.get("images") if isinstance(manifest, dict) else None
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != "circuitcare.component-annotation.v1"
            or not isinstance(images, list)):
        return {"status": "INVALID_MANIFEST", "errors": [{"code": "UNSUPPORTED_SCHEMA_OR_IMAGES"}]}
    split_rows = []
    counts = defaultdict(int)
    reviewed = 0
    reviewed_empty = 0
    required_metadata = ("dataset_id", "dataset_name", "dataset_source", "dataset_source_url",
        "source_version", "dataset_license", "annotation_source", "annotation_review_status",
        "split_name", "split_method", "board_grouping_policy", "split_policy")
    for field in required_metadata:
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            errors.append({"code": "MISSING_DATASET_METADATA", "field": field})
    if manifest.get("annotation_review_status") not in {
            "SOURCE_PROVIDED_UNADJUDICATED", "HUMAN_REVIEWED", "MIXED_REVIEW", "MODEL_PSEUDO_LABEL"}:
        errors.append({"code": "INVALID_DATASET_ANNOTATION_REVIEW_STATUS"})
    for image_index, image in enumerate(images):
        if not isinstance(image, dict):
            errors.append({"image_index": image_index, "code": "INVALID_IMAGE_RECORD"})
            continue
        required = ("image_id", "board_id", "board_revision", "board_side", "split", "image_sha256",
                    "image_size_px", "image_path", "label_state")
        missing = [key for key in required if image.get(key) is None]
        if missing:
            errors.append({"image_index": image_index, "code": "MISSING_IMAGE_PROVENANCE", "fields": missing})
            continue
        size = image.get("image_size_px")
        try:
            width, height = map(int, size) if isinstance(size, (list, tuple)) and len(size) == 2 else (0, 0)
        except (TypeError, ValueError):
            width, height = 0, 0
        if (not isinstance(image["board_side"], str)
                or image["board_side"] not in {"TOP", "BOTTOM", "UNKNOWN"}
                or width <= 0 or height <= 0):
            errors.append({"image_id": image["image_id"], "code": "INVALID_SIDE_OR_IMAGE_SIZE"})
        if not isinstance(image.get("image_sha256"), str) or not re.fullmatch(r"[a-fA-F0-9]{64}", image["image_sha256"]):
            errors.append({"image_id": image["image_id"], "code": "INVALID_IMAGE_SHA256"})
        if (not isinstance(image["label_state"], str)
                or image["label_state"] not in {"LABELED", "REVIEWED_EMPTY", "UNANNOTATED"}):
            errors.append({"image_id": image["image_id"], "code": "INVALID_LABEL_STATE"})
        if not isinstance(image.get("image_path"), str) or not image["image_path"].strip():
            errors.append({"image_id": image["image_id"], "code": "MISSING_IMAGE_PATH"})
        if image.get("label_state") == "REVIEWED_EMPTY":
            reviewed_empty += 1
            review = image.get("review_provenance")
            if (not isinstance(image.get("reviewer"), str) or not image["reviewer"].strip()
                    or not isinstance(review, dict) or not review.get("reviewed_at")
                    or not review.get("review_method")):
                errors.append({"image_id": image["image_id"], "code": "REVIEWED_EMPTY_WITHOUT_REVIEW_PROVENANCE"})
        split_rows.append({"image_id": image["image_id"], "board_id": image["board_id"],
                           "split": image["split"], "image_sha256": image["image_sha256"]})
        annotations = image.get("annotations")
        if not isinstance(annotations, list):
            errors.append({"image_id": image["image_id"], "code": "INVALID_ANNOTATION_LIST"})
            annotations = []
        if annotations and image["label_state"] != "LABELED":
            errors.append({"image_id": image["image_id"], "code": "ANNOTATIONS_CONFLICT_WITH_LABEL_STATE"})
        if not annotations and image["label_state"] == "LABELED":
            errors.append({"image_id": image["image_id"], "code": "LABELED_IMAGE_HAS_NO_ANNOTATIONS"})
        component_ids = set()
        for annotation_index, row in enumerate(annotations):
            if not isinstance(row, dict):
                errors.append({"image_id": image["image_id"], "annotation_index": annotation_index,
                               "code": "INVALID_ANNOTATION_RECORD"})
                continue
            location = {"image_id": image["image_id"], "annotation_index": annotation_index}
            for key in ("component_id", "class", "bbox_xyxy", "occlusion_state",
                        "annotation_status", "source"):
                if key not in row or row.get(key) is None:
                    errors.append({**location, "code": "MISSING_ANNOTATION_FIELD", "field": key})
            if "reviewer" not in row:
                errors.append({**location, "code": "MISSING_ANNOTATION_FIELD", "field": "reviewer"})
            if "small_component" not in row:
                errors.append({**location, "code": "MISSING_ANNOTATION_FIELD", "field": "small_component"})
            elif row["small_component"] is not None and not isinstance(row["small_component"], bool):
                errors.append({**location, "code": "INVALID_SMALL_COMPONENT_FLAG"})
            if row.get("class") not in class_names:
                errors.append({**location, "code": "UNSUPPORTED_CLASS", "class": row.get("class")})
            status = row.get("annotation_status")
            if not isinstance(status, str) or status not in ANNOTATION_STATUSES:
                errors.append({**location, "code": "INVALID_ANNOTATION_STATUS", "status": status})
            counts[str(status)] += 1
            if status == "REVIEWED_GROUND_TRUTH":
                reviewed += 1
                provenance = row.get("provenance")
                if (not isinstance(row.get("reviewer"), str) or not row["reviewer"].strip()
                        or not isinstance(provenance, dict) or not provenance.get("reviewed_at")
                        or not provenance.get("review_method")):
                    errors.append({**location, "code": "REVIEWED_LABEL_WITHOUT_REVIEW_PROVENANCE"})
                if row.get("small_component") is None:
                    errors.append({**location, "code": "REVIEWED_LABEL_MISSING_SMALL_COMPONENT_REVIEW"})
            component_id = row.get("component_id")
            if isinstance(component_id, str) and component_id and component_id in component_ids:
                errors.append({**location, "code": "DUPLICATE_COMPONENT_ID_IN_IMAGE", "component_id": component_id})
            if isinstance(component_id, str) and component_id:
                component_ids.add(component_id)
            box = row.get("bbox_xyxy")
            try:
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    raise ValueError("bbox must contain four coordinates")
                x1, y1, x2, y2 = map(float, box)
                valid_box = all(math.isfinite(value) for value in (x1, y1, x2, y2)) and 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
            except (TypeError, ValueError):
                valid_box = False
            if not valid_box:
                errors.append({**location, "code": "INVALID_BOUNDING_BOX"})
    split_check = validate_board_splits(split_rows)
    errors.extend(split_check["errors"])
    unresolved_board_ids = split_check["unresolved_board_image_count"] > 0
    all_annotations_reviewed = reviewed == sum(counts.values())
    has_reviewed_truth = reviewed > 0 or reviewed_empty > 0
    scoreable_status = ("BLOCKED_INVALID_MANIFEST" if errors else
        "BLOCKED_BOARD_ID_UNRESOLVED" if unresolved_board_ids else
        "READY_REVIEWED_LABELS" if all_annotations_reviewed and has_reviewed_truth else
        "BLOCKED_UNREVIEWED_ANNOTATIONS" if reviewed else
        "BLOCKED_NO_REVIEWED_GROUND_TRUTH")
    return {"status": "VALID" if not errors else "INVALID_OR_SPLIT_LEAKAGE",
            "image_count": len(images), "reviewed_ground_truth_count": reviewed,
            "reviewed_empty_image_count": reviewed_empty,
            "annotation_status_counts": dict(counts), "board_split": split_check, "errors": errors,
            "scoreable_status": scoreable_status}


def evaluate_detections(ground_truth_by_image: dict[str, list[dict[str, Any]]],
                        predictions_by_image: dict[str, list[dict[str, Any]]], *,
                        class_names: list[str], iou_threshold: float = 0.5,
                        confidence_threshold: float = 0.25,
                        confidence_thresholds: list[float] | None = None,
                        small_object_area_px2: float = 32.0 * 32.0) -> dict[str, Any]:
    """Class-aware, one-to-one detection metrics at fixed IoU and confidence cuts."""
    _validate_evaluation_inputs(ground_truth_by_image, predictions_by_image, class_names,
                               iou_threshold, confidence_threshold, confidence_thresholds,
                               small_object_area_px2)
    per_class: dict[str, Any] = {}
    iou_sweep = [round(value, 2) for value in np.arange(0.5, 0.951, 0.05)]
    ap_by_iou: dict[float, list[float]] = defaultdict(list)
    for class_id, class_name in enumerate(class_names):
        truths = [row for rows in ground_truth_by_image.values() for row in rows if int(row["class_id"]) == class_id]
        gt_count = len(truths)
        predictions = _class_predictions(predictions_by_image, class_name)
        for threshold in iou_sweep:
            ap = _average_precision_for_iou(ground_truth_by_image, predictions, class_id, threshold)
            if gt_count:
                ap_by_iou[threshold].append(ap)
        tp, fp = _count_at_threshold(ground_truth_by_image, predictions_by_image, class_id,
                                     class_name, confidence_threshold, iou_threshold)
        small_gt = [row for row in truths if _box_area(row["bbox_xyxy"]) <= small_object_area_px2]
        small_tp = _count_small_true_positives(ground_truth_by_image, predictions_by_image, class_id,
            class_name, confidence_threshold, iou_threshold, small_object_area_px2)
        per_class[class_name] = {"ground_truth_count": gt_count,
            "support_status": "INSUFFICIENT_SUPPORT" if gt_count < MIN_CLASS_SUPPORT_FOR_REPORTING else "MEASURED",
            "minimum_support_for_reporting": MIN_CLASS_SUPPORT_FOR_REPORTING,
            "prediction_count": len(predictions),
            "true_positive_count": tp, "false_positive_count": fp, "false_negative_count": gt_count - tp,
            "precision_at_confidence_threshold": tp / max(tp + fp, 1) if gt_count or predictions else None,
            "recall_at_confidence_threshold": tp / gt_count if gt_count else None,
            "ap50": _average_precision_for_iou(ground_truth_by_image, predictions, class_id, iou_threshold)
                if gt_count else None,
            "small_object_ground_truth_count": len(small_gt),
            "small_object_recall_at_confidence_threshold": small_tp / len(small_gt) if small_gt else None}
    valid_ap50 = [row["ap50"] for row in per_class.values() if row["ap50"] is not None]
    map_by_iou = {f"{threshold:.2f}": float(np.mean(values)) if values else None
                  for threshold, values in ap_by_iou.items()}
    sweep_thresholds = confidence_thresholds or [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
    threshold_curve = []
    for threshold in sweep_thresholds:
        tp, fp, gt = _aggregate_counts(ground_truth_by_image, predictions_by_image,
                                       class_names, threshold, iou_threshold)
        threshold_curve.append({"confidence_threshold": threshold, "true_positives": tp,
            "false_positives": fp, "false_negatives": gt - tp,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(gt, 1),
            "f1": 2 * tp / max(2 * tp + fp + gt - tp, 1)})
    total_tp, total_fp, total_gt = _aggregate_counts(ground_truth_by_image, predictions_by_image,
        class_names, confidence_threshold, iou_threshold)
    return {"images_evaluated": len(set(ground_truth_by_image) | set(predictions_by_image)), "iou_threshold": iou_threshold,
        "confidence_threshold_for_precision_recall": confidence_threshold,
        "precision": total_tp / max(total_tp + total_fp, 1), "recall": total_tp / max(total_gt, 1),
        "f1": 2 * total_tp / max(2 * total_tp + total_fp + total_gt - total_tp, 1),
        "true_positive_count": total_tp, "false_positive_count": total_fp,
        "false_negative_count": total_gt - total_tp,
        "map50": float(np.mean(valid_ap50)) if valid_ap50 else None,
        "map50_95": float(np.mean([value for value in map_by_iou.values() if value is not None]))
            if any(value is not None for value in map_by_iou.values()) else None,
        "map_by_iou_threshold": map_by_iou, "confidence_threshold_curve": threshold_curve,
        "small_object_area_px2_cutoff": small_object_area_px2,
        "classes": per_class,
        "note": "Scores apply only to supplied annotations/split. They are not board-independent without verified board-group provenance."}


def _class_predictions(predictions_by_image, class_name):
    rows = []
    for image_id, predictions in predictions_by_image.items():
        for row in predictions:
            predicted_class = _prediction_class(row)
            if predicted_class == class_name:
                rows.append((float(row["confidence"]), image_id, row["bbox_xyxy"]))
    return sorted(rows, key=lambda row: (-row[0], row[1], tuple(float(value) for value in row[2])))


def _prediction_class(row: dict[str, Any]) -> Any:
    for key in ("model_class_candidate", "predicted_class", "class_name"):
        if row.get(key) is not None:
            return row[key]
    return None


def _validate_evaluation_inputs(gt_by_image, predictions_by_image, class_names,
                                iou_threshold, confidence_threshold, confidence_thresholds,
                                small_object_area_px2) -> None:
    if not isinstance(class_names, list) or not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("class_names must be a non-empty list of unique class names")
    for name, value in (("iou_threshold", iou_threshold), ("confidence_threshold", confidence_threshold),
                        ("small_object_area_px2", small_object_area_px2)):
        if not math.isfinite(float(value)) or (name.endswith("threshold") and not 0 < value <= 1) or (
                name == "small_object_area_px2" and value <= 0):
            raise ValueError(f"{name} must be finite and within its valid range")
    thresholds = confidence_thresholds or []
    if any(not math.isfinite(float(value)) or not 0 <= value <= 1 for value in thresholds):
        raise ValueError("confidence_thresholds must contain finite values in [0, 1]")
    if not isinstance(gt_by_image, dict) or not isinstance(predictions_by_image, dict):
        raise ValueError("ground truth and predictions must be image-keyed mappings")
    class_count = len(class_names)
    for image_id, rows in gt_by_image.items():
        if not isinstance(rows, list):
            raise ValueError(f"Ground truth for {image_id} must be a list, including [] for reviewed negatives")
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or isinstance(row.get("class_id"), bool):
                raise ValueError(f"Invalid ground-truth record at {image_id}[{index}]")
            try:
                class_id = int(row["class_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid ground-truth class at {image_id}[{index}]") from exc
            if class_id < 0 or class_id >= class_count:
                raise ValueError(f"Ground-truth class outside taxonomy at {image_id}[{index}]")
            _validate_box(row.get("bbox_xyxy"), f"ground truth {image_id}[{index}]")
    known_classes = set(class_names)
    for image_id, rows in predictions_by_image.items():
        if not isinstance(rows, list):
            raise ValueError(f"Predictions for {image_id} must be a list")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"Invalid prediction record at {image_id}[{index}]")
            predicted_class = _prediction_class(row)
            if predicted_class not in known_classes:
                raise ValueError(f"Unsupported prediction class {predicted_class!r} at {image_id}[{index}]")
            score = row.get("confidence")
            if isinstance(score, bool) or not isinstance(score, (int, float, np.number)) or not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
                raise ValueError(f"Invalid prediction confidence at {image_id}[{index}]")
            _validate_box(row.get("bbox_xyxy"), f"prediction {image_id}[{index}]")


def _validate_box(box: Any, label: str) -> None:
    try:
        values = np.asarray(box, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid bounding box for {label}") from exc
    if (values.shape != (4,) or not np.isfinite(values).all()
            or not (values[0] < values[2] and values[1] < values[3])):
        raise ValueError(f"Invalid bounding box for {label}")


def _average_precision_for_iou(gt_by_image, predictions, class_id, iou_threshold):
    gt_count = sum(int(row["class_id"]) == class_id for rows in gt_by_image.values() for row in rows)
    if not gt_count:
        return 0.0
    matched: dict[str, set[int]] = defaultdict(set)
    tp, fp = [], []
    for _, image_id, box in predictions:
        candidates = [(index, gt) for index, gt in enumerate(gt_by_image.get(image_id, []))
                      if int(gt["class_id"]) == class_id and index not in matched[image_id]]
        best = max(candidates, key=lambda pair: box_iou_xyxy(box, pair[1]["bbox_xyxy"]), default=None)
        if best and box_iou_xyxy(box, best[1]["bbox_xyxy"]) >= iou_threshold:
            matched[image_id].add(best[0]); tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    cumulative_tp = np.cumsum(tp, dtype=float)
    cumulative_fp = np.cumsum(fp, dtype=float)
    return _average_precision(cumulative_tp / gt_count, cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12))


def _average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    changes = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))


def _count_at_threshold(gt_by_image, preds_by_image, class_id, class_name, confidence_threshold, iou_threshold):
    matched: dict[str, set[int]] = defaultdict(set)
    rows = sorted(((float(row["confidence"]), image_id, row["bbox_xyxy"])
                   for image_id, detections in preds_by_image.items() for row in detections
                   if _prediction_class(row) == class_name
                   and float(row["confidence"]) >= confidence_threshold),
                  key=lambda row: (-row[0], row[1], tuple(float(value) for value in row[2])))
    tp = fp = 0
    for _, image_id, box in rows:
        candidates = [(index, gt) for index, gt in enumerate(gt_by_image.get(image_id, []))
                      if int(gt["class_id"]) == class_id and index not in matched[image_id]]
        best = max(candidates, key=lambda pair: box_iou_xyxy(box, pair[1]["bbox_xyxy"]), default=None)
        if best and box_iou_xyxy(box, best[1]["bbox_xyxy"]) >= iou_threshold:
            matched[image_id].add(best[0]); tp += 1
        else:
            fp += 1
    return tp, fp


def _aggregate_counts(gt_by_image, preds_by_image, class_names, confidence_threshold, iou_threshold):
    total_tp = total_fp = total_gt = 0
    for class_id, class_name in enumerate(class_names):
        tp, fp = _count_at_threshold(gt_by_image, preds_by_image, class_id, class_name,
                                     confidence_threshold, iou_threshold)
        total_tp += tp; total_fp += fp
        total_gt += sum(int(row["class_id"]) == class_id for rows in gt_by_image.values() for row in rows)
    return total_tp, total_fp, total_gt


def _box_area(box):
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _count_small_true_positives(gt_by_image, preds_by_image, class_id, class_name,
                                confidence_threshold, iou_threshold, area_limit):
    tp = 0
    for image_id, predictions in preds_by_image.items():
        eligible = [gt for gt in gt_by_image.get(image_id, []) if int(gt["class_id"]) == class_id
                    and _box_area(gt["bbox_xyxy"]) <= area_limit]
        if not eligible:
            continue
        matches = set()
        rows = sorted((row for row in predictions if _prediction_class(row) == class_name
            and float(row["confidence"]) >= confidence_threshold),
            key=lambda row: (-float(row["confidence"]), tuple(float(value) for value in row["bbox_xyxy"])))
        for prediction in rows:
            candidates = [(index, gt) for index, gt in enumerate(eligible) if index not in matches]
            best = max(candidates, key=lambda pair: box_iou_xyxy(prediction["bbox_xyxy"], pair[1]["bbox_xyxy"]), default=None)
            if best and box_iou_xyxy(prediction["bbox_xyxy"], best[1]["bbox_xyxy"]) >= iou_threshold:
                matches.add(best[0]); tp += 1
    return tp
