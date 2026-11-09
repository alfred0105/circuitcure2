#!/usr/bin/env python3
"""Create a per-image FP/FN/class-confusion JSON and marked-up review set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from PIL import Image, ImageDraw
from circuitcare.vision.detection_evaluation import box_iou_xyxy
from circuitcare.vision.detection_postprocess import PCB_MODEL_CLASSES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True, help="full evaluator JSON including boxes")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    report = json.loads(args.evaluation.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    images = {row["image_id"]: row for row in manifest.get("images", [])}
    gt_by_image, pred_by_image = report.get("ground_truth"), report.get("predictions")
    if not isinstance(gt_by_image, dict) or not isinstance(pred_by_image, dict):
        parser.error("Evaluation JSON must retain ground_truth and predictions")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previews = args.output_dir / "images"
    previews.mkdir(exist_ok=True)
    failures = []
    for image_id in sorted(set(gt_by_image) | set(pred_by_image)):
        truth = gt_by_image.get(image_id, [])
        predictions = [row for row in pred_by_image.get(image_id, [])
                       if float(row.get("confidence", 0)) >= args.confidence]
        matched_truth: set[int] = set()
        matched_predictions: set[int] = set()
        order = sorted(range(len(predictions)), key=lambda index: (-float(predictions[index]["confidence"]),
            str(predictions[index].get("detection_id", "")), tuple(predictions[index]["bbox_xyxy"])))
        for prediction_index in order:
            prediction = predictions[prediction_index]
            predicted_class = prediction.get("model_class_candidate", prediction.get("predicted_class", prediction.get("class_name")))
            candidates = [(index, row) for index, row in enumerate(truth) if index not in matched_truth
                          and int(row["class_id"]) == PCB_MODEL_CLASSES.index(predicted_class)
                          and box_iou_xyxy(prediction["bbox_xyxy"], row["bbox_xyxy"]) >= args.iou]
            if candidates:
                index, _ = max(candidates, key=lambda item: box_iou_xyxy(prediction["bbox_xyxy"], item[1]["bbox_xyxy"]))
                matched_truth.add(index); matched_predictions.add(prediction_index)
        image_failures = []
        for index, row in enumerate(truth):
            if index in matched_truth:
                continue
            best = max(((box_iou_xyxy(row["bbox_xyxy"], pred["bbox_xyxy"]), pred)
                        for pred in predictions if pred.get("model_class_candidate", pred.get("predicted_class", pred.get("class_name")))
                        != PCB_MODEL_CLASSES[int(row["class_id"])]), key=lambda item: item[0], default=(0.0, None))
            image_failures.append({"error_type": "FALSE_NEGATIVE",
                "overlapping_wrong_class_detection": best[0] >= args.iou,
                "suggested_error_type": "CLASS_CONFUSION" if best[0] >= args.iou else "MISSED_COMPONENT",
                "class": PCB_MODEL_CLASSES[int(row["class_id"])], "bbox_xyxy": row["bbox_xyxy"],
                "annotation_id": row.get("annotation_id"), "best_wrong_class_iou": best[0]})
        for index, prediction in enumerate(predictions):
            if index in matched_predictions:
                continue
            class_name = prediction.get("model_class_candidate", prediction.get("predicted_class", prediction.get("class_name")))
            overlaps = max((box_iou_xyxy(prediction["bbox_xyxy"], row["bbox_xyxy"]) for row in truth), default=0.0)
            image_failures.append({"error_type": "FALSE_POSITIVE",
                "suggested_error_type": "CLASS_CONFUSION" if overlaps >= args.iou else "UNMATCHED_DETECTION",
                "class": class_name, "bbox_xyxy": prediction["bbox_xyxy"],
                "confidence": prediction.get("confidence"), "detection_id": prediction.get("detection_id"),
                "best_ground_truth_iou": overlaps})
        if not image_failures:
            continue
        image_row = images.get(image_id, {})
        failures.extend({"image_id": image_id, "board_id": image_row.get("board_id", "BOARD_ID_UNRESOLVED"),
                         "board_side": image_row.get("board_side", "UNKNOWN"), **failure}
                        for failure in image_failures)
        image_path = Path(image_row.get("image_path", ""))
        if not image_path.is_absolute():
            image_path = (args.manifest.parent / image_path).resolve()
        with Image.open(image_path) as opened:
            canvas = opened.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for row in truth:
            x1, y1, x2, y2 = map(float, row["bbox_xyxy"])
            draw.rectangle((x1, y1, x2, y2), outline="#21a675", width=3)
        for prediction in predictions:
            x1, y1, x2, y2 = map(float, prediction["bbox_xyxy"])
            draw.rectangle((x1, y1, x2, y2), outline="#e74949", width=3)
        canvas.save(previews / f"{Path(image_id).stem}_failures.jpg", quality=90)
    output = {"schema_version": "circuitcare.detection-failure-report.v1",
        "evaluation_status": report.get("evaluation_status"), "dataset": report.get("dataset"),
        "confidence_threshold": args.confidence, "iou_threshold": args.iou,
        "failure_count": len(failures), "failures": failures,
        "warning": "Error categories are geometric diagnostics against supplied labels; labels are not upgraded to reviewed ground truth."}
    (args.output_dir / "failure_report.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "failure_count": len(failures),
        "error_types": {kind: sum(row["error_type"] == kind for row in failures) for kind in {row["error_type"] for row in failures}}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
