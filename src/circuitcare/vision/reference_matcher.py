"""Explainable, non-probabilistic matching suggestions for PCB detections."""

from __future__ import annotations

import math
import re
from typing import Any


def suggest_reference_matches(detections: list[dict[str, Any]], ocr_records: list[dict[str, Any]],
                              registered_positions: list[dict[str, Any]], *,
                              image_size: tuple[int, int]) -> dict[str, Any]:
    """Rank references from location, literal OCR, and class compatibility evidence.

    Scores are heuristic ordering indices, not probabilities. No match is confirmed here.
    """
    width, height = image_size
    refs = {row["reference"]: row for row in registered_positions}
    pair_rows: list[dict[str, Any]] = []
    for detection in detections:
        box = [float(value) for value in detection["bbox_xyxy"]]
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        box_diagonal = math.hypot(box[2] - box[0], box[3] - box[1])
        for reference, position in refs.items():
            x, y = (float(value) for value in position["image_position_px"])
            if not (0 <= x < width and 0 <= y < height):
                continue
            gap_x = max(box[0] - x, 0, x - box[2])
            gap_y = max(box[1] - y, 0, y - box[3])
            distance = math.hypot(gap_x, gap_y)
            max_gap = max(24.0, box_diagonal * 0.8)
            location_score = max(0.0, 1.0 - distance / max_gap)
            exact_ocr = []
            for record in ocr_records:
                ocr_box = record.get("bbox_xyxy")
                if not ocr_box:
                    continue
                ocr_center = ((float(ocr_box[0]) + float(ocr_box[2])) / 2,
                              (float(ocr_box[1]) + float(ocr_box[3])) / 2)
                adjacent = math.hypot(ocr_center[0] - center[0], ocr_center[1] - center[1]) <= max(48.0, box_diagonal * 2.5)
                candidates = {normalize_reference(item) for item in record.get("reference_candidates", [])}
                if adjacent and normalize_reference(reference) in candidates:
                    exact_ocr.append(record)
            ocr_score = max((float(record.get("confidence", 0.0)) for record in exact_ocr), default=0.0)
            compatibility = component_type_compatibility(detection.get("class_name", ""), reference)
            score = 100 * (0.55 * location_score + 0.35 * ocr_score + 0.10 * compatibility)
            if score > 0:
                pair_rows.append({"detection_id": detection["detection_id"], "reference": reference,
                                  "heuristic_score_0_100": round(score, 2), "position_score_0_1": round(location_score, 4),
                                  "ocr_score_0_1": round(ocr_score, 4), "type_compatibility_0_1": compatibility,
                                  "center_gap_px": round(distance, 2),
                                  "matched_ocr_text": [record["raw_text"] for record in exact_ocr],
                                  "status": "suggested"})

    by_detection: dict[str, list[dict[str, Any]]] = {}
    for row in pair_rows:
        by_detection.setdefault(row["detection_id"], []).append(row)
    for rows in by_detection.values():
        rows.sort(key=lambda item: item["heuristic_score_0_100"], reverse=True)
    best_for_ref: dict[str, list[dict[str, Any]]] = {}
    for rows in by_detection.values():
        if rows:
            best_for_ref.setdefault(rows[0]["reference"], []).append(rows[0])
    conflicted_refs = set()
    for reference, rows in best_for_ref.items():
        rows.sort(key=lambda item: item["heuristic_score_0_100"], reverse=True)
        if len(rows) > 1:
            conflicted_refs.add(reference)
            if rows[0]["heuristic_score_0_100"] - rows[1]["heuristic_score_0_100"] < 8:
                for row in rows:
                    row["status"] = "ambiguous"
            else:
                for row in rows[1:]:
                    row["status"] = "ambiguous"
    for rows in by_detection.values():
        if not rows:
            continue
        top = rows[0]
        close_second = len(rows) > 1 and top["heuristic_score_0_100"] - rows[1]["heuristic_score_0_100"] < 8
        if close_second or top["reference"] in conflicted_refs:
            top["status"] = "ambiguous"
    return {"score_kind": "heuristic_ordering_index_not_probability", "assignments": by_detection,
            "candidate_pair_count": len(pair_rows), "status": "suggestions_only"}


def normalize_reference(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def component_type_compatibility(class_name: str, reference: str) -> float:
    ref = normalize_reference(reference)
    prefix = "".join(character for character in ref if character.isalpha())
    expected = {"R": "resistor", "C": "capacitor", "U": "integrated_circuit",
                "D": "diode", "J": "connector", "CN": "connector", "P": "connector"}
    return 1.0 if expected.get(prefix) == class_name else 0.0
