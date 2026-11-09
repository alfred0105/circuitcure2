"""PCB pad candidates and conservative component-conditioned pin hypotheses."""
from __future__ import annotations

import math

import cv2
import numpy as np


def _component_body_mask(shape: tuple[int, int], detections: list[dict]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    for d in detections:
        box = d.get("bbox_xyxy")
        if not box:
            continue
        x0, y0, x1, y1 = [int(round(float(v))) for v in box]
        bw, bh = x1 - x0, y1 - y0
        inset = max(2, int(round(min(bw, bh) * 0.12)))
        if bw * bh > h * w * 0.012:
            inset = max(2, int(round(min(bw, bh) * 0.07)))
        cv2.rectangle(mask, (max(0, x0 + inset), max(0, y0 + inset)),
                      (min(w - 1, x1 - inset), min(h - 1, y1 - inset)), 255, -1)
    return mask


def detect_pad_candidates(image: np.ndarray, detections: list[dict]) -> tuple[list[dict], np.ndarray]:
    """Find reflective pad/via-like blobs with scale-aware morphology checks."""
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value, sat = hsv[:, :, 2], hsv[:, :, 1]
    bright_metal = ((value >= 155) & (sat <= 105)).astype(np.uint8) * 255
    warm_metal = ((hsv[:, :, 0] <= 30) & (sat >= 45) & (value >= 100)).astype(np.uint8) * 255
    binary = cv2.bitwise_or(bright_metal, warm_metal)
    k = max(3, int(round(min(h, w) * 0.004)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(5.0, h * w * 0.000008)
    max_area = max(min_area + 1, h * w * 0.003)
    pads: list[dict] = []
    pad_mask = np.zeros((h, w), np.uint8)
    boxes = [d.get("bbox_xyxy") for d in detections if d.get("bbox_xyxy")]
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not min_area <= area <= max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 2 or bh < 2:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 4 * math.pi * area / (perimeter * perimeter + 1e-6)
        aspect = min(bw, bh) / max(bw, bh)
        if circularity < 0.12 and aspect < 0.28:
            continue
        cx, cy = x + bw / 2, y + bh / 2
        # Avoid component-body glare; edge leads are left available as anchors.
        if any(b[0] + (b[2] - b[0]) * 0.10 < cx < b[2] - (b[2] - b[0]) * 0.10 and
               b[1] + (b[3] - b[1]) * 0.10 < cy < b[3] - (b[3] - b[1]) * 0.10 for b in boxes):
            continue
        confidence = min(0.95, 0.35 + 0.35 * min(1, circularity) + 0.25 * aspect)
        index = len(pads) + 1
        pads.append({"id": index, "xy": [round(cx, 1), round(cy, 1)],
                     "bbox_xyxy": [x, y, x + bw, y + bh], "area_px": round(area, 1),
                     "circularity": round(circularity, 3), "confidence": round(confidence, 3)})
        cv2.drawContours(pad_mask, [contour], -1, 255, -1)
    pads.sort(key=lambda p: p["confidence"], reverse=True)
    for index, pad in enumerate(pads, 1):
        pad["id"] = index
    return pads, pad_mask


def _regular_edge_rows(box: list[float], pads: list[dict]) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Find evenly spaced observed pads along the perimeter of one component box."""
    x0, y0, x1, y1 = [float(v) for v in box]
    bw, bh = x1 - x0, y1 - y0
    band = max(4.0, min(bw, bh) * 0.14)
    bounds = {
        "top": (y0, x0 - bw * .08, x1 + bw * .08),
        "bottom": (y1, x0 - bw * .08, x1 + bw * .08),
        "left": (x0, y0 - bh * .08, y1 + bh * .08),
        "right": (x1, y0 - bh * .08, y1 + bh * .08),
    }
    sides: dict[str, list[dict]] = {k: [] for k in bounds}
    for pad in pads:
        px, py = [float(v) for v in pad["xy"]]
        candidates = []
        for side, (edge, lo, hi) in bounds.items():
            along = px if side in {"top", "bottom"} else py
            across = py if side in {"top", "bottom"} else px
            if lo <= along <= hi and abs(across - edge) <= band:
                candidates.append((abs(across - edge), side))
        if candidates:
            sides[min(candidates)[1]].append(pad)

    # A row is useful only if its measured pitch is reasonably regular. This
    # deliberately rejects partial/uneven IC detections instead of filling gaps.
    regular: dict[str, list[dict]] = {}
    for side, row in sides.items():
        row.sort(key=lambda p: p["xy"][0] if side in {"top", "bottom"} else p["xy"][1])
        if len(row) < 3:
            continue
        coords = np.asarray([p["xy"][0] if side in {"top", "bottom"} else p["xy"][1] for p in row], dtype=float)
        gaps = np.diff(coords)
        mean_gap = float(np.mean(gaps))
        if mean_gap <= 1 or float(np.std(gaps) / mean_gap) > 0.40 or float(np.max(gaps) / mean_gap) > 1.8:
            continue
        regular[side] = row
    return regular, sides


def _pin_priors(detections: list[dict], pads: list[dict] | None = None) -> list[dict]:
    """Estimate fixed terminal layouts; infer variable packages only from regular observed pad rows."""
    priors: list[dict] = []
    pads = pads or []
    two_terminal = {"resistor", "capacitor", "inductor", "diode", "led", "fuse", "ferrite_bead",
                    "emi_filter", "mov", "thermistor", "buzzer", "speaker", "battery"}
    three_terminal = {"transistor", "potentiometer", "voltage_regulator"}
    variable_rows = {"ic", "connector", "display", "relay", "switch", "transformer", "oscillator"}
    opposite_sides = {frozenset(("top", "bottom")), frozenset(("left", "right"))}

    for comp_i, d in enumerate(detections):
        box = d.get("bbox_xyxy")
        if not box:
            continue
        x0, y0, x1, y1 = [float(v) for v in box]
        bw, bh = x1 - x0, y1 - y0
        if bw < 6 or bh < 6:
            continue
        label = str(d.get("class_name", "component")).lower().replace(" ", "_")
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rows, raw_rows = _regular_edge_rows(box, pads)
        raw_pin_count = d.get("pin_count")
        declared_count = int(raw_pin_count) if isinstance(raw_pin_count, (int, float)) and 0 < raw_pin_count <= 128 else None
        points: list[tuple[list[float], list[int], str, str, float]] = []

        if label in two_terminal:
            if declared_count not in (None, 2):
                continue
            if bw >= bh:
                points = [([x0, cy], [-1, 0], "left", "fixed_2_terminal", .22),
                          ([x1, cy], [1, 0], "right", "fixed_2_terminal", .22)]
            else:
                points = [([cx, y0], [0, -1], "top", "fixed_2_terminal", .22),
                          ([cx, y1], [0, 1], "bottom", "fixed_2_terminal", .22)]
        elif label in three_terminal:
            if declared_count not in (None, 3):
                continue
            # Common three-lead footprint: two pads on one long edge and one
            # centered pad opposite. These remain low-confidence estimates.
            if bw >= bh:
                points = [([x0 + bw * .30, y0], [0, -1], "top_1", "fixed_3_terminal", .20),
                          ([x0 + bw * .70, y0], [0, -1], "top_2", "fixed_3_terminal", .20),
                          ([cx, y1], [0, 1], "bottom", "fixed_3_terminal", .20)]
            else:
                points = [([x0, y0 + bh * .30], [-1, 0], "left_1", "fixed_3_terminal", .20),
                          ([x0, y0 + bh * .70], [-1, 0], "left_2", "fixed_3_terminal", .20),
                          ([x1, cy], [1, 0], "right", "fixed_3_terminal", .20)]
        elif label in variable_rows:
            # ICs and multi-contact parts have no fixed count here. Require
            # multiple evenly spaced measured rows and preserve only observed pads.
            sides = set(rows)
            chosen_sides: list[str] = []
            if label == "ic":
                active = {side for side, row in raw_rows.items() if row}
                # If any evidence appears on all four sides, require a balanced
                # four-side pattern. Never silently accept only two sides of a
                # partially observed QFP-like IC.
                if active == {"top", "bottom", "left", "right"} and len(rows) == 4 and all(len(v) >= 3 for v in rows.values()):
                    counts = [len(rows[s]) for s in ("top", "bottom", "left", "right")]
                    if max(counts) - min(counts) <= max(1, int(round(max(counts) * .20))):
                        chosen_sides = ["top", "bottom", "left", "right"]
                else:
                    for pair in opposite_sides:
                        a, b = sorted(pair)
                        if active == set(pair) and pair.issubset(sides):
                            na, nb = len(rows[a]), len(rows[b])
                            if abs(na - nb) <= max(1, int(round(max(na, nb) * .20))):
                                chosen_sides = [a, b]
                                break
            elif label == "connector" and len(rows) == 1:
                # A single-row connector is valid when its one measured row is regular.
                chosen_sides = list(rows)
            else:
                for pair in opposite_sides:
                    if pair.issubset(sides):
                        a, b = sorted(pair)
                        na, nb = len(rows[a]), len(rows[b])
                        if abs(na - nb) <= max(1, int(round(max(na, nb) * .20))):
                            chosen_sides = [a, b]
                            break
            if not chosen_sides and label in {"connector", "display"} and len(rows) == 4 and all(len(v) >= 3 for v in rows.values()):
                counts = [len(rows[s]) for s in ("top", "bottom", "left", "right")]
                if max(counts) - min(counts) <= max(1, int(round(max(counts) * .20))):
                    chosen_sides = ["top", "bottom", "left", "right"]
            if chosen_sides:
                confidence = .42 if label == "ic" else .38
                for side in chosen_sides:
                    direction = {"top": [0, -1], "bottom": [0, 1], "left": [-1, 0], "right": [1, 0]}[side]
                    for pad in rows[side]:
                        points.append((list(map(float, pad["xy"])), direction, side,
                                       "observed_regular_pad_row", min(.65, confidence + .20 * float(pad.get("confidence", 0)))))

        for point, direction, side, prior_type, confidence in points:
            priors.append({"id": len(priors) + 1, "component_index": comp_i,
                           "component": label, "xy": [round(point[0], 1), round(point[1], 1)],
                           "direction_xy": direction, "side": side,
                           "confidence": round(confidence, 3),
                           "pin_count_template": len(points) if prior_type != "observed_regular_pad_row" else None,
                           "pin_count_observed": len(points) if prior_type == "observed_regular_pad_row" else None,
                           "prior_type": prior_type})
    return priors


def run_pin_pad_pipeline(image: np.ndarray, detections: list[dict]) -> dict:
    """Return pin/pad candidates only; no trace filtering, thinning, or net inference."""
    h, w = image.shape[:2]
    pads, _ = detect_pad_candidates(image, detections)
    priors = _pin_priors(detections, pads)
    matches: list[dict] = []
    if priors and pads:
        tolerance = max(8.0, min(h, w) * .025)
        cost = np.full((len(priors), len(pads) + len(priors)), .52, dtype=np.float64)
        cost[:, :len(pads)] = 10.0
        score_map: dict[tuple[int, int], tuple[float, float]] = {}
        for i, prior in enumerate(priors):
            px, py = prior["xy"]
            for j, pad in enumerate(pads):
                qx, qy = pad["xy"]
                dx, dy = qx - px, qy - py
                distance = math.hypot(dx, dy)
                if distance > tolerance:
                    continue
                direction = prior["direction_xy"]
                direction_score = (1.0 if distance <= 1.0 else
                                   max(0.0, min(1.0, (dx * direction[0] + dy * direction[1]) / distance)))
                score = (.50 * math.exp(-distance / max(1.0, tolerance * .55)) +
                         .30 * direction_score + .20 * float(pad["confidence"]))
                if direction_score < .15:
                    continue
                cost[i, j] = 1.0 - score
                score_map[(i, j)] = (score, distance)
        for prior_i, pad_i in _hungarian_assignment(cost):
            if pad_i >= len(pads) or (prior_i, pad_i) not in score_map:
                continue
            score, distance = score_map[(prior_i, pad_i)]
            if score < .48:
                continue
            matches.append({"prior_id": priors[prior_i]["id"], "pad_id": pads[pad_i]["id"],
                            "distance_px": round(distance, 1), "match_score": round(score, 3)})

    supported = {m["prior_id"] for m in matches}
    overlay = image.copy()
    for pad in pads:
        x, y = map(lambda v: int(round(v)), pad["xy"])
        cv2.circle(overlay, (x, y), 4, (30, 235, 90), 1, cv2.LINE_AA)
    for prior in priors:
        x, y = map(lambda v: int(round(v)), prior["xy"])
        is_supported = prior["id"] in supported or prior["prior_type"] == "observed_regular_pad_row"
        color, radius = ((220, 40, 220), 4) if is_supported else ((20, 190, 255), 2)
        cv2.circle(overlay, (x, y), radius, color, -1, cv2.LINE_AA)
        if is_supported:
            cv2.circle(overlay, (x, y), 5, (15, 20, 26), 1, cv2.LINE_AA)

    variable = {"ic", "connector", "display", "relay", "switch", "transformer", "oscillator"}
    available = {p["component_index"] for p in priors}
    deferred = [{"component_index": i, "class": str(d.get("class_name", "component")),
                 "reason": "NO_EVENLY_SPACED_OBSERVED_PAD_ROWS" if str(d.get("class_name", "")).lower() == "ic"
                          else "NO_CONFIDENT_PIN_LAYOUT"}
                for i, d in enumerate(detections)
                if str(d.get("class_name", "component")).lower().replace(" ", "_") in variable and i not in available]
    class_counts: dict[str, int] = {}
    for p in priors:
        class_counts[p["component"]] = class_counts.get(p["component"], 0) + 1
    diagnostics = {"pad_candidates": len(pads), "pin_priors": len(priors),
                   "pin_prior_estimated": sum(p["prior_type"] != "observed_regular_pad_row" for p in priors),
                   "pin_prior_observed": sum(p["prior_type"] == "observed_regular_pad_row" for p in priors),
                   "pin_prior_class_counts": class_counts, "pin_pad_matches": len(matches),
                   "pin_prior_unavailable_components": deferred,
                   "note": "노랑은 저신뢰도 위치 추정, 자홍은 패드 후보와 일치하거나 고른 행으로 관측된 핀 위치입니다."}
    return {"pad_overlay": overlay, "pads": pads, "pin_priors": priors,
            "matches": matches, "diagnostics": diagnostics}


def _hungarian_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost rectangular assignment (Kuhn-Munkres potentials form)."""
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        return []
    if matrix.shape[0] > matrix.shape[1]:
        pairs = _hungarian_assignment(matrix.T)
        return [(column, row) for row, column in pairs]
    rows, cols = matrix.shape
    u = np.zeros(rows + 1)
    v = np.zeros(cols + 1)
    p = np.zeros(cols + 1, dtype=np.int32)
    way = np.zeros(cols + 1, dtype=np.int32)
    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(cols + 1, np.inf)
        used = np.zeros(cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = np.inf, 0
            for j in range(1, cols + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return [(int(p[j] - 1), j - 1) for j in range(1, cols + 1) if p[j] != 0]
