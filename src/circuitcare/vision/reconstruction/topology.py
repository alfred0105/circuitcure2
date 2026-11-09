"""Observed same-side geometry graph; this is not an electrical netlist."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from circuitcare.vision.reconstruction.models import (
    ConnectivityEvidence, EvidenceStatus, ObservedTopologyGraph, PadViaCandidate,
)


def build_observed_topology(mask: np.ndarray, *, photo_id: str, board_side: str,
                            pad_candidates: Iterable[PadViaCandidate],
                            contact_tolerance_px: float = 2.0) -> ObservedTopologyGraph:
    """Create contact candidates only when same-side visible mask reaches a candidate."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError("연결 그래프의 입력 마스크는 2차원이어야 합니다.")
    if board_side not in {"front", "back", "unknown"}:
        raise ValueError("알 수 없는 PCB 면 값입니다.")
    features = [item for item in pad_candidates if item.photo_id == photo_id and item.board_side == board_side]
    nodes = [{"feature_id": item.candidate_id, "feature_type": item.candidate_kind,
              "board_side": board_side, "evidence_status": EvidenceStatus.INFERRED.value,
              "through_plated": item.through_plated} for item in features]
    edges = []
    kernel_size = max(1, int(round(contact_tolerance_px * 2 + 1)))
    grown = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
    h, w = binary.shape
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    for label in range(1, component_count):
        if stats[label, cv2.CC_STAT_AREA] < 8:
            continue
        trace_id = f"trace:{photo_id}:{board_side}:{label}"
        nodes.append({"feature_id": trace_id, "feature_type": "VISIBLE_COPPER_LIKE_MASK_COMPONENT",
                      "board_side": board_side, "evidence_status": EvidenceStatus.INFERRED.value})
    for item in features:
        x, y = (int(round(value)) for value in item.center_px)
        radius = max(1, int(round(item.radius_px)))
        x0, x1, y0, y1 = max(0, x-radius), min(w, x+radius+1), max(0, y-radius), min(h, y+radius+1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (xx-x)**2 + (yy-y)**2 <= radius**2
        touching = labels[y0:y1, x0:x1][disk & (grown[y0:y1, x0:x1] > 0)] if disk.size else []
        for label in np.unique(touching):
            if label <= 0 or stats[label, cv2.CC_STAT_AREA] < 8:
                continue
            edges.append(ConnectivityEvidence(
                source_feature_id=item.candidate_id,
                target_feature_id=f"trace:{photo_id}:{board_side}:{label}",
                relationship="VISIBLE_SAME_SIDE_CONTACT_CANDIDATE",
                evidence_type="PIXEL_MASK_PROXIMITY",
                verification_status=EvidenceStatus.INFERRED,
                source_photo_ids=(photo_id,), same_side_contact=True))
    return ObservedTopologyGraph(tuple(nodes), tuple(edges))
