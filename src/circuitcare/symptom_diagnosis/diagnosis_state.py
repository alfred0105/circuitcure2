"""Version-scoped diagnosis overlay state shared by the panel and PCB viewer."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def component_inventory_version(
    *, project_id: str, board_id: str | None, photo_id: str, image_sha256: str,
    components: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> str:
    """Version a photo-scoped reviewed component list without requiring a Component Map."""
    rows = []
    for row in components:
        if not isinstance(row, dict):
            continue
        rows.append({
            "component_id": str(row.get("component_id") or ""),
            "component_type": str(row.get("component_type", row.get("소자 종류", "unknown"))).casefold(),
            "bbox_xyxy": [round(float(value), 3) for value in row.get(
                "bbox_xyxy", [row.get("x1", 0), row.get("y1", 0), row.get("x2", 0), row.get("y2", 0)])],
        })
    payload = {"project_id": project_id, "board_id": board_id or "", "photo_id": photo_id,
               "image_sha256": image_sha256, "components": sorted(rows, key=lambda item: item["component_id"])}
    return "photo-inventory:" + hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def build_overlay_state(
    *, project_id: str, board_id: str | None, component_map_version: str | None,
    result: dict[str, Any], previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Separate the current Top-3 overlay from durable verified component history."""
    candidates = result.get("component_candidates", [])[:3]
    history = list((previous or {}).get("verified_component_history", []))
    historical_rows = [*(previous or {}).get("active_candidates", {}).values(), *candidates,
                       *result.get("verified_component_history", [])]
    known_history = {(row.get("component_id"), row.get("fault_hypothesis"), row.get("measurement_verdict"),
                      tuple(row.get("evidence_ids", []))) for row in history if isinstance(row, dict)}
    for row in historical_rows:
        if not isinstance(row, dict) or row.get("measurement_verdict") not in {"supports", "refutes"}:
            continue
        key = (row.get("component_id"), row.get("fault_hypothesis"), row.get("measurement_verdict"),
               tuple(row.get("evidence_ids", [])))
        if key not in known_history:
            history.append(dict(row, input_fingerprint=result.get("input_fingerprint")))
            known_history.add(key)
    return {
        "project_id": project_id,
        "board_id": board_id,
        "component_map_version": component_map_version,
        "diagnosis_id": result.get("diagnosis_id"),
        "input_fingerprint": result.get("input_fingerprint"),
        "active_candidates": {str(row["component_id"]): row for row in candidates},
        "verified_component_history": history[-1000:],
        "joint_candidates": result.get("joint_candidates", []),
    }


def current_overlay_candidates(
    value: Any, *, project_id: str, board_id: str | None,
    component_map_version: str | None, input_fingerprint: str | None,
) -> dict[str, dict[str, Any]]:
    """Return candidates only for a complete, exact current project/board/map/input scope."""
    if not isinstance(value, dict) or not project_id or not board_id or not component_map_version:
        return {}
    if value.get("project_id") != project_id or value.get("board_id") != board_id:
        return {}
    if value.get("component_map_version") != component_map_version:
        return {}
    if value.get("input_fingerprint") != input_fingerprint:
        return {}
    candidates = value.get("active_candidates")
    return candidates if isinstance(candidates, dict) else {}


def current_verified_component_history(
    value: Any, *, project_id: str, board_id: str | None,
    component_map_version: str | None, input_fingerprint: str | None,
) -> list[dict[str, Any]]:
    """Return retained verification rows for the exact live board inventory."""
    if not isinstance(value, dict) or not project_id or not board_id or not component_map_version or not input_fingerprint:
        return []
    if (value.get("project_id") != project_id or value.get("board_id") != board_id
            or value.get("component_map_version") != component_map_version):
        return []
    history = value.get("verified_component_history", [])
    return [row for row in history if isinstance(row, dict)] if isinstance(history, list) else []
