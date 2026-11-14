"""Bounded, coordinate-safe PCB context for local diagnostic assistance."""

from __future__ import annotations

import json
import math
from typing import Any


DEFAULT_TOKEN_BUDGET = 6000


def _box(row: dict[str, Any]) -> list[float] | None:
    value = row.get("bbox_xyxy")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _token_estimate(value: Any) -> int:
    # A conservative, dependency-free estimate for mixed Korean/JSON content.
    return max(1, math.ceil(len(json.dumps(value, ensure_ascii=False, default=str)) / 3.2))


def normalize_review_status(component: dict[str, Any]) -> str:
    """Normalize registry and viewer labels without treating tentative as confirmed."""
    labels = {str(component.get(key) or "").strip().upper() for key in
              ("presence_status", "review_status", "component_review_status")}
    if labels & {"EXCLUDED", "EXCLUDE", "RETIRED"}:
        return "EXCLUDED"
    if labels & {"STALE", "STALE_OR_UNRESOLVED", "UNRESOLVED", "RECONFIRMATION_REQUIRED"}:
        return "STALE_OR_UNRESOLVED"
    if labels & {"CONFIRMED", "USER_CONFIRMED"}:
        return "USER_CONFIRMED"
    if labels & {"DETECTED_UNVERIFIED", "UNCONFIRMED", "UNVERIFIED", "UNKNOWN"}:
        return "DETECTED_UNVERIFIED"
    return "STALE_OR_UNRESOLVED"


def build_pcb_context(
    *, components: list[dict[str, Any]], photo_records: tuple[Any, ...] | list[Any],
    diagnosis: dict[str, Any], suspected_components: list[dict[str, Any]] | None = None,
    board_registry: dict[str, Any] | None = None,
    token_budget: int = DEFAULT_TOKEN_BUDGET, max_focus_components: int = 24,
) -> dict[str, Any]:
    """Build Stage A summary and bounded Stage B detail without joining photo frames.

    Component IDs in this application are the IDs accepted by the active diagnosis
    snapshot. Photo-local coordinates are included only when the source frame and
    dimensions are known; spatial neighbors are scoped to a single photo/frame.
    """
    photos: dict[str, dict[str, Any]] = {}
    for photo in photo_records:
        photo_id = str(getattr(photo, "image_id", "") or "")
        if photo_id:
            dimensions = getattr(photo, "image_dimensions", None)
            photos[photo_id] = {
                "photo_id": photo_id, "side": getattr(photo, "side", "unknown"),
                "capture_type": getattr(photo, "capture_type", "unknown"),
                "width": dimensions[0] if dimensions and len(dimensions) == 2 else None,
                "height": dimensions[1] if dimensions and len(dimensions) == 2 else None,
            }

    blocks = {str(row.get("component_id")): row for row in diagnosis.get("functional_blocks", [])
              if isinstance(row, dict) and row.get("component_id")}
    registry_sources = {}
    if isinstance(board_registry, dict):
        registry_sources = {str(row.get("board_component_id")): list(row.get("source_components", []))
                            for row in board_registry.get("components", [])
                            if isinstance(row, dict) and row.get("board_component_id")}
    measures: dict[str, list[dict[str, Any]]] = {}
    for row in diagnosis.get("measurement_evidence", []):
        if isinstance(row, dict) and row.get("component_id"):
            valid_assessment = row.get("assessment") in {"TEST_NORMAL", "TEST_ABNORMAL"}
            measures.setdefault(str(row["component_id"]), []).append({
                key: row.get(key) for key in ("measurement_type", "value", "unit", "verification_status", "fault_id",
                             "measurement_target", "operating_conditions", "assessment", "verdict", "reason",
                             "reference_verification_status")
                if key in row
            } | {"evidence_eligibility": "VERIFIED_FOR_HYPOTHESIS" if valid_assessment else "NON_DIAGNOSTIC"})
    suspects = [row for row in (suspected_components or []) if isinstance(row, dict)]
    suspect_by_id: dict[str, dict[str, Any]] = {}
    for suspect in suspects:
        suspect_ids = {str(suspect.get("component_id") or ""),
                       str(suspect.get("board_component_id") or "")}
        for component in components:
            component_ids = {str(component.get("component_id") or ""),
                             str(component.get("board_component_id") or "")}
            if suspect_ids & component_ids:
                suspect_by_id[str(component.get("component_id"))] = suspect
    # Registry snapshots can expose the same board identity through more than
    # one photo row. Keep one analysis component while retaining every source.
    local_photos: dict[str, set[str]] = {}
    for source_row in components:
        if not isinstance(source_row, dict) or source_row.get("board_component_id"):
            continue
        local_id = str(source_row.get("component_id") or "")
        photo_id = str(source_row.get("photo_id") or "")
        if local_id and photo_id:
            local_photos.setdefault(local_id, set()).add(photo_id)
    unique_components: dict[str, dict[str, Any]] = {}
    source_rows_by_id: dict[str, list[dict[str, Any]]] = {}
    excluded_component_count = 0
    for source_row in components:
        if not isinstance(source_row, dict):
            continue
        board_identity = str(source_row.get("board_component_id") or "")
        local_identity = str(source_row.get("component_id") or "")
        source_photo = str(source_row.get("photo_id") or "")
        identity = board_identity or local_identity
        # A repeated local detector ID across photos is not proof of a shared
        # board component; keep those observations separate unless Registry IDs confirm it.
        if not board_identity and local_identity and source_photo and len(local_photos.get(local_identity, ())) > 1:
            identity = f"photo:{source_photo}:{local_identity}"
        if not identity:
            continue
        status = normalize_review_status(source_row)
        if status in {"EXCLUDED", "STALE_OR_UNRESOLVED"}:
            excluded_component_count += 1
            continue
        source_rows_by_id.setdefault(identity, []).append(source_row)
        previous = unique_components.get(identity)
        if previous is None or (status == "USER_CONFIRMED" and normalize_review_status(previous) != "USER_CONFIRMED"):
            unique_components[identity] = dict(source_row)
        elif previous is not None:
            for field in ("source_photo_ids", "source_backend_photo_ids", "photo_ids"):
                values = list(previous.get(field) or [])
                values.extend(source_row.get(field) or [])
                if values:
                    previous[field] = list(dict.fromkeys(str(value) for value in values if value))
            known_types = list(previous.get("candidate_component_types") or [])
            known_types.extend(source_row.get("candidate_component_types") or [])
            if known_types:
                previous["candidate_component_types"] = list(dict.fromkeys(known_types))
            if not previous.get("ocr_marking") and source_row.get("ocr_marking"):
                previous["ocr_marking"] = source_row["ocr_marking"]

    details: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {}
    tentative_type_counts: dict[str, int] = {}
    photo_observations: dict[str, set[str]] = {}
    for identity, source_rows in source_rows_by_id.items():
        for source_row in source_rows:
            photo_ids = source_row.get("source_backend_photo_ids", source_row.get(
                "source_photo_ids", source_row.get("photo_ids", []))) or []
            if source_row.get("photo_id"):
                photo_ids = [*photo_ids, source_row["photo_id"]]
            local_id = str(source_row.get("photo_local_component_id") or source_row.get("component_id") or identity)
            for source_photo_id in dict.fromkeys(str(value) for value in photo_ids if value):
                photo_observations.setdefault(source_photo_id, set()).add(local_id)
    for component_id, row in unique_components.items():
        if not component_id:
            continue
        status = normalize_review_status(row)
        kind = str(row.get("component_type", row.get("type")) or "unknown")
        type_counts[kind] = type_counts.get(kind, 0) + 1
        source_ids = row.get("source_backend_photo_ids", row.get("source_photo_ids", row.get("photo_ids", []))) or []
        photo_id = str(row.get("photo_id") or (source_ids[0] if len(source_ids) == 1 else ""))
        photo = photos.get(photo_id, {})
        bbox = _box(row)
        width, height = photo.get("width"), photo.get("height")
        normalized = None
        coordinate_frame = row.get("coordinate_frame", "photo_local_px" if photo_id else "UNKNOWN")
        if (bbox and photo_id and str(coordinate_frame).casefold() in {"source_photo_pixel", "source_image_px", "photo_local_px"}
                and width and height and width > 0 and height > 0):
            normalized = {
                "center_x": round(((bbox[0] + bbox[2]) / 2) / width, 5),
                "center_y": round(((bbox[1] + bbox[3]) / 2) / height, 5),
                "width": round((bbox[2] - bbox[0]) / width, 5),
                "height": round((bbox[3] - bbox[1]) / height, 5),
            }
        block = blocks.get(component_id, {})
        suspect = suspect_by_id.get(component_id)
        board_id = str(row.get("board_component_id") or component_id)
        raw_sources = registry_sources.get(board_id, [])
        local_source_ids = list(dict.fromkeys(str(source.get("component_id")) for source in raw_sources
                                              if source.get("component_id")))
        photo_local_id = (local_source_ids[0] if len(local_source_ids) == 1 else
                          row.get("photo_local_component_id") if not raw_sources else None)
        source_observations = []
        for source in raw_sources:
            source_photo_id = str(source.get("photo_id") or "")
            source_photo = photos.get(source_photo_id, {})
            source_box = _box(source)
            frame = str(source.get("coordinate_frame") or "UNKNOWN")
            source_normalized = None
            source_width, source_height = source_photo.get("width"), source_photo.get("height")
            if (source_box and frame.casefold() in {"source_photo_pixel", "source_image_px", "photo_local_px"}
                    and source_width and source_height and source_width > 0 and source_height > 0):
                source_normalized = {
                    "center_x": round(((source_box[0] + source_box[2]) / 2) / source_width, 5),
                    "center_y": round(((source_box[1] + source_box[3]) / 2) / source_height, 5),
                    "width": round((source_box[2] - source_box[0]) / source_width, 5),
                    "height": round((source_box[3] - source_box[1]) / source_height, 5),
                }
            source_observations.append({
                "photo_id": source_photo_id or None,
                "photo_size": {"width": source_width, "height": source_height},
                "photo_local_component_id": source.get("component_id"),
                "detection_id": source.get("detection_id"),
                "coordinate_frame": frame,
                "bbox_xyxy": source_box,
                "normalized_geometry": source_normalized,
                "type": source.get("component_type", "unknown"),
                "detection_confidence": source.get("detection_confidence"),
                "review_status": source.get("review_status", "UNCONFIRMED"),
                "presence_status": source.get("presence_status", "UNCONFIRMED"),
                "ocr_marking": source.get("ocr_marking"),
                "ocr_candidates": source.get("ocr_candidates", []),
                "possible_part_number": source.get("possible_part_number"),
                "source_observation_fingerprint": source.get("source_observation_fingerprint"),
            })
        tentative_types = sorted({str(source.get("component_type") or "unknown").casefold()
                                  for source in raw_sources
                                  if str(source.get("review_status", "")).casefold() not in {"excluded", "exclude"}
                                  and str(source.get("component_type") or "unknown").casefold() != "unknown"})
        candidate_types = ([kind.casefold()] if kind.casefold() != "unknown" else tentative_types)
        for tentative_type in candidate_types:
            if tentative_type != "unknown":
                tentative_type_counts[tentative_type] = tentative_type_counts.get(tentative_type, 0) + 1
        review_state = status
        confirmed_source_markings = [str(source.get("ocr_marking")) for source in raw_sources
                                     if source.get("ocr_marking") and str(source.get("review_status", "")).casefold()
                                     in {"user_confirmed", "confirmed"}]
        ocr_marking = row.get("ocr_marking") or (confirmed_source_markings[0] if confirmed_source_markings else None)
        details.append({
            "component_id": component_id,
            "board_component_id": row.get("board_component_id"),
            "photo_id": photo_id or None,
            "photo_local_component_id": photo_local_id,
            "photo": photo or None,
            "source_observations": source_observations,
            "coordinate_frame": coordinate_frame,
            "bbox_xyxy": bbox,
            "normalized_geometry": normalized,
            "type": kind, "detection_confidence": row.get("detection_confidence"),
            "type_evidence_status": "CONFIRMED" if review_state == "USER_CONFIRMED" else "TENTATIVE",
            "identification_status": row.get("identification_status", row.get("identity_status", row.get("presence_status", "UNKNOWN"))),
            "review_status": review_state,
            "component_review_status": review_state,
            "component_type_status": row.get("component_type_status", "UNCONFIRMED"),
            "candidate_component_types": candidate_types,
            "ocr_marking": ocr_marking,
            "ocr_marking_candidates": list(dict.fromkeys(
                str(source.get("ocr_marking")) for source in raw_sources if source.get("ocr_marking"))),
            "ocr_evidence_status": "CONFIRMED" if review_state == "USER_CONFIRMED" and ocr_marking else "TENTATIVE",
            "possible_part_number": row.get("possible_part_number"),
            "ocr_candidates": list(dict.fromkeys(str(item) for source in raw_sources
                                                   for item in (source.get("ocr_candidates") or []) if item)),
            "silkscreen_hints": [item for item in diagnosis.get("silkscreen_labels", [])
                                 if isinstance(item, dict) and item.get("component_id") == component_id][:4],
            "functional_roles": block.get("candidate_blocks", []),
            "functional_role_status": "INFERRED" if block else "UNKNOWN",
            "functional_role_evidence": block.get("role_evidence", {}),
            "nearby_component_ids": block.get("nearby_component_ids", []),
            "spatial_relation_scope": "same photo/frame only; not electrical connectivity",
            "spatial_connectivity_warning": "proximity does not establish electrical connection",
            "connectivity_status": "UNKNOWN",
            "measurements": measures.get(component_id, []),
            "user_suspicion": ({"fault_id": suspect.get("suspected_fault_mode"),
                                "source": "USER", "evidence_status": "UNVERIFIED"} if suspect else None),
        })

    spatial = _spatial_relations(details, photos)
    for row in details:
        row["nearby_components"] = spatial.get(row["component_id"], [])
    by_id = {row["component_id"]: row for row in details}
    confirmed = [row for row in details if row["review_status"] == "USER_CONFIRMED"]
    photo_counts = {photo_id: len(observation_ids) for photo_id, observation_ids in photo_observations.items()}
    summaries = {
        "component_count": len(details), "confirmed_component_count": len(confirmed),
        "tentative_component_count": sum(row["review_status"] == "DETECTED_UNVERIFIED" for row in details),
        "excluded_or_stale_component_count": excluded_component_count,
        "types": dict(sorted(type_counts.items())),
        "tentative_types": dict(sorted(tentative_type_counts.items())),
        "photos": dict(sorted(photo_counts.items())),
        "ocr_hints": [{"component_id": row["component_id"], "marking": row["ocr_marking"],
                        "part_number": row["possible_part_number"]} for row in details
                       if row.get("ocr_marking") or row.get("possible_part_number")][:12],
        "functional_role_hints": [{"component_id": row["component_id"], "roles": row["functional_roles"],
                                    "status": row["functional_role_status"]} for row in details
                                   if row["functional_roles"]][:12],
        "functional_block_group_count": len(diagnosis.get("functional_block_groups", [])),
    }
    focus_ids: list[str] = []
    for row in suspects:
        cid = str(row.get("board_component_id") or row.get("component_id") or "")
        if cid in by_id and cid not in focus_ids:
            focus_ids.append(cid)
    for suspect_component_id in list(focus_ids):
        for neighbor in by_id[suspect_component_id].get("nearby_components", []):
            cid = str(neighbor.get("component_id") or "")
            if cid in by_id and cid not in focus_ids:
                focus_ids.append(cid)
    for candidate in diagnosis.get("component_candidates", [])[:8]:
        cid = str(candidate.get("component_id") or "")
        if cid in by_id and cid not in focus_ids:
            focus_ids.append(cid)
    focus_ids.extend(row["component_id"] for row in confirmed if row["component_id"] not in focus_ids)
    for row in details:
        if row["functional_roles"] and row["component_id"] not in focus_ids:
            focus_ids.append(row["component_id"])
    focus_ids.extend(row["component_id"] for row in details if row["component_id"] not in focus_ids)

    symptom_codes = diagnosis.get("symptom_codes", diagnosis.get("symptoms", []))
    base = {
        "schema_version": "pcb_context_v1",
        "connectivity_status": "UNKNOWN",
        "symptoms": symptom_codes if isinstance(symptom_codes, list) else [],
        "symptom_codes": symptom_codes if isinstance(symptom_codes, list) else [],
        "symptom_text": str(diagnosis.get("symptom_text", ""))[:1200],
        "observations": diagnosis.get("observations", {}),
        "user_suspicions": [{"component_id": cid, "fault_id": row.get("suspected_fault_mode"),
                             "evidence_status": "UNVERIFIED"} for cid, row in suspect_by_id.items()
                            if cid in by_id],
        "hypotheses": [{"fault_id": row.get("fault_id"), "title": row.get("title"),
                        "supporting_evidence": row.get("supporting_evidence", [])[:4],
                        "opposing_evidence": row.get("opposing_evidence", [])[:4]}
                       for row in diagnosis.get("hypotheses", [])[:8]],
        "measurement_evidence": [row for row in diagnosis.get("measurement_evidence", [])
                                  if isinstance(row, dict) and row.get("assessment") in {"TEST_NORMAL", "TEST_ABNORMAL"}][:24],
        "functional_block_groups": [],
        "evidence_gaps": diagnosis.get("evidence_gaps", [])[:12],
        "evidence_ledger": diagnosis.get("evidence_ledger", [])[:40],
        "summary": summaries,
        "focus_components": [],
    }
    budget = max(600, int(token_budget))
    for cid in focus_ids:
        if len(base["focus_components"]) >= max(1, max_focus_components):
            break
        base["focus_components"].append(by_id[cid])
        if _token_estimate(base) > budget:
            base["focus_components"].pop()
            break
    included = {row["component_id"] for row in base["focus_components"]}
    for block in diagnosis.get("functional_block_groups", [])[:12]:
        member_ids = [str(value) for value in block.get("component_ids", []) if str(value) in included]
        if member_ids:
            base["functional_block_groups"].append({"block_id": block.get("block_id"),
                "role": block.get("role"), "component_ids": member_ids[:16],
                "verification_status": "INFERRED", "connectivity_status": "NOT_INFERRED",
                "evidence": block.get("supporting_evidence", [])[:4]})
    omitted = [row["component_id"] for row in details if row["component_id"] not in included]
    base["omitted_component_ids"] = omitted[:80]
    base["omitted_component_count"] = len(omitted)
    base["omitted_ids_truncated"] = len(omitted) > 80
    while _token_estimate(base) > budget:
        if base["focus_components"]:
            removed = base["focus_components"].pop()
            if removed["component_id"] not in base["omitted_component_ids"] and len(base["omitted_component_ids"]) < 80:
                base["omitted_component_ids"].append(removed["component_id"])
                base["omitted_component_count"] += 1
        elif base["omitted_component_ids"]:
            base["omitted_component_ids"] = base["omitted_component_ids"][:max(0, len(base["omitted_component_ids"]) // 2)]
            base["omitted_ids_truncated"] = True
        elif base["summary"]["ocr_hints"]:
            base["summary"]["ocr_hints"].pop()
        elif base["summary"]["functional_role_hints"]:
            base["summary"]["functional_role_hints"].pop()
        elif base["functional_block_groups"]:
            base["functional_block_groups"].pop()
        elif base["evidence_gaps"]:
            base["evidence_gaps"].pop()
        elif base["evidence_ledger"]:
            base["evidence_ledger"].pop()
        elif base["symptom_text"]:
            base["symptom_text"] = base["symptom_text"][:len(base["symptom_text"]) // 2]
        elif isinstance(base.get("observations"), dict) and base["observations"]:
            base["observations"] = dict(list(base["observations"].items())[:len(base["observations"]) // 2])
        else:
            break
    base["context_token_estimate"] = _token_estimate(base)
    base["token_budget"] = budget
    base["context_budget_exceeded"] = base["context_token_estimate"] > budget
    return base


def _spatial_relations(details: list[dict[str, Any]], photos: dict[str, dict[str, Any]], *,
                       neighbors_per_component: int = 5) -> dict[str, list[dict[str, Any]]]:
    """Compute spatial-only neighbors within each source photo coordinate frame."""
    observations: list[dict[str, Any]] = []
    for component in details:
        sources = component.get("source_observations") or []
        if sources:
            for source in sources:
                box = source.get("bbox_xyxy")
                if source.get("photo_id") and box:
                    observations.append({"component_id": component["component_id"],
                                         "photo_id": source["photo_id"],
                                         "frame": source.get("coordinate_frame"), "box": box})
        elif component.get("photo_id") and component.get("bbox_xyxy"):
            observations.append({"component_id": component["component_id"],
                                 "photo_id": component["photo_id"],
                                 "frame": component.get("coordinate_frame"),
                                 "box": component["bbox_xyxy"]})
    result: dict[str, list[dict[str, Any]]] = {}
    for source in observations:
        photo_id, frame = source["photo_id"], source["frame"]
        x1, y1, x2, y2 = source["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        photo = photos.get(photo_id, {})
        width, height = photo.get("width"), photo.get("height")
        diagonal = math.hypot(width or 0, height or 0) or 1.0
        candidates = []
        for other in observations:
            if (other["component_id"] == source["component_id"] or other["photo_id"] != photo_id
                    or other["frame"] != frame):
                continue
            ox1, oy1, ox2, oy2 = other["box"]
            dx, dy = (ox1 + ox2) / 2 - cx, (oy1 + oy2) / 2 - cy
            distance = math.hypot(dx, dy) / diagonal
            candidates.append({"component_id": other["component_id"], "photo_id": photo_id,
                               "distance_normalized": round(distance, 5),
                               "direction": ("right" if abs(dx) >= abs(dy) and dx >= 0 else
                                             "left" if abs(dx) >= abs(dy) else
                                             "below" if dy >= 0 else "above"),
                               "relation_status": "SPATIAL_ONLY"})
        candidates.sort(key=lambda row: (row["distance_normalized"], row["component_id"]))
        bucket = result.setdefault(source["component_id"], [])
        existing = {(row["component_id"], row["photo_id"]) for row in bucket}
        for row in candidates:
            key = (row["component_id"], row["photo_id"])
            if key not in existing and len(bucket) < neighbors_per_component:
                bucket.append(row)
                existing.add(key)
    return result
