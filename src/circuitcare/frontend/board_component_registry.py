"""Conservative board-level component identities across independent PCB photos.

Photo detections remain the source of geometry. Registry identities are merged
only through an explicit user-confirmed relation; geometric proximity is never
used to join detections from different coordinate frames.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any

from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel


REGISTRY_KEY = "circuitcare_v7_board_component_registry"


@dataclass(frozen=True)
class BoardDiagnosisComponent:
    """Engine-facing board identity without cross-photo pixel coordinates."""
    component_id: str
    board_component_id: str
    component_type: str
    bbox_xyxy: None
    ocr_marking: str | None
    possible_part_number: str | None
    component_info: str | None
    identification_status: str
    component_type_status: str
    presence_status: str
    confirmed_functional_roles: tuple[str, ...]
    evidence: tuple[str, ...]
    source_photo_ids: tuple[str, ...]
    identity_status: str
    overlay_source: str = "board_registry"
    detection_confidence: float | None = None
    candidate_component_types: tuple[str, ...] = ()
    ai_review_evidence: tuple[str, ...] = ()
    ocr_marking_candidates: tuple[str, ...] = ()
    candidate_component_subtypes: tuple[str, ...] = ()
    candidate_functional_roles: tuple[str, ...] = ()
    ai_review_statuses: tuple[str, ...] = ()
    candidate_board_features: tuple[str, ...] = ()
    detector_classes: tuple[str, ...] = ()
    confirmed_subtypes: tuple[str, ...] = ()
    possible_part_numbers: tuple[str, ...] = ()


def diagnosis_components(registry: dict[str, Any]) -> tuple[BoardDiagnosisComponent, ...]:
    """Project registry rows to the diagnosis engine using only confirmed facts."""
    result = []
    for row in registry.get("components", []):
        if row.get("presence_status") == "EXCLUDED":
            continue
        source_rows = sorted(row.get("source_components", []),
                             key=lambda item: item.get("review_status") == "EXCLUDED")
        resolved_type = str(row.get("component_type") or "unknown")
        if resolved_type == "board_feature" and row.get("component_type_status") in {
            "USER_CONFIRMED", "USER_CONFIRMED_EXTENSION"
        }:
            # PCB inspection structures remain in the Registry, but are not
            # physical fault-component candidates for diagnosis.
            continue
        confirmed = row.get("presence_status") == "CONFIRMED"
        type_confirmed = row.get("component_type_status") == "USER_CONFIRMED"
        if resolved_type == "other_component":
            resolved_type = "unknown"
        markings = [source.get("ocr_marking") for source in source_rows
                    if source.get("review_status") in {"USER_CONFIRMED", "user_confirmed"}
                    and source.get("ocr_marking")]
        confirmed_parts = list(dict.fromkeys(
            str(source.get("confirmed_part_number")) for source in source_rows
            if source.get("confirmed_part_number")
            and source.get("part_number_status") in {"USER_CONFIRMED", "VERIFIED_DATASHEET"}
        ))
        result.append(BoardDiagnosisComponent(
            component_id=str(row["board_component_id"]),
            board_component_id=str(row["board_component_id"]),
            component_type=str(resolved_type if type_confirmed else "unknown"),
            bbox_xyxy=None,
            ocr_marking=markings[0] if markings else None,
            possible_part_number=confirmed_parts[0] if len(confirmed_parts) == 1 else None,
            component_info=None,
            identification_status="USER_CONFIRMED" if confirmed else "SUGGESTED_CLASS_ONLY",
            component_type_status=("USER_CONFIRMED_EXTENSION" if row.get("component_type") in {"other_component", "board_feature"}
                                   else "USER_CONFIRMED" if type_confirmed else "UNCONFIRMED"),
            presence_status="CONFIRMED" if confirmed else "UNCONFIRMED",
            confirmed_functional_roles=tuple(row.get("functional_roles", ())) if confirmed else (),
            evidence=tuple(row.get("evidence", ())),
            source_photo_ids=tuple(row.get("source_photo_ids", ())),
            identity_status=str(row.get("identity_status", "PHOTO_LOCAL")),
            detection_confidence=None,
            candidate_component_types=tuple(dict.fromkeys(
                str(source.get("qwen_component_type_candidate")) for source in source_rows
                if source.get("qwen_component_type_candidate") not in {None, "not_component"}
                and source.get("qwen_component_type_candidate") != "board_feature"
                and source.get("qwen_component_type_status") in {"AI_PROPOSED_UNVERIFIED", "USER_CONFIRMED_EXTENSION"}
            )),
            ai_review_evidence=tuple(str(source.get("qwen_component_type_evidence")) for source in source_rows
                                     if source.get("qwen_component_type_evidence")),
            ocr_marking_candidates=tuple(dict.fromkeys(str(source.get("ocr_marking")) for source in source_rows
                                                        if source.get("ocr_marking"))),
            candidate_component_subtypes=tuple(dict.fromkeys(
                str(source.get("qwen_component_subtype_candidate")) for source in source_rows
                if source.get("qwen_component_subtype_candidate")
                and source.get("qwen_component_type_status") in {"AI_PROPOSED_UNVERIFIED", "USER_CONFIRMED_EXTENSION"}
            )),
            candidate_functional_roles=tuple(dict.fromkeys(
                str(role) for source in source_rows
                if source.get("qwen_component_type_status") in {"AI_PROPOSED_UNVERIFIED", "USER_CONFIRMED_EXTENSION"}
                for role in source.get("qwen_functional_role_candidates", ())
            )),
            ai_review_statuses=tuple(dict.fromkeys(str(source.get("qwen_review_status"))
                                                  for source in source_rows
                                                  if source.get("qwen_review_status"))),
            candidate_board_features=tuple(dict.fromkeys(
                str(source.get("component_id")) for source in source_rows
                if source.get("qwen_component_type_candidate") == "board_feature"
                and source.get("qwen_component_type_status") in {"AI_PROPOSED_UNVERIFIED", "USER_CONFIRMED_EXTENSION"}
            )),
            detector_classes=tuple(dict.fromkeys(str(source.get("detector_class") or source.get("component_type"))
                                                 for source in source_rows if source.get("component_type"))),
            confirmed_subtypes=tuple(dict.fromkeys(str(source.get("confirmed_subtype")) for source in source_rows
                                                    if source.get("confirmed_subtype"))),
            possible_part_numbers=tuple(dict.fromkeys(
                str(source.get("confirmed_part_number") or source.get("possible_part_number"))
                for source in source_rows if source.get("confirmed_part_number") or source.get("possible_part_number"))),
        ))
    return tuple(result)



def measurement_view_model_for_board_component(
    registry: dict[str, Any], *, board_component_id: str, preferred_photo_id: str | None = None,
) -> ComponentOverlayViewModel | None:
    """Resolve a board identity to a real, non-excluded photo observation for measurement UI."""
    component = next((row for row in registry.get("components", [])
                      if str(row.get("board_component_id")) == str(board_component_id)), None)
    if (not component or component.get("component_type") == "board_feature"
            or component.get("presence_status") == "EXCLUDED"):
        return None
    sources = []
    for row in component.get("source_components", []):
        if (row.get("review_status") == "EXCLUDED"
                or row.get("coordinate_frame") != "SOURCE_PHOTO_PIXEL"):
            continue
        box = row.get("bbox_xyxy")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in box)
        except (TypeError, ValueError):
            continue
        if (not all(math.isfinite(value) for value in (x1, y1, x2, y2))
                or x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1):
            continue
        photo_id = str(row.get("photo_id") or "")
        manifest = next((item for item in registry.get("photos", [])
                         if str(item.get("photo_id")) == photo_id), None)
        image_size = manifest.get("image_size_px") if isinstance(manifest, dict) else None
        if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
            width, height = image_size
            if (not all(isinstance(value, (int, float)) and math.isfinite(float(value))
                        for value in (width, height))
                    or x2 > float(width) or y2 > float(height)):
                continue
        sources.append(row)
    if preferred_photo_id:
        sources.sort(key=lambda row: row.get("photo_id") != preferred_photo_id)
    source = next((row for row in sources if all(isinstance(value, (int, float)) for value in row["bbox_xyxy"])), None)
    if source is None:
        return None
    box = tuple(float(value) for value in source["bbox_xyxy"])
    local_id = str(source.get("component_id") or "")
    photo_id = str(source.get("photo_id") or "")
    if not local_id or not photo_id:
        return None
    confirmed = source.get("review_status") in {"USER_CONFIRMED", "user_confirmed"}
    return ComponentOverlayViewModel(
        project_id=str(component.get("project_id") or ""), board_id=str(component.get("board_id") or ""),
        board_side=str(source.get("board_side") or "unknown"), component_id=local_id,
        board_component_id=str(board_component_id), component_type=str(source.get("component_type") or "unknown"),
        display_label=local_id, canonical_frame_id=photo_id, coordinate_frame="SOURCE_PHOTO_PIXEL",
        bbox_xyxy=box, polygon_xy=(), source_backend_photo_ids=(photo_id,), observation_ids=(),
        detection_confidence=source.get("detection_confidence"),
        identification_status="USER_CONFIRMED" if confirmed else "SUGGESTED_CLASS_ONLY",
        identity_match_status=str(component.get("identity_status") or "PHOTO_LOCAL"),
        ocr_marking=source.get("ocr_marking"), possible_part_number=source.get("possible_part_number"),
        evidence=tuple(source.get("evidence") or ()), overlay_status="CANDIDATE",
        ocr_status="UNVERIFIED" if source.get("ocr_marking") else None,
        ocr_candidates=tuple(source.get("ocr_candidates") or ()),
        presence_status=str(source.get("presence_status") or "UNCONFIRMED"),
        location_confidence=source.get("location_confidence"), type_confidence=source.get("type_confidence"),
        component_type_status=str(source.get("type_status") or "UNCONFIRMED"),
        overlay_source="board_registry_measurement_adapter",
    )

def measurement_component_aliases(registry: dict[str, Any]) -> dict[str, str | None]:
    """Map legacy local measurement IDs only when their identity is unambiguous."""
    candidates: dict[str, set[str]] = {}
    for row in registry.get("components", []):
        if row.get("component_type") == "board_feature":
            continue
        board_id = str(row.get("board_component_id") or "")
        for source in row.get("source_components", []):
            local_id = str(source.get("component_id") or "")
            if local_id:
                candidates.setdefault(local_id, set()).add(board_id)
    return {local_id: (next(iter(board_ids)) if len(board_ids) == 1 else None)
            for local_id, board_ids in candidates.items()}


def source_observation_fingerprint(source: dict[str, Any]) -> str:
    """Fingerprint component identity evidence, excluding UI-only state."""
    identity_fields = (
        "photo_id", "image_fingerprint", "component_id", "detection_id", "bbox_xyxy",
        "detection_version", "review_evidence_version",
        "detection_confidence", "location_confidence", "type_confidence", "source",
        "component_type", "review_status", "presence_status", "type_status",
        "ocr_marking", "ocr_candidates", "marking_candidates", "part_identity_status", "ic_identity_candidate",
        "possible_part_number", "functional_roles", "evidence",
        "qwen_component_type_candidate", "qwen_component_type_evidence", "qwen_component_type_status",
        "qwen_component_type_confidence", "qwen_component_type_decision",
        "qwen_component_subtype_candidate", "qwen_component_type_alternatives", "qwen_review_status",
        "qwen_visual_evidence", "qwen_ocr_evidence", "qwen_conflicting_evidence",
        "qwen_marking_candidates", "qwen_part_number_candidates", "qwen_functional_role_candidates",
        "user_component_type_id", "user_component_type_name", "user_component_parent_class",
        "ocr_inferred_value", "ocr_inference_method",
    )
    payload = {key: source.get(key) for key in identity_fields}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def current_source_observation_fingerprint(
    registry: dict[str, Any], *, photo_id: str | None, component_id: str,
) -> str | None:
    if not photo_id:
        return None
    key = source_component_key(photo_id, component_id)
    source = next((source for row in registry.get("components", [])
                   for source in row.get("source_components", []) if source.get("source_key") == key), None)
    return source_observation_fingerprint(source) if source else None


def resolve_measurement_board_component(
    registry: dict[str, Any], *, photo_id: str | None, component_id: str,
    photo_fingerprint: str | None = None, component_map_version: str | None = None,
    observation_fingerprint: str | None = None,
) -> tuple[str | None, str]:
    """Resolve a measurement source only by exact photo/local identity or a unique legacy alias."""
    if photo_id:
        manifest = next((row for row in registry.get("photos", [])
                         if row.get("photo_id") == photo_id), None)
        if photo_fingerprint and (
            manifest is None or manifest.get("image_fingerprint") != photo_fingerprint
        ):
            return None, "PHOTO_REPLACED_RECONFIRMATION_REQUIRED"
        key = source_component_key(photo_id, component_id)
        board_id = registry.get("source_to_board_id", {}).get(key)
        source = next((source for row in registry.get("components", [])
                       for source in row.get("source_components", [])
                       if source.get("source_key") == key), None)
        if source is None:
            return None, "RECONFIRMATION_REQUIRED"
        current_observation = source_observation_fingerprint(source)
        if not observation_fingerprint or observation_fingerprint != current_observation:
            return None, "RECONFIRMATION_REQUIRED"
        if (board_id and component_map_version and source
                and source.get("component_map_version") != component_map_version):
            return None, "PHOTO_COMPONENT_MAP_CHANGED_RECONFIRMATION_REQUIRED"
        if board_id:
            return str(board_id), "VERIFIED_SOURCE_IDENTITY"
        return None, "UNRESOLVED"
    alias = measurement_component_aliases(registry).get(component_id)
    if alias:
        candidates = [source for row in registry.get("components", [])
                      if row.get("board_component_id") == alias
                      for source in row.get("source_components", [])
                      if source.get("component_id") == component_id]
        if not observation_fingerprint:
            return None, "RECONFIRMATION_REQUIRED"
        matches = [source for source in candidates
                   if source_observation_fingerprint(source) == observation_fingerprint
                   and (not component_map_version or source.get("component_map_version") == component_map_version)]
        if len(matches) != 1:
            return None, "RECONFIRMATION_REQUIRED"
        return alias, "VERIFIED_UNIQUE_LEGACY_ALIAS"
    return None, "UNRESOLVED"


def project_measurements_to_registry(
    measurements: list[Any], registry: dict[str, Any],
) -> tuple[list[Any], list[Any]]:
    """Resolve stored readings against current photo observations once for UI and engine.

    Returns (all projected records for history, current verified records for diagnosis).
    Records with changed/missing source-observation fingerprints remain in the first
    list but are never passed as current evidence.
    """
    board_version = str(registry.get("version") or "")
    projected: list[Any] = []
    active: list[Any] = []

    def get(row: Any, name: str, default: Any = None) -> Any:
        return row.get(name, default) if isinstance(row, dict) else getattr(row, name, default)

    for row in measurements:
        source_photo = get(row, "source_photo_id")
        source_local = get(row, "source_component_id") or get(row, "component_id")
        board_id, resolution = resolve_measurement_board_component(
            registry, photo_id=source_photo, component_id=str(source_local or ""),
            photo_fingerprint=get(row, "source_photo_fingerprint"),
            observation_fingerprint=get(row, "source_observation_fingerprint"),
            component_map_version=(get(row, "component_map_version")
                                   if not get(row, "source_photo_fingerprint") else None),
        )
        values = {
            "component_id": str(board_id or ""),
            "board_component_id": str(board_id) if board_id else None,
            "component_map_version": board_version if board_id else get(row, "component_map_version"),
            "source_photo_id": source_photo,
            "source_component_id": source_local,
            "identity_resolution_status": resolution,
            "component_identity_status": (get(row, "component_identity_status")
                                           if board_id else "IDENTITY_AMBIGUOUS"),
        }
        projected_row = ({**row, **values} if isinstance(row, dict) else replace(row, **values))
        projected.append(projected_row)
        if board_id and resolution.startswith(("VERIFIED_", "USER_RECONFIRMED_")):
            active.append(projected_row)
    return projected, active


def _owner(project_id: str, board_id: str | None) -> str:
    return json.dumps([str(project_id), str(board_id or "")], separators=(",", ":"))


def source_component_key(photo_id: str, component_id: str) -> str:
    return json.dumps([str(photo_id), str(component_id)], ensure_ascii=False,
                      separators=(",", ":"))


def _base_id(project_id: str, board_id: str | None, source_key: str) -> str:
    raw = json.dumps([str(project_id), str(board_id or ""), source_key],
                     ensure_ascii=False, separators=(",", ":")).encode()
    return "bc-" + hashlib.sha256(raw).hexdigest()[:24]


def _source_rows(photo_snapshots: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for photo_id, snapshot in sorted(photo_snapshots.items()):
        for item in snapshot.get("registry_components", snapshot.get("components", ())):
            get = item.get if isinstance(item, dict) else lambda name, default=None: getattr(item, name, default)
            local_id = str(get("component_id", "") or "")
            if not local_id:
                continue
            key = source_component_key(photo_id, local_id)
            bbox = get("bbox_xyxy", ()) or ()
            sources[key] = {
                "source_key": key, "photo_id": str(photo_id), "component_id": local_id,
                "image_fingerprint": snapshot.get("image_fingerprint"),
                "review_version": snapshot.get("review_version"),
                "component_map_version": snapshot.get("version"),
                "board_side": str(snapshot.get("side", "unknown")),
                "capture_type": str(snapshot.get("capture_type", "unknown")),
                "detection_id": local_id, "component_type": str(get("component_type", "unknown")),
                "detector_class": get("detector_class"),
                "qwen_base_class": get("qwen_base_class"),
                "qwen_subtype": get("qwen_subtype"),
                "confirmed_base_class": get("confirmed_base_class"),
                "confirmed_subtype": get("confirmed_subtype"),
                "effective_class": get("effective_class"),
                "detection_version": get("detection_version", get("detection_model_version")),
                "review_evidence_version": get("review_evidence_version"),
                "detection_confidence": get("confidence", get("신뢰도")),
                "location_confidence": get("location_confidence"),
                "type_confidence": get("type_confidence"),
                "bbox_xyxy": list(bbox),
                "coordinate_frame": str(get("coordinate_frame", "SOURCE_PHOTO_PIXEL")),
                "review_status": str(get("identification_status", "UNCONFIRMED")),
                "presence_status": str(get("presence_status", "UNCONFIRMED")),
                "type_status": str(get("component_type_status", "UNCONFIRMED")),
                "component_type_status": str(get("component_type_status", "UNCONFIRMED")),
                "ocr_marking": get("ocr_marking"),
                "body_marking": get("body_marking"),
                "ocr_candidates": list(get("ocr_candidates", ()) or ()),
                "marking_candidates": list(get("marking_candidates", ()) or ()),
                "ocr_source": get("ocr_source"),
                "ocr_score": get("ocr_score"),
                "possible_part_number": get("possible_part_number"),
                "part_number_status": get("part_number_status"),
                "confirmed_part_number": get("confirmed_part_number"),
                "datasheet_source": get("datasheet_source"),
                "part_identity_status": get("part_identity_status"),
                "ic_identity_candidate": get("ic_identity_candidate"),
                "functional_roles": list(get("confirmed_functional_roles", ()) or ()),
                "evidence": list(get("evidence", ()) or ()),
                "source": get("overlay_source"),
                "qwen_component_type_candidate": get("qwen_component_type_candidate"),
                "qwen_component_type_evidence": get("qwen_component_type_evidence"),
                "qwen_component_type_confidence": get("qwen_component_type_confidence"),
                "qwen_component_type_decision": get("qwen_component_type_decision"),
                "qwen_component_type_status": get("qwen_component_type_status"),
                "qwen_component_subtype_candidate": get("qwen_component_subtype_candidate"),
                "qwen_component_type_alternatives": list(get("qwen_component_type_alternatives", ()) or ()),
                "qwen_review_status": get("qwen_review_status"),
                "qwen_visual_evidence": get("qwen_visual_evidence"),
                "qwen_ocr_evidence": get("qwen_ocr_evidence"),
                "qwen_conflicting_evidence": list(get("qwen_conflicting_evidence", ()) or ()),
                "qwen_marking_candidates": list(get("qwen_marking_candidates", ()) or ()),
                "qwen_part_number_candidates": list(get("qwen_part_number_candidates", ()) or ()),
                "qwen_functional_role_candidates": list(get("qwen_functional_role_candidates", ()) or ()),
                "user_component_type_id": get("user_component_type_id"),
                "user_component_type_name": get("user_component_type_name"),
                "user_component_parent_class": get("user_component_parent_class"),
                "ocr_inferred_value": get("ocr_inferred_value"),
                "ocr_inference_method": get("ocr_inference_method"),
            }
            sources[key]["source_observation_fingerprint"] = source_observation_fingerprint(sources[key])
    return sources


def _link_source_version(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_fingerprint": source.get("image_fingerprint"),
        "review_version": source.get("review_version"),
        "component_map_version": source.get("component_map_version"),
        "source_observation_fingerprint": source.get("source_observation_fingerprint"),
    }


def build_board_component_registry(
    state: Any, *, project_id: str, board_id: str | None,
    photo_snapshots: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the registry, preserving local IDs and confirmed identity links.

    ``confirmed_links`` stores source-key pairs after explicit user confirmation.
    Existing source-to-board IDs keep the canonical ID stable when a photo is
    added or removed. On a split, the lexicographically first remaining group
    retains the old ID and other groups receive their deterministic base IDs.
    """
    owner = _owner(project_id, board_id)
    root = state.get(REGISTRY_KEY, {})
    entry = dict(root.get(owner, {})) if isinstance(root, dict) else {}
    sources = _source_rows(photo_snapshots)
    source_ids = {key: _base_id(project_id, board_id, key) for key in sources}
    stable_ids = dict(entry.get("source_identity_ids", {}))
    for key, value in source_ids.items():
        stable_ids.setdefault(key, value)
    all_links = [tuple(sorted((str(link[0]), str(link[1])))) for link in entry.get("confirmed_links", [])
                 if isinstance(link, (list, tuple)) and len(link) == 2 and link[0] != link[1]]
    merge_anchors = entry.get("merge_anchors", {})
    if not isinstance(merge_anchors, dict):
        merge_anchors = {}
    link_versions = entry.get("link_versions", {})
    if not isinstance(link_versions, dict):
        link_versions = {}
    active_links = []
    inactive_links: list[dict[str, str]] = []
    for link in all_links:
        if link[0] not in sources or link[1] not in sources:
            inactive_links.append({"source_keys": list(link), "reason": "PHOTO_OR_COMPONENT_MISSING"})
            continue
        expected_versions = link_versions.get(json.dumps(list(link), separators=(",", ":")))
        if not expected_versions:
            inactive_links.append({"source_keys": list(link), "reason": "RECONFIRMATION_REQUIRED"})
            continue
        current_versions = [_link_source_version(sources[key]) for key in link]
        if expected_versions == current_versions:
            active_links.append(link)
        else:
            image_changed = any(
                expected.get("image_fingerprint") != current.get("image_fingerprint")
                for expected, current in zip(expected_versions or [], current_versions)
                if isinstance(expected, dict)
            )
            reason = ("PHOTO_REPLACED_RECONFIRMATION_REQUIRED" if image_changed
                      else "PHOTO_REDETECTED_RECONFIRMATION_REQUIRED")
            inactive_links.append({"source_keys": list(link), "reason": reason})
    links = active_links

    parent = {key: key for key in sources}
    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    group_photos = {key: {sources[key]["photo_id"]} for key in sources}
    accepted_links: list[tuple[str, str]] = []
    for left, right in links:
        a, b = find(left), find(right)
        if a != b:
            if group_photos[a] & group_photos[b]:
                inactive_links.append({"source_keys": [left, right],
                                       "reason": "SAME_PHOTO_COMPONENT_COLLISION_RECONFIRMATION_REQUIRED"})
                continue
            group_photos[min(a, b)] = group_photos[a] | group_photos[b]
            group_photos.pop(max(a, b), None)
            parent[max(a, b)] = min(a, b)
            accepted_links.append((left, right))
        else:
            accepted_links.append((left, right))
    links = accepted_links
    groups: dict[str, list[str]] = {}
    for key in sorted(sources):
        groups.setdefault(find(key), []).append(key)

    components: list[dict[str, Any]] = []
    source_to_board: dict[str, str] = {}
    used_ids: set[str] = set()
    for keys in sorted(groups.values(), key=lambda values: values[0]):
        group_links = [tuple(link) for link in links if link[0] in keys and link[1] in keys]
        preferred = [merge_anchors.get(json.dumps(list(link), separators=(",", ":")))
                     for link in group_links]
        anchor = next((key for key in preferred if key in keys), keys[0])
        board_component_id = str(stable_ids.get(anchor, source_ids[anchor]))
        if board_component_id in used_ids:
            board_component_id = _base_id(project_id, board_id, anchor + ":split")
        used_ids.add(board_component_id)
        for key in keys:
            source_to_board[key] = board_component_id
        group_sources = [sources[key] for key in keys]
        confirmed = any(row["review_status"] in {"USER_CONFIRMED", "user_confirmed"}
                        or row["presence_status"] == "CONFIRMED" for row in group_sources)
        confirmed_types = {row["component_type"] for row in group_sources
                           if row["review_status"] != "EXCLUDED"
                           and (row["type_status"] in {"USER_CONFIRMED", "CONFIRMED"}
                                or row["review_status"] in {"USER_CONFIRMED", "user_confirmed"})}
        group_presence = ("EXCLUDED" if all(row["review_status"] == "EXCLUDED" for row in group_sources)
                          else "CONFIRMED" if confirmed else "UNCONFIRMED")
        component_type = next(iter(confirmed_types)) if len(confirmed_types) == 1 else "unknown"
        functional_roles = sorted({role for row in group_sources if row["review_status"] != "EXCLUDED"
                                   for role in row["functional_roles"]})
        components.append({
            "board_component_id": board_component_id,
            "project_id": str(project_id), "board_id": str(board_id or ""),
            "source_components": group_sources,
            "source_photo_ids": list(dict.fromkeys(row["photo_id"] for row in group_sources)),
            "detection_ids": [row["detection_id"] for row in group_sources],
            "component_type": component_type,
            "component_type_status": "USER_CONFIRMED" if len(confirmed_types) == 1 else "UNCONFIRMED",
            "presence_status": group_presence,
            "identity_status": "USER_CONFIRMED" if len(keys) > 1 else "PHOTO_LOCAL",
            "ocr_markings": list(dict.fromkeys(row["ocr_marking"] for row in group_sources if row["ocr_marking"])),
            "candidate_component_types": list(dict.fromkeys(
                row["qwen_component_type_candidate"] for row in group_sources
                if row.get("qwen_component_type_candidate") not in {None, "not_component"}
                and row.get("qwen_component_type_status") == "AI_PROPOSED_UNVERIFIED")),
            "functional_roles": functional_roles,
            "evidence": list(dict.fromkeys(e for row in group_sources for e in row["evidence"])),
            "source_keys": keys,
        })
    photo_manifest = [{
        "photo_id": str(photo_id), "image_fingerprint": snapshot.get("image_fingerprint"),
        "image_size_px": snapshot.get("image_size_px"),
        "review_version": snapshot.get("review_version"),
        "detection_status": snapshot.get("detection_status"),
        "side": snapshot.get("side", "unknown"),
        "capture_type": snapshot.get("capture_type", "unknown"),
    } for photo_id, snapshot in sorted(photo_snapshots.items())]
    payload = {"project_id": str(project_id), "board_id": str(board_id or ""),
               "photos": photo_manifest, "components": components, "source_to_board_id": source_to_board,
               "confirmed_links": [list(link) for link in sorted(set(all_links))],
               "inactive_links": sorted(inactive_links, key=lambda item: (item["source_keys"], item["reason"]))}
    version_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"), default=str).encode()
    version = "board-registry:" + hashlib.sha256(version_payload).hexdigest()
    registry = {**payload, "version": version}
    root = dict(root) if isinstance(root, dict) else {}
    root[owner] = {**entry, "source_to_board_id": source_to_board,
                   "source_identity_ids": stable_ids,
                   "confirmed_links": payload["confirmed_links"], "registry": registry}
    state[REGISTRY_KEY] = root
    return registry


def confirm_component_match(
    state: Any, *, project_id: str, board_id: str | None,
    left_source_key: str, right_source_key: str,
) -> None:
    """Record a user-confirmed relation between two photo-local observations."""
    if left_source_key == right_source_key:
        raise ValueError("서로 다른 사진 관측을 선택해야 합니다.")
    owner = _owner(project_id, board_id)
    root = dict(state.get(REGISTRY_KEY, {}))
    entry = dict(root.get(owner, {}))
    registry = entry.get("registry", {})
    source_keys = set(registry.get("source_to_board_id", {}))
    if left_source_key not in source_keys or right_source_key not in source_keys:
        raise ValueError("현재 보드 Registry에서 두 소자 관측을 찾을 수 없습니다.")
    links = {tuple(sorted((str(a), str(b)))) for a, b in entry.get("confirmed_links", [])}
    link = tuple(sorted((left_source_key, right_source_key)))
    links.add(link)
    entry["confirmed_links"] = [list(link) for link in sorted(links)]
    anchors = dict(entry.get("merge_anchors", {}))
    link_key = json.dumps(list(link), separators=(",", ":"))
    anchors[link_key] = left_source_key
    entry["merge_anchors"] = anchors
    source_rows = {
        source.get("source_key"): source
        for row in registry.get("components", [])
        for source in row.get("source_components", [])
    }
    left_photo = source_rows[left_source_key].get("photo_id")
    right_photo = source_rows[right_source_key].get("photo_id")
    if left_photo == right_photo:
        raise ValueError("서로 다른 사진의 관측만 보드 소자로 연결할 수 있습니다.")
    left_board_id = registry["source_to_board_id"].get(left_source_key)
    right_board_id = registry["source_to_board_id"].get(right_source_key)
    left_group = next((row for row in registry.get("components", [])
                       if row.get("board_component_id") == left_board_id), {})
    right_group = next((row for row in registry.get("components", [])
                        if row.get("board_component_id") == right_board_id), {})
    left_photos = set(left_group.get("source_photo_ids", ()))
    right_photos = set(right_group.get("source_photo_ids", ()))
    if left_photos & right_photos:
        raise ValueError("두 소자 그룹에 같은 사진의 서로 다른 소자가 있어 병합할 수 없습니다.")
    versions = dict(entry.get("link_versions", {}))
    versions[link_key] = [_link_source_version(source_rows[key]) for key in link]
    entry["link_versions"] = versions
    root[owner] = entry
    state[REGISTRY_KEY] = root


def remove_component_match(
    state: Any, *, project_id: str, board_id: str | None,
    left_source_key: str, right_source_key: str,
) -> bool:
    owner = _owner(project_id, board_id)
    root = dict(state.get(REGISTRY_KEY, {}))
    entry = dict(root.get(owner, {}))
    target = tuple(sorted((left_source_key, right_source_key)))
    links = [tuple(link) for link in entry.get("confirmed_links", [])
             if isinstance(link, (list, tuple)) and len(link) == 2]
    if target not in links:
        return False
    entry["confirmed_links"] = [list(link) for link in links if link != target]
    versions = dict(entry.get("link_versions", {}))
    versions.pop(json.dumps(list(target), separators=(",", ":")), None)
    entry["link_versions"] = versions
    root[owner] = entry
    state[REGISTRY_KEY] = root
    return True


def source_key_for_component(photo_id: str, component_id: str) -> str:
    return source_component_key(photo_id, component_id)


def board_component_for_id(registry: dict[str, Any] | None, board_component_id: str) -> dict[str, Any] | None:
    if not isinstance(registry, dict):
        return None
    return next((row for row in registry.get("components", [])
                 if row.get("board_component_id") == board_component_id), None)


def primary_source_for_component(registry: dict[str, Any] | None, board_component_id: str) -> dict[str, Any] | None:
    row = board_component_for_id(registry, board_component_id)
    sources = row.get("source_components", []) if row else []
    return sources[0] if sources else None
