"""Resolve natural-language component references only against reviewed Registry rows."""

from __future__ import annotations

from typing import Any

from circuitcare.component_taxonomy import normalize_component_class


def resolve_symptom_targets(
    targets: list[dict[str, Any]] | None, *,
    registry: dict[str, Any] | None,
    component_rows: list[dict[str, Any]] | None = None,
    selected_board_component_ids: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return explicit resolution states; ambiguous ordinal references need user choice.

    Position and ordinal are retained as hints only. The resolver never decides
    that the third item means left-to-right or top-to-bottom without an explicit
    user selection or a unique eligible Registry component.
    """
    registry_rows = registry.get("components", []) if isinstance(registry, dict) else []
    visible_by_board = {
        str(row.get("board_component_id")): row
        for row in (component_rows or []) if isinstance(row, dict) and row.get("board_component_id")
    }
    selected = selected_board_component_ids or {}
    resolved_rows: list[dict[str, Any]] = []
    for raw in targets or []:
        if not isinstance(raw, dict):
            continue
        reference_id = str(raw.get("reference_id") or "").strip()
        target_type = normalize_component_class(raw.get("component_type"), allow_not_component=False)
        eligible = []
        excluded_ids = []
        for row in registry_rows:
            if not isinstance(row, dict) or not row.get("board_component_id"):
                continue
            board_id = str(row["board_component_id"])
            if row.get("presence_status") == "EXCLUDED":
                excluded_ids.append(board_id)
                continue
            if (row.get("presence_status") != "CONFIRMED"
                    or row.get("component_type_status") != "USER_CONFIRMED"
                    or normalize_component_class(row.get("component_type"), allow_not_component=False) != target_type
                    or board_id not in visible_by_board):
                continue
            attrs = row.get("visual_attributes") if isinstance(row.get("visual_attributes"), dict) else {}
            color = str(row.get("confirmed_color") or attrs.get("confirmed_color")
                        or attrs.get("color") or "").strip().casefold()
            region = str(row.get("confirmed_region") or attrs.get("confirmed_region")
                         or attrs.get("region") or "").strip().upper()
            confirmed_ref = str(row.get("user_reference_designator") or "").strip().upper()
            refs = [str(value).strip().upper() for value in row.get("reference_designators", [])]
            requested_color = str(raw.get("color") or "").strip().casefold()
            requested_region = str(raw.get("region") or "UNSPECIFIED").upper()
            requested_ref = str(raw.get("reference_designator") or "").strip().upper()
            if requested_color and color and requested_color not in color:
                continue
            if requested_region not in {"", "UNSPECIFIED"} and region and requested_region != region:
                continue
            if requested_ref and confirmed_ref and requested_ref != confirmed_ref:
                continue
            verified_attrs = []
            if requested_color and color:
                verified_attrs.append("color")
            if requested_region not in {"", "UNSPECIFIED"} and region:
                verified_attrs.append("region")
            if requested_ref and confirmed_ref == requested_ref:
                verified_attrs.append("reference_designator")
            unverified_attrs = [name for name, requested, actual in (
                ("color", requested_color, color),
                ("region", requested_region if requested_region != "UNSPECIFIED" else "", region),
                ("reference_designator", requested_ref, confirmed_ref),
            ) if requested and not actual]
            verified_group_id = str(row.get("verified_group_id") or "")
            group_status = str(row.get("group_membership_status") or "").upper()
            eligible.append({
                "board_component_id": board_id,
                "component_id": str(visible_by_board[board_id].get("component_id") or ""),
                "component_type": target_type,
                "reference_designators": refs,
                "ocr_markings": list(row.get("ocr_markings", [])),
                "verified_attributes": verified_attrs,
                "unverified_attributes": unverified_attrs,
                "verified_group_id": verified_group_id if group_status == "USER_CONFIRMED" else None,
            })
        eligible.sort(key=lambda row: (row["reference_designators"][:1], row["board_component_id"]))
        chosen_id = str(selected.get(reference_id) or "")
        chosen = next((row for row in eligible if row["board_component_id"] == chosen_id), None)
        group_size = raw.get("group_size") if isinstance(raw.get("group_size"), int) else 0
        ordinal = raw.get("ordinal") if isinstance(raw.get("ordinal"), int) else 0
        has_group_reference = bool(group_size or ordinal)
        if chosen:
            status = "RESOLVED_USER_CONFIRMED"
        elif len(eligible) == 1 and not has_group_reference:
            if not eligible[0].get("unverified_attributes"):
                chosen = eligible[0]
                status = "RESOLVED_UNIQUE"
            else:
                status = "NEEDS_ATTRIBUTE_CONFIRMATION"
        elif eligible:
            status = "AMBIGUOUS"
        elif excluded_ids:
            status = "EXCLUDED"
        else:
            status = "UNRESOLVED"
        resolved_rows.append({
            **raw,
            "component_type": target_type,
            "board_component_id": chosen["board_component_id"] if chosen else None,
            "component_id": chosen["component_id"] if chosen else None,
            "resolution_status": status,
            "candidate_components": eligible,
            "resolution_reason": (
                "사용자가 증상 대상을 Registry 소자와 연결했습니다." if status == "RESOLVED_USER_CONFIRMED" else
                "현재 사진에서 확인된 해당 종류의 소자가 하나뿐입니다." if status == "RESOLVED_UNIQUE" else
                "같은 종류의 확인 소자가 여러 개여서 박스 ID를 선택해야 합니다." if status == "AMBIGUOUS" else
                "증상에 언급된 색상·위치·실크스크린을 Registry에서 확인할 수 없어 사용자가 박스를 선택해야 합니다." if status == "NEEDS_ATTRIBUTE_CONFIRMATION" else
                "일치하는 소자가 제외 상태입니다." if status == "EXCLUDED" else
                "현재 사진에서 확인된 해당 종류의 소자를 찾지 못했습니다."
            ),
        })
    return resolved_rows


def bind_target_observations(
    parsed: dict[str, Any], resolved_targets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach only resolved Registry IDs to symptoms and explicitly reported peers."""
    by_reference = {str(row.get("reference_id")): row for row in resolved_targets
                    if isinstance(row, dict) and row.get("reference_id")}
    observations = []
    for raw in parsed.get("observations", []) if isinstance(parsed.get("observations"), list) else []:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        reference_id = str(row.get("target_reference") or "")
        target = by_reference.get(reference_id)
        if target and target.get("resolution_status") in {"RESOLVED_UNIQUE", "RESOLVED_USER_CONFIRMED"}:
            row["component_id"] = target.get("component_id")
            row["board_component_id"] = target.get("board_component_id")
            row["target_resolution_status"] = target.get("resolution_status")
        observations.append(row)

    peers = []
    for raw in parsed.get("peer_observations", []) if isinstance(parsed.get("peer_observations"), list) else []:
        if not isinstance(raw, dict):
            continue
        target = by_reference.get(str(raw.get("target_reference") or ""))
        if not target or target.get("resolution_status") not in {"RESOLVED_UNIQUE", "RESOLVED_USER_CONFIRMED"}:
            continue
        candidate_rows = target.get("candidate_components", [])
        expected_size = target.get("group_size")
        verified_group_ids = {str(row.get("verified_group_id") or "") for row in candidate_rows
                              if row.get("verified_group_id")}
        if (not isinstance(expected_size, int) or expected_size < 2
                or len(candidate_rows) != expected_size or len(verified_group_ids) != 1
                or any(not row.get("verified_group_id") for row in candidate_rows)):
            continue
        chosen_id = str(target.get("board_component_id") or "")
        for candidate in candidate_rows:
            board_id = str(candidate.get("board_component_id") or "")
            local_id = str(candidate.get("component_id") or "")
            if not board_id or not local_id or board_id == chosen_id:
                continue
            peers.append({**raw, "component_id": local_id, "board_component_id": board_id,
                          "target_resolution_status": "RESOLVED_PEER_GROUP"})
    return observations, peers
