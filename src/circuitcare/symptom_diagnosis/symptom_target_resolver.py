"""Resolve natural-language component references only against reviewed Registry rows."""

from __future__ import annotations

from typing import Any

from circuitcare.component_taxonomy import normalize_component_class


def resolve_symptom_targets(
    targets: list[dict[str, Any]] | None, *,
    registry: dict[str, Any] | None,
    component_rows: list[dict[str, Any]] | None = None,
    selected_board_component_ids: dict[str, str] | None = None,
    selected_photo_id: str | None = None,
) -> list[dict[str, Any]]:
    """Resolve explicit references and unambiguous ordinal positions to Registry boxes.

    Ordinals are used only when the parser provides an explicit direction and
    every eligible, confirmed component has a valid source-photo box in the same
    image. Otherwise the target remains ambiguous for user selection.
    """
    registry_rows = registry.get("components", []) if isinstance(registry, dict) else []
    visible_by_board = {
        str(row.get("board_component_id")): row
        for row in (component_rows or []) if isinstance(row, dict) and row.get("board_component_id")
    }
    # The diagnosis projection intentionally removes photo-local coordinates.
    # Recover the geometry only from the matching Registry source observation;
    # never infer a position from the board-level ID alone.
    source_observations: dict[str, list[dict[str, Any]]] = {}
    for registry_row in registry_rows:
        if not isinstance(registry_row, dict) or not registry_row.get("board_component_id"):
            continue
        board_id = str(registry_row["board_component_id"])
        for source in registry_row.get("source_components", []):
            if not isinstance(source, dict) or source.get("review_status") == "EXCLUDED":
                continue
            source_observations.setdefault(board_id, []).append({
                "board_component_id": board_id,
                "component_id": str(source.get("component_id") or ""),
                "photo_id": str(source.get("photo_id") or ""),
                "coordinate_frame": str(source.get("coordinate_frame") or "").upper(),
                "bbox_xyxy": source.get("bbox_xyxy"),
            })
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
                    or board_id not in visible_by_board and not source_observations.get(board_id)):
                continue
            view_row = visible_by_board.get(board_id, {})
            view_box = view_row.get("bbox_xyxy")
            view_photo = str(view_row.get("photo_id") or "")
            view_frame = str(view_row.get("coordinate_frame") or "").upper()
            usable_view_box = (isinstance(view_box, (list, tuple)) and len(view_box) == 4
                               and all(isinstance(value, (int, float)) for value in view_box)
                               and view_box[2] > view_box[0] and view_box[3] > view_box[1])
            source_rows = source_observations.get(board_id, [])
            if selected_photo_id:
                matching_sources = [source for source in source_rows
                                    if source["photo_id"] == str(selected_photo_id)]
            else:
                matching_sources = source_rows
            if usable_view_box and (not selected_photo_id or view_photo == str(selected_photo_id)):
                photo_id, coordinate_frame, bbox = view_photo, view_frame, view_box
                local_component_id = str(view_row.get("component_id") or "")
            elif matching_sources:
                unique_source_photos = {source["photo_id"] for source in matching_sources if source["photo_id"]}
                if len(unique_source_photos) == 1:
                    source = next((item for item in matching_sources if item["photo_id"] in unique_source_photos), {})
                elif selected_photo_id:
                    source = matching_sources[0]
                else:
                    source = {}
                photo_id = source.get("photo_id", "")
                coordinate_frame = source.get("coordinate_frame", "")
                bbox = source.get("bbox_xyxy")
                local_component_id = source.get("component_id", "")
            else:
                photo_id, coordinate_frame, bbox, local_component_id = view_photo, view_frame, view_box, ""
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
                "component_id": local_component_id or str(view_row.get("component_id") or ""),
                "component_type": target_type,
                "reference_designators": refs,
                "ocr_markings": list(row.get("ocr_markings", [])),
                "verified_attributes": verified_attrs,
                "unverified_attributes": unverified_attrs,
                "verified_group_id": verified_group_id if group_status == "USER_CONFIRMED" else None,
                "photo_id": str(photo_id or ""),
                "coordinate_frame": str(coordinate_frame or "").upper(),
                "bbox_xyxy": bbox,
            })
        eligible.sort(key=lambda row: (row["reference_designators"][:1], row["board_component_id"]))
        chosen_id = str(selected.get(reference_id) or "")
        chosen = next((row for row in eligible if row["board_component_id"] == chosen_id), None)
        group_size = raw.get("group_size") if isinstance(raw.get("group_size"), int) else 0
        ordinal = raw.get("ordinal") if isinstance(raw.get("ordinal"), int) else 0
        has_group_reference = bool(group_size or ordinal)
        positional_choice = None
        ordering = str(raw.get("ordering") or "UNSPECIFIED").upper()
        # Natural reports commonly say “the third of four LEDs” without
        # naming left-to-right. Infer an axis only when the confirmed boxes
        # themselves form one unambiguous row or column in the same photo.
        # Mixed/grid layouts and incomplete coordinate sets remain ambiguous.
        positional_boxes = [row.get("bbox_xyxy") for row in eligible]
        source_frames = {"SOURCE_PHOTO_PIXEL", "SOURCE_IMAGE_PX", "PHOTO_LOCAL_PX"}
        valid_boxes = all(
            isinstance(box, (list, tuple)) and len(box) == 4
            and all(isinstance(value, (int, float)) for value in box)
            and float(box[2]) > float(box[0]) and float(box[3]) > float(box[1])
            for box in positional_boxes
        )
        photo_ids = {row["photo_id"] for row in eligible}
        frames = {row["coordinate_frame"] for row in eligible}
        if (not chosen and ordinal > 0 and ordering == "UNSPECIFIED"
                and group_size == len(eligible) and len(eligible) > 1
                and len(photo_ids) == 1 and "" not in photo_ids
                and len(frames) == 1 and frames.issubset(source_frames) and valid_boxes):
            centers_x = [(float(box[0]) + float(box[2])) / 2 for box in positional_boxes]
            centers_y = [(float(box[1]) + float(box[3])) / 2 for box in positional_boxes]
            widths = [float(box[2]) - float(box[0]) for box in positional_boxes]
            heights = [float(box[3]) - float(box[1]) for box in positional_boxes]
            median_width = sorted(widths)[len(widths) // 2]
            median_height = sorted(heights)[len(heights) // 2]
            row_layout = max(centers_y) - min(centers_y) <= median_height * 0.35
            column_layout = max(centers_x) - min(centers_x) <= median_width * 0.35
            if row_layout != column_layout:
                ordering = "LEFT_TO_RIGHT" if row_layout else "TOP_TO_BOTTOM"
        if (not chosen and ordinal > 0 and ordering in {"LEFT_TO_RIGHT", "TOP_TO_BOTTOM"}
                and group_size == len(eligible) and len(eligible) > 1):
            if (len(photo_ids) == 1 and "" not in photo_ids and len(frames) == 1
                    and frames.issubset(source_frames) and valid_boxes and ordinal <= len(eligible)):
                axis = 0 if ordering == "LEFT_TO_RIGHT" else 1
                positioned = sorted(eligible, key=lambda row: (
                    (float(row["bbox_xyxy"][axis]) + float(row["bbox_xyxy"][axis + 2])) / 2,
                    row["board_component_id"],
                ))
                positional_choice = positioned[ordinal - 1]
                chosen = positional_choice
        if chosen:
            status = ("RESOLVED_USER_CONFIRMED" if chosen_id else
                      "RESOLVED_POSITIONAL" if positional_choice else "RESOLVED_UNIQUE")
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
                "같은 사진에서 확인된 박스 좌표가 단일 행·열을 이루어 순번 대상을 연결했습니다." if status == "RESOLVED_POSITIONAL" else
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
        if target and target.get("resolution_status") in {
            "RESOLVED_UNIQUE", "RESOLVED_USER_CONFIRMED", "RESOLVED_POSITIONAL",
        }:
            row["component_id"] = target.get("component_id")
            row["board_component_id"] = target.get("board_component_id")
            row["target_resolution_status"] = target.get("resolution_status")
        observations.append(row)

    peers = []
    for raw in parsed.get("peer_observations", []) if isinstance(parsed.get("peer_observations"), list) else []:
        if not isinstance(raw, dict):
            continue
        target = by_reference.get(str(raw.get("target_reference") or ""))
        if not target or target.get("resolution_status") not in {
            "RESOLVED_UNIQUE", "RESOLVED_USER_CONFIRMED", "RESOLVED_POSITIONAL",
        }:
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
