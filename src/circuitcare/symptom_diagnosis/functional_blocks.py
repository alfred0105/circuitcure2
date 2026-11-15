"""Low-confidence functional block hypotheses from labels, OCR hints, and layout."""

from __future__ import annotations

import hashlib
import json
from math import hypot
import re
from typing import Any

from circuitcare.symptom_diagnosis.ic_identity import resolve_ic_identity


def analyze_functional_blocks(
    components: list[dict[str, Any]], taxonomy: dict[str, list[str]], *,
    silkscreen_labels: list[dict[str, Any]] | None = None,
    ic_catalog: dict[str, dict] | None = None,
) -> list[dict[str, Any]]:
    """Return inferred role candidates with evidence and explicit uncertainty."""
    boxes = [row.get("bbox_xyxy", row.get("bbox")) for row in components]
    valid = [box for box in boxes if isinstance(box, (list, tuple)) and len(box) == 4]
    if valid:
        width = max(float(box[2]) for box in valid) - min(float(box[0]) for box in valid)
        height = max(float(box[3]) for box in valid) - min(float(box[1]) for box in valid)
        diagonal = max(hypot(width, height), 1.0)
    else:
        diagonal = 1.0

    identities: dict[str, dict] = {}
    result = []
    labels = silkscreen_labels or []
    for component in components:
        component_id = component.get("component_id")
        kind = str(component.get("component_type", "unknown"))
        own_roles = set(taxonomy.get(kind, []))
        confirmed_roles = {str(role) for role in component.get("confirmed_functional_roles", []) if str(role).strip()}
        ai_roles = {str(role) for role in component.get("ai_suggested_functional_roles", []) if str(role).strip()}
        identity = None
        if kind == "ic" and component.get("ocr_marking"):
            identity = resolve_ic_identity(str(component["ocr_marking"]), ic_catalog)
            own_roles.update(identity["role_candidates"])
        identities[str(component_id)] = {"roles": own_roles | confirmed_roles | ai_roles,
                                          "taxonomy_roles": own_roles,
                                          "confirmed_roles": confirmed_roles,
                                          "ai_roles": ai_roles,
                                          "identity": identity}

    for component in components:
        component_id = component.get("component_id")
        kind = str(component.get("component_type", "unknown"))
        box = component.get("bbox_xyxy", component.get("bbox"))
        nearby = []
        if isinstance(box, (list, tuple)) and len(box) == 4:
            cx, cy = (float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2
            for other in components:
                if other is component:
                    continue
                other_box = other.get("bbox_xyxy", other.get("bbox"))
                if not isinstance(other_box, (list, tuple)) or len(other_box) != 4:
                    continue
                ox, oy = (float(other_box[0]) + float(other_box[2])) / 2, (float(other_box[1]) + float(other_box[3])) / 2
                distance = hypot(cx - ox, cy - oy) / diagonal
                if distance <= 0.18:
                    nearby.append((other, distance))

        votes: dict[str, float] = {}
        evidence_by_role: dict[str, list[str]] = {}
        own = identities.get(str(component_id), {})
        for role in own.get("taxonomy_roles", set()):
            votes[role] = votes.get(role, 0.0) + 0.5
            evidence_by_role.setdefault(role, []).append("소자 종류 기반 역할 후보")
        for role in own.get("ai_roles", set()):
            votes[role] = votes.get(role, 0.0) + 0.5
            evidence_by_role.setdefault(role, []).append("Qwen 기능 역할 제안 · 미확정")
        for role in own.get("confirmed_roles", set()):
            votes[role] = votes.get(role, 0.0) + 2.0
            evidence_by_role.setdefault(role, []).append("사용자 확인 기능 역할")
        if own.get("identity"):
            for role in own["identity"].get("role_candidates", []):
                verified_identity = own["identity"].get("verified") is True
                votes[role] = votes.get(role, 0.0) + (1.0 if verified_identity else 0.35)
                evidence_by_role.setdefault(role, []).append(
                    "검증된 IC 카탈로그 역할 후보 · 실제 회로 기능 미확정" if verified_identity
                    else "IC 각인 문자열 힌트 · 미확정")
        related_ids = []
        for other, distance in nearby:
            other_id = str(other.get("component_id"))
            other_info = identities.get(other_id, {})
            shared_roles = set(own.get("roles", set())) & set(other_info.get("roles", set()))
            complement_roles = set(other_info.get("roles", set()))
            for role in complement_roles:
                # Neighbor support is deliberately weaker than a part's own role.
                votes[role] = votes.get(role, 0.0) + max(0.1, 0.35 * (1.0 - distance / 0.18))
                evidence_by_role.setdefault(role, []).append(f"인접 부품 {other_id}")
            if shared_roles:
                related_ids.append(other_id)

        cx = cy = None
        if isinstance(box, (list, tuple)) and len(box) == 4:
            cx, cy = (float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2
        nearby_text = []
        if cx is not None:
            for label in labels:
                text_box = label.get("bbox_xyxy", label.get("bbox"))
                if not isinstance(text_box, (list, tuple)) or len(text_box) != 4:
                    continue
                tx, ty = (float(text_box[0]) + float(text_box[2])) / 2, (float(text_box[1]) + float(text_box[3])) / 2
                if hypot(cx - tx, cy - ty) / diagonal <= 0.08:
                    nearby_text.append(str(label.get("text", "")))

        # Reference designators and nearby signal labels are weak hints, not
        # confirmed connectivity. Apply them to the component whose box is nearby.
        reference_roles = {
            "Q": ("driver",), "U": ("control", "power"), "FB": ("power",),
            "L": ("power",), "D": ("power",), "C": ("power",),
            "R": ("power",), "BZ": ("buzzer",), "K": ("relay",),
            "X": ("control_timing",), "Y": ("control_timing",),
            "J": ("power",), "SW": ("control_input",),
        }
        hint_text = " ".join(nearby_text).upper()
        for prefix, roles_for_prefix in reference_roles.items():
            if re.search(rf"(?<![A-Z0-9]){prefix}\s*\d+", hint_text):
                for role in roles_for_prefix:
                    votes[role] = votes.get(role, 0.0) + 0.5
                    evidence_by_role.setdefault(role, []).append(f"근처 실크스크린 참조명 {prefix}")
        if re.search(r"\b(VIN|VOUT|VCC|GND|VBAT|3V3|5V)\b", hint_text):
            votes["power"] = votes.get("power", 0.0) + 0.6
            evidence_by_role.setdefault("power", []).append("근처 실크스크린 전원 신호명")

        roles = sorted(votes, key=lambda role: (-votes[role], role))[:4]
        identity = own.get("identity")
        result.append({
            "component_id": component_id, "component_type": kind,
            "candidate_blocks": roles, "nearby_component_ids": [str(row.get("component_id")) for row, _ in nearby],
            "related_component_ids": related_ids, "nearby_silkscreen": nearby_text,
            "role_evidence": {role: list(dict.fromkeys(evidence_by_role[role])) for role in roles},
            "role_status_by_candidate": {role: ("USER_CONFIRMED" if role in own.get("confirmed_roles", set())
                                                else "AI_SUGGESTED" if role in own.get("ai_roles", set())
                                                else "INFERRED") for role in roles},
            "ic_identity": identity, "status": "INFERRED", "verified": False,
        })
    return result


def build_functional_block_groups(
    components: list[dict[str, Any]], component_roles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group reviewed components by inferred role; membership never implies wiring."""
    roles_by_id = {}
    for role_row in component_roles:
        if not isinstance(role_row, dict):
            continue
        for alias in (role_row.get("component_id"), role_row.get("board_component_id")):
            if alias:
                roles_by_id[str(alias)] = role_row
    groups: list[dict[str, Any]] = []
    valid_components = [component for component in components if isinstance(component, dict)
                        and str(component.get("component_type") or "").casefold() != "not_component"
                        and str(component.get("presence_status") or "").upper() != "EXCLUDED"]
    board_points = []
    for component in valid_components:
        box = component.get("bbox_xyxy", component.get("bbox"))
        if isinstance(box, (list, tuple)) and len(box) == 4:
            try:
                board_points.append(((float(box[0]) + float(box[2])) / 2,
                                     (float(box[1]) + float(box[3])) / 2))
            except (TypeError, ValueError, OverflowError):
                continue
    board_diagonal = (max(hypot(max(point[0] for point in board_points) - min(point[0] for point in board_points),
                                max(point[1] for point in board_points) - min(point[1] for point in board_points)), 1.0)
                      if board_points else 1.0)
    for role in ("power", "control", "control_timing", "control_input", "input",
                 "output", "driver", "protection", "thermal", "buzzer", "relay", "input_output"):
        members = []
        for component in valid_components:
            local_component_id = str(component.get("component_id") or "")
            component_id = str(component.get("board_component_id") or local_component_id)
            if not component_id:
                continue
            role_row = roles_by_id.get(local_component_id) or roles_by_id.get(component_id, {})
            if role not in role_row.get("candidate_blocks", []):
                continue
            identity_verified = (
                str(component.get("component_type_status") or "").upper() == "USER_CONFIRMED"
                and str(component.get("presence_status") or "").upper() in {"CONFIRMED", "USER_CONFIRMED"}
            )
            role_verified = role in {str(value) for value in component.get("confirmed_functional_roles", [])}
            role_source = ("USER_CONFIRMED" if role_verified else
                           "AI_SUGGESTED" if role in {str(value) for value in component.get("ai_suggested_functional_roles", [])}
                           else "INFERRED")
            box = component.get("bbox_xyxy", component.get("bbox"))
            center = None
            if isinstance(box, (list, tuple)) and len(box) == 4:
                try:
                    center = ((float(box[0]) + float(box[2])) / 2,
                              (float(box[1]) + float(box[3])) / 2)
                except (TypeError, ValueError, OverflowError):
                    center = None
            members.append({"board_component_id": str(component.get("board_component_id") or component_id),
                            "component_id": component_id,
                            "local_component_id": local_component_id or component_id,
                            "component_type": component.get("component_type", "unknown"),
                            "identity_status": "VERIFIED" if identity_verified else "UNKNOWN",
                            "evidence_status": "VERIFIED" if identity_verified else "UNKNOWN",
                            "role_status": role_source,
                            "role_verified": role_verified,
                            "photo_id": component.get("photo_id"),
                            "bbox_xyxy": box, "center_xy": center,
                            "ocr_marking": component.get("ocr_marking"),
                            "reference_designators": list(component.get("reference_designators", [])),
                            "role_evidence": list(role_row.get("role_evidence", {}).get(role, [])),
                            "ai_suggested_functional_roles": list(component.get("ai_suggested_functional_roles", [])),
                            "functional_instance_id": component.get("functional_instance_id"),
                            "verified_group_id": component.get("verified_group_id"),
                            "group_confirmation_status": str(component.get("group_confirmation_status") or "UNKNOWN")})
        if not members:
            continue
        # Use explicit instance labels first. Otherwise, cluster only with a
        # conservative complete-link distance rule; proximity is grouping
        # evidence, never proof of electrical connectivity. Without coordinates
        # keep separate unresolved instances instead of merging by role alone.
        diagonal = board_diagonal
        explicit_clusters: list[list[dict[str, Any]]] = []
        unlabelled_clusters: list[list[dict[str, Any]]] = []
        named_clusters: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for member in members:
            group_id = str(member.get("verified_group_id") or "").strip()
            instance_id = str(member.get("functional_instance_id") or "").strip()
            explicit_id = group_id or instance_id
            explicit_kind = "verified_group" if group_id else "instance"
            if explicit_id:
                named_clusters.setdefault((explicit_kind, explicit_id), []).append(member)
            else:
                unlabelled_clusters.append([member])
        explicit_clusters = [named_clusters[key] for key in sorted(named_clusters)]
        located = [cluster for cluster in unlabelled_clusters
                   if all(member.get("center_xy") for member in cluster)]
        unlocated = [cluster for cluster in unlabelled_clusters if cluster not in located]
        merged: list[list[dict[str, Any]]] = []
        for cluster in sorted(located, key=lambda group: group[0]["component_id"]):
            for existing in merged:
                if all(hypot(left["center_xy"][0] - right["center_xy"][0],
                             left["center_xy"][1] - right["center_xy"][1]) / diagonal <= 0.12
                       for left in cluster for right in existing):
                    existing.extend(cluster)
                    break
            else:
                merged.append(list(cluster))
        # Explicit instance IDs are authoritative partitions: proximity may
        # combine only unlabelled layout hints, never distinct named instances.
        clusters = explicit_clusters + merged + unlocated
        for cluster in clusters:
            cluster.sort(key=lambda row: row["component_id"])
            member_ids = sorted({row["board_component_id"] for row in cluster})
            seed = json.dumps([role, member_ids], ensure_ascii=False, separators=(",", ":"))
            block_id = f"block-{hashlib.sha256(seed.encode()).hexdigest()[:12]}"
            evidence = list(dict.fromkeys(text for member in cluster for text in member["role_evidence"]))
            role_verified = bool(cluster) and all(member["role_verified"] for member in cluster)
            verified_membership = bool(cluster) and all(
                member.get("verified_group_id")
                and member.get("group_confirmation_status") == "USER_CONFIRMED"
                for member in cluster
            )
            explicit_instance = bool(cluster) and all(member.get("functional_instance_id") for member in cluster)
            partition_status = ("USER_CONFIRMED_GROUP" if verified_membership else
                                "EXPLICIT_INSTANCE" if explicit_instance
                                else "LAYOUT_CLUSTER" if all(member.get("center_xy") for member in cluster)
                                else "UNRESOLVED_NO_POSITION")
            version_seed = json.dumps([role, partition_status, member_ids,
                                       [(member["component_id"], member["role_status"], member["role_evidence"],
                                         member.get("functional_instance_id"), member.get("verified_group_id"),
                                         member.get("group_confirmation_status"))
                                        for member in cluster]], ensure_ascii=False, separators=(",", ":"))
            block_version = hashlib.sha256(version_seed.encode()).hexdigest()[:16]
            membership_status = ("USER_CONFIRMED" if verified_membership else
                                 "INFERRED" if partition_status in {"EXPLICIT_INSTANCE", "LAYOUT_CLUSTER"}
                                 else "UNKNOWN")
            role_status = "USER_CONFIRMED" if role_verified else "INFERRED"
            groups.append({"block_id": block_id, "functional_block_id": block_id, "role": role,
                           "version": block_version,
                           "title": role.replace("_", " ").title(), "component_ids": member_ids,
                           "board_component_ids": member_ids, "members": cluster,
                           "supporting_evidence": evidence, "opposing_evidence": [],
                           # Legacy fields describe role only; membership and
                           # electrical connectivity have independent status.
                           "verification_status": "VERIFIED" if role_verified else "INFERRED",
                           "status": "VERIFIED" if role_verified else "INFERRED",
                           "role_status": role_status,
                           "membership_status": membership_status,
                           "verified": role_verified, "partition_status": partition_status,
                           "connectivity_status": "UNKNOWN",
                           "unknown_connectivity_component_ids": member_ids,
                           "unverified_relations": [],
                           "connectivity_note": "역할 그룹은 전기적 연결 관계를 의미하지 않습니다."})
    return groups


def evaluate_functional_block_states(
    functional_blocks: list[dict[str, Any]], observations: list[dict[str, Any]] | None = None,
    measurements: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Attach independent, scoped behavior states to existing block groups.

    Only explicit functional-block observations and verified measurements
    directly scoped to a block contribute. Member proximity never propagates
    a behavior or electrical relationship.
    """
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for row in observations or []:
        if not isinstance(row, dict) or str(row.get("scope") or "").casefold() != "functional_block":
            continue
        subject = str(row.get("subject_id") or "")
        if subject and str(row.get("status") or "observed").casefold() in {"observed", "measured"}:
            by_subject.setdefault(subject, []).append(row)
    for row in measurements or []:
        if not isinstance(row, dict) or not row.get("functional_block_id"):
            continue
        if str(row.get("verification_status") or "").upper() not in {"VERIFIED", "VALIDATED"}:
            continue
        by_subject.setdefault(str(row["functional_block_id"]), []).append({
            "id": row.get("measurement_id"), "scope": "functional_block",
            "subject_id": row["functional_block_id"], "behavior": row.get("behavior_status"),
            "value": row.get("value"), "status": "measured", "source": row.get("source"),
            "operating_condition": row.get("operating_conditions"),
        })

    result = []
    for block in functional_blocks:
        block_id = str(block.get("block_id") or "")
        evidence = by_subject.get(block_id, [])
        directions: dict[str, set[str]] = {}
        values_by_fact: dict[tuple[str, str], set[str]] = {}
        abnormal_measured = False
        for row in evidence:
            condition = str(row.get("operating_condition") or "unspecified")
            code = str(row.get("code") or row.get("test_id") or "")
            if code:
                values_by_fact.setdefault((condition, code), set()).add(json.dumps(
                    row.get("value", row.get("behavior_status")), sort_keys=True, default=str))
            behavior = str(row.get("behavior") or row.get("behavior_status") or "").upper()
            if not behavior:
                code_upper = str(row.get("code") or "").upper()
                if code_upper.endswith(("_WORKING", "_VERIFIED_NORMAL")):
                    behavior = "WORKING"
                elif code_upper.endswith(("_FAILED", "_ABNORMAL", "_NO_OUTPUT")):
                    behavior = "ABNORMAL"
            if behavior in {"REPORTED_WORKING", "WORKING", "NORMAL"}:
                directions.setdefault(condition, set()).add("WORKING")
            elif behavior in {"REPORTED_ABNORMAL", "ABNORMAL", "FAILED", "NO_OUTPUT"}:
                directions.setdefault(condition, set()).add("ABNORMAL")
                abnormal_measured = abnormal_measured or str(row.get("status")).casefold() == "measured"
        if (any(len(values) > 1 for values in directions.values())
                or any(len(values) > 1 for values in values_by_fact.values())):
            status = "CONFLICTED"
        elif {direction for values in directions.values() for direction in values} == {"WORKING", "ABNORMAL"}:
            status = "PARTIAL_FAILURE"
        elif any("WORKING" in values for values in directions.values()):
            status = "REPORTED_WORKING"
        elif any("ABNORMAL" in values for values in directions.values()):
            status = "MEASUREMENT_SUPPORTED_ABNORMAL" if abnormal_measured else "REPORTED_ABNORMAL"
        else:
            status = "UNKNOWN"
        operating_conditions = list(dict.fromkeys(
            str(row.get("operating_condition") or row.get("measurement_conditions") or "")
            for row in evidence if row.get("operating_condition") or row.get("measurement_conditions")
        ))
        result.append({**block, "behavior_status": status,
                       "behavior_scope": "functional_block", "behavior_evidence": evidence,
                       "operating_conditions": operating_conditions})
    return result
