"""Evidence-only component-center and risk overlay renderer."""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw, ImageFont

RISK_COLORS = {
    "EXCEEDED": (220, 38, 38),
    "NEAR_LIMIT": (245, 130, 32),
    "WITHIN_LIMIT": (70, 120, 165),
    "UNKNOWN": (130, 140, 150),
    "KNOWN_FAILED": (150, 45, 175),
}


def risk_by_reference(risk_assessment: dict[str, Any] | None, *, expected_fault: str | None,
                      expected_fault_type: str | None) -> tuple[dict[str, dict[str, Any]], str | None]:
    if not risk_assessment:
        return {}, "현재 회로/고장 조건에 사용할 위험 평가 결과가 없습니다."
    if risk_assessment.get("simulation_status") != "success":
        return {}, "위험 평가가 성공한 시뮬레이션 결과가 아닙니다."
    if expected_fault and risk_assessment.get("failed_component") != expected_fault:
        return {}, "위험 보고서의 고장 부품이 현재 선택과 일치하지 않습니다."
    if expected_fault_type and risk_assessment.get("fault_type") != expected_fault_type:
        return {}, "위험 보고서의 고장 유형이 현재 선택과 일치하지 않습니다."
    result = {}
    for row in risk_assessment.get("secondary_candidates", []):
        ref = row.get("component_ref")
        if ref:
            result[ref] = dict(row)
    failed = risk_assessment.get("failed_component")
    if failed:
        result[failed] = {"component_ref": failed, "risk_level": "KNOWN_FAILED", "priority_rank": 0,
                          "reason": "사용자가 현재 분석의 고장 부품으로 지정했습니다.", "damage_confirmed": False}
    return result, None


def render_risk_overlay(image: Image.Image, component_locations: dict[str, Any],
                        risk_assessment: dict[str, Any] | None = None, *,
                        expected_fault: str | None = None,
                        expected_fault_type: str | None = None,
                        show_risk: bool = True,
                        selected_reference: str | None = None,
                        inspection_priority_by_reference: dict[str, dict[str, Any]] | None = None) -> tuple[Image.Image, dict[str, Any]]:
    records = component_locations.get("components", [])
    risks, risk_issue = risk_by_reference(
        risk_assessment, expected_fault=expected_fault, expected_fault_type=expected_fault_type,
    ) if show_risk else ({}, None)
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    matched, unmatched = [], []
    inspection_priority_by_reference = inspection_priority_by_reference or {}
    for location in records:
        ref = location["reference"]
        risk = risks.get(ref)
        u, v = map(float, location["image_position_px"])
        radius = 13 if ref == selected_reference else 9
        color = RISK_COLORS.get(risk.get("risk_level"), RISK_COLORS["UNKNOWN"]) if risk else RISK_COLORS["UNKNOWN"]
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=color, outline=(255, 255, 255), width=3)
        priority = inspection_priority_by_reference.get(ref)
        label = f"{ref} · inspect #{priority['priority_rank']}" if priority and priority.get("priority_rank") else ref
        if risk and risk.get("priority_rank") is not None and risk.get("priority_rank") != 0:
            label += f" #{risk['priority_rank']}"
        box = draw.textbbox((u + radius + 3, v - radius), label, font=font, stroke_width=1)
        draw.rectangle((box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1), fill=(20, 25, 32))
        draw.text((u + radius + 3, v - radius), label, fill=(255, 255, 255), font=font, stroke_width=1)
        if risk:
            matched.append({**location, "risk": risk, "inspection_priority": priority})
        else:
            unmatched.append({**location, "inspection_priority": priority})
    return output, {"matched": matched, "without_risk_result": unmatched, "risk_issue": risk_issue,
                    "risk_overlay_enabled": show_risk and risk_issue is None}


def render_detection_risk_overlay(image: Image.Image, detections: list[dict[str, Any]], *,
                                  confirmed_references: dict[str, str],
                                  risk_assessment: dict[str, Any] | None = None,
                                  expected_fault: str | None = None,
                                  expected_fault_type: str | None = None,
                                  show_risk: bool = True,
                                  inspection_priority_by_reference: dict[str, dict[str, Any]] | None = None,
                                  selected_reference: str | None = None) -> tuple[Image.Image, dict[str, Any]]:
    """Draw model boxes; only user-confirmed Reference assignments can carry risk."""
    risks, risk_issue = risk_by_reference(
        risk_assessment, expected_fault=expected_fault, expected_fault_type=expected_fault_type,
    ) if show_risk else ({}, None)
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    mapped, unmatched = [], []
    inspection_priority_by_reference = inspection_priority_by_reference or {}
    for detection in detections:
        detection_id = detection["detection_id"]
        reference = confirmed_references.get(detection_id)
        risk = risks.get(reference) if reference else None
        inspection_priority = inspection_priority_by_reference.get(reference) if reference else None
        color = RISK_COLORS.get(risk.get("risk_level"), RISK_COLORS["UNKNOWN"]) if risk else (80, 170, 205)
        x1, y1, x2, y2 = (float(value) for value in detection["bbox_xyxy"])
        is_selected = reference == selected_reference
        draw.rectangle((x1, y1, x2, y2), outline=color, width=5 if is_selected else 4 if risk else 2)
        confidence = detection.get("confidence")
        score_label = f"{float(confidence):.2f}" if confidence is not None else "manual"
        label = f"{detection.get('corrected_class') or detection['class_name']} {score_label}"
        if reference:
            label += f" · {reference}"
        if risk:
            label += f" · {risk.get('risk_level', 'UNKNOWN')}"
        if inspection_priority and inspection_priority.get("priority_rank"):
            label += f" · inspect #{inspection_priority['priority_rank']}"
        if is_selected:
            label = "▶ " + label
        label_box = draw.textbbox((x1, max(0, y1 - 15)), label, font=font)
        draw.rectangle(label_box, fill=color)
        draw.text((x1, max(0, y1 - 15)), label, fill=(255, 255, 255), font=font)
        record = {**detection, "confirmed_reference": reference,
                  "reference_status": "user_confirmed" if reference else "unmatched",
                  "risk": risk if risk else None,
                  "inspection_priority": inspection_priority}
        (mapped if reference and risk else unmatched).append(record)
    return output, {"confirmed_reference_count": sum(bool(value) for value in confirmed_references.values()),
                    "risk_mapped_count": len(mapped), "unmatched_or_no_risk_count": len(unmatched),
                    "inspection_priority_count": sum(bool(item.get("inspection_priority")) for item in (*mapped, *unmatched)),
                    "risk_issue": risk_issue, "risk_overlay_enabled": show_risk and risk_issue is None,
                    "source": "current_scenario_risk_report_only"}
