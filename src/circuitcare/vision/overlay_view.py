"""Confirmed registration and risk overlay presentation workflow."""

from __future__ import annotations

import json
from pathlib import Path


def render_confirmed_overlay(st, view_context, data, *, png_bytes, safe_result_id) -> None:
    from circuitcare.vision.overlay import (
        RISK_COLORS,
        render_detection_risk_overlay,
        render_risk_overlay,
        risk_by_reference,
    )
    from circuitcare.vision.registration import map_component_positions

    registration = data.registration
    loaded = data.loaded_image
    rectified_image = data.rectified_image
    parsed = data.parsed_positions
    positions = data.positions
    netlist_component_refs = view_context.netlist_component_refs
    board_key = data.board_key
    csv_hash = data.csv_hash
    side = data.board_side
    context_key = view_context.context_key
    failed_component = view_context.failed_component
    fault_type = view_context.fault_type
    current_result_status = view_context.simulation_status
    risk_path = view_context.risk_path
    risk_assessment = view_context.overlay_risk
    inspection_ranking = view_context.inspection_ranking
    selected_component = view_context.selected_component
    detection_record = data.detection_record
    rectified_state = data.rectified_state
    ai_state = data.ai_state
    ai_reference_state = data.ai_reference_state
    ai_reference_confirmations = data.ai_reference_confirmations
    matching = data.matching
    rectified_sha256 = data.rectified_image_sha256
    if view_context.blocked_reason:
        st.warning(f"위험도 색상은 연결하지 않았습니다: {view_context.blocked_reason}")

    try:
        registration["status"] = "user_confirmed"
        registration["image_sha256"] = loaded.sha256
        registration["image_size_px"] = [rectified_image.width, rectified_image.height]
        registration["kicad_csv_sha256"] = csv_hash
        registration["position_source"] = parsed["source"]
        registration["board_identity_user_assertion"] = data.board_identity
        registration["board_side"] = side
        circuit_positions = [row for row in positions if row.reference in netlist_component_refs]
        locations = map_component_positions(circuit_positions, registration["homography"], board_side=side,
                                             image_size=(rectified_image.width, rectified_image.height), confirmed=True)
        inspection_by_ref = {row["component_ref"]: row for row in (inspection_ranking or {}).get("posterior_weighted_inspection_order", [])}
        for location in locations["components"]:
            priority = inspection_by_ref.get(location["reference"])
            location["inspection_priority"] = ({"rank": priority["priority_rank"],
                                                  "score_not_probability": priority["posterior_weighted_priority_score"],
                                                  "source": "bayesian_what_if_conditional_rank"} if priority else None)
        locations["netlist_reference_match"] = matching
        locations["scenario_context"] = {"context_key": context_key, "failed_component": failed_component,
                                          "fault_type": fault_type, "simulation_result_status": current_result_status,
                                          "risk_source": risk_path}
    except ValueError as exc:
        st.error(f"부품 좌표 변환에 실패했습니다: {exc}")
        return
    current_risks, risk_issue = risk_by_reference(risk_assessment, expected_fault=failed_component,
                                                  expected_fault_type=fault_type)
    inspection_by_ref = {row["component_ref"]: row for row in (inspection_ranking or {}).get("posterior_weighted_inspection_order", [])}
    if current_result_status not in {"SUCCESS", "CACHED_SIMULATION"}:
        risk_issue = "현재 조건에 성공 또는 저장된 위험 결과가 없어 위치만 표시합니다."
        current_risks = {}
    location_by_ref = {record["reference"] for record in locations["components"]}
    if locations["outside_image"]:
        st.warning("정합 좌표가 보정 이미지 밖에 있어 표시하지 않은 부품: " +
                   ", ".join(record["reference"] for record in locations["outside_image"]))
    if not location_by_ref:
        st.error("현재 이미지 범위 안에 표시할 매칭 부품이 없습니다. 사진 면, 좌표 원점과 기준점을 확인하세요.")
        return
    st.caption(f"확정 정합 · {len(location_by_ref)}개 부품 중심이 이미지 안에 매핑됨 · 위험 결과: {'연결' if current_risks and not risk_issue else '미연결'} · {current_result_status}")
    if risk_issue:
        st.warning(f"위험도 색상은 연결하지 않았습니다: {risk_issue}")
    ordered_refs = sorted(location_by_ref)
    selected_index = ordered_refs.index(selected_component) if selected_component in ordered_refs else 0
    selected_reference = st.selectbox("상세 정보 부품", ordered_refs, index=selected_index,
                                      key=f"vision_detail_{board_key}_{csv_hash[:8]}_{side}_{selected_component or 'none'}")
    show_risk = st.checkbox("현재 분석의 위험도 표시", value=bool(current_risks), disabled=not bool(current_risks), key=f"vision_show_risk_{board_key}_{csv_hash[:8]}_{side}")
    overlay, overlay_meta = render_risk_overlay(rectified_image, locations, risk_assessment if current_risks else None,
                                                expected_fault=failed_component, expected_fault_type=fault_type,
                                                show_risk=show_risk, selected_reference=selected_reference,
                                                inspection_priority_by_reference=inspection_by_ref)
    if overlay_meta["without_risk_result"] and show_risk:
        st.caption("위치만 있고 현재 위험 보고서에 없는 Reference: " +
                   ", ".join(item["reference"] for item in overlay_meta["without_risk_result"]))
    left, right = st.columns(2)
    with left:
        st.image(rectified_image, caption="원근 보정 이미지", use_container_width=True)
    with right:
        st.image(overlay, caption="위험 오버레이 · 부품 중심점만 표시", use_container_width=True)
    if ai_state.get("detections"):
        detector_overlay, detector_meta = render_detection_risk_overlay(
            rectified_image, ai_state["detections"], confirmed_references=ai_reference_confirmations,
            risk_assessment=risk_assessment if current_risks else None,
            expected_fault=failed_component, expected_fault_type=fault_type, show_risk=show_risk,
            inspection_priority_by_reference=inspection_by_ref,
            selected_reference=selected_reference,
        )
        st.image(detector_overlay, caption="AI 검출 박스 · Reference를 개별 확인한 경우에만 현재 시나리오 위험 색상 표시", use_container_width=True)
        st.caption(f"AI 검출 {len(ai_state['detections'])}개 · 확인 Reference {detector_meta['confirmed_reference_count']}개 · 위험 결과 연결 {detector_meta['risk_mapped_count']}개 · {detector_meta['risk_issue'] or '현재 시나리오 위험 근거'}")
    legend = [{"상태": name, "표시": f"RGB{color}"} for name, color in RISK_COLORS.items()]
    st.dataframe(legend, use_container_width=True, hide_index=True)
    risk_map, _ = risk_by_reference(risk_assessment if current_risks else None,
                                    expected_fault=failed_component, expected_fault_type=fault_type)
    detail = next(record for record in locations["components"] if record["reference"] == selected_reference)
    detail_risk = risk_map.get(selected_reference)
    st.markdown(f"#### {selected_reference} 상세 근거")
    if detail_risk:
        metrics = detail_risk.get("metrics", {})
        rating_note = next((metric.get("rating_source", {}).get("note") for metric in metrics.values()
                            if metric.get("rating_source", {}).get("note")), None)
        selected_risk_detail = {
            "component": selected_reference,
            "simulation_stress_and_rating_metrics": metrics,
            "risk_level": detail_risk.get("risk_level"),
            "inspection_priority_rank": detail_risk.get("priority_rank"),
            "inspection_priority_score_not_probability": detail_risk.get("priority_score"),
            "reason": detail_risk.get("reason"),
            "damage_confirmed": detail_risk.get("damage_confirmed"),
            "rating_source_note": rating_note,
            "registered_position": detail,
            "source_risk_file": risk_path,
        }
        st.json(selected_risk_detail)
    else:
        st.info("현재 위험 결과에 이 Reference가 없습니다. 위치 정보만 표시합니다.")
        st.json(detail)
    st.caption("마커는 KiCad footprint 중심 위치입니다. 위험 표시는 계산된 스트레스·설정 정격 기반이며 물리적 손상 확정이나 사진 기반 검출이 아닙니다.")

    png = png_bytes(overlay)
    st.download_button("오버레이 PNG 다운로드", png, file_name="pcb_risk_overlay.png", mime="image/png")
    registration_record = {**registration, "status": "user_confirmed"}
    board_record = {**detection_record, "perspective": {key: value for key, value in rectified_state.items() if key != "image_bytes"},
                    "board_side": side, "kicad_csv_sha256": csv_hash}
    result_bundle = {"board_detection": board_record, "registration": registration_record,
                     "component_locations": locations,
                     "overlay": {"image_file": "pcb_risk_overlay.png", "risk_source_file": risk_path,
                                 "scenario_id": risk_assessment.get("scenario_id") if risk_assessment else None,
                                 "metadata": overlay_meta}}
    if ai_state.get("detections"):
        suggestions = ai_reference_state["suggestions"].get("assignments", {})
        result_bundle["ai_component_detection"] = {
            "model": ai_state.get("model"), "runtime": "onnxruntime-cpu",
            "image_sha256": rectified_sha256, "ocr": {"status": ai_state.get("ocr", {}).get("status"),
                                                           "records": ai_state.get("ocr_records", [])},
            "score_kind": ai_reference_state["suggestions"].get("score_kind"),
            "detections": [{**row, "confirmed_reference": ai_reference_confirmations.get(row["detection_id"]),
                            "reference_status": "user_confirmed" if row["detection_id"] in ai_reference_confirmations else "suggested_or_unmatched",
                            "reference_suggestions": suggestions.get(row["detection_id"], [])}
                           for row in ai_state["detections"]],
            "verification_level": "model_inference_performed_reference_requires_user_confirmation",
        }
    final_json = json.dumps(result_bundle, ensure_ascii=False, indent=2).encode()
    st.download_button("PCB 정합·AI 매칭 결과 JSON 다운로드", final_json,
                       file_name="pcb_vision_ai_result.json", mime="application/json",
                       key=f"vision_ai_final_download_{board_key}_{csv_hash[:8]}_{side}")
    st.download_button("Vision 결과 JSON 다운로드", json.dumps(result_bundle, ensure_ascii=False, indent=2).encode(),
                       file_name="pcb_vision_result.json", mime="application/json")
    if st.button("Vision 결과를 로컬 data/vision_results에 저장", key=f"vision_save_{board_key}_{csv_hash[:8]}_{side}"):
        root = Path(__file__).resolve().parents[3]
        save_context = f"{context_key}:{board_key}:{registration['homography']}"
        folder = root / "data" / "vision_results" / safe_result_id(loaded.sha256, csv_hash, side, save_context)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "board_detection.json").write_text(json.dumps(board_record, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "registration.json").write_text(json.dumps(registration_record, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "component_locations.json").write_text(json.dumps(locations, ensure_ascii=False, indent=2), encoding="utf-8")
        if result_bundle.get("ai_component_detection"):
            (folder / "ai_component_detection.json").write_text(
                json.dumps(result_bundle["ai_component_detection"], ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "pcb_risk_overlay.png").write_bytes(png)
        st.success(f"원본 사진을 제외한 결과를 `{folder.relative_to(root)}`에 저장했습니다.")

