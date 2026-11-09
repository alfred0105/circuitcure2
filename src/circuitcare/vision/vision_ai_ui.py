"""Streamlit controls for optional local detection, OCR, and human verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from circuitcare.vision.component_detector import (
    CORRECTABLE_CLASSES, add_manual_detection, apply_detection_corrections, detect_components,
)
from circuitcare.vision.detection_postprocess import PCB_MODEL_CLASSES
from circuitcare.vision.detector_backend import DetectorUnavailable, model_file_info
from circuitcare.vision.ocr_backend import OCRUnavailable, run_local_ocr
from circuitcare.vision.reference_matcher import suggest_reference_matches


def default_detector_model_path() -> Path:
    """Return the repository-root model path used by the preparation script."""
    return Path(__file__).resolve().parents[3] / "models" / "pcb_component_detector.onnx"


def render_detection_section(st: Any, image: Any, *, image_sha256: str) -> dict[str, Any]:
    """Render image-scoped inference and return saved predictions for later matching."""
    st.markdown("#### AI 부품 탐지 및 실크스크린 OCR")
    model_default = default_detector_model_path()
    model_path = st.text_input("로컬 ONNX 부품 탐지 모델 경로", value=str(model_default), key=f"vision_ai_model_path_{image_sha256[:12]}")
    model_info = model_file_info(model_path)
    if model_info["status"] == "MODEL_UNAVAILABLE":
        st.warning("MODEL_UNAVAILABLE · 실제 학습 가중치가 없어 AI 탐지는 실행되지 않습니다. 설치/내보내기 절차는 docs/PCB_AI_DETECTION.md를 확인하세요.")
    else:
        st.caption(f"모델: {model_info['model_id']} · SHA-256 `{model_info['sha256'][:16]}…` · 런타임 ONNX Runtime CPU")

    detection_key = f"vision_ai_detections_{image_sha256}_{model_info.get('sha256', 'missing')}"
    ocr_key = f"vision_ai_ocr_{image_sha256}"
    col_detect, col_ocr = st.columns(2)
    if col_detect.button("실제 모델로 부품 탐지", disabled=model_info["status"] != "READY",
                         key=f"vision_ai_detect_btn_{image_sha256[:12]}_{model_info.get('sha256','missing')[:8]}"):
        try:
            st.session_state[detection_key] = detect_components(image, model_path)
        except DetectorUnavailable as exc:
            st.session_state[detection_key] = {"status": "MODEL_UNAVAILABLE", "error": str(exc), "detections": []}
    if col_ocr.button("로컬 OCR 실행", key=f"vision_ai_ocr_btn_{image_sha256[:12]}"):
        try:
            st.session_state[ocr_key] = run_local_ocr(image)
        except OCRUnavailable as exc:
            st.session_state[ocr_key] = {"status": "OCR_UNAVAILABLE", "error": str(exc), "records": []}

    detection_result = st.session_state.get(detection_key, {"status": model_info["status"], "detections": []})
    ocr_result = st.session_state.get(ocr_key, {"status": "OCR_UNAVAILABLE", "records": []})
    if detection_result.get("error"):
        st.error(detection_result["error"])
    detections = detection_result.get("detections", [])
    corrections_key = f"vision_ai_corrections_{image_sha256}_{model_info.get('sha256', 'missing')}"
    corrections = st.session_state.setdefault(corrections_key, {})
    if detections:
        st.caption(f"탐지 후보 {len(detections)}개 · confidence는 모델 점수이며 정답률이나 고장 확률이 아닙니다.")
        for detection in detections:
            score_label = f"{detection['confidence']:.3f}" if detection.get("confidence") is not None else "manual"
            with st.expander(f"{detection['detection_id']} · {detection['class_name']} · {score_label}", expanded=False):
                current = corrections.get(detection["detection_id"], {})
                corrected = st.selectbox("부품 종류 수정", [""] + list(CORRECTABLE_CLASSES),
                                         index=(1 + list(CORRECTABLE_CLASSES).index(current["corrected_class"]) if current.get("corrected_class") in CORRECTABLE_CLASSES else 0),
                                         format_func=lambda value: value or "모델 예측 유지",
                                         key=f"vision_ai_class_{image_sha256[:10]}_{detection['detection_id']}")
                rejected = st.checkbox("오검출 삭제", value=bool(current.get("rejected")),
                                       key=f"vision_ai_reject_{image_sha256[:10]}_{detection['detection_id']}")
                if st.button("수정 저장", key=f"vision_ai_save_{image_sha256[:10]}_{detection['detection_id']}"):
                    current.update({"corrected_class": corrected or None, "rejected": rejected})
                    corrections[detection["detection_id"]] = current
                    st.session_state[corrections_key] = corrections
        detections = apply_detection_corrections(detections, corrections)

    with st.expander("누락 부품 수동 추가", expanded=False):
        add_class = st.selectbox("수동 부품 종류", CORRECTABLE_CLASSES, key=f"vision_ai_add_class_{image_sha256[:10]}")
        width, height = image.size
        columns = st.columns(4)
        x1 = columns[0].number_input("X1 px", 0.0, float(width), 0.0, key=f"vision_ai_x1_{image_sha256[:10]}")
        y1 = columns[1].number_input("Y1 px", 0.0, float(height), 0.0, key=f"vision_ai_y1_{image_sha256[:10]}")
        x2 = columns[2].number_input("X2 px", 0.0, float(width), float(width / 4), key=f"vision_ai_x2_{image_sha256[:10]}")
        y2 = columns[3].number_input("Y2 px", 0.0, float(height), float(height / 4), key=f"vision_ai_y2_{image_sha256[:10]}")
        if st.button("수동 부품 후보 추가", key=f"vision_ai_add_{image_sha256[:10]}"):
            try:
                manual = add_manual_detection(detections, class_name=add_class, bbox_xyxy=[x1, y1, x2, y2], image_size=image.size)
                if detection_key not in st.session_state:
                    st.session_state[detection_key] = {"status": "MANUAL_CANDIDATES", "model": model_info,
                                                       "detections": []}
                st.session_state[detection_key]["detections"].append(manual)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    if ocr_result.get("error"):
        st.warning(ocr_result["error"])
    records = ocr_result.get("records", [])
    if records:
        st.caption(f"OCR 결과 {len(records)}개 · Reference처럼 보이는 문자열은 원문 보존 후보입니다.")
        st.dataframe([{"원문": row["raw_text"], "후보 Reference": ", ".join(row["reference_candidates"]),
                       "OCR 점수": row["confidence"], "상태": row["verification_status"]} for row in records],
                     hide_index=True, use_container_width=True)
    elif ocr_result.get("status") == "OCR_COMPLETE":
        st.info("문자 후보를 찾지 못했습니다. 부품 탐지 결과는 유지됩니다.")

    preview, preview_meta = _draw_boxes(image, detections)
    if detections:
        st.image(preview, caption="AI component candidates · unverified · no electrical risk linked", use_container_width=True)
    bundle = {"status": detection_result.get("status"), "model": detection_result.get("model"),
              "image_sha256": image_sha256, "detections": detections,
              "ocr": {"status": ocr_result.get("status"), "records": records}, "preview": preview_meta,
              "verification_level": "inference_only_not_accuracy_evaluation"}
    st.download_button("부품 탐지·OCR JSON 다운로드", json.dumps(bundle, ensure_ascii=False, indent=2).encode(),
                       file_name="pcb_ai_detection.json", mime="application/json",
                       key=f"vision_ai_download_{image_sha256[:12]}")
    return {**bundle, "detections": detections, "ocr_records": records,
            "corrections": corrections, "detection_key": detection_key}


def render_reference_confirmation(st: Any, ai_state: dict[str, Any], *,
                                  registered_locations: dict[str, Any], image_size: tuple[int, int],
                                  state_suffix: str) -> dict[str, str]:
    """Show ranked candidates and persist explicit, unique, per-detection confirmations."""
    detections = ai_state.get("detections", [])
    if not detections:
        return {"confirmed_references": {}, "suggestions": {}}
    suggestions = suggest_reference_matches(detections, ai_state.get("ocr_records", []),
                                            registered_locations.get("components", []), image_size=image_size)
    st.markdown("#### AI 후보와 KiCad Reference 비교")
    st.caption("점수는 위치·OCR·유형 근거의 정렬 지수이며 확률이 아닙니다. 제안은 자동 승인되지 않으며, 서로 다른 검출은 같은 Reference로 확정할 수 없습니다.")
    refs = [row["reference"] for row in registered_locations.get("components", [])]
    prefix = f"vision_ai_reference_{state_suffix}"
    state = st.session_state.setdefault(prefix, {})
    confirmed: dict[str, str] = {}
    for detection in detections:
        if detection.get("is_rejected"):
            continue
        detection_id = detection["detection_id"]
        candidates = suggestions["assignments"].get(detection_id, [])
        if candidates:
            st.caption(f"{detection_id}: " + " · ".join(
                f"{row['reference']} {row['heuristic_score_0_100']:.1f} ({row['status']})" for row in candidates[:3]))
        current = state.get(detection_id, {})
        unavailable_refs = {row.get("reference") for key, row in state.items()
                            if key != detection_id and row.get("confirmed")}
        choices = [""] + [ref for ref in refs if ref not in unavailable_refs]
        suggested = candidates[0]["reference"] if candidates else ""
        default = current.get("reference") or suggested
        index = choices.index(default) if default in choices else 0
        columns = st.columns([2, 1, 1])
        chosen = columns[0].selectbox(f"{detection_id} Reference", choices, index=index,
                                     key=f"vision_ai_ref_select_{state_suffix}_{detection_id}")
        confirm = columns[1].checkbox("확정", value=bool(current.get("confirmed")),
                                      key=f"vision_ai_ref_confirm_{state_suffix}_{detection_id}")
        if columns[2].button("적용", key=f"vision_ai_ref_apply_{state_suffix}_{detection_id}"):
            if confirm and chosen:
                if chosen in unavailable_refs:
                    st.error(f"{chosen}는 이미 다른 검출에 확정되어 있습니다.")
                else:
                    state[detection_id] = {"reference": chosen, "confirmed": True,
                                           "verification_source": "user_manual_confirmation",
                                           "suggestion_evidence": candidates[:5]}
            elif chosen:
                state[detection_id] = {"reference": chosen, "confirmed": False,
                                       "verification_source": None, "suggestion_evidence": candidates[:5]}
            else:
                state.pop(detection_id, None)
        saved = state.get(detection_id, {})
        if saved.get("confirmed") and saved.get("reference"):
            confirmed[detection_id] = saved["reference"]
    st.session_state[prefix] = state
    return {"confirmed_references": confirmed, "suggestions": suggestions}


def _draw_boxes(image: Any, detections: list[dict[str, Any]]):
    from PIL import ImageDraw, ImageFont

    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    for row in detections:
        x1, y1, x2, y2 = row["bbox_xyxy"]
        draw.rectangle((x1, y1, x2, y2), outline=(20, 150, 215), width=3)
        score = f"{row['confidence']:.2f}" if row.get("confidence") is not None else "manual"
        label = f"{row.get('corrected_class') or row['class_name']} {score}"
        draw.text((x1, max(0, y1 - 13)), label, fill=(255, 255, 255), font=font,
                  stroke_width=2, stroke_fill=(20, 70, 105))
    return output, {"detection_count": len(detections), "drawn_bbox_count": len(detections)}
