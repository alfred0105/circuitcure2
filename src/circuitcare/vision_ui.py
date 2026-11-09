"""Optional Streamlit workflow for aligning PCB photographs to KiCad data."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import re
from typing import Any

from circuitcare.vision.ui_context import ConfirmedOverlayContext, VisionViewContext


def render_vision_tab(st: Any, *, netlist_component_refs: list[str], risk_assessment: dict | None,
                      risk_path: str, failed_component: str, fault_type: str,
                      context_key: str, current_result_status: str,
                      inspection_ranking: dict | None = None,
                      selected_component: str | None = None,
                      result_integrity: str = "PROVENANCE_UNVERIFIED") -> None:
    view_context = VisionViewContext(
        context_key=context_key,
        netlist_component_refs=tuple(netlist_component_refs),
        risk_assessment=risk_assessment,
        risk_path=risk_path,
        failed_component=failed_component,
        fault_type=fault_type,
        simulation_status=current_result_status,
        result_integrity=result_integrity,
        inspection_ranking=inspection_ranking,
        selected_component=selected_component,
    )
    risk_assessment = view_context.overlay_risk
    st.subheader("PCB 사진 인식")
    st.caption("선택 설치된 로컬 학습 모델은 사진의 부품 후보를 탐지하고, OCR은 실크스크린 Reference 후보를 읽습니다. 사진은 회로 연결이나 손상을 판정하지 않습니다. 위험 표시는 같은 회로임을 확인하고 부품 Reference까지 승인한 뒤에만 연결합니다.")
    try:
        from circuitcare.vision.board_detector import (
            detect_board_candidates,
            validate_corners,
        )
        from circuitcare.vision.image_loader import ImageLoadError, load_board_image
        from circuitcare.vision.pcb_positions import (
            ComponentPosition,
            match_netlist_references,
            parse_position_csv,
        )
        from circuitcare.vision.perspective import rectify_board
        from circuitcare.vision.registration import fit_registration
    except ImportError as exc:
        st.error(f"Vision 선택 의존성을 설치해야 합니다: {exc}. `python -m pip install -e '.[vision,ui]'` 실행 후 앱을 다시 시작하세요.")
        return

    source_mode = st.radio("PCB 사진 입력", ["사진 업로드", "카메라 촬영"], horizontal=True,
                           key=f"vision_source_{context_key}")
    st.caption("개별 파일 20 MB 및 2,500만 픽셀 이하 · 파일명은 저장 경로에 사용하지 않으며, 원본은 자동 저장/전송하지 않습니다.")
    upload_limit = {"max_upload_size": 20} if "max_upload_size" in inspect.signature(st.file_uploader).parameters else {}
    camera_limit = {"max_upload_size": 20} if "max_upload_size" in inspect.signature(st.camera_input).parameters else {}
    uploaded = (st.file_uploader("PCB 사진 업로드", type=["jpg", "jpeg", "png"],
                                 key=f"vision_upload_{context_key}", **upload_limit) if source_mode == "사진 업로드"
                else st.camera_input("PCB 촬영", key=f"vision_camera_{context_key}", **camera_limit))
    if uploaded is None:
        st.info("JPG/JPEG/PNG 이미지를 선택하거나 카메라로 촬영하세요. 처리와 분석은 이 컴퓨터 안에서 실행됩니다.")
        return
    raw_image = uploaded.getvalue()
    try:
        loaded = load_board_image(raw_image)
    except ImageLoadError as exc:
        st.error(str(exc))
        return
    image_key = loaded.sha256[:12]
    previous_image = st.session_state.get("vision_active_image")
    if previous_image and previous_image != image_key:
        heavy_prefixes = ("vision_rectified_", "vision_ai_detections_", "vision_ai_ocr_",
                          "vision_ai_corrections_", "vision_registration_", "vision_locations_",
                          "vision_overlay_")
        for state_key in list(st.session_state):
            if state_key.startswith(heavy_prefixes) and previous_image in state_key:
                st.session_state.pop(state_key, None)
    st.session_state["vision_active_image"] = image_key
    st.image(loaded.image, caption=f"{loaded.format} · 원본 {loaded.original_size[0]}×{loaded.original_size[1]} px · 처리 {loaded.processed_size[0]}×{loaded.processed_size[1]} px · EXIF 방향 보정 · RGB", use_container_width=True)
    st.caption(f"이미지 SHA-256: `{loaded.sha256}` · 좌표 축소율 x={loaded.scale_xy[0]:.5f}, y={loaded.scale_xy[1]:.5f}")

    side = st.radio("PCB 면", ["front", "back"], horizontal=True, format_func=lambda value: "앞면 (front)" if value == "front" else "뒷면 (back)",
                    key=f"vision_side_{context_key}_{image_key}")
    detection = detect_board_candidates(loaded.image)
    if detection["board_detected"]:
        st.info(f"기판 외곽 후보 {len(detection['candidates'])}개를 찾았습니다. 자동 검출은 후보이며 모서리를 확인하거나 직접 수정하세요.")
        candidate_index = st.selectbox("자동 외곽 후보", list(range(len(detection["candidates"]))),
                                       format_func=lambda index: f"후보 {index + 1} · 면적 {detection['candidates'][index]['area_ratio']:.1%}",
                                       key=f"vision_candidate_{image_key}")
        default_corners = detection["candidates"][candidate_index]["corners_px"]
        st.image(_draw_candidate(loaded.image, default_corners), caption="자동 외곽 후보 미리보기 · 아직 확정되지 않음", use_container_width=True)
        board_key = f"{image_key}_candidate_{candidate_index}"
    else:
        st.warning("자동 기판 외곽 후보를 찾지 못했습니다. 사진 좌표를 직접 입력해 영역을 지정할 수 있습니다.")
        width, height = loaded.processed_size
        default_corners = [[width * .08, height * .08], [width * .92, height * .08],
                           [width * .92, height * .92], [width * .08, height * .92]]
        board_key = f"{image_key}_manual"
    corner_key = f"vision_corners_{board_key}"
    stored_corners = st.session_state.get(corner_key, default_corners)
    st.caption("모서리 좌표는 현재 처리 이미지의 픽셀 단위입니다. 꼭짓점을 시계 방향으로 지정하세요.")
    with st.form(f"vision_board_form_{board_key}"):
        corner_cols = st.columns(4)
        edited_corners = []
        for index, (x_value, y_value) in enumerate(stored_corners):
            with corner_cols[index]:
                st.markdown(f"**모서리 {index + 1}**")
                x = st.number_input(f"X {index + 1} px", min_value=0.0, max_value=float(loaded.processed_size[0] - 1), value=min(float(x_value), float(loaded.processed_size[0] - 1)), key=f"vision_corner_x_{board_key}_{index}")
                y = st.number_input(f"Y {index + 1} px", min_value=0.0, max_value=float(loaded.processed_size[1] - 1), value=min(float(y_value), float(loaded.processed_size[1] - 1)), key=f"vision_corner_y_{board_key}_{index}")
                edited_corners.append([x, y])
        physical_enabled = st.checkbox("실제 보드 크기를 알고 있습니다", key=f"vision_phys_enabled_{board_key}")
        physical_size = None
        if physical_enabled:
            physical_cols = st.columns(2)
            physical_width = physical_cols[0].number_input("보드 폭 (mm)", min_value=0.1, value=50.0, key=f"vision_phys_w_{board_key}")
            physical_height = physical_cols[1].number_input("보드 높이 (mm)", min_value=0.1, value=30.0, key=f"vision_phys_h_{board_key}")
            physical_size = (physical_width, physical_height)
        apply_corners = st.form_submit_button("기판 영역 확정 및 원근 보정")
    if apply_corners:
        try:
            ordered = validate_corners(edited_corners, loaded.processed_size)
            st.session_state[corner_key] = ordered.tolist()
            rectified = rectify_board(loaded.image, ordered, physical_size_mm=physical_size)
            rectified_key = f"vision_rectified_{board_key}"
            st.session_state[rectified_key] = {
                "image_bytes": _png_bytes(rectified["image"]),
                "transform_matrix": rectified["transform_matrix"],
                "corners_px": rectified["corners_px"],
                "output_size_px": rectified["output_size_px"],
                "physical_size_mm": rectified["physical_size_mm"],
                "aspect_source": rectified["aspect_source"],
                "status": "user_confirmed_board_region",
            }
            for prefix in ("vision_registration_", "vision_confirmed_", "vision_locations_", "vision_overlay_",
                           "vision_identity_", "vision_user_confirmed_", "vision_board_identity_",
                           "vision_confirm_button_", "vision_show_risk_", "vision_detail_"):
                for state_key in list(st.session_state):
                    if state_key.startswith(f"{prefix}{board_key}_"):
                        st.session_state.pop(state_key, None)
            st.rerun()
        except (ValueError, RuntimeError) as exc:
            st.error(f"기판 영역을 적용할 수 없습니다: {exc}")

    detection_record = {
        **detection,
        "image_sha256": loaded.sha256,
        "image_size_original_px": list(loaded.original_size),
        "image_size_processed_px": list(loaded.processed_size),
        "image_scale_xy": list(loaded.scale_xy),
        "corners_source": "opencv_candidate_user_confirmed" if detection["board_detected"] else "manual_corner_input",
        "selected_corners_px": st.session_state.get(corner_key),
        "status": "user_confirmed_board_region" if st.session_state.get(f"vision_rectified_{board_key}") else detection["status"],
    }
    rectified_state = st.session_state.get(f"vision_rectified_{board_key}")
    if not rectified_state:
        return
    rectified_image = _image_from_png(rectified_state["image_bytes"])
    st.image(rectified_image, caption=f"확정된 기판 영역 · 원근 보정 미리보기 {rectified_image.width}×{rectified_image.height} px · 실제 mm 비율: {'반영' if rectified_state['physical_size_mm'] else '미상, 촬영 모서리로 비율 근사'}", use_container_width=True)

    from circuitcare.vision.vision_ai_ui import render_detection_section
    rectified_sha256 = hashlib.sha256(rectified_state["image_bytes"]).hexdigest()
    ai_state = render_detection_section(st, rectified_image, image_sha256=rectified_sha256)

    st.markdown("#### KiCad 부품 위치 CSV")
    pos_file = st.file_uploader("`kicad-cli pcb export pos`로 만든 CSV", type=["csv"], key=f"vision_pos_{context_key}_{image_key}_{side}")
    if pos_file is None:
        st.info("CSV가 없으면 사진 보정까지 가능하고, 부품 위치 및 위험 오버레이는 사용할 수 없습니다.")
        return
    csv_bytes = pos_file.getvalue()
    csv_hash = hashlib.sha256(csv_bytes).hexdigest()
    with st.form(f"vision_csv_settings_{csv_hash[:12]}_{side}"):
        csv_cols = st.columns(3)
        units = csv_cols[0].selectbox("CSV 좌표 단위", ["mm", "in"], help="KiCad 내보내기 때 지정한 단위를 선택하세요. CLI 기본 단위는 inch입니다.", key=f"vision_units_{csv_hash[:12]}")
        csv_side = csv_cols[1].selectbox("CSV에 포함된 면", ["both", "front", "back"], key=f"vision_csv_side_{csv_hash[:12]}")
        origin = csv_cols[2].selectbox("KiCad 좌표 원점", ["absolute origin", "drill/place origin", "user origin / other"], key=f"vision_origin_{csv_hash[:12]}")
        origin_note = st.text_input("원점 참고 (예: KiCad export option)", value=origin, key=f"vision_origin_note_{csv_hash[:12]}")
        parse_clicked = st.form_submit_button("CSV 읽기")
    parsed_key = f"vision_positions_{csv_hash}_{side}"
    if parse_clicked:
        try:
            rows, source_info = parse_position_csv(csv_bytes.decode("utf-8-sig"), units=units, board_side=csv_side, origin_description=origin_note)
            source_info.update({"sha256": csv_hash, "selected_board_side": side})
            st.session_state[parsed_key] = {"rows": [row.to_dict() for row in rows], "source": source_info}
            for suffix in ("registration", "confirmed", "locations", "overlay"):
                st.session_state.pop(f"vision_{suffix}_{image_key}_{csv_hash}_{side}", None)
            st.rerun()
        except (UnicodeDecodeError, ValueError) as exc:
            st.error(f"Position CSV를 읽지 못했습니다: {exc}")
    parsed = st.session_state.get(parsed_key)
    if not parsed:
        st.info("단위, 내보낸 PCB 면과 원점 정보를 확인한 뒤 CSV 읽기를 누르세요.")
        return
    positions = [ComponentPosition(**row) for row in parsed["rows"]]
    matching = match_netlist_references(positions, netlist_component_refs)
    st.caption(f"KiCad 위치 {len(positions)}개 · 넷리스트 매칭 {len(matching['matched'])}개 · CSV 해시 `{csv_hash[:12]}` · 원점: {parsed['source']['origin_description']}")
    st.dataframe([row.to_dict() for row in positions], use_container_width=True, hide_index=True)
    if matching["in_netlist_without_position"]:
        st.warning("넷리스트에는 있으나 선택한 면 CSV에 없는 부품: " + ", ".join(matching["in_netlist_without_position"]))
    if matching["position_without_netlist_component"]:
        st.warning("위치 CSV에는 있으나 현재 넷리스트에 없는 Reference: " + ", ".join(matching["position_without_netlist_component"]))

    side_positions = [row for row in positions if row.side == side and row.reference in netlist_component_refs]
    if len(side_positions) < 4:
        st.warning("사진과 설계 좌표를 정합하려면 현재 면·회로에 공통인 위치가 4개 이상 필요합니다.")
        return
    registration_key = f"vision_registration_{board_key}_{csv_hash}_{side}"
    confirmed_key = f"vision_confirmed_{board_key}_{csv_hash}_{side}"
    st.markdown("#### 사진–설계 좌표 대응점")
    st.caption("보정된 큰 사진에서 기준 부품의 중심을 읽고 좌표를 입력합니다. 클릭 좌표 도구가 설치되지 않은 환경에서도 쓸 수 있도록 숫자 입력을 제공합니다. 4개 이상을 보드 전체에 넓게 배치하세요.")
    refs = [row.reference for row in side_positions]
    with st.form(f"vision_registration_form_{board_key}_{csv_hash[:12]}_{side}"):
        selected_refs = []
        image_points = []
        for index in range(4):
            cols = st.columns([1, 1, 1])
            ref = cols[0].selectbox(f"기준점 {index + 1} Reference", refs, index=min(index, len(refs) - 1), key=f"vision_ref_{board_key}_{csv_hash[:8]}_{side}_{index}")
            u = cols[1].number_input(f"사진 X {index + 1} px", min_value=0.0, max_value=float(rectified_image.width - 1), value=float(rectified_image.width * (0.2 + index * 0.18)), key=f"vision_u_{board_key}_{csv_hash[:8]}_{side}_{index}")
            v = cols[2].number_input(f"사진 Y {index + 1} px", min_value=0.0, max_value=float(rectified_image.height - 1), value=float(rectified_image.height * (0.25 + (index % 2) * 0.4)), key=f"vision_v_{board_key}_{csv_hash[:8]}_{side}_{index}")
            selected_refs.append(ref)
            image_points.append((u, v))
        add_validation = st.checkbox("독립 확인점 1개 입력 (정합 계산에는 사용하지 않음)", key=f"vision_add_validation_{board_key}_{csv_hash[:8]}_{side}")
        validation_ref = validation_u = validation_v = None
        if add_validation:
            remaining = [ref for ref in refs if ref not in selected_refs]
            if remaining:
                val_cols = st.columns(3)
                validation_ref = val_cols[0].selectbox("검증 Reference", remaining, key=f"vision_validation_ref_{board_key}_{csv_hash[:8]}_{side}")
                validation_u = val_cols[1].number_input("검증 사진 X px", min_value=0.0, max_value=float(rectified_image.width - 1), value=float(rectified_image.width / 2), key=f"vision_validation_u_{board_key}_{csv_hash[:8]}_{side}")
                validation_v = val_cols[2].number_input("검증 사진 Y px", min_value=0.0, max_value=float(rectified_image.height - 1), value=float(rectified_image.height / 2), key=f"vision_validation_v_{board_key}_{csv_hash[:8]}_{side}")
            else:
                st.warning("독립 검증용으로 선택하지 않은 Reference가 없습니다.")
        registration_clicked = st.form_submit_button("Homography 정합 계산")
    if registration_clicked:
        if len(set(selected_refs)) != 4:
            st.error("서로 다른 네 개의 Reference를 선택하세요.")
        else:
            position_map = {row.reference: row for row in side_positions}
            correspondences = []
            for ref, (u, v) in zip(selected_refs, image_points):
                row = position_map[ref]
                correspondences.append({"reference": ref, "x_mm": row.x_mm, "y_mm": row.y_mm, "u_px": u, "v_px": v})
            validation = []
            if validation_ref:
                row = position_map[validation_ref]
                validation.append({"reference": validation_ref, "x_mm": row.x_mm, "y_mm": row.y_mm, "u_px": validation_u, "v_px": validation_v})
            try:
                result = fit_registration(correspondences, board_side=side, validation_points=validation)
                st.session_state[registration_key] = result.to_dict()
                st.session_state.pop(confirmed_key, None)
                st.rerun()
            except ValueError as exc:
                st.error(f"정합을 계산할 수 없습니다: {exc}")
    registration = st.session_state.get(registration_key)
    if not registration:
        return
    st.info(f"정합 상태: pending_verification · 기준점 재투영 RMSE {registration['reprojection_error_px']:.3f} px · 이 값만으로 정합 정확도를 보증하지 않습니다.")
    if registration.get("validation_points"):
        st.dataframe(registration["validation_points"], use_container_width=True, hide_index=True)
    preview_positions = _preview_locations(positions, registration["homography"], side, (rectified_image.width, rectified_image.height), netlist_component_refs)
    neutral = _draw_pending_locations(rectified_image, preview_positions)
    st.image(neutral, caption="미확정 정합 미리보기 · 중립 중심점만 표시 · 위험도 비활성", use_container_width=True)
    st.caption("기준점 표시가 실제 부품 중심과 맞는지 사진에서 확인하세요. 기준점 4개일 때 계산 오차는 0에 가까울 수 있어 독립 검증점이 더 유용합니다.")
    from circuitcare.vision.vision_ai_ui import render_reference_confirmation
    ai_reference_state = render_reference_confirmation(
        st, ai_state, registered_locations=preview_positions,
        image_size=(rectified_image.width, rectified_image.height),
        state_suffix=f"{context_key}_{board_key}_{csv_hash[:10]}_{side}_{rectified_sha256[:10]}_"
                     f"{(ai_state.get('model') or {}).get('sha256', 'missing')[:10]}_"
                     f"{hashlib.sha256(json.dumps(registration['homography'], sort_keys=True).encode()).hexdigest()[:10]}",
    )
    ai_reference_confirmations = ai_reference_state["confirmed_references"]
    st.warning("Reference 이름이 겹친다는 사실만으로 사진·Position CSV·선택 회로가 같은 PCB라고 증명되지 않습니다. 아래에서 실제 같은 KiCad 보드/회로인지 직접 확인해야 위험도를 연결합니다.")
    board_identity = st.text_input("KiCad 프로젝트/보드 이름 및 Revision", key=f"vision_board_identity_{board_key}_{csv_hash[:8]}_{side}")
    user_confirmed = st.checkbox("사진 기준점과 부품 중심이 맞고, 이 사진·CSV가 현재 선택 회로와 같은 실제 PCB 설계임을 확인했습니다", key=f"vision_user_confirmed_{board_key}_{csv_hash[:8]}_{side}")
    if st.button("정합 확정 및 위험 오버레이 활성화", disabled=not (user_confirmed and board_identity.strip()), key=f"vision_confirm_button_{board_key}_{csv_hash[:8]}_{side}"):
        st.session_state[confirmed_key] = True
        st.session_state[f"vision_identity_{board_key}_{csv_hash}_{side}"] = board_identity.strip()
        st.rerun()
    identity_key = f"vision_identity_{board_key}_{csv_hash}_{side}"
    confirmation_active = (st.session_state.get(confirmed_key, False)
                           and st.session_state.get(identity_key) == board_identity.strip()
                           and user_confirmed and bool(board_identity.strip()))
    if not confirmation_active:
        st.session_state.pop(confirmed_key, None)
        return

    from circuitcare.vision.overlay_view import render_confirmed_overlay
    render_confirmed_overlay(
        st,
        view_context,
        ConfirmedOverlayContext(
            registration=registration,
            loaded_image=loaded,
            rectified_image=rectified_image,
            parsed_positions=parsed,
            positions=positions,
            board_key=board_key,
            csv_hash=csv_hash,
            board_side=side,
            board_identity=board_identity.strip(),
            detection_record=detection_record,
            rectified_state=rectified_state,
            ai_state=ai_state,
            ai_reference_state=ai_reference_state,
            ai_reference_confirmations=ai_reference_confirmations,
            matching=matching,
            rectified_image_sha256=rectified_sha256,
        ),
        png_bytes=_png_bytes,
        safe_result_id=_safe_result_id,
    )


def _preview_locations(positions: list[Any], homography: list[list[float]], side: str,
                       image_size: tuple[int, int], netlist_refs: list[str]) -> dict:
    import cv2
    import numpy as np

    eligible = [item for item in positions if item.side == side and item.reference in netlist_refs]
    points = np.asarray([[item.x_mm, item.y_mm] for item in eligible], dtype=np.float64).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(points, np.asarray(homography, dtype=np.float64)).reshape(-1, 2)
    width, height = image_size
    return {"components": [{"reference": item.reference, "image_position_px": [float(u), float(v)],
                            "position_status": "pending_verification"}
                           for item, (u, v) in zip(eligible, mapped) if 0 <= u < width and 0 <= v < height]}


def _draw_pending_locations(image: Any, locations: dict) -> Any:
    from PIL import ImageDraw

    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    for item in locations["components"]:
        u, v = item["image_position_px"]
        draw.ellipse((u - 7, v - 7, u + 7, v + 7), outline=(20, 100, 220), width=3)
        draw.text((u + 9, v), item["reference"], fill=(20, 60, 160))
    return output


def _draw_candidate(image: Any, points: list[list[float]]) -> Any:
    from PIL import ImageDraw

    output = image.copy()
    draw = ImageDraw.Draw(output)
    rounded = [(round(x), round(y)) for x, y in points]
    draw.line(rounded + [rounded[0]], fill=(255, 190, 0), width=max(3, image.width // 500))
    for index, (x, y) in enumerate(rounded, start=1):
        radius = max(5, image.width // 200)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(230, 50, 45), outline="white", width=2)
        draw.text((x + radius, y + radius), str(index), fill="white", stroke_width=1, stroke_fill="black")
    return output


def _png_bytes(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _image_from_png(value: bytes) -> Any:
    from PIL import Image

    return Image.open(io.BytesIO(value)).convert("RGB")


def _safe_result_id(image_hash: str, csv_hash: str, side: str, context_key: str) -> str:
    token = hashlib.sha256(f"{image_hash}:{csv_hash}:{side}:{context_key}".encode()).hexdigest()[:16]
    return re.sub(r"[^a-z0-9_-]", "_", f"{side}-{token}")
