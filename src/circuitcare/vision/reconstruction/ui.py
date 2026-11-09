"""Streamlit entry point for the multi-photo PCB reconstruction MVP."""

from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
from typing import Any
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from circuitcare.vision.board_detector import detect_board_candidates
from circuitcare.vision.perspective import rectify_board
from circuitcare.vision.reconstruction.copper_segmentation import (
    detect_pad_via_candidates, segment_visible_copper, vectorize_copper,
)
from circuitcare.vision.reconstruction.multiview_registration import (
    back_to_front_transform, register_manual_correspondences, register_same_side,
)
from circuitcare.vision.reconstruction.photo_collection import CAPTURE_TYPES, collect_photos
from circuitcare.vision.reconstruction.quality import inspect_photo_quality
from circuitcare.vision.reconstruction.topology import build_observed_topology


_SIDE_LABELS = {"front": "앞면", "back": "뒷면", "unknown": "미지정"}
_CAPTURE_LABELS = {
    "full_view": "기판 전체", "close_up": "주변 확대", "oblique": "경사 촬영",
    "marking_detail": "마킹 확대",
}


def render_multiview_reconstruction(st: Any) -> None:
    """Render local multi-photo upload, quality, alignment, and trace candidates."""
    st.subheader("다중 사진 PCB 복원")
    st.caption("원본 사진을 보존하고, 사진에서 보이는 형상 후보만 표시합니다. 이 초기 복원 그래프는 전기적 넷리스트가 아니며 배선 연속성은 별도 측정이 필요합니다.")
    input_mode = st.radio("입력", ["내 사진 업로드", "예제 PCB로 테스트"], horizontal=True,
                          key="pcb_reconstruction_input_mode")
    if input_mode == "예제 PCB로 테스트":
        _render_prism_example(st)
        return
    uploads = st.file_uploader("PCB 사진 여러 장 추가", type=["jpg", "jpeg", "png", "webp"],
                               accept_multiple_files=True, key="pcb_reconstruction_uploads")
    if not uploads:
        st.info("앞면·뒷면 전체 사진을 한 장씩 올리면 정렬을 시도할 수 있습니다. 한 장으로도 사진 품질과 가시 배선 후보를 살펴볼 수 있습니다.")
        st.caption("EXIF 방향 보정 · 파일당 20 MB · 사진당 2,500만 픽셀 · 원본 바이트 자동 저장/전송 없음")
        return
    if len(uploads) > 20:
        st.error("한 작업에는 사진을 최대 20장까지 추가할 수 있습니다.")
        return

    role_map: dict[str, tuple[str, str]] = {}
    role_columns = st.columns(min(3, len(uploads)))
    for index, upload in enumerate(uploads):
        column = role_columns[index % len(role_columns)]
        key = f"pcb_reconstruction_role_{index}_{upload.name}"
        with column:
            st.markdown(f"**사진 {index + 1} · {upload.name}**")
            side = st.selectbox("기판 면", ["unknown", "front", "back"],
                                format_func=lambda item: _SIDE_LABELS[item], key=f"{key}_side")
            capture = st.selectbox("촬영 유형", CAPTURE_TYPES,
                                   format_func=lambda item: _CAPTURE_LABELS[item], key=f"{key}_capture")
            role_map[f"{index}:{upload.name}"] = (side, capture)

    try:
        records = collect_photos(
            uploads,
            side_by_name={name: side for name, (side, _) in role_map.items()},
            capture_by_name={name: capture for name, (_, capture) in role_map.items()},
        )
    except (ValueError, OSError) as exc:
        st.error(f"사진을 읽을 수 없습니다: {exc}")
        return

    duplicate_count = sum(record.duplicate_of is not None for record, _, _ in records)
    st.caption(f"{len(records)}장 처리 · 중복 사진 {duplicate_count}장 · 사진 원본의 SHA-256과 보정 결과의 출처를 함께 관리합니다.")
    processed = []
    tabs = st.tabs([f"사진 {i + 1} · {_SIDE_LABELS[record.side]}" for i, (record, _, _) in enumerate(records)])
    for tab, (record, image, _) in zip(tabs, records):
        with tab:
            detection = detect_board_candidates(image)
            candidate_index = 0
            if len(detection["candidates"]) > 1:
                candidate_index = st.selectbox(
                    "기판 외곽 후보 선택", range(len(detection["candidates"])),
                    format_func=lambda value: f"후보 {value + 1} · 면적 {detection['candidates'][value]['area_ratio']:.1%}",
                    key=f"reconstruction_board_candidate_{record.photo_id}")
            corners = detection["candidates"][candidate_index]["corners_px"] if detection["candidates"] else None
            quality = inspect_photo_quality(image, board_corners=corners)
            st.image(image, caption=f"원본 미리보기 · {record.image_format} · {record.image_size_px[0]}×{record.image_size_px[1]} px",
                     use_container_width=True)
            st.caption(f"SHA-256 `{record.file_sha256}` · EXIF 방향 보정 · 촬영 유형 {_CAPTURE_LABELS[record.capture_type]}")
            with st.expander("부품 마킹 후보 확인 (선택 실행)"):
                run_ocr = st.checkbox("이 사진에서 로컬 OCR 실행", key=f"reconstruction_ocr_{record.photo_id}")
                if run_ocr:
                    try:
                        from circuitcare.vision.ocr_backend import run_local_ocr
                        ocr = run_local_ocr(image)
                        st.write({"OCR 상태": ocr["status"], "엔진": ocr["engine"],
                                  "후보 텍스트": [{"text": row["text"],
                                                "Reference 후보": row["reference_candidates"],
                                                "상태": row["verification_status"]}
                                               for row in ocr["records"]]})
                        st.caption("OCR 결과는 문자 후보이며 Reference나 전기적 연결을 자동 승인하지 않습니다.")
                    except RuntimeError as exc:
                        st.warning(str(exc))
            if record.duplicate_of:
                st.warning("이 사진은 앞서 선택한 이미지와 내용이 같습니다.")
            if corners:
                use_corners = st.checkbox("가장 큰 자동 외곽 후보를 사용해 정면 보정", value=True,
                                          key=f"reconstruction_rectify_{record.photo_id}")
                st.caption("자동 외곽은 후보입니다. 선택한 외곽이 실제 기판 경계인지 확인하세요.")
                if use_corners:
                    rectified = rectify_board(image, corners)
                    analysis_image = rectified["image"]
                    board_frame_status = "BOARD_BOUNDARY_CANDIDATE"
                    board_mask = np.full((analysis_image.height, analysis_image.width), 255, dtype=np.uint8)
                else:
                    analysis_image = image
                    board_frame_status = "UNRECTIFIED_IMAGE_FRAME"
                    board_mask = np.zeros((image.height, image.width), dtype=np.uint8)
                    cv2.fillConvexPoly(board_mask, np.asarray(corners, dtype=np.int32), 255)
            else:
                st.warning("기판 외곽을 찾지 못했습니다. 사진 좌표를 유지하며 자동 정합은 시도하지 않습니다.")
                analysis_image = image
                board_frame_status = "BOARD_BOUNDARY_UNKNOWN"
                board_mask = None
            st.markdown("**사진 품질**")
            st.caption(f"흐림 지표 {quality['blur_laplacian_variance']:.1f} · 반사 후보 {quality['glare_candidate_fraction']:.1%} · 해상도 {quality['resolution_px'][0]}×{quality['resolution_px'][1]} px")
            with st.expander("품질 분석 세부 정보"):
                st.json(quality)
            if quality["flags"]:
                st.warning("촬영 참고: " + ", ".join(quality["flags"]))
            else:
                st.success("뚜렷한 품질 경고가 없습니다. 이는 배선 검증을 의미하지 않습니다.")
            if quality["flags"]:
                for flag in quality["flags"]:
                    if flag == "POSSIBLE_BLUR":
                        st.caption("부품 마킹이나 가는 배선이 흐릴 수 있습니다. 해당 영역을 가까이서 다시 촬영해 보세요.")
                    elif flag == "POSSIBLE_GLARE":
                        st.caption("반사광 후보 영역이 있습니다. 조명 각도를 바꾼 사진을 추가하세요.")
                    elif flag == "LOW_RESOLUTION":
                        st.caption("더 높은 해상도의 사진이 작은 패드와 마킹 분석에 도움이 됩니다.")
            processed.append({"record": record, "image": analysis_image,
                              "source_image": image,
                              "board_mask": board_mask,
                              "frame_status": board_frame_status, "quality": quality})

    full_views = [item for item in processed if item["record"].capture_type == "full_view"]
    front_views = [item for item in full_views if item["record"].side == "front"]
    back_views = [item for item in full_views if item["record"].side == "back"]
    if len(front_views) > 1 or len(back_views) > 1:
        st.markdown("#### 같은 면 사진 정렬")
        for side_views in (front_views, back_views):
            if len(side_views) < 2:
                continue
            reference = side_views[0]
            for moving in side_views[1:]:
                result = register_same_side(
                    reference["image"], moving["image"],
                    source_photo_id=moving["record"].photo_id,
                    reference_photo_id=reference["record"].photo_id,
                    board_side=reference["record"].side)
                st.write({"사진": moving["record"].photo_id,
                          "상태": result.registration_status,
                          "특징점 inlier": result.inlier_count,
                          "재투영 오차 px": result.reprojection_error_px,
                          "겹침 면적 비율": result.overlap_ratio,
                          "근거": result.evidence_status.value})
                st.caption("특징점 정합은 사진 위치 추정입니다. 사용자가 검토하기 전 좌표를 확정하지 않습니다.")
                if result.transformation is not None:
                    ref_w, ref_h = reference["image"].size
                    warped = cv2.warpPerspective(np.asarray(moving["image"]),
                                                  np.asarray(result.transformation), (ref_w, ref_h))
                    comparison = cv2.addWeighted(np.asarray(reference["image"]), 0.5, warped, 0.5, 0)
                    st.image(Image.fromarray(comparison),
                             caption="정합 중첩 미리보기 · 후보 상태, 사용자가 확인해야 함",
                             use_container_width=True)
                if result.registration_status == "REGISTRATION_UNVERIFIED":
                    with st.expander("정합 실패 — 대응점을 직접 지정"):
                        st.caption("두 사진에서 같은 네 지점을 찾아 각각의 픽셀 좌표를 입력하세요. 보드 전체에 넓게 분포해야 합니다.")
                        ref_w, ref_h = reference["image"].size
                        mov_w, mov_h = moving["image"].size
                        ref_points, mov_points = [], []
                        with st.form(f"manual_photo_registration_{moving['record'].photo_id}"):
                            for point_index, fraction in enumerate(((.1, .1), (.9, .1), (.9, .9), (.1, .9))):
                                st.markdown(f"**대응점 {point_index + 1}**")
                                cols = st.columns(4)
                                ru = cols[0].number_input(f"기준 X {point_index+1} px", min_value=0.0, max_value=float(ref_w-1), value=float(ref_w*fraction[0]), key=f"manual_ref_x_{moving['record'].photo_id}_{point_index}")
                                rv = cols[1].number_input(f"기준 Y {point_index+1} px", min_value=0.0, max_value=float(ref_h-1), value=float(ref_h*fraction[1]), key=f"manual_ref_y_{moving['record'].photo_id}_{point_index}")
                                mu = cols[2].number_input(f"대상 X {point_index+1} px", min_value=0.0, max_value=float(mov_w-1), value=float(mov_w*fraction[0]), key=f"manual_mov_x_{moving['record'].photo_id}_{point_index}")
                                mv = cols[3].number_input(f"대상 Y {point_index+1} px", min_value=0.0, max_value=float(mov_h-1), value=float(mov_h*fraction[1]), key=f"manual_mov_y_{moving['record'].photo_id}_{point_index}")
                                ref_points.append((ru, rv))
                                mov_points.append((mu, mv))
                            manual_submitted = st.form_submit_button("수동 정합 후보 계산")
                        if manual_submitted:
                            try:
                                manual = register_manual_correspondences(
                                    ref_points, mov_points, source_photo_id=moving["record"].photo_id,
                                    reference_photo_id=reference["record"].photo_id,
                                    board_side=reference["record"].side)
                                st.write({"수동 정합 상태": manual.registration_status,
                                          "재투영 오차 px": manual.reprojection_error_px,
                                          "근거 상태": manual.evidence_status.value})
                            except ValueError as exc:
                                st.error(str(exc))

    if front_views and back_views:
        st.markdown("#### 앞면·뒷면 기준 방향")
        st.caption("반대 면은 단순 뒤집기로 자동 정합하지 않습니다. 보드 외곽을 정면 보정한 뒤 기준 방향을 선택하고 사용자가 확인해야 좌표 변환 후보가 활성화됩니다.")
        rotation = st.selectbox("뒷면 기준 회전", [0, 90, 180, 270], format_func=lambda x: f"{x}°",
                                key="pcb_reconstruction_back_rotation")
        mirror = st.checkbox("좌우 반전", value=True, key="pcb_reconstruction_back_mirror")
        confirm = st.checkbox("사진의 기준 모서리와 커넥터 방향을 비교해 위 변환을 확인했습니다",
                              key="pcb_reconstruction_back_confirm")
        transform = back_to_front_transform(rotation_degrees=rotation, mirror_x=mirror,
                                            board_size_px=front_views[0]["image"].size)
        back_image = back_views[0]["image"]
        fw, fh = front_views[0]["image"].size
        back_array = cv2.resize(np.asarray(back_image), (fw, fh), interpolation=cv2.INTER_AREA)
        aligned_back = cv2.warpPerspective(back_array, np.asarray(transform["matrix"]), (fw, fh))
        st.image(Image.fromarray(aligned_back), caption="뒷면 좌표 변환 미리보기 · 정합 검증 상태 아님", use_container_width=True)
        st.write({**transform, "status": "USER_CONFIRMED_ORIENTATION" if confirm else "REGISTRATION_UNVERIFIED",
                  "evidence_status": "USER_CONFIRMED" if confirm else "UNKNOWN"})
        st.caption("사용자 확인은 방향 입력만 확정합니다. 패드·비아의 실제 전기적 접속은 확정하지 않습니다.")

    st.markdown("#### 관측 가능한 배선·패드 후보")
    st.caption("밝은 저채도/주변 대비 Baseline입니다. 실크스크린·반사·솔더마스크와 혼동될 수 있어 모든 형상은 후보로만 표시합니다.")
    view_mode = st.radio("복원 보기", ["원본 사진", "배선 후보", "Pad/Via 후보", "접촉 후보"],
                         horizontal=True, key="pcb_reconstruction_view_mode")
    opacity = st.slider("배선 표시 투명도", 0.15, 0.85, 0.45, 0.05,
                        disabled=view_mode == "원본 사진", key="pcb_reconstruction_trace_opacity")
    overlay_cols = st.columns(min(2, len(processed)))
    for index, item in enumerate(processed):
        record, image = item["record"], item["image"]
        segmented = segment_visible_copper(image, board_side=record.side,
                                           board_mask=item["board_mask"])
        traces = vectorize_copper(segmented.mask, photo_id=record.photo_id, board_side=record.side)
        pads = detect_pad_via_candidates(image, photo_id=record.photo_id,
                                         board_side=record.side, copper_mask=segmented.mask)
        graph = build_observed_topology(segmented.mask, photo_id=record.photo_id,
                                        board_side=record.side, pad_candidates=pads)
        display_image = item["source_image"] if view_mode == "원본 사진" else image
        display = np.asarray(display_image).copy()
        if view_mode != "원본 사진":
            tint = np.zeros_like(display)
            tint[:, :, 1] = 235
            active = segmented.mask
            display[active] = ((1-opacity) * display[active] + opacity * tint[active]).astype(np.uint8)
        if view_mode in {"Pad/Via 후보", "접촉 후보"}:
            for candidate in pads:
                center = tuple(int(round(value)) for value in candidate.center_px)
                cv2.circle(display, center, max(3, int(round(candidate.radius_px))), (255, 150, 20), 2)
        if view_mode == "접촉 후보":
            for edge in graph.edges:
                candidate = next((row for row in pads if row.candidate_id == edge.source_feature_id), None)
                if candidate:
                    center = tuple(int(round(value)) for value in candidate.center_px)
                    cv2.circle(display, center, 5, (255, 50, 50), -1)
        column = overlay_cols[index % len(overlay_cols)]
        with column:
            coordinate_note = "원본 픽셀 좌표" if view_mode == "원본 사진" else item["frame_status"]
            _render_zoomable_canvas(
                st, Image.fromarray(display),
                caption=f"{_SIDE_LABELS[record.side]} · {view_mode} · 좌표계: {coordinate_note}")
            st.write({"관측 상태": segmented.status, "사진 좌표계": item["frame_status"],
                      "배선 영역": len(traces), "패드/비아 후보": len(pads),
                      "같은 면 접촉 후보": len(graph.edges),
                      "연결 그래프": graph.status,
                      "미관측 영역": "솔더마스크/가림 아래 배선은 NOT_OBSERVABLE"})
            st.caption("접촉 후보는 전기적 Net이나 연속성 확인이 아닙니다. 사진이 바뀌면 후보와 근거도 새로 계산됩니다.")


def _render_prism_example(st: Any) -> None:
    """Run pinned PRISM sample images through the local image-only pipeline."""
    from circuitcare.vision.reconstruction.prism_dataset import SAMPLES, load_references, load_sample

    repo_root = Path(__file__).resolve().parents[4]
    dataset_root = repo_root / "data" / "external" / "prism"
    if not (dataset_root / "Dataset").is_dir():
        st.info("예제 데이터는 PRISM 저장소에서 고정 버전으로 자동 다운로드합니다. 원본 데이터는 CircuitCare Git 이력에 추가하지 않습니다.")
        if st.button("PRISM 예제 데이터 다운로드", key="pcb_prism_download"):
            script = repo_root / "scripts" / "fetch_prism_dataset.py"
            result = subprocess.run([sys.executable, str(script)], cwd=repo_root,
                                    capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                st.success(result.stdout.strip())
                st.rerun()
            else:
                st.error(result.stderr.strip() or "PRISM 예제 데이터 다운로드에 실패했습니다.")
        return

    sample_id = st.selectbox("예제 PCB", list(SAMPLES), key="pcb_prism_sample")
    st.caption("출처: PRISM · 고정 Commit 9189c87bbec6995b745b809db99cef6d3269401f · 실제 회로 기판이 아닌 렌더 기반 예제입니다.")
    st.caption("KiCad 정답은 이미지 복원 입력으로 전달되지 않습니다. 이미지 좌표와 KiCad mm 좌표의 검증된 변환이 없어 Pixel 정확도 점수는 표시하지 않습니다.")
    if not st.button("이미지 복원 후보 실행", key=f"pcb_prism_run_{sample_id}"):
        try:
            sample = load_sample(dataset_root, sample_id)
            col_a, col_b = st.columns(2)
            col_a.image(sample.top_image, caption="TOP 원본 이미지", use_container_width=True)
            col_b.image(sample.bottom_image, caption="BOTTOM 원본 이미지", use_container_width=True)
        except (OSError, ValueError) as exc:
            st.error(f"예제 파일을 읽을 수 없습니다: {exc}")
        return

    with st.spinner(f"{sample_id}의 TOP/BOTTOM 이미지를 처리하고 있습니다…"):
        script = repo_root / "scripts" / "benchmark_prism.py"
        result = subprocess.run([sys.executable, str(script), "--dataset-root", str(dataset_root),
                                 "--sample", sample_id], cwd=repo_root,
                                capture_output=True, text=True, timeout=300)
    if result.returncode:
        st.error(result.stderr.strip() or "이미지 후보 처리가 실패했습니다.")
        return
    output_root = repo_root / "data" / "benchmarks" / "prism"
    sample = load_sample(dataset_root, sample_id)
    report_path = output_root / "baseline" / "sample_results" / f"{sample_id}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reference_stats = load_references(dataset_root, sample_id)["expected_stats"]
    st.info("결과는 이미지에서 관측한 후보입니다. 정답 geometry와의 pixel 비교는 좌표 정합이 검증되지 않아 수행하지 않았습니다.")
    st.write({"예제": sample_id, "정답 부품 수": reference_stats.get("component_count", "미기재"),
              "정답 배선 segment 수": reference_stats.get("segment_count", "미기재"),
              "정답 via 수": reference_stats.get("via_count", "미기재"),
              "Connectivity 평가": "NOT_VERIFIED"})
    for side in ("TOP", "BOTTOM"):
        source = sample.top_image if side == "TOP" else sample.bottom_image
        overlay = output_root / "baseline" / "overlays" / f"{sample_id}_{side}.png"
        col_a, col_b = st.columns(2)
        col_a.image(source, caption=f"{side} 원본", use_container_width=True)
        if overlay.is_file():
            col_b.image(overlay, caption=f"{side} 배선 후보 overlay · baseline", use_container_width=True)
        st.write({"면": side, "파이프라인": report["baseline"][side]["status"],
                  "배선 후보 component": report["baseline"][side].get("trace_component_count"),
                  "Pad/Via 후보": report["baseline"][side].get("pad_via_candidate_count"),
                  "동일 면 접촉 후보": report["baseline"][side].get("same_side_contact_candidate_count")})


def _render_zoomable_canvas(st: Any, image: Image.Image, *, caption: str) -> None:
    """Render the current photo layer with browser-local wheel zoom and drag pan."""
    image.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
      *{{box-sizing:border-box}} body{{margin:0;background:#0e141b;color:#d8e2ec;font:13px sans-serif}}
      .toolbar{{height:40px;display:flex;align-items:center;gap:6px;padding:4px 8px;background:#141e29;border:1px solid #344354;border-bottom:0}}
      button{{height:28px;min-width:34px;border:1px solid #415266;border-radius:4px;background:#202d3a;color:#edf3f8;cursor:pointer}}
      button:hover{{background:#2b3d4e}} #zoom{{min-width:48px;text-align:center;color:#aebdca}}
      #caption{{margin-left:auto;color:#aebdca;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
      #viewport{{height:460px;width:100%;overflow:hidden;position:relative;background:#f7f6f2;border:1px solid #344354;touch-action:none;cursor:grab}}
      #viewport.dragging{{cursor:grabbing}} #stage{{position:absolute;left:0;top:0;transform-origin:0 0;will-change:transform}}
      #photo{{display:block;max-width:none;max-height:none;user-select:none;-webkit-user-drag:none;pointer-events:none}}
    </style></head><body>
      <div class="toolbar" role="toolbar" aria-label="사진 캔버스 조작">
        <button id="fit" aria-label="화면 맞춤" title="화면 맞춤">맞춤</button>
        <button id="minus" aria-label="축소" title="축소">−</button>
        <span id="zoom" aria-live="polite">100%</span>
        <button id="plus" aria-label="확대" title="확대">＋</button>
        <span id="caption">{caption}</span>
      </div>
      <div id="viewport" aria-label="이동 및 확대 가능한 PCB 사진">
        <div id="stage"><img id="photo" alt="PCB 사진과 선택한 후보 레이어" src="data:image/jpeg;base64,{encoded}"></div>
      </div>
      <script>
        const viewport=document.getElementById('viewport'),stage=document.getElementById('stage'),img=document.getElementById('photo'),zoomLabel=document.getElementById('zoom');
        let scale=1,x=0,y=0,fitScale=1,drag=null;
        function apply(){{stage.style.transform=`translate(${{x}}px,${{y}}px) scale(${{scale}})`;zoomLabel.textContent=`${{Math.round(scale/fitScale*100)}}%`;}}
        function fit(){{const r=viewport.getBoundingClientRect();fitScale=Math.min(r.width/img.naturalWidth,r.height/img.naturalHeight);scale=fitScale;x=(r.width-img.naturalWidth*scale)/2;y=(r.height-img.naturalHeight*scale)/2;apply();}}
        function around(point,nextScale){{const rect=viewport.getBoundingClientRect(),px=point.clientX-rect.left,py=point.clientY-rect.top,ratio=nextScale/scale;x=px-(px-x)*ratio;y=py-(py-y)*ratio;scale=nextScale;apply();}}
        if(img.complete)fit();else img.addEventListener('load',fit);window.addEventListener('resize',fit);document.getElementById('fit').onclick=fit;
        document.getElementById('plus').onclick=()=>around({{clientX:viewport.clientWidth/2+viewport.getBoundingClientRect().left,clientY:viewport.clientHeight/2+viewport.getBoundingClientRect().top}},Math.min(scale*1.25,fitScale*8));
        document.getElementById('minus').onclick=()=>around({{clientX:viewport.clientWidth/2+viewport.getBoundingClientRect().left,clientY:viewport.clientHeight/2+viewport.getBoundingClientRect().top}},Math.max(scale/1.25,fitScale*.2));
        viewport.addEventListener('wheel',event=>{{event.preventDefault();around(event,Math.max(fitScale*.2,Math.min(scale*(event.deltaY<0?1.12:.89),fitScale*8)));}},{{passive:false}});
        viewport.addEventListener('pointerdown',event=>{{drag={{x:event.clientX,y:event.clientY,originX:x,originY:y}};viewport.classList.add('dragging');viewport.setPointerCapture(event.pointerId);}});
        viewport.addEventListener('pointermove',event=>{{if(!drag)return;x=drag.originX+event.clientX-drag.x;y=drag.originY+event.clientY-drag.y;apply();}});
        function stop(){{drag=null;viewport.classList.remove('dragging');}}viewport.addEventListener('pointerup',stop);viewport.addEventListener('pointercancel',stop);
      </script></body></html>"""
    st.components.v1.html(html, height=510, scrolling=False)
