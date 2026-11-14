import io
from types import SimpleNamespace

from PIL import Image

from circuitcare.dashboard.components.pcb_workspace import (
    OVERLAPPING_SELECTION_KEY, coordinate_status_for_viewer, render_component_viewer,
    scale_overlay_to_display, validate_component_selection_event,
)
from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel
from circuitcare.frontend.component_overlay_models import display_identity_status
from circuitcare.dashboard.views.unified_workspace import (
    _bind_review_models_to_map, _prepare_board_registry_snapshot, _prepare_photo_component_snapshot,
)
from circuitcare.frontend.models import ImageRecord, ProjectContext, SelectionState
from circuitcare.frontend.workbench_state import (
    BOARD_COMPONENT_SELECTION_KEY, SELECTION_KEY, update_component_selection,
)
from circuitcare.vision.box_reconciliation import board_visible_review_rows
from circuitcare.vision.component_recognition_panel import (
    _store_predictions, ensure_review_row_for_component, invalidate_component_review_state,
    record_box_exclusion, review_keys, restore_review_row, undo_last_box_exclusion, update_review_row,
    _apply_qwen_proposal_to_row, _selected_classification_detections,
)


class _ProgressStub:
    def write(self, *_args, **_kwargs):
        pass

    def update(self, *_args, **_kwargs):
        pass


def _model():
    return ComponentOverlayViewModel(
        project_id="p1", board_id="b1", board_side="TOP", component_id="c1",
        component_type="resistor", display_label="R4", canonical_frame_id="frame1",
        coordinate_frame="EXIF_CORRECTED_IMAGE_PIXEL", bbox_xyxy=(10, 20, 30, 50),
        polygon_xy=((10, 20), (30, 20), (30, 50), (10, 50)),
        source_backend_photo_ids=("bp1",), observation_ids=("o1",),
        detection_confidence=.8, identification_status="SUGGESTED_UNCONFIRMED",
        identity_match_status="NEW_COMPONENT", ocr_marking=None,
        possible_part_number=None, evidence=("MODEL_DETECTION_CANDIDATE",),
        overlay_status="DETECTION_CANDIDATE",
    )


def test_original_coordinates_scale_with_known_exif_corrected_resize():
    scaled = scale_overlay_to_display((_model(),), source_size=(100, 100), display_size=(50, 50))[0]
    assert scaled.bbox_xyxy == (5, 10, 15, 25)
    assert scaled.polygon_xy == ((5, 10), (15, 10), (15, 25), (5, 25))


def test_click_and_clear_selection_events_are_scoped_and_monotonic():
    event = {"project_id": "p1", "board_id": "b1", "board_side": "TOP",
             "canonical_frame_id": "frame1", "component_id": "c1",
             "selection_action": "select", "event_seq": 1, "event_id": "evt-1"}
    kwargs = dict(project_id="p1", board_id="b1", board_side="TOP", frame_id="frame1",
                  allowed_ids={"c1"}, last_sequence=0)
    assert validate_component_selection_event(event, **kwargs) == (True, None)
    overlapping = {**event, "overlapping_component_ids": ["c1", "c2"]}
    assert validate_component_selection_event(overlapping, **{**kwargs, "allowed_ids": {"c1", "c2"}}) == (True, None)
    # Other overlap candidates may be stale between the browser click and the
    # current render; the selected, owner-validated ID must remain selectable.
    assert validate_component_selection_event(overlapping, **kwargs) == (True, None)
    assert validate_component_selection_event(
        {**event, "component_id": "missing", "overlapping_component_ids": ["missing", "c2"]},
        **{**kwargs, "allowed_ids": {"c1"}},
    )[0] is False
    assert validate_component_selection_event(
        {**event, "event_epoch": "old"}, **{**kwargs, "event_epoch": "current"}
    ) == (False, "이전 Viewer 인스턴스의 선택 이벤트")
    assert validate_component_selection_event({**event, "project_id": "other"}, **kwargs)[0] is False
    assert validate_component_selection_event({**event, "board_id": "other"}, **kwargs)[0] is False
    assert validate_component_selection_event({**event, "board_side": "BOTTOM"}, **kwargs)[0] is False
    assert validate_component_selection_event({**event, "canonical_frame_id": "old-frame"}, **kwargs)[0] is False
    assert validate_component_selection_event({**event, "component_id": "missing"}, **kwargs)[0] is False
    assert validate_component_selection_event(event, **{**kwargs, "last_sequence": 1})[0] is False
    assert validate_component_selection_event({k: v for k, v in event.items() if k != "event_id"}, **kwargs)[0] is False
    clear = {**event, "component_id": None, "selection_action": "clear", "event_seq": 2}
    assert validate_component_selection_event(clear, **{**kwargs, "last_sequence": 1}) == (True, None)


def test_viewer_click_event_updates_real_streamlit_selection_and_overlap_stack(monkeypatch):
    import circuitcare.dashboard.components.pcb_workspace as workspace

    streamlit = SimpleNamespace(session_state={})
    project = ProjectContext(project_id="p1", board_id="b1")
    streamlit.session_state[SELECTION_KEY] = SelectionState(project_id="p1", board_id="b1")
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(output, format="PNG")
    components = tuple(ComponentOverlayViewModel(**{**_model().__dict__, "component_id": item})
                       for item in ("c1", "c2"))

    def fake_render(_st, **kwargs):
        key = kwargs["key"]
        streamlit.session_state[key] = {"selection_event": {
            "project_id": "p1", "board_id": "b1", "board_side": "TOP",
            "canonical_frame_id": "photo1", "component_id": "c1", "selection_action": "select",
            "overlapping_component_ids": ["c1", "c2"], "event_seq": 1, "event_id": "click-1",
            "event_epoch": kwargs["event_epoch"],
        }}
        kwargs["on_selection_event_change"]()
        return object()

    monkeypatch.setattr(workspace, "render_pcb_viewer", fake_render)
    available, viewer_key, _status = render_component_viewer(
        streamlit, image=None, source_image_bytes=output.getvalue(), canonical_png_bytes=None,
        image_size_px=(32, 32), components=components, project=project, board_side="TOP",
        frame_id="photo1", selected_component_id=None, map_cache_key="map1", image_mode="원본 사진",
    )
    assert available is True
    selection = streamlit.session_state[SELECTION_KEY]
    assert selection.selected_component == "c1"
    assert streamlit.session_state[f"{OVERLAPPING_SELECTION_KEY}_{viewer_key}"] == ["c1", "c2"]


def test_viewer_click_keeps_valid_selection_and_filters_stale_overlap_ids(monkeypatch):
    import circuitcare.dashboard.components.pcb_workspace as workspace

    streamlit = SimpleNamespace(session_state={})
    project = ProjectContext(project_id="p1", board_id="b1")
    streamlit.session_state[SELECTION_KEY] = SelectionState(project_id="p1", board_id="b1")
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(output, format="PNG")
    components = (_model(),)

    def fake_render(_st, **kwargs):
        key = kwargs["key"]
        streamlit.session_state[key] = {"selection": {
            "project_id": "p1", "board_id": "b1", "board_side": "TOP",
            "canonical_frame_id": "photo1", "component_id": "c1", "selection_action": "select",
            "overlapping_component_ids": ["c1", "deleted-stale-id"], "event_seq": 1,
            "event_id": "click-stale-overlap", "event_epoch": kwargs["event_epoch"],
        }}
        kwargs["on_selection_event_change"]()
        return object()

    monkeypatch.setattr(workspace, "render_pcb_viewer", fake_render)
    _available, viewer_key, _status = render_component_viewer(
        streamlit, image=None, source_image_bytes=output.getvalue(), canonical_png_bytes=None,
        image_size_px=(32, 32), components=components, project=project, board_side="TOP",
        frame_id="photo1", selected_component_id=None, map_cache_key="map1", image_mode="원본 사진",
    )
    assert streamlit.session_state[SELECTION_KEY].selected_component == "c1"
    assert streamlit.session_state[f"{OVERLAPPING_SELECTION_KEY}_{viewer_key}"] == ["c1"]


def test_exclusion_then_new_box_click_updates_local_and_board_selection_before_rerun(monkeypatch):
    import circuitcare.dashboard.components.pcb_workspace as workspace

    project = ProjectContext("p1", "b1")
    state = {SELECTION_KEY: SelectionState("p1", "b1", selected_component="c1"),
             BOARD_COMPONENT_SELECTION_KEY: "bc-c1"}
    # The exclusion action must retire the old board identity atomically.
    update_component_selection(state, project=project, selected_component=None)
    assert state[SELECTION_KEY].selected_component is None
    assert BOARD_COMPONENT_SELECTION_KEY not in state

    streamlit = SimpleNamespace(session_state=state)
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(output, format="PNG")
    components = tuple(ComponentOverlayViewModel(**{
        **_model().__dict__, "component_id": local_id, "board_component_id": board_id,
    }) for local_id, board_id in (("c1", "bc-c1"), ("c2", "bc-c2")))

    def fake_render(_st, **kwargs):
        key = kwargs["key"]
        streamlit.session_state[key] = {"selection_event": {
            "project_id": "p1", "board_id": "b1", "board_side": "TOP",
            "canonical_frame_id": "photo1", "component_id": "c2", "selection_action": "select",
            "overlapping_component_ids": ["c2"], "event_seq": 1, "event_id": "after-exclude-click",
            "event_epoch": kwargs["event_epoch"],
        }}
        kwargs["on_selection_event_change"]()
        return object()

    monkeypatch.setattr(workspace, "render_pcb_viewer", fake_render)
    workspace.render_component_viewer(
        streamlit, image=None, source_image_bytes=output.getvalue(), canonical_png_bytes=None,
        image_size_px=(32, 32), components=components, project=project, board_side="TOP",
        frame_id="photo1", selected_component_id=None, map_cache_key="map-after-exclusion",
        image_mode="원본 사진",
    )
    assert state[SELECTION_KEY].selected_component == "c2"
    assert state[BOARD_COMPONENT_SELECTION_KEY] == "bc-c2"


def test_component_result_is_applied_without_forcing_a_second_rerun(monkeypatch):
    import circuitcare.dashboard.components.pcb_workspace as workspace

    reruns = []
    streamlit = SimpleNamespace(session_state={}, rerun=lambda: reruns.append(True))
    project = ProjectContext(project_id="p1", board_id="b1")
    from circuitcare.frontend.workbench_state import SELECTION_KEY
    streamlit.session_state[SELECTION_KEY] = SelectionState(project_id="p1", board_id="b1")
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(output, format="PNG")
    event = {"project_id": "p1", "board_id": "b1", "board_side": "TOP",
             "canonical_frame_id": "photo1", "component_id": "c1", "selection_action": "select",
             "overlapping_component_ids": ["c1"], "event_seq": 1, "event_id": "result-click",
             "event_epoch": "unused"}

    def fake_render(_st, **kwargs):
        return SimpleNamespace(selection={**event, "event_epoch": kwargs["event_epoch"]}, viewer_status=None)

    monkeypatch.setattr(workspace, "render_pcb_viewer", fake_render)
    workspace.render_component_viewer(
        streamlit, image=None, source_image_bytes=output.getvalue(), canonical_png_bytes=None,
        image_size_px=(32, 32), components=(_model(),), project=project, board_side="TOP",
        frame_id="photo1", selected_component_id=None, map_cache_key="map1", image_mode="원본 사진",
    )
    assert streamlit.session_state[SELECTION_KEY].selected_component == "c1"
    assert reruns == []


def test_ai_progress_slot_never_clears_the_pcb_viewer():
    from circuitcare.dashboard.views.unified_workspace import _LeftProgressSlot

    class _Slot:
        def __init__(self):
            self.cleared = False
            self.container_value = object()

        def empty(self):
            self.cleared = True

        def container(self):
            return self.container_value

    status_slot, viewer_slot = _Slot(), _Slot()
    assert _LeftProgressSlot(status_slot, viewer_slot).container() is status_slot.container_value
    assert viewer_slot.cleared is False


def test_yolo_and_qwen_share_one_fixed_height_progress_slot():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "src/circuitcare/dashboard/views/unified_workspace.py").read_text()
    assert "workflow_status_slot = st.empty()" in source
    assert 'st.container(height=54, border=False, key="cc_workflow_progress")' in source
    assert "qwen_progress_slot = workflow_status_slot" in source
    assert "with qwen_progress_slot.container():" in source
    assert "workflow_status_slot.empty()" in source
    assert "with status_slot.container():" not in source


def test_workbench_hides_internal_status_lines_and_keeps_ai_loading_on_one_line():
    from pathlib import Path

    root = Path(__file__).parents[1]
    workspace = (root / "src/circuitcare/dashboard/views/unified_workspace.py").read_text()
    progress = (root / "src/circuitcare/dashboard/components/review_progress.py").read_text()
    viewer_html = (root / "src/circuitcare/dashboard/components/assets/pcb_viewer.html").read_text()
    assert "백엔드 미연결 · 원본 사진 보기" not in workspace
    assert "Registry {board_registry.get" not in workspace
    assert 'st.caption(_diagnosis_status_label(st, project, analysis_snapshot=live_analysis_snapshot))' not in workspace
    assert "display:flex;align-items:center;gap:9px" in progress
    assert "height=44" in progress
    assert "OCR/Qwen 검수 진행</strong></div>" in viewer_html


def test_viewer_identity_and_image_asset_are_independent_from_overlay_version(monkeypatch):
    import circuitcare.dashboard.components.pcb_workspace as workspace

    st = SimpleNamespace(session_state={SELECTION_KEY: SelectionState("p1", "b1")})
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(output, format="PNG")
    captured = []
    monkeypatch.setattr(workspace, "render_pcb_viewer",
                        lambda _st, **kwargs: captured.append(kwargs) or object())
    args = dict(st=st, image=None, source_image_bytes=output.getvalue(), canonical_png_bytes=None,
                image_size_px=(32, 32), components=(_model(),), project=ProjectContext("p1", "b1"),
                board_side="TOP", frame_id="photo1", selected_component_id=None,
                image_mode="원본 사진")
    workspace.render_component_viewer(**args, map_cache_key="overlay-v1")
    changed_box = ComponentOverlayViewModel(**{**_model().__dict__, "bbox_xyxy": (11, 21, 31, 51)})
    workspace.render_component_viewer(**{**args, "components": (changed_box,)}, map_cache_key="overlay-v2")
    workspace.render_component_viewer(**{**args, "components": (changed_box,)}, map_cache_key="overlay-v2")

    assert captured[0]["key"] == captured[1]["key"]
    assert captured[0]["asset_key"] == captured[1]["asset_key"]
    assert captured[0]["overlay_version"] != captured[1]["overlay_version"]
    assert captured[1]["overlay_version"] == captured[2]["overlay_version"]
    profile = st.session_state[f"cc_viewer_payload_profile_{captured[2]['key']}"]
    assert profile["render_count"] == 3
    assert st.session_state["cc_performance_events"][-1]["image_unchanged"] is True
    assert st.session_state["cc_performance_events"][-1]["overlay_unchanged"] is True
    assert st.session_state["cc_performance_events"][-1]["image_href_expected_change"] is False


def test_viewer_updates_regions_incrementally_without_replacing_svg_layer():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "src/circuitcare/dashboard/components/assets/pcb_viewer.js").read_text()
    assert "layer.replaceChildren()" not in source
    assert "state.regionNodes = new Map()" in source
    assert "entry.shape.classList.toggle(\"cc-selected\", id === selectedId)" in source
    assert "if (!overlayChanged && !selectionChanged) return;" in source
    assert "state.imageGeometryKey !== geometryKey" in source
    assert 'state.setStateValue("selection_event", payload)' in source
    assert 'image.setAttribute("href"' in source


def test_pcb_viewer_uses_shared_hit_test_from_real_click_and_does_not_capture_plain_click():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "src/circuitcare/dashboard/components/assets/pcb_viewer.js").read_text()
    assert 'listen(viewport,"pointerup"' in source
    assert 'const hits=hitComponents(event)' in source
    assert "[pcb-selection] pointerup_select" in source
    assert "viewport.setPointerCapture(event.pointerId)" in source
    # Pointer capture is acquired only after the movement crosses the drag threshold.
    pointerdown = source.split('listen(viewport,"pointerdown"', 1)[1].split('listen(viewport,"pointermove"', 1)[0]
    plain_pan_start = pointerdown.split("state.drag={", 1)[1]
    assert "setPointerCapture" not in plain_pan_start


def test_review_row_is_single_source_for_selected_box_edit_exclusion_and_restore():
    box = _model()
    rows = []
    row = ensure_review_row_for_component(rows, box)
    row.update({"qwen_component_type_status": "USER_CONFIRMED_EXTENSION",
                "qwen_component_type_candidate": "other_component",
                "confirmed_base_class": "other_component", "confirmed_subtype": "ic_socket",
                "qwen_user_confirmed_extension": True})
    update_review_row(row, component_type="capacitor", status="확인", bbox_xyxy=[12, 22, 34, 56])
    assert row["component_id"] == "c1"
    assert row["소자 종류"] == "capacitor" and row["상태"] == "확인"
    assert [row[key] for key in ("x1", "y1", "x2", "y2")] == [12, 22, 34, 56]
    assert row["confirmed_base_class"] == "capacitor"
    assert row["component_type_status"] == "USER_CONFIRMED"
    assert "qwen_user_confirmed_extension" not in row
    state = {}
    record_box_exclusion(state, scope="scope", row=row)
    assert row["상태"] == "제외" and row["user_edited"] is True
    assert board_visible_review_rows(rows) == []
    assert undo_last_box_exclusion(state, scope="scope", rows=rows) == "c1"
    assert row["상태"] == "확인" and board_visible_review_rows(rows) == [row]

    rerun_state = {"rows": [row]}
    record_box_exclusion(rerun_state, scope="scope", row=row)
    restore_review_row(row)
    assert row["상태"] == "확인"
    record_box_exclusion(rerun_state, scope="scope", row=row)
    _store_predictions(rerun_state, "rows", {"detections": [{
        "class_name": "resistor", "confidence": .91, "bbox_xyxy": [11, 21, 31, 51],
    }]})
    assert rerun_state["rows"][0]["component_id"] == "c1"
    assert rerun_state["rows"][0]["상태"] == "제외"
    assert board_visible_review_rows(rerun_state["rows"]) == []


def test_component_change_invalidates_diagnosis_but_preserves_measurement_history():
    state = {"circuitcare_v5_measurements": {"owner": ["measurement"]},
             "cc_symptom_diagnosis_p1": {"result": {"status": "old"}, "input_fingerprint": "old"}}
    invalidate_component_review_state(state, scope="scope", project_id="p1", board_id="b1")
    assert state["cc_symptom_diagnosis_p1"]["result"] is None
    assert state["circuitcare_v5_measurements"]["owner"] == ["measurement"]


def test_map_and_detector_boxes_bind_to_one_review_identity():
    map_box = ComponentOverlayViewModel(**{**_model().__dict__, "component_id": "map-id",
                                           "fault_candidate_status": "SUSPECTED_FAULT",
                                           "fault_history": ({"visual_status": "FAULT_REFUTED"},)})
    review_box = ComponentOverlayViewModel(**{**_model().__dict__, "component_id": "detector-id",
                                              "component_type": "capacitor", "bbox_xyxy": (11, 21, 31, 51),
                                              "overlay_source": "yolo"})
    rows = [{"component_id": "detector-id", "소자 종류": "capacitor", "상태": "확인",
             "x1": 11, "y1": 21, "x2": 31, "y2": 51}]
    merged, rows = _bind_review_models_to_map((map_box,), (review_box,), rows)
    assert len(merged) == 1
    assert merged[0].component_id == "detector-id"
    assert merged[0].bbox_xyxy == (11, 21, 31, 51)
    assert merged[0].fault_candidate_status is None
    assert merged[0].fault_history == ({"visual_status": "FAULT_REFUTED"},)
    assert rows[0]["associated_component_ids"] == ["map-id"]
    rows[0]["상태"] = "제외"
    hidden, _rows = _bind_review_models_to_map((map_box,), (review_box,), rows)
    assert hidden == ()


def test_ambiguous_map_geometry_is_not_arbitrarily_bound_or_drawn_twice():
    map_box = ComponentOverlayViewModel(**{**_model().__dict__, "component_id": "map-id"})
    left = ComponentOverlayViewModel(**{**_model().__dict__, "component_id": "review-a",
                                        "overlay_source": "yolo"})
    right = ComponentOverlayViewModel(**{**_model().__dict__, "component_id": "review-b",
                                         "overlay_source": "yolo"})
    rows = [
        {"component_id": item, "소자 종류": "resistor", "상태": "검수 필요",
         "x1": 10, "y1": 20, "x2": 30, "y2": 50}
        for item in ("review-a", "review-b")
    ]
    merged, rows = _bind_review_models_to_map((map_box,), (left, right), rows)
    assert {item.component_id for item in merged} == {"review-a", "review-b"}
    assert all("associated_component_ids" not in row for row in rows)
    assert all(row["map_match_status"] == "AMBIGUOUS_GEOMETRY" for row in rows)


def test_photo_component_snapshots_are_prepared_without_rendering_the_viewer(monkeypatch):
    image_bytes = []
    for color in ("red", "blue"):
        output = io.BytesIO()
        Image.new("RGB", (40, 30), color).save(output, format="PNG")
        image_bytes.append(output.getvalue())
    project = ProjectContext(
        "p1", "b1", image_records=(
            ImageRecord("photo-1", side="front"), ImageRecord("photo-2", side="back"),
        ),
    )
    state = {"circuitcare_v2_active_photo_id": "photo-1",
             "circuitcare_v2_photo_choice": "photo-1"}
    st = SimpleNamespace(session_state=state, status=lambda *_args, **_kwargs: _ProgressStub())
    detection_calls = []

    def fake_detection(st_api, *, image_bytes, project_id, photo_id,
                       review_image_bytes=None, review_image_size=None, progress=None):
        detection_calls.append(photo_id)
        scope = unified_workspace.component_review_scope(review_image_bytes, project_id, photo_id)
        _result_key, rows_key, _error_key = review_keys(scope)
        component_id = "R1" if photo_id == "photo-1" else "C1"
        st_api.session_state[rows_key] = [{
            "component_id": component_id, "소자 종류": "resistor" if component_id == "R1" else "capacitor",
            "상태": "검수 필요", "x1": 1, "y1": 1, "x2": 10, "y2": 10,
            "body_marking": "AMS1117", "possible_part_number": "AMS1117",
            "part_number_status": "UNVERIFIED_PART_ID", "ocr_source": "RapidOCR", "ocr_score": .88,
            "qwen_marking_candidates": ["AMS1117-3.3"],
            "qwen_conflicting_evidence": ["OCR 후보가 일치하지 않음"],
        }]
        return {"detections": []}, None

    from circuitcare.dashboard.views import unified_workspace
    monkeypatch.setattr(unified_workspace, "ensure_component_detection", fake_detection)
    photos = [{"image_id": "photo-1", "data": image_bytes[0]},
              {"image_id": "photo-2", "data": image_bytes[1]}]
    first = _prepare_photo_component_snapshot(st, project, photos)
    assert [item.component_id for item in first["components"]] == ["R1"]

    state["circuitcare_v2_photo_choice"] = "photo-2"
    state["circuitcare_v2_active_photo_id"] = "photo-2"
    second = _prepare_photo_component_snapshot(st, project, photos)
    assert [item.component_id for item in second["components"]] == ["C1"]
    snapshots = state["circuitcare_v6_photo_component_snapshots"]
    assert first["analysis_scope_key"] != second["analysis_scope_key"]
    assert snapshots[first["analysis_scope_key"]]["photo_id"] == "photo-1"
    assert snapshots[second["analysis_scope_key"]]["photo_id"] == "photo-2"
    first_source = snapshots[first["analysis_scope_key"]]["registry_components"][0]
    assert first_source["possible_part_number"] == "AMS1117"
    assert first_source["part_number_status"] == "UNVERIFIED_PART_ID"
    assert first_source["ocr_source"] == "RapidOCR"
    assert first_source["qwen_marking_candidates"] == ["AMS1117-3.3"]
    from circuitcare.frontend.board_component_registry import build_board_component_registry
    registry = build_board_component_registry({}, project_id="p1", board_id="b1", photo_snapshots={
        "photo-1": snapshots[first["analysis_scope_key"]],
    })
    board_source = registry["components"][0]["source_components"][0]
    assert board_source["possible_part_number"] == "AMS1117"
    assert board_source["qwen_conflicting_evidence"] == ["OCR 후보가 일치하지 않음"]


def test_initial_yolo_status_uses_dedicated_slot_and_finishes_explicitly(monkeypatch):
    from contextlib import nullcontext
    from circuitcare.dashboard.views import unified_workspace

    class Status:
        def __init__(self):
            self.writes = []
            self.updates = []

        def write(self, message):
            self.writes.append(message)

        def update(self, **kwargs):
            self.updates.append(kwargs)

    status = Status()
    class Slot:
        def container(self):
            return nullcontext()

    output = io.BytesIO()
    Image.new("RGB", (32, 24), "navy").save(output, format="PNG")
    photo = output.getvalue()
    project = ProjectContext("p-status", "b-status", image_records=(ImageRecord("photo-status"),))
    state = {"circuitcare_v2_photo_choice": "photo-status"}
    st = SimpleNamespace(session_state=state, status=lambda *_a, **_k: status)

    def fake_detection(st_api, *, image_bytes, project_id, photo_id, progress=None, **_kwargs):
        assert progress is status
        scope = unified_workspace.component_review_scope(
            unified_workspace._cached_photo_pixels(image_bytes)[3], project_id, photo_id)
        _result, rows_key, _error = review_keys(scope)
        st_api.session_state[rows_key] = []
        return {"detections": [], "image_width": 32, "image_height": 24}, None

    monkeypatch.setattr(unified_workspace, "ensure_component_detection", fake_detection)
    unified_workspace._prepare_photo_component_snapshot(
        st, project, [{"image_id": "photo-status", "data": photo}],
        workflow_status_slot=Slot(),
    )
    assert status.updates[-1]["state"] == "complete"
    assert "YOLO 탐지 완료" in status.updates[-1]["label"]

    failed_status = Status()
    st.status = lambda *_a, **_k: failed_status
    st.session_state = {"circuitcare_v2_photo_choice": "photo-status"}
    monkeypatch.setattr(unified_workspace, "ensure_component_detection",
                        lambda *_a, **_k: (None, "weights missing"))
    unified_workspace._prepare_photo_component_snapshot(
        st, project, [{"image_id": "photo-status", "data": photo}],
        workflow_status_slot=Slot(),
    )
    assert failed_status.updates[-1]["state"] == "error"


def test_unvalidated_qwen_cleanup_never_auto_excludes_and_protects_user_confirmed_rows():
    assert qwen_cleanup_disposition({"decision": "suggest", "proposed_class": "not_component"}) == (
        "REVIEW_REQUIRED", "실제 데이터로 검증된 자동 정리 기준이 없어 사용자 확인 필요")
    assert qwen_cleanup_disposition({"decision": "same_component"}, geometry_valid=False)[0] == "REVIEW_REQUIRED"
    assert qwen_cleanup_disposition({"decision": "different_components"})[0] == "KEEP"
    assert qwen_cleanup_disposition({"decision": "same_component"}, protected=True)[0] == "USER_CONFIRMED"
    validated = {"validated": True, "policy_version": "measured-v1",
                 "independent_dataset_fingerprint": "sha256:fixture", "evaluation_mode": "REAL_LABELED_PCB",
                 "metrics": {"sample_count": 1000, "duplicate_precision_lower_bound": .999,
                             "normal_component_exclusion_rate_upper_bound": 0,
                             "group_or_connector_false_exclusion_rate_upper_bound": 0}}
    assert qwen_cleanup_disposition({"decision": "suggest", "proposed_class": "resistor"},
                                    validation=validated)[0] == "REVIEW_REQUIRED"
    assert qwen_cleanup_disposition({"decision": "same_component"}, geometry_valid=True,
                                    validation=validated)[0] == "REVIEW_REQUIRED"
    assert qwen_cleanup_disposition({"decision": "same_component", "evidence": "same package body"},
                                    geometry_valid=True, validation=validated)[0] == "AUTO_EXCLUDED"


def test_qwen_class_suggestion_applies_automatically_without_confirming_and_keeps_fingerprint_stable():
    row = {"component_id": "c1", "소자 종류": "unknown", "상태": "검수 필요",
           "x1": 1, "y1": 2, "x2": 20, "y2": 30}
    before = _selected_classification_detections([row], {"c1"})
    assert _apply_qwen_proposal_to_row(row, {
        "detection_id": "c1", "object_status": "COMPONENT", "proposed_class": "capacitor",
        "proposed_subtype": "electrolytic", "confidence": .8, "evidence": "visible can package",
    })
    assert row["소자 종류"] == "capacitor"
    assert row["상태"] == "검수 필요"
    assert row["qwen_component_type_status"] == "AI_PROPOSED_UNVERIFIED"
    assert _selected_classification_detections([row], {"c1"}) == before
    assert not _apply_qwen_proposal_to_row(row, row["ai_review"])


def test_qwen_false_positive_requires_explicit_object_status_and_user_rows_are_protected():
    row = {"component_id": "c1", "소자 종류": "unknown", "상태": "검수 필요"}
    assert not _apply_qwen_proposal_to_row(row, {"proposed_class": "not_component"})
    assert _apply_qwen_proposal_to_row(row, {
        "detection_id": "c1", "proposed_class": "not_component", "object_status": "NOT_COMPONENT",
    })
    confirmed = {"component_id": "c2", "소자 종류": "resistor", "상태": "확인"}
    assert not _apply_qwen_proposal_to_row(confirmed, {
        "detection_id": "c2", "proposed_class": "capacitor", "object_status": "COMPONENT",
    })
    edited = {"component_id": "c3", "소자 종류": "ic", "상태": "검수 필요", "user_edited": True}
    assert not _apply_qwen_proposal_to_row(edited, {
        "detection_id": "c3", "proposed_class": "resistor", "object_status": "COMPONENT",
    })


def test_board_registry_prioritizes_selected_photo_and_reuses_processed_snapshots(monkeypatch):
    image_bytes = []
    for color in ("red", "blue", "green"):
        output = io.BytesIO()
        Image.new("RGB", (40, 30), color).save(output, format="PNG")
        image_bytes.append(output.getvalue())
    photo_ids = ("front", "back", "close-up")
    project = ProjectContext(
        "p-board", "b-board",
        image_records=tuple(ImageRecord(photo_id, side="unknown",
                                       capture_type=("close_up" if photo_id == "close-up" else "full_view"))
                            for photo_id in photo_ids),
    )
    state = {"circuitcare_v2_active_photo_id": "front",
             "circuitcare_v2_photo_choice": "front"}
    st = SimpleNamespace(session_state=state, status=lambda *_args, **_kwargs: _ProgressStub())
    detection_calls = []

    def fake_detection(st_api, *, image_bytes, project_id, photo_id,
                       review_image_bytes=None, review_image_size=None, progress=None):
        detection_calls.append(photo_id)
        scope = unified_workspace.component_review_scope(review_image_bytes, project_id, photo_id)
        _result_key, rows_key, _error_key = review_keys(scope)
        st_api.session_state[rows_key] = [{
            "component_id": f"det-{photo_id}", "소자 종류": "resistor",
            "상태": "확인" if photo_id == "front" else "검수 필요",
            "x1": 1, "y1": 1, "x2": 10, "y2": 10,
        }]
        return {"detections": []}, None

    from circuitcare.dashboard.views import unified_workspace
    monkeypatch.setattr(unified_workspace, "ensure_component_detection", fake_detection)
    photos = [{"image_id": photo_id, "data": data}
              for photo_id, data in zip(photo_ids, image_bytes)]

    registry, selected = _prepare_board_registry_snapshot(st, project, photos)

    assert selected["photo_id"] == "front"
    assert {item["photo_id"] for item in registry["photos"]} == set(photo_ids)
    assert {source["photo_id"] for row in registry["components"]
            for source in row["source_components"]} == {"front"}
    assert detection_calls == ["front"]
    assert len(state["circuitcare_v6_photo_component_snapshots"]) == 3
    assert {item["photo_id"] for item in state["circuitcare_v7_photo_snapshots_by_id"].values()} == set(photo_ids)


def test_redetection_keeps_exclusion_and_does_not_create_a_resurrection_duplicate():
    old_rows = [{"component_id": "excluded-1", "소자 종류": "resistor", "상태": "제외",
                 "x1": 10, "y1": 20, "x2": 30, "y2": 50, "user_edited": True,
                 "source": "yolo", "exclusion_reason": "qwen_not_component"}]
    detections = [
        {"class_name": "resistor", "confidence": .91, "bbox_xyxy": [10, 20, 30, 50]},
        {"class_name": "resistor", "confidence": .89, "bbox_xyxy": [10, 20, 30, 50]},
    ]
    from circuitcare.vision.box_reconciliation import preserve_review_rows
    rows = preserve_review_rows(old_rows, detections)
    assert len(rows) == 1
    assert rows[0]["component_id"] == "excluded-1"
    assert rows[0]["상태"] == "제외"
    assert board_visible_review_rows(rows) == []


def test_all_exclusion_sources_share_undo_snapshot_and_preserve_reason():
    row = {"component_id": "c1", "소자 종류": "resistor", "상태": "확인",
           "x1": 1, "y1": 2, "x2": 10, "y2": 12}
    state = {}
    record_box_exclusion(state, scope="scope", row=row, reason="qwen_duplicate_review",
                         review={"decision": "same_component"}, duplicate_of="c0")
    assert row["상태"] == "제외" and row["duplicate_of"] == "c0"
    assert undo_last_box_exclusion(state, scope="scope", rows=[row]) == "c1"
    assert row["상태"] == "확인" and "duplicate_of" not in row


def test_auto_excluded_box_remains_reversible_and_keeps_review_metadata():
    row = {"component_id": "auto1", "소자 종류": "resistor", "상태": "검수 필요",
           "x1": 1, "y1": 2, "x2": 10, "y2": 12,
           "ai_review": {"decision": "same_component", "evidence": "same physical body"},
           "duplicate_of": "canonical1"}
    state = {}
    record_box_exclusion(state, scope="scope", row=row, reason="AUTO_EXCLUDED_VALIDATED_POLICY",
                         review=row["ai_review"], duplicate_of="canonical1")
    assert row["상태"] == "제외"
    restore_review_row(row)
    assert row["상태"] == "검수 필요"
    assert row["component_id"] == "auto1"
    assert row["ai_review"]["decision"] == "same_component"


def test_hover_identity_status_is_user_facing_korean_and_unknown_is_preserved():
    assert display_identity_status("SUGGESTED_UNCONFIRMED") == "미확인 후보"
    assert display_identity_status("IDENTITY_MATCHED") == "동일 부품 후보"
    assert display_identity_status("IDENTITY_AMBIGUOUS") == "식별 모호"
    assert display_identity_status("FUTURE_STATUS") == "FUTURE_STATUS"


def test_coordinate_status_requires_client_mount_and_valid_frame():
    assert coordinate_status_for_viewer(api_available=True, client_ready=False) == "UNVERIFIED_RESPONSIVE_CSS"
    assert coordinate_status_for_viewer(api_available=True, client_ready=True) == "VERIFIED_SHARED_SVG_VIEWBOX"
    assert coordinate_status_for_viewer(api_available=False, client_ready=False) == "VIEWER_UNAVAILABLE"
    assert coordinate_status_for_viewer(api_available=True, client_ready=True, frame_valid=False) == "INVALID_COORDINATE_FRAME"
