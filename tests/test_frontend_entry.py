from __future__ import annotations

from pathlib import Path
import hashlib
import io

import pytest
from PIL import Image

from circuitcare.frontend.intake import create_preview_context, create_project_context
from circuitcare.frontend.models import ImageRecord, ProjectContext
from circuitcare.frontend.state import INTAKE_IMAGES_KEY, INTAKE_PROJECT_KEY
from circuitcare.frontend.component_map_adapter import (
    COMPONENT_MAP_STATE_KEY, component_map_cache_key, photo_input_fingerprint,
)
from circuitcare.frontend.photo_adapter import adapt_component_map_input


def test_v2_flag_routes_to_real_entry_screen(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    project_root = Path(__file__).parents[1]
    monkeypatch.setenv("CIRCUITCARE_UI_V2", "1")
    app = AppTest.from_file(str(project_root / "src/circuitcare/ui.py"), default_timeout=30).run()

    assert not app.exception
    assert any("새 PCB 진단" in item.value for item in app.markdown)
    assert any(button.label == "예제 회로 체험" for button in app.button)
    assert any(button.label == "진단 시작" for button in app.button)

    app.radio[0].set_value("overheating").run()
    app.text_area[0].set_value("전원 연결 후 보드가 뜨거워짐").run()
    assert app.radio[0].value == "overheating"
    assert app.text_area[0].value == "전원 연결 후 보드가 뜨거워짐"

    demo_button = next(button for button in app.button if button.label == "예제 회로 체험")
    demo_button.click().run()
    assert not app.exception
    assert any(selectbox.label == "회로 데이터셋" for selectbox in app.selectbox)


def test_default_entry_keeps_legacy_ui_when_v2_flag_is_unset(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    project_root = Path(__file__).parents[1]
    monkeypatch.delenv("CIRCUITCARE_UI_V2", raising=False)
    app = AppTest.from_file(str(project_root / "src/circuitcare/ui.py"), default_timeout=30).run()

    assert not app.exception
    # The existing demo case chooser is still present on the legacy route.
    assert any(selectbox.label == "회로 데이터셋" for selectbox in app.selectbox)


def test_existing_project_draft_shows_prepared_state_without_claiming_analysis(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    image = Image.new("RGB", (40, 30), "blue")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    raw = stream.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    record = {
        "image_id": digest[:20], "sha256": digest, "filename": "board.png",
        "data": raw, "role": "unknown", "media_type": "image/png",
        "width": 40, "height": 30,
    }
    project = create_project_context(
        images=[record], symptom_category="unknown", symptom_description=None,
    )
    project_root = Path(__file__).parents[1]
    monkeypatch.setenv("CIRCUITCARE_UI_V2", "1")
    app = AppTest.from_file(str(project_root / "src/circuitcare/ui.py"), default_timeout=30)
    app.session_state[INTAKE_IMAGES_KEY] = [record]
    app.session_state[INTAKE_PROJECT_KEY] = project
    app.run()

    assert not app.exception
    assert any("프로젝트 생성 완료 · 진단 미실행" in item.value for item in app.success)
    assert any("통합 Viewer는 다음 단계에서 연결" in item.value for item in app.info)


def test_photo_upload_role_assignment_and_start_create_session_project(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    def png(color: str) -> bytes:
        image = Image.new("RGB", (64, 48), color)
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()

    project_root = Path(__file__).parents[1]
    monkeypatch.setenv("CIRCUITCARE_UI_V2", "1")
    app = AppTest.from_file(str(project_root / "src/circuitcare/ui.py"), default_timeout=30).run()
    app.file_uploader[0].set_value([
        ("front.png", png("red"), "image/png"),
        ("detail.png", png("blue"), "image/png"),
    ]).run()

    assert not app.exception
    assert len(app.session_state[INTAKE_IMAGES_KEY]) == 2
    assert all(item["side"] == "unknown" for item in app.session_state[INTAKE_IMAGES_KEY])
    assert all(item["capture_type"] == "unknown" for item in app.session_state[INTAKE_IMAGES_KEY])
    assert next(button for button in app.button if button.label == "진단 시작").disabled is False

    app.selectbox[0].set_value("front").run()
    app.selectbox[1].set_value("close_up").run()
    assert app.session_state[INTAKE_IMAGES_KEY][0]["side"] == "front"
    assert app.session_state[INTAKE_IMAGES_KEY][0]["capture_type"] == "close_up"
    app.radio[0].set_value("overheating").run()
    app.text_area[0].set_value("  전원을 켜면   과열됨  ").run()
    next(button for button in app.button if button.label == "진단 시작").click().run()

    assert not app.exception
    project = app.session_state[INTAKE_PROJECT_KEY]
    assert project.symptom_category == "overheating"
    assert project.symptom_description == "전원을 켜면 과열됨"
    assert project.image_records[0].side == "front"
    assert project.image_records[0].capture_type == "close_up"
    assert project.input_fingerprint
    assert app.session_state["circuitcare_v2_screen"] == "workbench"


def test_photo_removal_preserves_symptom_input(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    image = Image.new("RGB", (32, 24), "green")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    project_root = Path(__file__).parents[1]
    monkeypatch.setenv("CIRCUITCARE_UI_V2", "1")
    app = AppTest.from_file(str(project_root / "src/circuitcare/ui.py"), default_timeout=30).run()
    app.file_uploader[0].set_value(("board.png", stream.getvalue(), "image/png")).run()
    app.radio[0].set_value("power_failure").run()
    next(button for button in app.button if button.label == "사진 제거").click().run()

    assert not app.exception
    assert app.session_state[INTAKE_IMAGES_KEY] == []
    assert app.radio[0].value == "power_failure"
    assert next(button for button in app.button if button.label == "진단 시작").disabled


def test_edit_preserves_inputs_and_new_project_clears_inputs_and_widget_state(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    image = Image.new("RGB", (40, 30), "blue")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    raw = stream.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    record = {
        "image_id": "photo-1", "sha256": digest, "filename": "board.png", "data": raw,
        "side": "back", "capture_type": "oblique", "media_type": "image/png",
        "width": 40, "height": 30, "image_dimensions": (40, 30), "image_format": "PNG",
    }
    project = create_project_context(
        images=[record], symptom_category="overheating", symptom_description="뜨거움",
    )
    project_root = Path(__file__).parents[1]
    monkeypatch.setenv("CIRCUITCARE_UI_V2", "1")
    app = AppTest.from_file(str(project_root / "src/circuitcare/ui.py"), default_timeout=30)
    app.session_state[INTAKE_IMAGES_KEY] = [record]
    app.session_state[INTAKE_PROJECT_KEY] = project
    app.session_state["circuitcare_v2_symptom_category"] = "overheating"
    app.session_state["circuitcare_v2_symptom_description"] = "뜨거움"
    app.session_state["circuitcare_v2_analysis_status"] = "STALE"
    app.run()

    next(button for button in app.button if button.label == "입력 수정").click().run()
    assert len(app.session_state[INTAKE_IMAGES_KEY]) == 1
    assert app.session_state["circuitcare_v2_symptom_category"] == "overheating"
    assert "circuitcare_v2_analysis_status" not in app.session_state

    # Recreate a completed draft and perform a clean new-project transition.
    project = create_project_context(
        images=app.session_state[INTAKE_IMAGES_KEY], symptom_category="overheating",
        symptom_description="뜨거움",
    )
    app.session_state[INTAKE_PROJECT_KEY] = project
    app.run()
    app.session_state["cc_v2_photo_upload_0"] = "old-upload-widget-state"
    next(button for button in app.button if button.label == "새 프로젝트").click().run()
    assert app.session_state[INTAKE_IMAGES_KEY] == []
    assert app.session_state["circuitcare_v2_symptom_category"] == "unknown"
    assert app.session_state["circuitcare_v2_symptom_description"] == ""
    assert "cc_v2_photo_upload_0" not in app.session_state


def test_f2_start_enters_pcb_workbench_and_keeps_photo_across_tab_switch(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    def png(color: str) -> bytes:
        image = Image.new("RGB", (64, 48), color)
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()

    monkeypatch.setenv("CIRCUITCARE_UI_V2", "1")
    app = AppTest.from_file(str(Path(__file__).parents[1] / "src/circuitcare/ui.py"), default_timeout=30).run()
    app.file_uploader[0].set_value([
        ("front.png", png("red"), "image/png"),
        ("detail.png", png("blue"), "image/png"),
    ]).run()
    app.selectbox[0].set_value("front").run()
    app.selectbox[1].set_value("full_view").run()
    app.selectbox[2].set_value("back").run()
    app.selectbox[3].set_value("close_up").run()
    app.radio[0].set_value("output_anomaly").run()
    app.text_area[0].set_value("출력 전압이 낮음").run()
    next(button for button in app.button if button.label == "진단 시작").click().run()

    assert not app.exception
    project = app.session_state[INTAKE_PROJECT_KEY]
    assert app.session_state["circuitcare_v2_screen"] == "workbench"
    assert project.project_id == app.session_state["circuitcare_v2_selection"].project_id
    assert project.symptom_category == "output_anomaly"
    assert len(app.session_state[INTAKE_IMAGES_KEY]) == 2
    assert app.session_state["circuitcare_v2_active_tab"] == "PCB"
    assert any(button.label == "입력 수정" for button in app.button)
    assert any(button.label == "새 프로젝트" for button in app.button)
    assert any("진단 미실행" in item.value for item in app.info)
    assert any("64×48 px" in item.value for item in app.caption)

    photo_id = app.session_state["circuitcare_v2_active_photo_id"]
    app.session_state["circuitcare_v2_active_tab"] = "회로도"
    app.run()
    assert not app.exception
    assert app.session_state["circuitcare_v2_active_photo_id"] == photo_id
    assert any("회로도 없음" in item.value for item in app.info)
    app.session_state["circuitcare_v2_active_tab"] = "PCB"
    app.run()
    assert not app.exception
    assert any("64×48 px" in item.value for item in app.caption)


def test_workbench_edit_preserves_project_identity_and_new_project_clears_state(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    image = Image.new("RGB", (32, 24), "green")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    monkeypatch.setenv("CIRCUITCARE_UI_V2", "1")
    app = AppTest.from_file(str(Path(__file__).parents[1] / "src/circuitcare/ui.py"), default_timeout=30).run()
    app.file_uploader[0].set_value(("board.png", stream.getvalue(), "image/png")).run()
    next(button for button in app.button if button.label == "진단 시작").click().run()
    first = app.session_state[INTAKE_PROJECT_KEY]
    next(button for button in app.button if button.label == "입력 수정").click().run()
    assert INTAKE_PROJECT_KEY not in app.session_state
    assert app.session_state[INTAKE_IMAGES_KEY]
    app.text_area[0].set_value("전원 LED 꺼짐").run()
    next(button for button in app.button if button.label == "진단 시작").click().run()
    revised = app.session_state[INTAKE_PROJECT_KEY]
    assert revised.project_id == first.project_id
    assert revised.board_id == first.board_id
    assert revised.symptom_description == "전원 LED 꺼짐"

    next(button for button in app.button if button.label == "새 프로젝트").click().run()
    assert not app.session_state[INTAKE_IMAGES_KEY]
    assert INTAKE_PROJECT_KEY not in app.session_state
    assert "circuitcare_v2_selected_component" not in app.session_state
    assert app.session_state["circuitcare_v2_screen"] == "entry"


def test_photo_free_preview_and_add_photo_returns_to_same_project(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    def png() -> bytes:
        image = Image.new("RGB", (64, 48), "purple")
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()

    monkeypatch.setenv("CIRCUITCARE_UI_V2", "1")
    app = AppTest.from_file(str(Path(__file__).parents[1] / "src/circuitcare/ui.py"), default_timeout=30).run()
    app.radio[0].set_value("overheating").run()
    app.text_area[0].set_value("팬이 돌지 않음").run()
    next(button for button in app.button if button.label == "사진 없이 작업공간 보기").click().run()

    assert not app.exception
    preview = app.session_state[INTAKE_PROJECT_KEY]
    assert preview.image_records == ()
    assert preview.symptom_category == "overheating"
    assert preview.symptom_description == "팬이 돌지 않음"
    assert app.session_state["circuitcare_v2_workbench_mode"] == "PREVIEW_ONLY"
    assert app.session_state["circuitcare_v2_photo_status"] == "NO_IMAGES"
    assert app.session_state["circuitcare_v2_active_tab"] == "PCB"
    assert app.session_state["circuitcare_v2_view_transform"]["coordinate_status"] == "NO_IMAGE"
    assert not app.selectbox
    assert not app.slider
    assert any("미리보기 · 진단 미실행" in item.value for item in app.info)

    next(button for button in app.button if button.label == "사진 추가").click().run()
    assert app.session_state["circuitcare_v2_screen"] == "entry"
    assert INTAKE_PROJECT_KEY not in app.session_state
    assert app.session_state["circuitcare_v2_edit_project_id"] == preview.project_id
    assert app.radio[0].value == "overheating"
    assert app.text_area[0].value == "팬이 돌지 않음"

    app.file_uploader[0].set_value(("board.png", png(), "image/png")).run()
    next(button for button in app.button if button.label == "진단 시작").click().run()
    completed = app.session_state[INTAKE_PROJECT_KEY]
    assert not app.exception
    assert len(completed.image_records) == 1
    assert completed.project_id == preview.project_id
    assert completed.board_id == preview.board_id
    assert app.session_state["circuitcare_v2_workbench_mode"] == "DIAGNOSTIC"
    assert app.session_state["circuitcare_v2_photo_status"] == "READY"
    next(button for button in app.button if button.label == "새 프로젝트").click().run()
    assert not app.session_state[INTAKE_IMAGES_KEY]
    assert INTAKE_PROJECT_KEY not in app.session_state
    assert "circuitcare_v2_workbench_mode" not in app.session_state
    assert "circuitcare_v2_view_transform" not in app.session_state


def test_preview_context_is_separate_from_strict_diagnostic_factory():
    preview = create_preview_context(
        symptom_category="unknown", symptom_description=None,
        project_id="preview-project", board_id="preview-board",
    )
    assert preview.image_records == ()
    assert preview.project_id == "preview-project"
    assert preview.board_id == "preview-board"
    with pytest.raises(ValueError, match="한 장 이상"):
        create_project_context(images=[], symptom_category="unknown", symptom_description=None)


@pytest.mark.parametrize("sides, expected", [
    (["front", "front", "back"], {"앞면": 2, "뒷면": 1}),
    (["front", "unknown"], {"앞면": 1, "미지정": 1}),
    (["front", "back", "unknown"], {"앞면": 1, "뒷면": 1, "미지정": 1}),
    (["unknown", "unknown", "unknown"], {"미지정": 3}),
])
def test_workbench_exposes_originals_for_every_present_side(monkeypatch, sides, expected):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CIRCUITCARE_UI_V2", "1")
    project_root = Path(__file__).parents[1]
    app = AppTest.from_file(str(project_root / "src/circuitcare/ui.py"), default_timeout=30)
    records = []
    photos = []
    for index, side in enumerate(sides):
        image = Image.new("RGB", (24, 16), (30 + index, 70, 110))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        data = stream.getvalue()
        image_id = f"photo-{index}"
        digest = hashlib.sha256(data).hexdigest()
        records.append(ImageRecord(
            image_id=image_id, side=side, capture_type="unknown", filename=f"{image_id}.png",
            content_sha256=digest, media_type="image/png", image_dimensions=(24, 16), image_format="PNG",
        ))
        photos.append({"image_id": image_id, "data": data})
    project = ProjectContext("project-side-test", "board-side-test", tuple(records), input_fingerprint="photo-fp")
    app.session_state[INTAKE_PROJECT_KEY] = project
    app.session_state[INTAKE_IMAGES_KEY] = photos
    app.session_state["circuitcare_v2_screen"] = "workbench"
    app.run()

    assert not app.exception
    side_widget = next(widget for widget in app.selectbox if widget.label == "면")
    assert side_widget.options == list(expected)
    original_widget = next(widget for widget in app.selectbox if widget.label == "원본 사진")
    assert len(original_widget.options) == next(iter(expected.values()))


def test_broken_canonical_artifact_falls_back_to_source_without_dropping_map(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CIRCUITCARE_UI_V2", "1")
    image = Image.new("RGB", (24, 16), "orange")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    data = stream.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    record = ImageRecord("photo-1", "front", "full_view", "board.png", digest,
                         "image/png", (24, 16), "PNG")
    project = ProjectContext("fallback-project", "fallback-board", (record,), input_fingerprint="unused")
    request = adapt_component_map_input(record, board_id=project.board_id, data=data).request
    photo_fp = photo_input_fingerprint([record])
    cache_key = component_map_cache_key(
        project_id=project.project_id, board_id=project.board_id, inputs=[request], photo_manifest=[record],
    )
    bad_map = {
        "schema_version": "circuitcare.pcb-component-map.v1", "board_id": project.board_id,
        "photos": [], "sides": {"TOP": {"registrations": [], "canonical_image": {
            "image_id": "frame-bad", "coordinate_frame": "MOSAIC_PIXEL", "image_size_px": [1, 1],
            "reference_photo_id": None, "image_png_base64": "not-base64"}}},
    }
    app = AppTest.from_file(str(Path(__file__).parents[1] / "src/circuitcare/ui.py"), default_timeout=30)
    app.session_state[INTAKE_PROJECT_KEY] = project
    app.session_state[INTAKE_IMAGES_KEY] = [{"image_id": record.image_id, "data": data}]
    app.session_state["circuitcare_v2_screen"] = "workbench"
    app.session_state["circuitcare_v3_view_mode"] = "통합 이미지"
    app.session_state[COMPONENT_MAP_STATE_KEY] = {
        "project_id": project.project_id, "board_id": project.board_id,
        "input_fingerprint": photo_fp, "component_map_cache_key": cache_key,
        "component_map": bad_map, "photo_id_map": [], "failed_artifacts": [],
    }
    app.run()

    assert not app.exception
    assert app.session_state["circuitcare_v3_view_mode"] == "원본 사진"
    assert any("원본 사진으로 전환" in warning.value for warning in app.warning)
    assert app.session_state[COMPONENT_MAP_STATE_KEY]["component_map"] == bad_map
    assert app.session_state[COMPONENT_MAP_STATE_KEY]["failed_artifacts"]
    assert app.session_state["circuitcare_v2_view_transform"]["source_frame_id"] == record.image_id
    app.run()
    assert not app.exception
    assert not app.warning
    assert app.session_state["circuitcare_v2_view_transform"]["source_frame_id"] == record.image_id

