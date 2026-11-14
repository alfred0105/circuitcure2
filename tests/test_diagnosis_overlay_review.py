from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel
from circuitcare.dashboard.views.unified_workspace import _attach_verified_history
from circuitcare.symptom_diagnosis.diagnosis_state import (
    build_overlay_state,
    component_inventory_version,
    current_overlay_candidates,
    current_verified_component_history,
)
from circuitcare.symptom_diagnosis.fault_hypothesis_validator import validate_component_hypothesis
from circuitcare.symptom_diagnosis import diagnose, load_knowledge
from pathlib import Path


def test_photo_inventory_version_scopes_overlay_and_history_without_component_map():
    version = component_inventory_version(
        project_id="p1", board_id="b1", photo_id="photo-1", image_sha256="abc",
        components=[{"component_id": "D1", "component_type": "diode", "bbox_xyxy": [1, 2, 3, 4]}],
    )
    same_version = component_inventory_version(
        project_id="p1", board_id="b1", photo_id="photo-1", image_sha256="abc",
        components=[{"component_id": "D1", "component_type": "DIODE", "bbox_xyxy": [1, 2, 3, 4]}],
    )
    changed_version = component_inventory_version(
        project_id="p1", board_id="b1", photo_id="photo-1", image_sha256="abc",
        components=[{"component_id": "D1", "component_type": "diode", "bbox_xyxy": [1, 2, 3, 5]}],
    )
    assert version == same_version
    assert version != changed_version

    state = build_overlay_state(
        project_id="p1", board_id="b1", component_map_version=version,
        result={"input_fingerprint": "diagnosis-1", "component_candidates": [
            {"component_id": "D1", "visual_status": "SUSPECTED", "fault_hypothesis": "DIODE_OPEN_FAILURE"}
        ], "verified_component_history": [
            {"component_id": "D2", "visual_status": "FAULT_REFUTED",
             "fault_hypothesis": "DIODE_LEAKAGE_FAILURE", "measurement_verdict": "refutes"}
        ]},
    )
    args = {"project_id": "p1", "board_id": "b1", "component_map_version": version,
            "input_fingerprint": "diagnosis-1"}
    assert set(current_overlay_candidates(state, **args)) == {"D1"}
    history = current_verified_component_history(state, **args)
    assert [row["component_id"] for row in history] == ["D2"]
    assert current_overlay_candidates(state, **{**args, "component_map_version": changed_version}) == {}
    assert current_verified_component_history(state, **{**args, "input_fingerprint": "stale"}) == history


def test_presence_type_and_part_identity_are_independent_inputs():
    mode = {"candidate_component_types": ["diode"], "required_component_types": ["diode"]}
    base = {"component_id": "D1", "component_type": "diode"}
    assert not validate_component_hypothesis(
        mode=mode, component={**base, "component_type_status": "UNCONFIRMED"},
        confirmed_component_ids={"D1"}, confirmed_component_types={"diode"},
    )["eligible"]
    assert validate_component_hypothesis(
        mode=mode, component={**base, "component_type_status": "USER_CONFIRMED",
                              "part_identity_status": "UNCONFIRMED"},
        confirmed_component_ids={"D1"}, confirmed_component_types={"diode"},
    )["eligible"]


def test_engine_requires_confirmed_component_type_and_keeps_driver_role_explicit():
    diode = {"component_id": "D1", "component_type": "diode", "component_type_status": "UNCONFIRMED"}
    result = diagnose(symptoms=["OVERHEATING"], components=[diode], confirmed_component_ids=["D1"])
    assert all(row["component_id"] != "D1" for row in result["component_candidates"])

    parts = [
        {"component_id": "Q1", "component_type": "transistor", "component_type_status": "USER_CONFIRMED"},
        {"component_id": "K1", "component_type": "relay", "component_type_status": "USER_CONFIRMED"},
    ]
    inferred = diagnose(symptoms=["RELAY_NO_ACTUATION"], components=parts,
                        confirmed_component_ids=["Q1", "K1"])
    driver = next(row for row in inferred["component_candidates"]
                  if row["fault_hypothesis"] == "RELAY_DRIVER_FAILURE")
    assert driver["component_id"] == "Q1"
    assert driver["role_status"] == "INFERRED"
    confirmed_parts = [dict(parts[0], confirmed_functional_roles=["driver"]), parts[1]]
    confirmed = diagnose(symptoms=["RELAY_NO_ACTUATION"], components=confirmed_parts,
                         confirmed_component_ids=["Q1", "K1"])
    driver = next(row for row in confirmed["component_candidates"]
                  if row["fault_hypothesis"] == "RELAY_DRIVER_FAILURE")
    assert driver["role_status"] == "CONFIRMED"


def test_diode_reverse_leakage_requires_its_own_profile_and_relay_driver_requires_role():
    knowledge = load_knowledge()
    profiles = knowledge["component_test_profiles"]["profiles"]["diode"]["tests"]
    assert "DIODE_LEAKAGE_FAILURE" not in profiles["forward_drop"]["fault_relations"]
    assert profiles["reverse_leakage"]["measurement_type"] == "current"
    assert profiles["reverse_leakage"]["fault_relations"]["DIODE_LEAKAGE_FAILURE"] == {
        "TEST_NORMAL": "refutes", "TEST_ABNORMAL": "supports",
    }
    assert "reverse_leakage" in knowledge["fault_modes"]["fault_modes"]["DIODE_LEAKAGE_FAILURE"][
        "required_measurement_tests"]
    assert knowledge["fault_modes"]["fault_modes"]["RELAY_DRIVER_FAILURE"]["functional_roles"] == ["driver"]


def test_viewer_model_keeps_history_separate_from_active_fault_status():
    item = ComponentOverlayViewModel(
        project_id="p", board_id="b", board_side="TOP", component_id="D1", component_type="diode",
        display_label="D1", canonical_frame_id="photo", coordinate_frame="SOURCE_PHOTO_PIXEL",
        bbox_xyxy=(1, 2, 3, 4), polygon_xy=(), source_backend_photo_ids=("photo",),
        observation_ids=(), detection_confidence=0.9, identification_status="SUGGESTED_CLASS_ONLY",
        identity_match_status=None, ocr_marking=None, possible_part_number=None, evidence=(),
        overlay_status="CANDIDATE", fault_candidate_status="SUSPECTED_FAULT",
        fault_history=({"visual_status": "FAULT_REFUTED", "fault_hypothesis": "DIODE_LEAKAGE_FAILURE"},),
    )
    assert item.fault_candidate_status == "SUSPECTED_FAULT"
    assert item.fault_history[0]["visual_status"] == "FAULT_REFUTED"


def test_verified_history_never_becomes_a_current_pcb_verdict():
    item = ComponentOverlayViewModel(
        project_id="p", board_id="b", board_side="TOP", component_id="D1", component_type="diode",
        display_label="D1", canonical_frame_id="photo", coordinate_frame="SOURCE_PHOTO_PIXEL",
        bbox_xyxy=(1, 2, 3, 4), polygon_xy=(), source_backend_photo_ids=("photo",),
        observation_ids=(), detection_confidence=0.9, identification_status="SUGGESTED_CLASS_ONLY",
        identity_match_status=None, ocr_marking=None, possible_part_number=None, evidence=(),
        overlay_status="CANDIDATE", fault_candidate_status=None,
        diagnosis_display_status="DIAGNOSIS_NOT_RUN",
    )
    historical = {"component_id": "D1", "visual_status": "FAULT_REFUTED",
                  "measurement_verdict": "refutes", "fault_hypothesis": "DIODE_LEAKAGE_FAILURE"}
    updated = _attach_verified_history((item,), {"D1": [historical]})[0]
    assert updated.fault_candidate_status is None
    assert updated.diagnosis_display_status == "DIAGNOSIS_NOT_RUN"
    assert updated.fault_history == (historical,)


def test_viewer_assets_render_supported_refuted_and_unresolved_states():
    assets = Path(__file__).resolve().parents[1] / "src/circuitcare/dashboard/components/assets"
    javascript = (assets / "pcb_viewer.js").read_text(encoding="utf-8")
    styles = (assets / "pcb_viewer.css").read_text(encoding="utf-8")
    markup = (assets / "pcb_viewer.html").read_text(encoding="utf-8")
    for status in ("FAULT_SUPPORTED", "FAULT_REFUTED", "INCONCLUSIVE"):
        assert status in javascript
    for color_class in ("cc-fault-supported", "cc-fault-refuted", "cc-fault-inconclusive"):
        assert color_class in styles and color_class in javascript
    assert "초록 · 고장 가설 반박" in markup
    assert "fault_history" in javascript
