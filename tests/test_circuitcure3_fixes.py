from circuitcare.symptom_diagnosis.measurement_evaluator import evaluate_component_measurements
from circuitcare.symptom_diagnosis.observation_store import ObservationStore
from circuitcare.vision.component_recognition_panel import _store_predictions


def test_rerun_reuses_component_id_and_review_status_by_iou():
    state = {"rows": [{"component_id": "auto-existing", "소자 종류": "resistor", "상태": "확인",
                       "신뢰도": .8, "x1": 10, "y1": 10, "x2": 40, "y2": 40}]}
    _store_predictions(state, "rows", {"detections": [{"class_name": "unknown", "confidence": 0,
        "location_confidence": .9, "bbox_xyxy": [11, 10, 41, 40]}]})
    assert state["rows"][0]["component_id"] == "auto-existing"
    assert state["rows"][0]["상태"] == "확인"
    assert state["rows"][0]["소자 종류"] == "resistor"


def test_rerun_keeps_manual_and_qwen_boxes_when_detector_misses_them():
    state = {"rows": [
        {"component_id": "manual-1", "소자 종류": "unknown", "상태": "검수 필요",
         "source": "manual", "x1": 70, "y1": 70, "x2": 90, "y2": 90},
        {"component_id": "qwen-1", "소자 종류": "capacitor", "상태": "검수 필요",
         "source": "qwen_suggestion", "x1": 110, "y1": 70, "x2": 130, "y2": 90},
    ]}
    _store_predictions(state, "rows", {"detections": []})
    assert [row["component_id"] for row in state["rows"]] == ["manual-1", "qwen-1"]


def test_conflicted_measurement_verdict_is_not_overwritten_by_later_measurement():
    def measurement(identifier, value):
        return {"component_id": "R1", "fault_hypothesis": "R_FAIL", "test_id": "resistance",
                "board_component_id": "bc-r1", "identity_resolution_status": "USER_RECONFIRMED_VERIFIED",
                "measurement_target": "A-B", "operating_conditions": "power off",
                "measurement_id": identifier, "source": "user_measurement", "verification_status": "USER_REPORTED",
                "component_identity_status": "USER_CONFIRMED", "location_verified": True,
                "uncertainty": .01, "uncertainty_source": "meter", "reference_verification_status": "DATASHEET_VERIFIED",
                "reference_source_id": "ds-1", "reference_source": "datasheet", "reference_conditions": "power off",
                "component_type": "resistor", "measurement_type": "ohm", "value": value,
                "reference_min": 1, "reference_max": 2}
    profiles = {"profiles": {"resistor": {"tests": {"resistance": {
        "measurement_type": "ohm", "target_keywords": ["A-B"], "condition_keywords": ["power off"],
        "fault_relations": {"R_FAIL": {"TEST_NORMAL": "supports", "TEST_ABNORMAL": "refutes"}}}}}}}
    result = evaluate_component_measurements([measurement("m1", 1.5), measurement("m2", 9), measurement("m3", 1.5)], profiles=profiles)
    verdict = next(iter(result.values()))
    assert verdict["status"] == "CONFLICTED"
    assert verdict["verdict"] == "conflicted"
    assert verdict["assessment_status"] == "CONFLICTED"
    assert verdict["decision_scope"] == "FAULT_HYPOTHESIS_ONLY"
    assert verdict["evidence_ids"] == ["m1", "m2", "m3"]


def test_observation_store_keeps_current_upsert_and_append_only_change_history(tmp_path):
    store = ObservationStore(tmp_path / "observations.sqlite")
    base = {"id": "o1", "code": "VOLTAGE", "value": 3.3, "status": "observed", "source": "user"}
    store.save(diagnosis_id="d1", project_id="p1", board_id=None, status="open", result={}, observations=[base])
    changed = {**base, "value": 0.0}
    store.save(diagnosis_id="d1", project_id="p1", board_id=None, status="open", result={}, observations=[changed])
    assert len(store.get("d1")["observations"]) == 1
    import sqlite3
    with sqlite3.connect(tmp_path / "observations.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM observation_change_history").fetchone()[0] == 1
