from dataclasses import replace
import math

import pytest

from circuitcare.frontend.measurement_models import create_measurement_record
from circuitcare.frontend.measurement_state import (
    diagnosis_state, get_measurements, save_measurement, update_measurement,
    delete_measurement, current_measurement_set_sha256, MEASUREMENT_HISTORY_KEY,
)
from circuitcare.dashboard.views.measurement_panel import (
    _measurement_verification_gaps, _sigma_for_input_unit,
    _reset_measurement_type_state, _sync_profile_test_state,
)
from circuitcare.analysis.fingerprints import build_analysis_fingerprints
from circuitcare.analysis.models import AnalysisScope, AnalysisStatus
from circuitcare.analysis.state_manager import (
    begin_analysis, commit_analysis_result, get_current_result, get_current_state,
)


def _measurement(**overrides):
    values = {
        "project_id": "p1", "board_id": "b1", "component_id": "c1",
        "component_identity_status": "SUGGESTED_UNCONFIRMED",
        "component_map_version": "map-v1", "measurement_type": "voltage",
        "value": "0", "unit": "V", "uncertainty": "0.1",
        "source": "user_measurement", "measurement_id": "m1", "timestamp": "t1",
    }
    values.update(overrides)
    return create_measurement_record(**values)


@pytest.mark.parametrize(("value", "unit", "expected", "base_unit"), [
    ("0", "V", 0.0, "V"), ("-0.25", "V", -0.25, "V"),
    ("1250", "mV", 1.25, "V"), ("3.5", "kΩ", 3500.0, "Ω"),
    ("250", "mA", 0.25, "A"),
])
def test_measurement_value_accepts_zero_negative_and_converts_units(value, unit, expected, base_unit):
    measurement_type = {"V": "voltage", "mV": "voltage", "kΩ": "resistance", "mA": "current"}[unit]
    row = _measurement(value=value, unit=unit, measurement_type=measurement_type)
    assert row.value == expected
    assert row.unit == base_unit
    assert row.entered_value == float(value)
    assert row.entered_unit == unit
    assert row.node_id is None and row.probe_id is None
    assert row.mapping_status == "NODE_MAPPING_UNAVAILABLE"


@pytest.mark.parametrize("value", ["", "NaN", "Infinity", "-Infinity", "abc"])
def test_invalid_numeric_values_are_rejected(value):
    with pytest.raises(ValueError):
        _measurement(value=value)


@pytest.mark.parametrize("uncertainty", ["0", "-0.1", "NaN", "Infinity", "abc"])
def test_invalid_uncertainty_is_rejected(uncertainty):
    with pytest.raises(ValueError):
        _measurement(uncertainty=uncertainty)


def test_uncertainty_may_be_absent_for_record_only_measurement():
    row = _measurement(uncertainty="")
    assert row.uncertainty is None
    assert row.verification_status == "USER_REPORTED"
    assert "측정 불확실도 또는 출처 미입력" in _measurement_verification_gaps(row)
    assert "전기적 측정 지점 연결 미확인" not in _measurement_verification_gaps(row)


def test_measurement_record_can_be_saved_without_verification_data():
    row = _measurement(uncertainty="", measurement_target="")
    assert row.measurement_id == "m1"
    assert len(_measurement_verification_gaps(row)) >= 3


def test_measurement_history_is_project_board_and_component_scoped_and_mutations_invalidate():
    state = {}
    row = _measurement()
    save_measurement(state, row)
    first_set = current_measurement_set_sha256(state, project_id="p1", board_id="b1")
    assert len(get_measurements(state, project_id="p1", board_id="b1", component_id="c1")) == 1
    assert get_measurements(state, project_id="p2", board_id="b1") == []
    assert get_measurements(state, project_id="p1", board_id="b2") == []
    assert diagnosis_state(state, project_id="p1", board_id="b1")["status"] == "STALE_RESULT"

    update_measurement(state, replace(row, value=2.5, entered_value=2.5, changed_at="t2"))
    updated_set = current_measurement_set_sha256(state, project_id="p1", board_id="b1")
    assert updated_set != first_set
    assert get_measurements(state, project_id="p1", board_id="b1")[0].value == 2.5
    assert delete_measurement(state, project_id="p1", board_id="b1", measurement_id="m1")
    assert get_measurements(state, project_id="p1", board_id="b1") == []
    history = state[MEASUREMENT_HISTORY_KEY]
    assert [event["event_type"] for event in history] == ["CREATED", "UPDATED", "DELETED"]
    assert history[0]["record"]["value"] == 0.0
    assert history[1]["previous_record"]["value"] == 0.0
    assert history[1]["record"]["value"] == 2.5
    assert history[2]["record"]["value"] == 2.5


def test_board_measurement_requires_source_photo_and_local_id_and_preserves_identity():
    state = {}
    row = _measurement(board_component_id="bc-1", source_photo_id="photo-front",
                       source_component_id="local-1",
                       identity_resolution_status="USER_SELECTED_VERIFIED")
    save_measurement(state, row)
    stored = get_measurements(state, project_id="p1", board_id="b1", board_component_id="bc-1")
    assert len(stored) == 1
    assert stored[0].board_component_id == row.board_component_id
    assert stored[0].source_photo_id == row.source_photo_id
    assert stored[0].source_component_id == row.source_component_id
    with pytest.raises(ValueError, match="원본 사진"):
        save_measurement(state, _measurement(measurement_id="m2", board_component_id="bc-2"))


def test_measurement_cannot_be_relinked_to_another_component_on_edit():
    state = {}
    row = _measurement()
    save_measurement(state, row)
    with pytest.raises(ValueError, match="부품 연결"):
        update_measurement(state, replace(row, component_id="c2"))


def test_measurement_cannot_be_relinked_to_a_new_map_or_identity_on_edit():
    state = {}
    row = _measurement()
    save_measurement(state, row)
    with pytest.raises(ValueError, match="Component Map"):
        update_measurement(state, replace(row, component_map_version="map-v2"))
    with pytest.raises(ValueError, match="Component Map"):
        update_measurement(state, replace(row, component_identity_status="USER_CONFIRMED"))


def test_same_measurement_facts_keep_current_analysis_but_changed_facts_stale_it():
    state = {}
    row = _measurement()
    save_measurement(state, row)
    scope = AnalysisScope("p1", "b1", "photo-1")
    fingerprint = build_analysis_fingerprints(
        input_payload={"symptoms": ["no_power"]}, measurements=[row],
    )
    run_id = begin_analysis(state, scope=scope, fingerprints=fingerprint)
    assert commit_analysis_result(state, scope=scope, run_id=run_id,
                                  fingerprints=fingerprint, result={"status": "OK"})

    update_measurement(state, replace(row, changed_at="later", measurement_set_id="different"))
    assert get_current_result(state, scope=scope, fingerprints=fingerprint) == {"status": "OK"}
    assert get_current_state(state, scope)["status"] == AnalysisStatus.CURRENT

    update_measurement(state, replace(row, value=2.5, entered_value=2.5, changed_at="changed"))
    assert get_current_result(state, scope=scope, fingerprints=fingerprint) is None
    assert get_current_state(state, scope)["status"] == AnalysisStatus.STALE
    assert get_current_state(state, scope)["history"][-1]["results"]["symptom"] == {"status": "OK"}


def test_ui_uncertainty_stays_in_base_unit_when_measurement_unit_changes():
    assert float(_sigma_for_input_unit("0.1", "voltage", "mV")) == pytest.approx(100.0)
    assert float(_sigma_for_input_unit("5", "resistance", "kΩ")) == pytest.approx(0.005)


def test_registered_measurement_profile_callback_updates_type_unit_before_widget_creation():
    state = {"test": "resistor_open", "type": "voltage", "unit": "mV", "sigma": "0.2"}
    _sync_profile_test_state(
        state, test_key="test", type_key="type", unit_key="unit", sigma_key="sigma",
        tests={"resistor_open": {"measurement_type": "resistance"}},
    )
    assert state == {"test": "resistor_open", "type": "resistance", "unit": "Ω", "sigma": ""}


def test_measurement_type_change_resets_only_dependent_unit_and_uncertainty():
    state = {"type": "voltage", "unit": "mV", "sigma": "1.5", "value": "3.3"}
    _reset_measurement_type_state(state, type_key="type", unit_key="unit", sigma_key="sigma")
    assert state == {"type": "voltage", "unit": "V", "sigma": "", "value": "3.3"}


def test_measurement_profile_and_fault_are_saved_through_real_streamlit_widgets():
    app_testing = pytest.importorskip("streamlit.testing.v1")
    from pathlib import Path
    from circuitcare.frontend.measurement_state import MEASUREMENTS_KEY

    app_path = Path(__file__).parent / "e2e" / "f5_measurement_profile_app.py"
    app = app_testing.AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
    assert next(item for item in app.selectbox if item.label == "측정 종류").value == "resistance"
    assert next(item for item in app.selectbox if item.label == "검증할 고장 가설").value == "RESISTOR_FAILURE"
    assert next(item for item in app.selectbox if item.label == "부품별 검사 항목").value == "isolated_resistance"

    owner = "f5-profile-apptest-f5-profile-board-BC-R1-RESISTOR_FAILURE"
    app.selectbox(key=f"cc_v5_power_state_{owner}").select("POWER_OFF_CONFIRMED").run()
    app.selectbox(key=f"cc_v5_stored_energy_{owner}").select("SAFE_CONFIRMED").run()
    app.selectbox(key=f"cc_v5_isolation_{owner}").select("ISOLATED_CONFIRMED").run()
    app.checkbox(key=f"cc_v5_location_verified_{owner}").check().run()
    app.checkbox(key=f"cc_v5_reference_verified_{owner}").check().run()

    app.text_input(key=f"cc_v5_value_{owner}").set_value("10")
    next(item for item in app.text_input if item.label == "측정 대상/단자").set_value("R1 양단")
    condition = next(item for item in app.text_input if item.label == "측정 조건 (선택)")
    condition.set_value("전원 차단 · 주변 25 °C")
    assert sum(item.label in {"측정 조건", "측정 조건 (선택)"} for item in app.text_input) == 1
    app.button(key=f"FormSubmitter:cc_v5_add_measurement_{owner}-측정 기록 저장").click().run()

    assert not app.exception
    records = app.session_state[MEASUREMENTS_KEY]
    saved = next(iter(records.values()))[0]
    assert saved.measurement_type == "resistance"
    assert saved.test_id == "isolated_resistance"
    assert saved.fault_hypothesis == "RESISTOR_FAILURE"
    assert saved.operating_conditions == "전원 차단 · 주변 25 °C"
