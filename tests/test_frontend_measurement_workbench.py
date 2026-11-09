from dataclasses import replace
import math

import pytest

from circuitcare.frontend.measurement_models import create_measurement_record
from circuitcare.frontend.measurement_state import (
    diagnosis_state, get_measurements, save_measurement, update_measurement,
    delete_measurement, current_measurement_set_sha256,
)
from circuitcare.dashboard.views.measurement_panel import _sigma_for_input_unit


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


def test_ui_uncertainty_stays_in_base_unit_when_measurement_unit_changes():
    assert float(_sigma_for_input_unit("0.1", "voltage", "mV")) == pytest.approx(100.0)
    assert float(_sigma_for_input_unit("5", "resistance", "kΩ")) == pytest.approx(0.005)
