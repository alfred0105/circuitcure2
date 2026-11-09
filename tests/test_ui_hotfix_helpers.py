import subprocess

import pytest

from circuitcare.ui import (
    SimulationRunError,
    _advisor_matches,
    _analysis_context_key,
    _cached_download_bytes,
    _download_file_digest,
    _measurement_set_id,
    _simulation_result_id,
    _simulation_status,
    _supports_bayesian_demo,
    _upsert_measurement,
    _validate_live_result,
)


def test_csv_download_cache_is_bounded_and_invalidates_on_content_change(tmp_path):
    path = tmp_path / "comparison.csv"
    _cached_download_bytes.cache_clear()
    path.write_bytes(b"time,value\n0,1\n")
    first_digest = _download_file_digest(path)
    assert _cached_download_bytes(str(path), first_digest, "context-a") == path.read_bytes()
    path.write_bytes(b"time,value\n0,2\n")
    second_digest = _download_file_digest(path)
    assert second_digest != first_digest
    assert _cached_download_bytes(str(path), second_digest, "context-a") == path.read_bytes()
    assert _cached_download_bytes.cache_info().maxsize == 4


def test_csv_download_cache_is_context_scoped_and_detects_racing_change(tmp_path, monkeypatch):
    path = tmp_path / "comparison.csv"
    path.write_bytes(b"a,b\n1,2\n")
    digest = _download_file_digest(path)
    _cached_download_bytes.cache_clear()
    assert _cached_download_bytes(str(path), digest, "circuit-a") == path.read_bytes()
    assert _cached_download_bytes(str(path), digest, "circuit-b") == path.read_bytes()
    assert _cached_download_bytes.cache_info().misses == 2

    _cached_download_bytes.cache_clear()
    original_open = type(path).open

    def mutate_after_open(target, *args, **kwargs):
        opened = original_open(target, *args, **kwargs)
        if target == path and args and args[0] == "rb":
            target.write_bytes(b"a,b\n3,4\n")
        return opened

    monkeypatch.setattr(type(path), "open", mutate_after_open)
    with pytest.raises(ValueError, match="changed"):
        _cached_download_bytes(str(path), digest, "circuit-a")


def test_csv_download_size_limit_is_enforced(tmp_path):
    from circuitcare.ui import MAX_DOWNLOAD_CSV_BYTES

    path = tmp_path / "large.csv"
    with path.open("wb") as output:
        output.truncate(MAX_DOWNLOAD_CSV_BYTES + 1)
    with pytest.raises(ValueError, match="64 MiB"):
        _download_file_digest(path)


def test_ui_context_and_measurement_updates_are_isolated():
    case = {"netlist": "tests/fixtures/divider_netlist.xml", "mapping": "data/circuits/divider_mapping.json"}
    first = _analysis_context_key("divider", case, "R1", "resistance_drift", "10k")
    changed_fault = _analysis_context_key("divider", case, "R2", "resistance_drift", "10k")
    changed_value = _analysis_context_key("divider", case, "R1", "resistance_drift", "20k")
    assert first != changed_fault != changed_value

    measurements = _upsert_measurement([], {"node": "VOUT", "value": 7.9, "source": "simulated_demo"})
    measurements = _upsert_measurement(measurements, {"node": "LOAD", "value": 3.95, "source": "simulated_demo"})
    measurements = _upsert_measurement(measurements, {"node": "vout", "value": 7.8, "source": "user_measurement"})
    assert [item["node"] for item in measurements] == ["vout", "LOAD"]
    assert measurements[0]["value"] == 7.8


def test_ui_does_not_activate_nonzero_subprocess_result(tmp_path):
    completed = subprocess.CompletedProcess(["python", "-m", "circuitcare.cli"], -11, "", "ngspice segfault")
    with pytest.raises(SimulationRunError, match="종료 코드: -11"):
        _validate_live_result(completed, tmp_path, "R1", "resistance_drift")


def test_ui_status_and_saved_advice_follow_current_analysis(tmp_path):
    measurements = [{"node": "VOUT", "value": 7.9, "unit": "V", "source": "simulated_demo"}]
    context = ("regulator", "U1", "regulator_output_rise", "8")
    saved = {
        "analysis_context_key": context,
        "measurement_set_id": _measurement_set_id(measurements),
        "simulation_result_id": _simulation_result_id(tmp_path),
    }
    assert _advisor_matches(saved, context, measurements, tmp_path)
    assert not _advisor_matches(saved, context, [], tmp_path)
    assert _simulation_status(False, "ngspice failed", True) == "FAILED"
    assert _simulation_status(False, None, True) == "CACHED"
    assert _simulation_status(False, None, False) == "NOT_RUN"
    assert _simulation_status(True, None, False) == "SUCCESS"


def test_ui_only_exposes_bayesian_for_its_matching_regulator_demo():
    case = "Regulator + divider + switch (cached demo)"
    assert _supports_bayesian_demo(case, "U1", "regulator_output_rise", "8")
    assert not _supports_bayesian_demo(case, "U1", "regulator_output_drop", "3")
    assert not _supports_bayesian_demo("Independent resistor divider (evaluation circuit A)", "R1", "resistance_drift", "10k")
