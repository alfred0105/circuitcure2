from __future__ import annotations

import subprocess

from circuitcare.dashboard.case_loader import DEMO_CASES
from circuitcare.services import simulation_service


def test_opted_in_divider_fingerprint_includes_ngspice_version(monkeypatch):
    def fake_run(command, **kwargs):
        assert command == ["ngspice", "--version"]
        assert kwargs["timeout"] == 5
        return subprocess.CompletedProcess(command, 0, "******\n** ngspice-47 : Circuit level simulator\n", "")

    monkeypatch.setattr(simulation_service.subprocess, "run", fake_run)
    case = DEMO_CASES["Independent resistor divider (evaluation circuit A)"]
    fingerprint = simulation_service._current_simulation_fingerprint(
        case, "R1", "resistance_drift", "10k"
    )
    assert fingerprint["simulation_settings"]["ngspice_version"] == "ngspice-47"
    assert fingerprint["fault_scenario"] == {
        "component_ref": "R1", "fault_type": "resistance_drift", "parameter": "10k"
    }
    assert fingerprint["fingerprint_sha256"]


def test_opted_in_divider_fingerprint_fails_closed_without_engine_version(monkeypatch):
    def fake_run(command, **kwargs):
        raise FileNotFoundError("ngspice unavailable")

    monkeypatch.setattr(simulation_service.subprocess, "run", fake_run)
    case = DEMO_CASES["Independent resistor divider (evaluation circuit A)"]
    assert simulation_service._current_simulation_fingerprint(
        case, "R1", "resistance_drift", "10k"
    ) == {"fingerprint_sha256": None}
