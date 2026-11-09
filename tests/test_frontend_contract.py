from circuitcare.frontend.adapters import MockDiagnosisAdapter
from circuitcare.frontend.models import MeasurementViewModel, SelectionState


def test_zero_is_present_and_none_is_missing():
    measured = MeasurementViewModel("p", "U1", "VOUT", 0.0, "V", "user_measurement")
    missing = MeasurementViewModel("p", "U1", "VOUT", None, "V", "not_entered")
    assert measured.value == 0.0
    assert missing.value is None


def test_mock_adapter_makes_no_diagnostic_claims():
    result = MockDiagnosisAdapter().diagnosis_summary(project_id="project-1")
    assert result.provenance["source"] == "MOCK"
    assert result.verification_status == "DEMO_ONLY"
    assert result.candidate_components == ()
    assert result.suggested_measurements == ()


def test_selection_state_uses_only_workbench_tabs():
    assert SelectionState("project-1").active_tab == "pcb"
    assert SelectionState("project-1", active_tab="schematic").active_tab == "schematic"
