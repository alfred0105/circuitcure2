from dataclasses import replace

from circuitcare.frontend.diagnosis_adapter import (
    DiagnosisContext, DiagnosisStatus, adapt_fault_overlay, assess_diagnosis_readiness,
    current_diagnosis_context, is_diagnosis_result_current, refresh_diagnosis,
    validate_diagnosis_result,
)
from circuitcare.frontend.measurement_models import create_measurement_record, measurement_set_sha256


def _record():
    return create_measurement_record(
        project_id="p1", board_id="b1", component_id="c1",
        component_identity_status="USER_CONFIRMED", component_map_version="map-v1",
        measurement_type="voltage", value="0.5", unit="V", uncertainty="0.02",
        source="user_measurement", measurement_id="m1", timestamp="t1",
    )


def _mapped_record():
    return replace(_record(), probe_id="probe-1", node_id="node-1", mapping_status="MAPPED")


def _context():
    return DiagnosisContext(
        project_id="p1", board_id="b1", analysis_context_id="ctx-1",
        circuit_model_id="circuit-1", circuit_model_version="v1",
        hypothesis_set_id="hyp-1", component_map_version="map-v1", mapping_status="MAPPED",
    )


def test_measurement_without_node_or_model_is_blocked_and_never_promoted_to_diagnosis():
    row = _record()
    status = assess_diagnosis_readiness(context=_context(), measurements=[row], provider_available=False)
    assert status.status == DiagnosisStatus.NODE_MAPPING_UNAVAILABLE
    assert any("Probe/Node Mapping" in blocker for blocker in status.blockers)
    assert any("Provider" in blocker for blocker in status.blockers)


def test_readiness_requires_measurement_node_model_uncertainty_source_and_map_version():
    assert assess_diagnosis_readiness(context=_context(), measurements=[]).status == DiagnosisStatus.NO_MEASUREMENT
    row = replace(_mapped_record(), uncertainty=None)
    status = assess_diagnosis_readiness(context=_context(), measurements=[row], provider_available=True)
    assert status.status == DiagnosisStatus.MEASUREMENT_RECORDED
    assert any("불확실도" in blocker for blocker in status.blockers)
    stale_map = replace(_mapped_record(), component_map_version="old-map")
    stale = assess_diagnosis_readiness(context=_context(), measurements=[stale_map], provider_available=True)
    assert any("Component Map" in blocker for blocker in stale.blockers)


def test_readiness_only_accepts_explicit_voltage_measurement_mapping():
    current = create_measurement_record(
        project_id="p1", board_id="b1", component_id="c1", component_identity_status="USER_CONFIRMED",
        component_map_version="map-v1", measurement_type="current", value="2", unit="mA",
        uncertainty="0.1", source="user_measurement", measurement_id="m2", timestamp="t2",
    )
    current = replace(current, probe_id="probe-1", node_id="node-1", mapping_status="MAPPED")
    status = assess_diagnosis_readiness(context=_context(), measurements=[current], provider_available=True)
    assert any("V 단위 전압" in blocker for blocker in status.blockers)


class _Provider:
    def __init__(self, *, alter=None):
        self.call = None
        self.alter = alter

    def refresh_diagnosis(self, **kwargs):
        self.call = kwargs
        result = {
            "status": "DIAGNOSIS_COMPLETE", "project_id": kwargs["project_id"],
            "board_id": kwargs["board_id"], "analysis_context_id": kwargs["analysis_context_id"],
            "circuit_model_id": kwargs["circuit_model_id"],
            "circuit_model_version": kwargs["circuit_model_version"],
            "hypothesis_set_id": kwargs["hypothesis_set_id"],
            "component_map_version": kwargs["component_map_version"],
            "measurement_set_sha256": kwargs["measurement_set_sha256"],
            "candidates": [{"hypothesis_id": "hyp-1", "ranking": 0.7}],
        }
        if self.alter:
            result.update(self.alter)
        return result


def test_guarded_provider_receives_only_explicitly_mapped_measurements():
    rows = [_mapped_record()]
    provider = _Provider()
    result, readiness, error = refresh_diagnosis(provider, context=_context(), measurements=rows)
    assert error is None
    assert readiness.status == DiagnosisStatus.DIAGNOSIS_COMPLETE
    assert result["candidates"]
    assert provider.call["measurements"] == [{
        "node": "node-1", "value": 0.5, "unit": "V", "uncertainty": 0.02,
        "source": "user_measurement", "measurement_id": "m1", "component_id": "c1",
        "probe_id": "probe-1",
    }]


def test_stale_provider_result_is_rejected():
    provider = _Provider(alter={"board_id": "other-board"})
    result, readiness, _ = refresh_diagnosis(provider, context=_context(), measurements=[_mapped_record()])
    assert result is None
    assert readiness.status == DiagnosisStatus.STALE_RESULT


def test_fault_overlay_requires_diagnosis_binding_frame_and_mapped_component():
    context = _context()
    row = _mapped_record()
    set_sha = measurement_set_sha256([row], project_id="p1", board_id="b1")
    diagnosis = {
        "status": "DIAGNOSIS_COMPLETE", "project_id": "p1", "board_id": "b1",
        "analysis_context_id": "ctx-1", "circuit_model_id": "circuit-1",
        "circuit_model_version": "v1", "hypothesis_set_id": "hyp-1",
        "component_map_version": "map-v1", "measurement_set_sha256": set_sha,
        "candidates": [{"hypothesis_id": "hyp-1"}],
    }
    overlay = {"schema_version": "circuitcare.pcb-fault-overlay.v1", "board_id": "b1",
               "board_side": "TOP", "candidates": [
        {"board_id": "b1", "board_side": "TOP", "canonical_image_reference": "frame-1",
         "component_id": "c1", "mapping_status": "MAPPED", "fault_candidate_status": "SUSPECTED_FAULT"},
        {"board_id": "b1", "board_side": "TOP", "canonical_image_reference": "old-frame",
         "component_id": "c1", "mapping_status": "MAPPED", "fault_candidate_status": "SUSPECTED_FAULT"},
        {"board_id": "b1", "board_side": "TOP", "canonical_image_reference": "frame-1",
         "component_id": "unknown", "mapping_status": "MAPPED", "fault_candidate_status": "SUSPECTED_FAULT"},
    ]}
    envelope = {"diagnosis": diagnosis, "fault_overlay": overlay,
                "canonical_frame_id": "frame-1", "component_map_version": "map-v1"}
    mapped, error = adapt_fault_overlay(envelope, context=context, measurement_set_sha256=set_sha,
                                        board_side="TOP", canonical_frame_id="frame-1", component_ids={"c1"})
    assert error is None
    assert list(mapped) == ["c1"]
    assert validate_diagnosis_result(diagnosis, context=context, measurement_set_sha256=set_sha)[0]


def test_unverified_confirmed_fault_is_not_shown_and_model_mismatch_overlay_is_suppressed():
    context = _context()
    row = _mapped_record()
    set_sha = measurement_set_sha256([row], project_id="p1", board_id="b1")
    diagnosis = {
        "status": "DIAGNOSIS_COMPLETE", "project_id": "p1", "board_id": "b1",
        "analysis_context_id": "ctx-1", "circuit_model_id": "circuit-1",
        "circuit_model_version": "v1", "hypothesis_set_id": "hyp-1",
        "component_map_version": "map-v1", "measurement_set_sha256": set_sha,
        "candidates": [{"hypothesis_id": "hyp-1"}],
    }
    overlay = {"schema_version": "circuitcare.pcb-fault-overlay.v1", "board_id": "b1",
               "board_side": "TOP", "candidates": [
        {"board_id": "b1", "board_side": "TOP", "canonical_image_reference": "frame-1",
         "component_id": "c1", "mapping_status": "MAPPED", "fault_candidate_status": "CONFIRMED_FAULT"},
    ]}
    envelope = {"diagnosis": diagnosis, "fault_overlay": overlay,
                "canonical_frame_id": "frame-1", "component_map_version": "map-v1"}
    mapped, _ = adapt_fault_overlay(envelope, context=context, measurement_set_sha256=set_sha,
                                    board_side="TOP", canonical_frame_id="frame-1", component_ids={"c1"})
    assert mapped["c1"]["fault_candidate_status"] == "UNDETERMINED"
    diagnosis["status"] = "MODEL_MISMATCH"
    mapped, _ = adapt_fault_overlay(envelope, context=context, measurement_set_sha256=set_sha,
                                    board_side="TOP", canonical_frame_id="frame-1", component_ids={"c1"})
    assert mapped == {}


def test_current_diagnosis_result_requires_full_live_context_and_measurement_set():
    context = _context()
    row = _mapped_record()
    set_sha = measurement_set_sha256([row], project_id="p1", board_id="b1")
    diagnosis = {
        "status": "DIAGNOSIS_COMPLETE", "project_id": "p1", "board_id": "b1",
        "analysis_context_id": "ctx-1", "circuit_model_id": "circuit-1",
        "circuit_model_version": "v1", "hypothesis_set_id": "hyp-1",
        "component_map_version": "map-v1", "measurement_set_sha256": set_sha,
        "candidates": [],
    }
    stored = {"result": diagnosis, "context": context, "measurement_set_sha256": set_sha}
    assert is_diagnosis_result_current(
        stored_result=stored, current_context=context, current_measurement_set=set_sha,
    )
    for changed in (
        replace(context, analysis_context_id="ctx-2"),
        replace(context, circuit_model_id="circuit-2"),
        replace(context, circuit_model_version="v2"),
        replace(context, hypothesis_set_id="hyp-2"),
        replace(context, component_map_version="map-v2"),
        replace(context, board_id="board-2"),
        replace(context, project_id="project-2"),
    ):
        assert not is_diagnosis_result_current(
            stored_result=stored, current_context=changed, current_measurement_set=set_sha,
        )
    assert not is_diagnosis_result_current(
        stored_result=stored, current_context=context, current_measurement_set="new-set",
    )


def test_live_context_uses_current_project_and_map_not_stored_result_owner():
    from circuitcare.frontend.measurement_state import DIAGNOSIS_CONTEXT_KEY, owner_key

    state = {DIAGNOSIS_CONTEXT_KEY: {owner_key("p1", "b1"): {
        "circuit_model_id": "circuit-1", "circuit_model_version": "v1",
        "hypothesis_set_id": "hyp-1", "mapping_status": "MAPPED",
    }}}
    current = current_diagnosis_context(
        state, project_id="p1", board_id="b1", analysis_context_id="ctx-new",
        component_map_version="map-new",
    )
    assert current.analysis_context_id == "ctx-new"
    assert current.component_map_version == "map-new"
    assert current.circuit_model_id == "circuit-1"
