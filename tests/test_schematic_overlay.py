from __future__ import annotations

import json
from pathlib import Path

import pytest

from circuitcare.models import Circuit, Component, Connection
from circuitcare.schematic.geometry import GeometryResult, SymbolHitRegion
from circuitcare.schematic.measurement_geometry import MeasurementGeometryResult
from circuitcare.schematic.overlay import (
    build_fault_overlay,
    build_measurement_overlay,
    build_risk_overlay,
)

ROOT = Path(__file__).resolve().parents[1]


def _circuit() -> Circuit:
    return Circuit(
        components={ref: Component(ref, value) for ref, value in
                    (("J1", "5V"), ("R1", "1k"), ("R2", "1k"), ("U2", "IC"))},
        connections=[Connection("J1", "1", "VIN"), Connection("R1", "1", "VIN"),
                     Connection("R1", "2", "VOUT"), Connection("R2", "1", "VOUT"),
                     Connection("R2", "2", "GND"), Connection("U2", "1", "GND")],
    )


def _geometry(status="VERIFIED_GEOMETRY") -> GeometryResult:
    refs = ("J1", "R1", "R2", "U2")
    regions = tuple(SymbolHitRegion(ref, f"uuid-{ref}", "/sheet", 1, "Device:R",
                                    ((index * 10.0, 0.0), (index * 10.0 + 4, 0.0),
                                     (index * 10.0 + 4, 4.0), (index * 10.0, 4.0)))
                    for index, ref in enumerate(refs))
    return GeometryResult(status, "test geometry", (0, 0, 100, 100), regions)


def _risk() -> dict:
    return json.loads((ROOT / "data/evaluations/divider/simulation_results/fault_r1_resistance_drift/risk_assessment.json").read_text())


def _risk_gate(risk=None, *, status="CACHED_SIMULATION", integrity="VERIFIED", geometry=None,
               equivalence="VERIFIED_EQUIVALENT", comparison=None, manifest=None):
    report = risk or _risk()
    return build_risk_overlay(
        circuit=_circuit(), failed_component="R1", fault_type="resistance_drift",
        fault_value="10k",
        risk=report, comparison=comparison or {"scenario_id": report["scenario_id"], "simulation_success": True},
        simulation_status=status, simulation_integrity=integrity,
        manifest=manifest or {"fingerprint": {"fault_scenario": {
            "component_ref": "R1", "fault_type": "resistance_drift", "parameter": "10k"},
            "fingerprint_sha256": "fp-a"}},
        current_fingerprint={"fingerprint_sha256": "fp-a"},
        geometry=geometry or _geometry(), electrical_equivalence=equivalence,
        context_id="ctx-a", asset_key="svg-a",
    )


def test_fault_overlay_distinguishes_source_direct_and_indirect_candidates():
    payload = build_fault_overlay(circuit=_circuit(), failed_component="R1", geometry=_geometry(),
                                  electrical_equivalence="VERIFIED_EQUIVALENT",
                                  context_id="ctx-a", asset_key="svg-a", max_hops=3)
    by_ref = {row["component_ref"]: row for row in payload["markers"]}
    assert payload["status"] == "READY"
    assert by_ref["R1"]["kind"] == "FAULT_SOURCE"
    assert by_ref["J1"]["kind"] == "DIRECT_CANDIDATE"
    assert by_ref["R2"]["kind"] == "DIRECT_CANDIDATE"
    assert by_ref["U2"]["kind"] == "INDIRECT_CANDIDATE"
    assert by_ref["U2"]["hop_count"] == 2
    assert by_ref["U2"]["evidence_level"] == "connectivity_only"
    assert payload["wire_overlay"] is False
    assert all(row["damage_confirmed"] is False for row in payload["markers"])


@pytest.mark.parametrize(("equivalence", "geometry_status"), [
    ("MISMATCH", "VERIFIED_GEOMETRY"),
    ("NOT_VERIFIED", "VERIFIED_GEOMETRY"),
    ("VERIFIED_EQUIVALENT", "MULTI_SHEET_MAPPING_PARTIAL"),
    ("VERIFIED_EQUIVALENT", "UNVERIFIED_GEOMETRY"),
])
def test_fault_overlay_fails_closed_on_unverified_context(equivalence, geometry_status):
    payload = build_fault_overlay(circuit=_circuit(), failed_component="R1", geometry=_geometry(geometry_status),
                                  electrical_equivalence=equivalence, context_id="ctx-a",
                                  asset_key="svg-a")
    assert payload["status"] == "BLOCKED"
    assert payload["markers"] == []


def test_fault_overlay_rejects_unknown_fault_reference_and_marks_partial_paths():
    payload = build_fault_overlay(circuit=_circuit(), failed_component="Q9", geometry=_geometry(),
                                  electrical_equivalence="VERIFIED_EQUIVALENT",
                                  context_id="ctx-a", asset_key="svg-a")
    assert payload["status"] == "BLOCKED"

    payload = build_fault_overlay(circuit=_circuit(), failed_component="R1", geometry=_geometry(),
                                  electrical_equivalence="VERIFIED_EQUIVALENT",
                                  context_id="ctx-a", asset_key="svg-a", max_hops=1)
    assert payload["path_search"]["status"] == "PARTIAL_DEPTH_LIMIT"
    assert any(row["component_ref"] == "R2" for row in payload["markers"])


def test_divider_risk_overlay_uses_existing_levels_and_preserves_rating_evidence():
    payload = _risk_gate()
    rows = {row["component_ref"]: row for row in payload["markers"]}
    assert payload["status"] == "READY"
    assert rows["R1"]["kind"] == "FAULT_SOURCE"
    assert rows["R2"]["risk_level"] == "WITHIN_LIMIT"
    assert rows["J1"]["risk_level"] == "UNKNOWN"
    assert rows["R2"]["rating_source_types"] == ["demo_assumption"]
    assert rows["R2"]["metrics"]["max_power_w"]["rated_limit"] == 0.125
    assert rows["R2"]["metrics"]["max_current_a"]["status"] == "unknown"
    assert payload["damage_confirmed"] is False


@pytest.mark.parametrize(("status", "integrity"), [
    ("STALE", "STALE"), ("PROVENANCE_UNVERIFIED", "PROVENANCE_UNVERIFIED"),
    ("FAILED", "FAILED"), ("NOT_RUN", "PROVENANCE_UNVERIFIED"),
    ("CACHED_SIMULATION", "STALE"),
])
def test_risk_overlay_blocks_unverified_or_non_success_snapshot(status, integrity):
    assert _risk_gate(status=status, integrity=integrity)["status"] == "BLOCKED"


def test_risk_overlay_blocks_scenario_geometry_and_reference_mismatches():
    risk = _risk()
    assert _risk_gate(comparison={"scenario_id": "OTHER", "simulation_success": True})["status"] == "BLOCKED"
    assert _risk_gate(manifest={"fingerprint": {"fault_scenario": {
        "component_ref": "R2", "fault_type": "resistance_drift", "parameter": "10k"},
        "fingerprint_sha256": "fp-a"}})["status"] == "BLOCKED"
    assert _risk_gate(geometry=_geometry("MULTI_SHEET_MAPPING_PARTIAL"))["status"] == "BLOCKED"
    risk["secondary_candidates"][0]["component_ref"] = "Q9"
    assert _risk_gate(risk=risk)["status"] == "BLOCKED"


@pytest.mark.parametrize("level", ["EXCEEDED", "NEAR_LIMIT", "WITHIN_LIMIT", "UNKNOWN"])
def test_risk_overlay_supports_only_explicit_risk_levels(level):
    risk = _risk()
    risk["secondary_candidates"] = [{"component_ref": "R2", "risk_level": level,
                                      "metrics": {}, "rating_source_types": []}]
    payload = _risk_gate(risk=risk)
    assert payload["status"] == "READY"
    assert next(row for row in payload["markers"] if row["component_ref"] == "R2")["risk_level"] == level


@pytest.mark.parametrize("bad_row", [
    {"component_ref": "R2", "risk_level": ["UNKNOWN"], "metrics": {}},
    {"component_ref": "R2", "risk_level": "UNKNOWN", "metrics": []},
    {"component_ref": "R2", "risk_level": "UNKNOWN", "metrics": {"power": None}},
])
def test_risk_overlay_fails_closed_for_malformed_metric_rows(bad_row):
    risk = _risk()
    risk["secondary_candidates"] = [bad_row]
    assert _risk_gate(risk=risk)["status"] == "BLOCKED"


def test_measurement_overlay_shows_every_logical_pin_anchor_and_scoped_inspection_rank():
    recommendation = {"node": "VOUT", "measurement_type": "voltage", "unit": "V",
                      "expected_information_gain_nats": 0.8, "uncertainty_source": "demo_assumption"}
    anchors = MeasurementGeometryResult("VERIFIED_LOGICAL_NET_CANDIDATES", "mapped", "VOUT", (
        {"component_ref": "R1", "pin": "2", "net": "VOUT", "x": 100.0, "y": 77.62},
        {"component_ref": "R2", "pin": "1", "net": "VOUT", "x": 130.0, "y": 70.0},
    ))
    ranking = {"posterior_weighted_inspection_order": [
        {"component_ref": "R2", "priority_rank": 1,
         "posterior_weighted_priority_score": 0.72, "posterior_evidence_coverage": 1.0}]}
    payload = build_measurement_overlay(
        circuit=_circuit(), failed_component="R1", geometry=_geometry(),
        electrical_equivalence="VERIFIED_EQUIVALENT", context_id="ctx-a", asset_key="svg-a",
        support_status="VERIFIED_CURRENT_CONTEXT", source_class="simulated_demo",
        measurement_set_id="set-a", recommendation=recommendation, anchors=anchors,
        inspection_ranking=ranking,
    )
    assert payload["status"] == "READY"
    candidates = [row for row in payload["markers"] if row["kind"] == "MEASUREMENT_NODE_CANDIDATE"]
    assert {(row["component_ref"], row["pin"]) for row in candidates} == {("R1", "2"), ("R2", "1")}
    assert all(row["physical_probe_point_confirmed"] is False for row in candidates)
    priority = next(row for row in payload["markers"] if row["kind"] == "INSPECTION_PRIORITY")
    assert priority["priority_rank"] == 1
    assert priority["damage_probability"] is None


@pytest.mark.parametrize(("support", "source"), [
    ("NOT_SUPPORTED", "prior_only"), ("STALE", "user_measurement"),
    ("VERIFIED_CURRENT_CONTEXT", "what_if_preview"),
])
def test_measurement_overlay_rejects_unsupported_or_preview_sources(support, source):
    payload = build_measurement_overlay(
        circuit=_circuit(), failed_component="R1", geometry=_geometry(),
        electrical_equivalence="VERIFIED_EQUIVALENT", context_id="ctx-a", asset_key="svg-a",
        support_status=support, source_class=source, measurement_set_id=None,
        recommendation=None, anchors=None, inspection_ranking=None,
    )
    assert payload["status"] == "BLOCKED"


def test_measurement_overlay_blocks_unverified_location_and_model_mismatch():
    recommendation = {"node": "VOUT", "measurement_type": "voltage", "unit": "V"}
    blocked = build_measurement_overlay(
        circuit=_circuit(), failed_component="R1", geometry=_geometry(),
        electrical_equivalence="VERIFIED_EQUIVALENT", context_id="ctx-a", asset_key="svg-a",
        support_status="VERIFIED_CURRENT_CONTEXT", source_class="prior_only", measurement_set_id="empty",
        recommendation=recommendation,
        anchors=MeasurementGeometryResult("UNVERIFIED_MEASUREMENT_LOCATION", "pin mismatch", "VOUT"),
        inspection_ranking=None,
    )
    assert blocked["status"] == "BLOCKED"
    mismatch = build_measurement_overlay(
        circuit=_circuit(), failed_component="R1", geometry=_geometry(),
        electrical_equivalence="VERIFIED_EQUIVALENT", context_id="ctx-a", asset_key="svg-a",
        support_status="VERIFIED_CURRENT_CONTEXT", source_class="simulated_demo", measurement_set_id="set-a",
        recommendation=recommendation, anchors=None, inspection_ranking=None, model_mismatch=True,
    )
    assert mismatch["status"] == "BLOCKED"
