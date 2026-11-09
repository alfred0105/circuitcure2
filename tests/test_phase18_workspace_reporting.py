from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from circuitcare.models import Circuit, Component
from circuitcare.reporting.exporters import (
    export_bundle,
    export_html,
    export_json,
    export_markdown,
)
from circuitcare.reporting.report_builder import (
    _git_sha,
    build_diagnostic_snapshot,
    classify_parameter_grid_coverage,
)
from circuitcare.reporting.report_validator import validate_report, verify_bundle
from circuitcare.workspace.context import (
    build_workspace_context,
    store_uploaded_netlist,
)
from circuitcare.workspace.measurement_provenance import (
    measurement_set_sha256,
    measurement_source_class,
    normalize_measurement,
)
from circuitcare.workspace.provenance import artifact_record
from circuitcare.workspace.session import reconcile_session_workspace
from circuitcare.workspace.simulation_context import (
    build_simulation_fingerprint,
    verify_cached_fingerprint,
    verify_simulation_manifest,
)
from circuitcare.workspace.state_validator import stale_analysis_records
from circuitcare.workspace.workflow import workflow_steps


def _report(tmp_path: Path) -> dict:
    return {
        "schema_version": "circuitcare-diagnosis-report-v1",
        "generated_at": "2026-11-04T00:00:00Z",
        "workspace": {"workspace_id": "ws-1"},
        "circuit": {"hash": "a" * 64, "name": "demo"},
        "evidence": [{"evidence_id": "EV1", "evidence_type": "SPICE_SIMULATION"}],
        "findings": [{"finding_id": "F1", "source_evidence_ids": ["EV1"]}],
        "secondary_damage_risk": {"damage_confirmed": False},
        "limitations": ["Synthetic demo only."],
        "reproducibility": {"git_sha": "abc"},
    }


def test_workspace_hash_and_session_invalidation(tmp_path: Path) -> None:
    netlist = tmp_path / "board.xml"
    model = tmp_path / "model.cir"
    netlist.write_text("<netlist/>")
    model.write_text("R1 a b 1k")
    first = build_workspace_context(circuit_id="board", netlist_path=netlist,
                                    input_paths={"spice_model": model}, fault={"ref": "U1"})
    state = {}
    reconcile_session_workspace(state, {**first.to_dict(), "component_refs": ["U1", "R1"]})
    model.write_text("R1 a b 2k")
    changed = build_workspace_context(circuit_id="board", netlist_path=netlist,
                                      input_paths={"spice_model": model}, fault={"ref": "U1"})
    current = reconcile_session_workspace(state, {**changed.to_dict(), "component_refs": ["R1"]})
    assert first.workspace_id != changed.workspace_id
    assert "spice_fault_comparison" in current["invalidated_analyses"]
    assert "component_reference_context" in current["invalidated_analyses"]
    assert "graph_analysis" not in current["invalidated_analyses"]


def test_dependency_scoped_stale_checks() -> None:
    records = [
        {"analysis_type": "spice_fault_comparison", "status": "COMPLETE",
         "circuit_hash": "same", "input_hashes": {"mapping": "old", "ratings": "r1"}},
        {"analysis_type": "bayesian_diagnosis", "status": "COMPLETE",
         "circuit_hash": "same", "input_hashes": {"prior": "p1"}},
    ]
    result = stale_analysis_records(records, current_circuit_hash="same",
                                    current_input_hashes={"mapping": "new", "ratings": "r1", "prior": "p1"})
    assert result[0]["status"] == "STALE"
    assert result[1]["status"] == "COMPLETE"


def test_parameter_grid_report_distinguishes_in_grid_and_off_grid() -> None:
    candidates = [{"component_ref": "R1", "fault_type": "resistance_drift",
                   "parameters": {"fault_resistance": "2k"}},
                  {"component_ref": "R1", "fault_type": "resistance_drift",
                   "parameters": {"fault_resistance": "500"}}]
    assert classify_parameter_grid_coverage(component_ref="R1", fault_type="resistance_drift",
        fault_parameter="2k", rule_candidates=candidates)["status"] == "CATALOG_CONTAINS_EXACT_PARAMETER"
    off_grid = classify_parameter_grid_coverage(component_ref="R1", fault_type="resistance_drift",
        fault_parameter="1.37k", rule_candidates=candidates)
    assert off_grid["status"] == "EXACT_FAULT_PARAMETER_NOT_IDENTIFIED"
    assert classify_parameter_grid_coverage(component_ref="R2", fault_type="resistance_drift",
        fault_parameter="1.37k", rule_candidates=candidates)["status"] == "NOT_EVALUATED"


def test_report_exports_and_bundle_hash_verification(tmp_path: Path) -> None:
    report = _report(tmp_path)
    assert validate_report(report, expected_circuit_hash="a" * 64) == []
    assert "circuitcare-diagnosis-report-v1" in export_json(report)
    assert "손상 확정 여부: 아니요" in export_markdown(report)
    assert "<!doctype html>" in export_html(report)
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"value": 1}))
    bundle = export_bundle(report, root=tmp_path, selected_artifacts={"measurements.json": evidence})
    manifest = verify_bundle(bundle)
    assert "evidence/measurements.json" in manifest["artifacts"]


def test_bundle_never_replaces_original_evidence_hash_with_export_hash(tmp_path: Path) -> None:
    report = _report(tmp_path)
    artifact = tmp_path / "measurements.json"
    artifact.write_text('{"measurement": 5}\n', encoding="utf-8")
    original_hash = "b" * 64
    report["evidence"][0].update(source_artifact="measurements.json", artifact_sha256=original_hash)
    bundle = export_bundle(report, root=tmp_path, selected_artifacts={"measurements.json": artifact})
    from io import BytesIO
    with zipfile.ZipFile(BytesIO(bundle)) as archive:
        bundled_report = json.loads(archive.read("diagnosis.json"))
    evidence = bundled_report["evidence"][0]
    assert evidence["artifact_sha256"] == original_hash
    assert evidence["bundle_verification_status"] == "ARTIFACT_HASH_MISMATCH"


def test_report_snapshot_portable_paths_and_simulated_observation_are_labelled(tmp_path: Path) -> None:
    circuit = Circuit(components={}, connections=[])
    workspace = {"workspace_id": "ws", "circuit_id": "demo", "circuit_hash": "a" * 64,
                 "netlist_path": "demo.xml", "input_hashes": {}, "validation_status": "READY"}
    report = build_diagnostic_snapshot(root=tmp_path, workspace=workspace, circuit=circuit,
        fault={"component_ref": "U1", "fault_type": "regulator_output_rise", "parameter": "8"},
        simulation_status="SUCCESS", comparison={"scenario_id": "S1", "baseline": {"circuit": "/private/tmp/base.cir"}},
        risk_assessment={"risk_assessment_json": "/private/tmp/risk_assessment.json"},
        observations=[{"node": "VOUT", "value": 7.9, "source": "simulated_demo"}],
        active_fault_candidates={"hypotheses": [], "private_notes": "operator private memo",
                                 "llm_metadata": {"error": "failed at /Users/alice/private/model.bin"}})
    assert report["spice_verification"]["comparison"]["baseline"]["circuit"] == "base.cir"
    assert report["secondary_damage_risk"]["risk_assessment_json"] == "risk_assessment.json"
    assert report["fault_or_symptom"]["observation_classification"] == "simulated_demo"
    assert report["evidence"][1]["evidence_type"] == "HEURISTIC_ESTIMATE"
    assert "측정값" in export_markdown(report) and "진단 순위" in export_markdown(report)
    assert "<table>" in export_html(report) and "가설 및 검증 상태" in export_html(report)
    serialized = export_json(report)
    assert "operator private memo" not in serialized
    assert "/Users/alice/private" not in serialized


def test_report_snapshot_uses_verified_source_file_digest_not_reserialized_object(tmp_path: Path) -> None:
    circuit = Circuit(components={}, connections=[])
    workspace = {"workspace_id": "ws", "circuit_id": "demo", "circuit_hash": "a" * 64,
                 "netlist_path": "demo.xml", "input_hashes": {}, "validation_status": "READY"}
    raw_comparison_hash = "c" * 64
    report = build_diagnostic_snapshot(root=tmp_path, workspace=workspace, circuit=circuit,
        fault={"component_ref": "U1", "fault_type": "regulator_output_rise"},
        simulation_status="SUCCESS", comparison={"scenario_id": "S1"},
        simulation_context={"status": "VERIFIED", "fingerprint": {"fingerprint_sha256": "a" * 64},
                            "result_sha256": {"comparison.json": raw_comparison_hash}})
    assert report["evidence"][0]["artifact_sha256"] == raw_comparison_hash


def test_export_security_rejects_paths_outside_root_and_unknown_names(tmp_path: Path) -> None:
    report = _report(tmp_path)
    other = tmp_path.parent / "private.txt"
    other.write_text("private")
    with pytest.raises(ValueError, match="inside"):
        export_bundle(report, root=tmp_path, selected_artifacts={"measurements.json": other})
    local = tmp_path / "allowed.json"
    local.write_text("{}")
    with pytest.raises(ValueError, match="approved"):
        export_bundle(report, root=tmp_path, selected_artifacts={"../private.txt": local})


def test_provenance_and_report_validator_fail_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "spice.json"
    artifact.write_text("{}")
    record = artifact_record(artifact, workspace_root=tmp_path,
                             evidence_type="SPICE_SIMULATION", verification_status="RECORDED")
    assert record["path"] == "spice.json"
    with pytest.raises(ValueError, match="inside"):
        artifact_record(tmp_path.parent / "private.txt", workspace_root=tmp_path,
                        evidence_type="USER_REPORTED_OBSERVATION", verification_status="RECORDED")
    report = _report(tmp_path)
    report["findings"][0]["source_evidence_ids"] = ["MISSING"]
    assert "MISSING_FINDING_EVIDENCE:F1" in validate_report(report)
    report["evidence"][0]["circuit_hash"] = "b" * 64
    assert "CROSS_CIRCUIT_EVIDENCE:EV1" in validate_report(report)


def test_uploaded_netlist_is_content_addressed_private_and_rejects_dtd(tmp_path: Path) -> None:
    stored = store_uploaded_netlist(b"<export><components/></export>", workspace_root=tmp_path)
    assert stored.parent.parts[-3:] == ("private", "workspaces", "netlists")
    assert stored.name.endswith(".xml")
    assert store_uploaded_netlist(b"<export><components/></export>", workspace_root=tmp_path) == stored
    with pytest.raises(ValueError, match="DTD or entity"):
        store_uploaded_netlist(b"<!DOCTYPE export><export/>", workspace_root=tmp_path)
    with pytest.raises(ValueError, match="25 MiB"):
        store_uploaded_netlist(b"", workspace_root=tmp_path)


def test_workflow_shows_spice_failure_without_blocking_report() -> None:
    steps = workflow_steps(circuit_status="READY", graph_ready=True,
                           fault_status="COMPLETE", spice_status="FAILED", report_ready=True)
    assert steps[3]["status"] == "FAILED"
    assert steps[4]["status"] == "READY"


def test_measurement_provenance_preserves_zero_and_rejects_nonfinite_values() -> None:
    from circuitcare.fault_ai_ui import _collect_observation
    assert _collect_observation(False, "VOUT", 0.0, 0.2, "user_measurement") == []
    assert _collect_observation(True, "VOUT", 0.0, 0.2, "user_measurement")[0]["value"] == 0.0
    zero = normalize_measurement({"node": "VOUT", "value": 0, "unit": "V", "source": "실제 측정"})
    assert zero["value"] == 0.0
    assert zero["source"] == "user_measurement"
    assert zero["verification_status"] == "USER_REPORTED"
    assert normalize_measurement({"node": "VOUT", "value": -1, "source": "simulation"})["value"] == -1
    with pytest.raises(ValueError):
        normalize_measurement({"node": "VOUT", "value": float("nan"), "source": "unknown"})
    with pytest.raises(ValueError):
        normalize_measurement({"node": "VOUT", "value": 1, "uncertainty": -0.1})
    assert measurement_source_class([zero, {"node": "VIN", "value": 5, "source": "simulation"}]) == "MIXED_SOURCES"
    assert measurement_source_class([{"node": "VOUT", "value": 0, "source": "mystery"}]) == "unknown"


def test_active_measurement_and_what_if_preview_stay_separate(tmp_path: Path) -> None:
    circuit = Circuit(components={}, connections=[])
    actual = [{"node": "VOUT", "value": 5.0, "unit": "V", "source": "user_measurement"}]
    preview = [{"node": "VOUT", "value": 7.9, "unit": "V", "source": "simulated_demo"}]
    context = {"circuit_hash": "a" * 64, "measurement_set_sha256": measurement_set_sha256(actual),
               "mode": "ACTIVE_DIAGNOSIS"}
    report = build_diagnostic_snapshot(root=tmp_path,
        workspace={"workspace_id": "ws", "circuit_id": "demo", "circuit_hash": "a" * 64,
                   "netlist_path": "demo.xml", "input_hashes": {}, "validation_status": "READY"},
        circuit=circuit, fault={"component_ref": "U1", "fault_type": "regulator_output_rise"},
        simulation_status="NOT_EVALUATED", observations=actual,
        diagnosis={"status": "ACTIVE_CONDITIONAL", "posterior": [{"hypothesis_id": "h5", "posterior": 1.0}]},
        diagnosis_context=context,
        preview_context={"measurement_source": "simulated_demo", "measurement_set_sha256": measurement_set_sha256(preview)},
        preview_diagnosis={"posterior": [{"hypothesis_id": "h79", "posterior": 1.0}]})
    assert report["fault_or_symptom"]["observations"][0]["value"] == 5.0
    assert report["diagnosis"]["posterior"][0]["hypothesis_id"] == "h5"
    assert report["preview_context"]["status"] == "PREVIEW_ONLY"
    assert report["preview_diagnosis"]["posterior"][0]["hypothesis_id"] == "h79"
    assert "MEASUREMENT_DIAGNOSIS_MISMATCH" not in validate_report(report)
    report["diagnosis_context"]["measurement_set_sha256"] = "f" * 64
    assert "MEASUREMENT_DIAGNOSIS_MISMATCH" in validate_report(report)


def test_dashboard_report_only_includes_context_matched_inspection_ranking(tmp_path: Path) -> None:
    import hashlib

    from circuitcare.services.report_service import (
        ReportSnapshotContext,
        build_report_snapshot_for_dashboard,
    )

    analysis_context_key = ("demo", "U1", "regulator_output_rise", "8")
    context_id = hashlib.sha256(repr(analysis_context_key).encode()).hexdigest()[:16]
    observations = [{"node": "VOUT", "value": 0.0, "unit": "V", "source": "user_measurement"}]
    measurement_hash = measurement_set_sha256(observations)
    ranking = {"posterior_weighted_inspection_order": [{"component_ref": "U2", "rank": 1}]}
    state = {
        "analysis_context_key": analysis_context_key,
        "measurements": observations,
        "diagnosis_result": {
            "analysis_context_id": context_id,
            "measurement_set_sha256": measurement_hash,
            "diagnosis": {"status": "ACTIVE_CONDITIONAL"},
            "inspection_ranking": ranking,
        },
    }
    context = ReportSnapshotContext(
        root=tmp_path,
        workspace={"workspace_id": "ws", "circuit_id": "demo", "circuit_hash": "a" * 64,
                   "netlist_path": "demo.xml", "input_hashes": {}, "validation_status": "READY"},
        circuit=Circuit(components={"U1": Component("U1", "REG", "Regulator")}, connections=[]), failed_component="U1",
        fault_type="regulator_output_rise", fault_value="8", simulation_status="NOT_RUN",
        comparison={}, risk_assessment={}, simulation_context={}, analysis_context_key=analysis_context_key,
        bayesian_supported=True, include_preview=True,
        what_if_result={"diagnosis": {"posterior": [{"hypothesis_id": "preview"}]},
                        "recommendations": {"inspection_ranking": {"component_ref": "PREVIEW"}}},
        what_if_measurements=[{"node": "VOUT", "value": 7.9, "unit": "V", "source": "simulated_demo"}],
        connection_paths=[],
    )

    report = build_report_snapshot_for_dashboard(context, state)
    assert report["recommended_inspection"]["ranking"] == ranking
    assert report["fault_or_symptom"]["observations"][0]["value"] == 0.0
    assert report["preview_context"]["status"] == "PREVIEW_ONLY"
    assert report["preview_diagnosis"]["posterior"][0]["hypothesis_id"] == "preview"

    state["diagnosis_result"]["measurement_set_sha256"] = "f" * 64
    stale_by_measurement = build_report_snapshot_for_dashboard(context, state)
    assert stale_by_measurement["recommended_inspection"]["status"] == "NOT_EVALUATED"
    assert stale_by_measurement["diagnosis"]["status"] == "NOT_EVALUATED"

    state["diagnosis_result"]["measurement_set_sha256"] = measurement_hash
    state["diagnosis_result"]["analysis_context_id"] = "wrong-context"
    stale_by_context = build_report_snapshot_for_dashboard(context, state)
    assert stale_by_context["recommended_inspection"]["status"] == "NOT_EVALUATED"


def test_simulation_fingerprint_detects_same_size_content_change(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("board.xml", "deck.cir", "mapping.json", "probes.json")]
    for path in paths:
        path.write_text("AAAA")
    fault = {"component_ref": "R1", "fault_type": "resistance_drift", "parameter": "10k"}
    first = build_simulation_fingerprint(netlist=paths[0], spice_deck=paths[1], component_mapping=paths[2],
        probes=paths[3], fault=fault, settings={"stop": "10ms"})
    stat = paths[1].stat()
    paths[1].write_text("BBBB")
    import os
    os.utime(paths[1], ns=(stat.st_atime_ns, stat.st_mtime_ns))
    changed = build_simulation_fingerprint(netlist=paths[0], spice_deck=paths[1], component_mapping=paths[2],
        probes=paths[3], fault=fault, settings={"stop": "10ms"})
    assert verify_cached_fingerprint(first, first)[0] == "VERIFIED"
    assert verify_cached_fingerprint(first, changed)[0] == "STALE"
    assert verify_cached_fingerprint(None, changed)[0] == "PROVENANCE_UNVERIFIED"
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    result_file = result_dir / "comparison.json"
    result_file.write_text("saved")
    import hashlib
    manifest = {"fingerprint": first, "result_sha256": {"comparison.json": hashlib.sha256(b"saved").hexdigest()}}
    assert verify_simulation_manifest(result_dir, manifest, first)[0] == "VERIFIED"
    result_file.write_text("tampered")
    assert verify_simulation_manifest(result_dir, manifest, first)[0] == "STALE"


def test_legacy_cached_simulation_is_not_reported_as_verified(tmp_path: Path) -> None:
    report = build_diagnostic_snapshot(root=tmp_path,
        workspace={"workspace_id": "ws", "circuit_id": "demo", "circuit_hash": "a" * 64,
                   "netlist_path": "demo.xml", "input_hashes": {}, "validation_status": "READY"},
        circuit=Circuit(components={}, connections=[]),
        fault={"component_ref": "U1", "fault_type": "regulator_output_rise", "parameter": "8"},
        simulation_status="CACHED", comparison={"scenario_id": "S1"},
        risk_assessment={"scenario_id": "S1", "damage_confirmed": False})
    assert report["spice_verification"]["status"] == "PROVENANCE_UNVERIFIED"
    assert report["spice_verification"]["comparison"] is None
    assert report["secondary_damage_risk"]["status"] == "NOT_EVALUATED"


@pytest.mark.integration
def test_live_ngspice_result_manifest_is_verified_then_detects_tamper(tmp_path: Path) -> None:
    if not shutil.which("ngspice"):
        pytest.skip("ngspice is optional; live simulation integration requires the executable")
    root = Path(__file__).resolve().parents[1]
    case = {"netlist": "tests/fixtures/fault_demo_netlist.xml", "base": "data/circuits/fault_demo.cir",
            "mapping": "data/circuits/fault_demo_mapping.json", "probes": "data/circuits/fault_demo_probes.json",
            "ratings": "data/component_specs/fault_demo_ratings.json"}
    output = tmp_path / "live-run"
    command = [sys.executable, "-m", "circuitcare.cli", "--netlist", case["netlist"], "--fault", "R1",
        "--fault-type", "resistance_drift", "--fault-value", "10k", "--spice-base", case["base"],
        "--spice-mapping", case["mapping"], "--output", str(output), "--simulate", "--probes", case["probes"],
        "--ratings", case["ratings"], "--assess-risk"]
    completed = subprocess.run(command, check=False, cwd=root, capture_output=True, text=True, timeout=120)
    from circuitcare.ui import _validate_live_result
    verified_dir = _validate_live_result(completed, output, "R1", "resistance_drift", case=case, fault_value="10k")
    fingerprint = build_simulation_fingerprint(netlist=root / case["netlist"], spice_deck=root / case["base"],
        component_mapping=root / case["mapping"], probes=root / case["probes"],
        fault={"component_ref": "R1", "fault_type": "resistance_drift", "parameter": "10k"},
        settings={"engine": "ngspice", "analysis": "transient", "runner": "circuitcare-cli"},
        ratings=root / case["ratings"])
    manifest = json.loads((verified_dir / "simulation_context.json").read_text())
    assert "comparison.csv" in manifest["result_sha256"]
    assert verify_simulation_manifest(verified_dir, manifest, fingerprint)[0] == "VERIFIED"
    csv_path = verified_dir / "comparison.csv"
    csv_bytes = csv_path.read_bytes()
    csv_path.write_bytes(csv_bytes + b"\n")
    assert verify_simulation_manifest(verified_dir, manifest, fingerprint)[0] == "STALE"
    csv_path.write_bytes(csv_bytes)
    (verified_dir / "comparison.json").write_text("{}")
    assert verify_simulation_manifest(verified_dir, manifest, fingerprint)[0] == "STALE"


def test_report_rejects_cross_scenario_and_cross_component_risk(tmp_path: Path) -> None:
    report = _report(tmp_path)
    report["fault_or_symptom"] = {"component_ref": "U1", "fault_type": "regulator_output_rise"}
    report["spice_verification"] = {"scenario_id": "S1", "comparison": {"scenario_id": "S1"}}
    report["secondary_damage_risk"] = {"damage_confirmed": False, "scenario_id": "S2", "failed_component": "R1"}
    errors = validate_report(report)
    assert "CROSS_SCENARIO_EVIDENCE" in errors
    assert "SIMULATION_CONTEXT_MISMATCH" in errors


def test_bundle_evidence_references_match_included_artifact(tmp_path: Path) -> None:
    report = _report(tmp_path)
    report["evidence"][0].update(source_artifact="comparison.json", artifact_sha256=None)
    artifact = tmp_path / "comparison.json"
    artifact.write_text('{"scenario_id":"S1"}')
    bundle = export_bundle(report, root=tmp_path, selected_artifacts={"spice_results.json": artifact})
    manifest = verify_bundle(bundle)
    assert manifest["evidence_sources"][0]["included_in_bundle"] is True
    assert manifest["evidence_sources"][0]["source_artifact"] == "evidence/spice_results.json"
    with zipfile.ZipFile(BytesIO(bundle)) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    contents["evidence/spice_results.json"] = b"tampered"
    changed = BytesIO()
    with zipfile.ZipFile(changed, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in contents.items():
            archive.writestr(name, value)
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bundle(changed.getvalue())


def test_bundle_verifier_rejects_duplicate_and_traversal_paths() -> None:
    duplicate = BytesIO()
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("manifest.json", "{}")
        archive.writestr("report.md", "one")
        archive.writestr("report.md", "two")
    with pytest.raises(ValueError, match="Duplicate"):
        verify_bundle(duplicate.getvalue())
    traversal = BytesIO()
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("../private.txt", "private")
        archive.writestr("manifest.json", "{}")
    with pytest.raises(ValueError, match="Unsafe"):
        verify_bundle(traversal.getvalue())


def test_streamlit_app_keeps_original_tabs_and_displays_workspace_reports() -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    script = Path(__file__).resolve().parents[1] / "src/circuitcare/ui.py"
    app = AppTest.from_file(str(script), default_timeout=30).run()
    assert not app.exception
    assert any("출처 확인 필요" in item.value for item in app.markdown)
    assert app.session_state["active_analysis_view"] == "회로도 Workbench"
    next(item for item in app.button if item.label == "AI 성능 평가").click().run()
    assert not app.exception
    assert app.session_state["active_analysis_view"] == "AI 성능 평가"
    next(item for item in app.radio if item.key == "main_area").set_value("보고서").run()
    assert any(button.label == "Evidence Bundle ZIP" for button in app.download_button)
    selector = next(item for item in app.selectbox if item.key == "case_name")
    selector.select("KiCad XML 업로드 (연결 그래프 전용)").run()
    assert not app.exception
    assert any("KiCad XML을 올리면" in item.value for item in app.info)


def test_git_sha_reads_checkout_metadata_without_subprocess(tmp_path, monkeypatch):
    git_dir = tmp_path / ".git"
    (git_dir / "refs/heads").mkdir(parents=True)
    sha = "a" * 40
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="ascii")
    (git_dir / "refs/heads/main").write_text(sha + "\n", encoding="ascii")

    def reject_subprocess(*args, **kwargs):
        raise AssertionError("report rendering must not spawn git")

    monkeypatch.setattr("subprocess.run", reject_subprocess)
    assert _git_sha(tmp_path) == sha
