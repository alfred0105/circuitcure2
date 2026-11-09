import subprocess
from pathlib import Path

from circuitcare.dashboard.analysis_loader import load_analysis_snapshot
from circuitcare.dashboard.session_state import (
    analysis_context_keys_to_clear,
    reconcile_analysis_context,
    simulation_context_keys_to_clear,
)
from circuitcare.dashboard.simulation_controller import (
    SimulationRequest,
    build_simulation_command,
    execute_simulation_request,
)
from circuitcare.dashboard.workspace_adapter import build_analysis_records
from circuitcare.vision.ui_context import VisionViewContext


def test_analysis_context_invalidation_keeps_unrelated_state() -> None:
    state = {
        "analysis_context_key": "circuit-a",
        "live_result_dir": "/tmp/a",
        "what_if_voltage_VOUT": 7.9,
        "vision_active_image": "image-a",
        "unrelated_setting": True,
    }
    assert analysis_context_keys_to_clear(list(state)) == {
        "live_result_dir", "what_if_voltage_VOUT",
    }
    assert reconcile_analysis_context(state, "circuit-b") is True
    assert "live_result_dir" not in state
    assert "what_if_voltage_VOUT" not in state
    assert state["vision_active_image"] == "image-a"
    assert state["unrelated_setting"] is True
    assert state["measurements"] == []
    assert reconcile_analysis_context(state, "circuit-b") is False


def test_simulation_start_clears_only_live_and_preview_state() -> None:
    existing = ["live_result_dir", "live_context_key", "last_advisor",
                "what_if_result", "what_if_voltage_VOUT", "vision_active_image"]
    assert simulation_context_keys_to_clear(existing) == set(existing) - {"vision_active_image"}


def test_analysis_loader_accepts_verified_cache_and_blocks_unverified_cache(tmp_path: Path) -> None:
    current = {"fingerprint_sha256": "a" * 64}
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    (result_dir / "comparison.json").write_text('{"scenario_id":"S1"}', encoding="utf-8")
    (result_dir / "risk_assessment.json").write_text('{"scenario_id":"S1"}', encoding="utf-8")
    (result_dir / "component_stress.json").write_text('{"simulation_success":true}', encoding="utf-8")
    # Missing a manifest is fail-closed even when result JSON is readable.
    blocked = load_analysis_snapshot(result_dir=result_dir, current_fingerprint=current,
                                     live_used=False, live_error=None, cached_selection=True)
    assert blocked.status == "PROVENANCE_UNVERIFIED"
    assert blocked.comparison == blocked.risk == blocked.stress == {}

    for filename in ("comparison.json", "risk_assessment.json", "component_stress.json"):
        import hashlib
        digest = hashlib.sha256((result_dir / filename).read_bytes()).hexdigest()
        if filename == "comparison.json":
            digests = {filename: digest}
        else:
            digests[filename] = digest
    import json
    manifest = {"fingerprint": current, "result_sha256": digests}
    (result_dir / "simulation_context.json").write_text(json.dumps(manifest), encoding="utf-8")
    verified = load_analysis_snapshot(result_dir=result_dir, current_fingerprint=current,
                                      live_used=False, live_error=None, cached_selection=True)
    assert verified.status == "CACHED_SIMULATION"
    assert verified.integrity == "VERIFIED"
    assert verified.comparison["scenario_id"] == "S1"


def test_analysis_loader_blocks_stale_and_damaged_results(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    (result_dir / "comparison.json").write_text('{"scenario_id":"S1"}', encoding="utf-8")
    snapshot = load_analysis_snapshot(result_dir=result_dir,
        current_fingerprint={"fingerprint_sha256": "b" * 64}, live_used=False,
        live_error=None, cached_selection=True)
    assert snapshot.status == "PROVENANCE_UNVERIFIED"
    assert snapshot.comparison == {}
    (result_dir / "risk_assessment.json").write_text("not-json", encoding="utf-8")
    damaged = load_analysis_snapshot(result_dir=result_dir,
        current_fingerprint={"fingerprint_sha256": "b" * 64}, live_used=False,
        live_error="ngspice failed", cached_selection=True)
    assert damaged.status == "FAILED"
    assert damaged.risk == {}


def test_analysis_loader_rejects_valid_manifest_for_different_context(tmp_path: Path) -> None:
    import hashlib
    import json

    result_dir = tmp_path / "results"
    result_dir.mkdir()
    contents = {name: '{"scenario_id":"S1"}' for name in
                ("comparison.json", "risk_assessment.json", "component_stress.json")}
    digests = {}
    for name, content in contents.items():
        (result_dir / name).write_text(content, encoding="utf-8")
        digests[name] = hashlib.sha256(content.encode()).hexdigest()
    (result_dir / "simulation_context.json").write_text(json.dumps({
        "fingerprint": {"fingerprint_sha256": "a" * 64}, "result_sha256": digests,
    }), encoding="utf-8")
    stale = load_analysis_snapshot(result_dir=result_dir,
        current_fingerprint={"fingerprint_sha256": "b" * 64}, live_used=False,
        live_error=None, cached_selection=True)
    assert stale.status == "STALE"
    assert stale.integrity == "STALE"
    assert stale.comparison == stale.risk == stale.stress == {}


def _write_verified_analysis(result_dir: Path, fingerprint: dict) -> None:
    import hashlib
    import json

    contents = {name: '{"scenario_id":"S1"}' for name in
                ("comparison.json", "risk_assessment.json", "component_stress.json")}
    digests = {}
    for name, content in contents.items():
        (result_dir / name).write_text(content, encoding="utf-8")
        digests[name] = hashlib.sha256(content.encode()).hexdigest()
    csv_path = result_dir / "comparison.csv"
    if csv_path.exists():
        digests[csv_path.name] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    (result_dir / "simulation_context.json").write_text(json.dumps({
        "fingerprint": fingerprint, "result_sha256": digests,
    }), encoding="utf-8")


def test_live_analysis_must_pass_manifest_before_success_or_data_use(tmp_path: Path) -> None:
    import json

    fingerprint = {"fingerprint_sha256": "a" * 64}
    result_dir = tmp_path / "live"
    result_dir.mkdir()
    _write_verified_analysis(result_dir, fingerprint)
    valid = load_analysis_snapshot(result_dir=result_dir, current_fingerprint=fingerprint,
                                   live_used=True, live_error=None, cached_selection=False)
    assert (valid.status, valid.integrity) == ("SUCCESS", "VERIFIED")
    assert valid.comparison and valid.risk and valid.stress

    (result_dir / "comparison.json").write_text('{"scenario_id":"TAMPERED"}', encoding="utf-8")
    tampered = load_analysis_snapshot(result_dir=result_dir, current_fingerprint=fingerprint,
                                      live_used=True, live_error=None, cached_selection=False)
    assert (tampered.status, tampered.integrity) == ("STALE", "STALE")
    assert tampered.comparison == tampered.risk == tampered.stress == {}

    (result_dir / "simulation_context.json").unlink()
    missing = load_analysis_snapshot(result_dir=result_dir, current_fingerprint=fingerprint,
                                     live_used=True, live_error=None, cached_selection=False)
    assert (missing.status, missing.integrity) == ("PROVENANCE_UNVERIFIED", "PROVENANCE_UNVERIFIED")
    assert missing.comparison == missing.risk == missing.stress == {}

    _write_verified_analysis(result_dir, fingerprint)
    manifest_path = result_dir / "simulation_context.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result_sha256"].pop("risk_assessment.json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    incomplete_manifest = load_analysis_snapshot(result_dir=result_dir, current_fingerprint=fingerprint,
                                                 live_used=True, live_error=None, cached_selection=False)
    assert (incomplete_manifest.status, incomplete_manifest.integrity) == (
        "PROVENANCE_UNVERIFIED", "PROVENANCE_UNVERIFIED")
    assert incomplete_manifest.comparison == incomplete_manifest.risk == incomplete_manifest.stress == {}


def test_analysis_loader_keeps_failed_and_verified_cached_statuses(tmp_path: Path) -> None:
    fingerprint = {"fingerprint_sha256": "a" * 64}
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    _write_verified_analysis(result_dir, fingerprint)
    cached = load_analysis_snapshot(result_dir=result_dir, current_fingerprint=fingerprint,
                                    live_used=False, live_error=None, cached_selection=True)
    assert cached.status == "CACHED_SIMULATION" and cached.integrity == "VERIFIED"
    failed = load_analysis_snapshot(result_dir=result_dir, current_fingerprint=fingerprint,
                                    live_used=False, live_error="ngspice failed", cached_selection=True)
    assert failed.status == "FAILED"


def test_analysis_loader_verifies_cached_csv_and_preserves_json_only_cache(tmp_path: Path) -> None:
    import json

    fingerprint = {"fingerprint_sha256": "a" * 64}
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    _write_verified_analysis(result_dir, fingerprint)
    json_only = load_analysis_snapshot(result_dir=result_dir, current_fingerprint=fingerprint,
        live_used=False, live_error=None, cached_selection=True)
    assert json_only.status == "CACHED_SIMULATION"
    assert json_only.csv_integrity == "NOT_PRESENT"

    csv_path = result_dir / "comparison.csv"
    csv_path.write_text("time_s,baseline_V(OUT)_V,fault_V(OUT)_V\n0,1,2\n", encoding="utf-8")
    _write_verified_analysis(result_dir, fingerprint)
    verified = load_analysis_snapshot(result_dir=result_dir, current_fingerprint=fingerprint,
        live_used=False, live_error=None, cached_selection=True)
    assert (verified.status, verified.integrity, verified.csv_integrity) == (
        "CACHED_SIMULATION", "VERIFIED", "VERIFIED")

    csv_path.write_text("time_s,baseline_V(OUT)_V,fault_V(OUT)_V\n0,8,9\n", encoding="utf-8")
    tampered = load_analysis_snapshot(result_dir=result_dir, current_fingerprint=fingerprint,
        live_used=False, live_error=None, cached_selection=True)
    assert (tampered.status, tampered.integrity, tampered.csv_integrity) == ("STALE", "STALE", "STALE")
    assert tampered.comparison == tampered.risk == tampered.stress == {}

    _write_verified_analysis(result_dir, fingerprint)
    csv_path.unlink()
    deleted = load_analysis_snapshot(result_dir=result_dir, current_fingerprint=fingerprint,
        live_used=False, live_error=None, cached_selection=True)
    assert (deleted.status, deleted.integrity, deleted.csv_integrity) == ("STALE", "STALE", "STALE")

    csv_path.write_text("time_s,baseline_V(OUT)_V,fault_V(OUT)_V\n0,1,2\n", encoding="utf-8")
    manifest_path = result_dir / "simulation_context.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result_sha256"].pop("comparison.csv")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    legacy = load_analysis_snapshot(result_dir=result_dir, current_fingerprint=fingerprint,
        live_used=False, live_error=None, cached_selection=True)
    assert (legacy.status, legacy.integrity, legacy.csv_integrity) == (
        "PROVENANCE_UNVERIFIED", "PROVENANCE_UNVERIFIED", "PROVENANCE_UNVERIFIED")
    assert legacy.comparison == legacy.risk == legacy.stress == {}


def test_datasheet_failed_reassessment_clears_only_exact_scoped_key() -> None:
    from circuitcare.datasheet.rating_adapter import RatingApplicationError
    from circuitcare.ui import _clear_failed_datasheet_cache

    key = "rating_review_applied_context-a_stress-a"
    state = {key: ["approved"], "rating_review_applied_context-b_stress-b": ["keep"],
             "analysis_context_key": "context-a"}
    for error in (RatingApplicationError("bad rating"), ValueError(), KeyError(), TypeError()):
        try:
            raise error
        except (RatingApplicationError, ValueError, KeyError, TypeError):
            _clear_failed_datasheet_cache(state, key)
    assert key not in state
    assert state["rating_review_applied_context-b_stress-b"] == ["keep"]
    assert state["analysis_context_key"] == "context-a"


def test_datasheet_application_cache_identity_is_shared_and_scoped() -> None:
    from circuitcare.datasheet.rating_review import applied_rating_session_key

    context = "a1b2c3d4e5f60718"
    first = applied_rating_session_key(context, "stress-a")
    assert first == applied_rating_session_key(context, "stress-a")
    assert first != applied_rating_session_key(context, "stress-b")
    assert first != applied_rating_session_key("other-context", "stress-a")

    # Exercise the same token + signature passed through ui.py and the datasheet UI.
    from circuitcare.ui import hashlib as ui_hashlib
    raw_analysis_context = ("case-a", "R1", "resistance_drift", "13k")
    token = ui_hashlib.sha256(repr(raw_analysis_context).encode()).hexdigest()[:16]
    ui_cache_key = applied_rating_session_key(token, "stress-a")
    application_cache_key = applied_rating_session_key(token, "stress-a")
    assert ui_cache_key == application_cache_key


def test_datasheet_reassessment_is_blocked_without_verified_risk_and_stress(tmp_path: Path) -> None:
    from circuitcare.datasheet.ui_application import render_rating_application

    class FakeStreamlit:
        def __init__(self):
            self.warnings = []
            self.buttons = []
        def warning(self, value): self.warnings.append(value)
        def button(self, label, **kwargs):
            self.buttons.append(label)
            return True

    fake = FakeStreamlit()
    render_rating_application(
        fake, root=tmp_path, case={"netlist": "demo.xml"}, risk={}, stress={},
        component_ref="R1", selected_index=0,
        ds_state={"reviews": {"0": {"decision": "approved"}}}, conditions_fit=True,
        context_key="ctx", state_id="state", stress_signature="stress", current_risk=None,
    )
    assert fake.warnings
    assert fake.buttons == []


def test_datasheet_apply_revoke_reapply_invalidates_same_ui_cache_key(monkeypatch, tmp_path: Path) -> None:
    import circuitcare.datasheet.ui_application as application
    from circuitcare.datasheet.rating_review import applied_rating_session_key

    monkeypatch.setattr(application, "_specs_for_case", lambda *_: {})
    monkeypatch.setattr(application, "_mapping_for_case", lambda *_: {})
    monkeypatch.setattr(application, "apply_approved_rating", lambda specs, *_args, **_kwargs: specs)
    monkeypatch.setattr(application, "reassess_existing_risk", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(application, "save_application", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(application, "_risk_row", lambda *_args: {})
    monkeypatch.setattr(application, "_comparison_row", lambda *args: args)

    class FakeStreamlit:
        def __init__(self):
            self.session_state = {}
            self.clicked = ""
        def button(self, label, **kwargs): return label == self.clicked
        def success(self, *_args): pass
        def warning(self, *_args): pass
        def rerun(self): pass
        def subheader(self, *_args): pass
        def dataframe(self, *_args, **_kwargs): pass
        def caption(self, *_args): pass
        def write(self, *_args): pass
        def json(self, *_args): pass

    fake = FakeStreamlit()
    token, stress_signature = "8bf307f136d9abce", "stress-a"
    cache_key = applied_rating_session_key(token, stress_signature)
    case = {"netlist": "demo.xml"}
    risk = {"scenario_id": "S1"}
    current_risk = {"scenario_id": "S1", "reassessment_source": "approved_datasheet",
                    "rating_checks": []}
    kwargs = {"root": tmp_path, "case": case, "risk": risk,
              "stress": {"simulation_success": True}, "component_ref": "R1",
              "selected_index": 0, "ds_state": {"reviews": {"0": {"decision": "approved"}}},
              "conditions_fit": True, "context_key": token, "state_id": "review",
              "stress_signature": stress_signature, "current_risk": current_risk}

    for action in ("승인 정격 적용 및 위험 재평가", "승인 정격 적용 해제",
                   "승인 정격 적용 및 위험 재평가"):
        fake.session_state[cache_key] = [{"component_ref": "R1"}]
        fake.clicked = action
        application.render_rating_application(fake, **kwargs)
        assert cache_key not in fake.session_state


def test_circuit_view_uses_live_fault_type_and_parameter(monkeypatch, tmp_path: Path) -> None:
    import circuitcare.dashboard.views.circuit as view

    class FakeStreamlit:
        def __init__(self): self.output = []
        def subheader(self, value): pass
        def write(self, value): self.output.append(str(value))
        def graphviz_chart(self, *args, **kwargs): pass
        def selectbox(self, *args, **kwargs): return "R1"
        def json(self, value): pass
        def caption(self, value): pass
        def warning(self, value): pass

    monkeypatch.setattr(view, "direct_candidates", lambda *_: [])
    monkeypatch.setattr(view, "find_paths_result", lambda *_args, **_kwargs: type("P", (), {
        "paths": [], "status": "COMPLETE", "expanded_states": 0,
        "peak_queue_size": 0, "limit_reason": None})())
    monkeypatch.setattr(view, "build_graph", lambda *_: {})
    monkeypatch.setattr(view, "to_dot", lambda *_: "digraph G {}")
    fake = FakeStreamlit()
    fake.output = []
    view.render_circuit_view(fake, circuit=type("C", (), {
        "components": {"R1": type("Part", (), {"ref": "R1", "value": "10k", "library_id": "Device:R"})()},
        "connections": []})(), case_name="A", case={"fault_type": "old", "value": "10k", "netlist": "demo.xml"},
        selected_fault="R1", fault_type="resistance_drift", fault_value="13k",
        root=tmp_path, comparison_path=tmp_path / "comparison.json")
    assert "resistance_drift = 13k" in fake.output[0]

    fake.output = []
    view.render_circuit_view(fake, circuit=type("C", (), {
        "components": {"R1": type("Part", (), {"ref": "R1", "value": "10k", "library_id": "Device:R"})()},
        "connections": []})(), case_name="A", case={"fault_type": "old", "value": "10k", "netlist": "demo.xml"},
        selected_fault="R1", fault_type="open", fault_value=None,
        root=tmp_path, comparison_path=tmp_path / "comparison.json")
    assert "open = 해당 없음" in fake.output[0]


def test_download_view_blocks_raw_csv_when_simulation_is_unverified(tmp_path: Path) -> None:
    import hashlib

    from circuitcare.dashboard.views.downloads import render_downloads

    class Block:
        def __enter__(self): return self
        def __exit__(self, *args): return False

    class FakeStreamlit:
        def __init__(self):
            self.downloads = []
            self.captions = []
        def expander(self, *_args, **_kwargs): return Block()
        def download_button(self, label, *_args, **_kwargs): self.downloads.append(label)
        def caption(self, value): self.captions.append(value)
        def warning(self, *_args, **_kwargs): pass

    result_dir = tmp_path / "results"
    result_dir.mkdir()
    csv_content = b"time,value\n0,1\n"
    (result_dir / "comparison.csv").write_bytes(csv_content)
    csv_sha256 = hashlib.sha256(csv_content).hexdigest()
    fake = FakeStreamlit()
    fake.downloads = []
    fake.captions = []
    render_downloads(fake, {"netlist": "demo.xml"}, result_dir, {}, {}, None,
                     "R1", "resistance_drift", "13k", simulation_verified=False)
    assert "시뮬레이션 CSV" not in fake.downloads
    assert any("다운로드를 차단" in caption for caption in fake.captions)

    fake.downloads = []
    render_downloads(fake, {"netlist": "demo.xml"}, result_dir, {}, {}, None,
                     "R1", "resistance_drift", "13k", simulation_verified=True, csv_verified=False,
                     csv_sha256=csv_sha256)
    assert "시뮬레이션 CSV" not in fake.downloads
    render_downloads(fake, {"netlist": "demo.xml"}, result_dir, {}, {}, None,
                     "R1", "resistance_drift", "13k", simulation_verified=True, csv_verified=True,
                     csv_sha256=csv_sha256)
    assert "시뮬레이션 CSV" in fake.downloads


def test_waveform_view_does_not_read_unverified_csv(tmp_path: Path, monkeypatch) -> None:
    import circuitcare.dashboard.views.waveforms as view

    calls = []
    monkeypatch.setattr(view, "csv_rows", lambda path, **kwargs: calls.append((path, kwargs)) or [])

    class FakeStreamlit:
        def __init__(self): self.warnings = []
        def subheader(self, *_args): pass
        def selectbox(self, *_args): return "OUT"
        def warning(self, message): self.warnings.append(message)
        def columns(self, count): return [self for _ in range(count)]
        def metric(self, *_args): pass
        def caption(self, *_args): pass
        def dataframe(self, *_args, **_kwargs): pass

    st = FakeStreamlit()
    view.render_waveform_view(st, comparison={"simulation_success": True, "voltage_nodes": {
        "OUT": {"baseline": {"last": 1}, "fault": {"last": 2},
                "delta_fault_minus_baseline": {"last": 1}}}},
        result_dir=tmp_path, stress={}, csv_verified=False)
    assert calls == []
    assert any("CSV 출처를 확인" in message for message in st.warnings)
    view.render_waveform_view(st, comparison={"simulation_success": True, "voltage_nodes": {
        "OUT": {"baseline": {"last": 1}, "fault": {"last": 2},
                "delta_fault_minus_baseline": {"last": 1}}}},
        result_dir=tmp_path, stress={}, csv_verified=True, csv_sha256="c" * 64)
    assert calls == [(tmp_path / "comparison.csv", {"expected_sha256": "c" * 64})]


def test_waveform_csv_reader_rejects_content_changed_after_snapshot(tmp_path: Path) -> None:
    import hashlib

    from circuitcare.dashboard.formatters import csv_rows

    path = tmp_path / "comparison.csv"
    path.write_text("time_s,baseline,fault\n0,1,2\n", encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text("time_s,baseline,fault\n0,7,8\n", encoding="utf-8")
    assert csv_rows(path, expected_sha256=expected) == []


def test_simulation_controller_builds_absolute_command_and_optional_fault_value(tmp_path: Path) -> None:
    command = build_simulation_command(
        python_executable="python", project_root=tmp_path,
        netlist=tmp_path / "input.xml", case={"base": "base.cir", "mapping": "map.json",
            "probes": "probes.json", "ratings": "ratings.json"},
        component_ref="R1", fault_type="resistance_drift", output_dir=tmp_path / "out",
        fault_value="13k",
    )
    assert command[command.index("--netlist") + 1] == str((tmp_path / "input.xml").resolve())
    assert command[command.index("--fault-value") + 1] == "13k"
    no_value = build_simulation_command(
        python_executable="python", project_root=tmp_path, netlist=tmp_path / "input.xml",
        case={"base": "base.cir", "mapping": "map.json", "probes": "probes.json", "ratings": "ratings.json"},
        component_ref="U1", fault_type="open", output_dir=tmp_path / "out", fault_value=None,
    )
    assert "--fault-value" not in no_value


def test_simulation_controller_normalizes_success_failure_and_timeout(monkeypatch, tmp_path: Path) -> None:
    import circuitcare.dashboard.simulation_controller as controller

    request = SimulationRequest(command=("python",), output_dir=tmp_path / "out",
        component_ref="R1", fault_type="open", fault_value=None, case={})
    monkeypatch.setattr(controller, "execute_simulation", lambda *args, **kwargs: tmp_path / "result")
    success = execute_simulation_request(request)
    assert success.status == "SUCCESS" and success.result_dir == tmp_path / "result"

    for error in (RuntimeError("failed"), subprocess.TimeoutExpired(["python"], 1)):
        def fail(*args, error=error, **kwargs):
            raise error
        monkeypatch.setattr(controller, "execute_simulation", fail)
        outcome = execute_simulation_request(request)
        assert outcome.status == "FAILED"
        assert outcome.error


def test_workspace_adapter_preserves_analysis_status_contract() -> None:
    ws_context = {"circuit_hash": "c" * 64, "input_hashes": {"netlist": "n"}}
    active = {"generated_at": "2026-11-04T00:00:00Z"}
    records = build_analysis_records(workspace_context=ws_context, active_workspace=active,
        comparison={"scenario_id": "S1"}, risk={"scenario_id": "S1"},
        status="CACHED_SIMULATION")
    assert [row["analysis_type"] for row in records] == [
        "circuit_validation", "graph_analysis", "spice_fault_comparison", "risk_assessment"]
    assert records[2]["status"] == "COMPLETE"
    assert records[2]["scenario_id"] == records[3]["scenario_id"] == "S1"


def test_vision_overlay_requires_manifest_verification_for_live_and_cached() -> None:
    base = {"context_key": "c", "netlist_component_refs": ("R1",), "risk_assessment": {"scenario_id": "S1"},
            "risk_path": "risk_assessment.json", "failed_component": "R1", "fault_type": "resistance_drift",
            "result_integrity": "VERIFIED", "inspection_ranking": None, "selected_component": None}
    assert VisionViewContext(**base, simulation_status="SUCCESS").overlay_risk is not None
    assert VisionViewContext(**base, simulation_status="CACHED_SIMULATION").overlay_risk is not None
    for status in ("STALE", "PROVENANCE_UNVERIFIED", "FAILED", "NOT_RUN", "CACHED"):
        context = VisionViewContext(**base, simulation_status=status)
        assert context.overlay_risk is None
    assert VisionViewContext(**{**base, "result_integrity": "STALE"},
                             simulation_status="SUCCESS").overlay_risk is None
