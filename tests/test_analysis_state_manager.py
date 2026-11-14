from types import SimpleNamespace

from circuitcare.analysis.fingerprints import build_analysis_fingerprints
from circuitcare.analysis.models import AnalysisScope, AnalysisStatus
from circuitcare.analysis.state_manager import (
    begin_analysis, commit_analysis_result, fail_analysis, get_current_result,
    get_current_state, invalidate_analysis,
)
from circuitcare.dashboard.views.symptom_diagnosis_panel import current_analysis_snapshot
from circuitcare.dashboard.views import symptom_diagnosis_panel as symptoms
from circuitcare.frontend.models import ImageRecord, ProjectContext, SelectionState
from circuitcare.frontend.board_component_registry import (
    build_board_component_registry, current_source_observation_fingerprint,
    diagnosis_components, measurement_component_aliases,
)
from circuitcare.frontend.measurement_models import create_measurement_record
from circuitcare.frontend.measurement_state import reconfirm_measurement_identity, save_measurement


def _fingerprints(*, symptoms="no_power", measurements=(), rules="rules-v1"):
    return build_analysis_fingerprints(
        input_payload={"symptoms": symptoms, "component": "U1", "review": "CONFIRMED"},
        measurements=measurements,
        extra_rules_payload={"engine_profile": rules},
    )


def test_changed_inputs_stale_result_and_preserve_history():
    state = {}
    scope = AnalysisScope("p1", "b1", "photo-1")
    first = _fingerprints()
    run_id = begin_analysis(state, scope=scope, fingerprints=first)
    assert get_current_state(state, scope)["status"] == AnalysisStatus.RUNNING
    assert commit_analysis_result(state, scope=scope, run_id=run_id, fingerprints=first,
                                  current_fingerprints=first, result={"status": "OK"})
    assert get_current_result(state, scope=scope, fingerprints=first) == {"status": "OK"}

    changed = _fingerprints(symptoms="no_led")
    assert get_current_result(state, scope=scope, fingerprints=changed) is None
    record = get_current_state(state, scope)
    assert record["status"] == AnalysisStatus.STALE
    assert record["history"][-1]["results"]["symptom"] == {"status": "OK"}


def test_analysis_changed_while_running_cannot_commit_old_result():
    state = {}
    scope = AnalysisScope("p1", "b1", "photo-1")
    started = _fingerprints()
    run_id = begin_analysis(state, scope=scope, fingerprints=started)
    invalidate_analysis(state, scope=scope, reason="component changed")
    assert not commit_analysis_result(state, scope=scope, run_id=run_id,
                                      fingerprints=started, result={"status": "OLD"})
    assert get_current_result(state, scope=scope, fingerprints=started) is None
    assert get_current_state(state, scope)["status"] == AnalysisStatus.STALE


def test_commit_rejects_changed_live_fingerprint_even_if_run_token_is_valid():
    state = {}
    scope = AnalysisScope("p1", "b1", "photo-1")
    started = _fingerprints()
    changed = _fingerprints(symptoms="no_led")
    run_id = begin_analysis(state, scope=scope, fingerprints=started)
    assert not commit_analysis_result(state, scope=scope, run_id=run_id,
                                      fingerprints=started, current_fingerprints=changed,
                                      result={"status": "OLD"})
    assert get_current_result(state, scope=scope, fingerprints=started) is None


def test_commit_rejects_a_result_when_the_active_analysis_scope_changed():
    state = {}
    scope = AnalysisScope("p1", "b1", "photo-1")
    other_photo = AnalysisScope("p1", "b1", "photo-2")
    fingerprints = _fingerprints()
    run_id = begin_analysis(state, scope=scope, fingerprints=fingerprints)
    assert not commit_analysis_result(state, scope=scope, run_id=run_id,
                                      fingerprints=fingerprints, current_fingerprints=fingerprints,
                                      current_scope=other_photo, result={"status": "OLD"})
    assert get_current_state(state, scope)["status"] == AnalysisStatus.STALE


def test_identical_semantic_inputs_and_display_only_scopes_do_not_invalidate():
    state = {}
    scope = AnalysisScope("p1", "b1", "photo-1")
    fingerprints = _fingerprints(measurements=[{"value": 3.3, "unit": "V"}])
    run_id = begin_analysis(state, scope=scope, fingerprints=fingerprints)
    assert commit_analysis_result(state, scope=scope, run_id=run_id,
                                  fingerprints=fingerprints, result={"status": "OK"})
    # Zoom, tab and selected box are intentionally absent from the analysis payload.
    same = _fingerprints(measurements=[{"unit": "V", "value": 3.3}])
    assert get_current_result(state, scope=scope, fingerprints=same) == {"status": "OK"}


def test_project_board_and_photo_scopes_are_isolated():
    state = {}
    first = AnalysisScope("p1", "b1", "photo-1")
    second_photo = AnalysisScope("p1", "b1", "photo-2")
    other_board = AnalysisScope("p1", "b2", "photo-1")
    other_project = AnalysisScope("p2", "b1", "photo-1")
    fingerprints = _fingerprints()
    run_id = begin_analysis(state, scope=first, fingerprints=fingerprints)
    assert commit_analysis_result(state, scope=first, run_id=run_id,
                                  fingerprints=fingerprints, result={"status": "OK"})
    invalidate_analysis(state, scope=second_photo, reason="other photo changed")
    assert get_current_result(state, scope=first, fingerprints=fingerprints) == {"status": "OK"}
    for other in (second_photo, other_board, other_project):
        assert get_current_result(state, scope=other, fingerprints=fingerprints) is None


def test_rule_fingerprint_and_failure_status_are_tracked():
    state = {}
    scope = AnalysisScope("p1", "b1", "photo-1")
    first = _fingerprints()
    run_id = begin_analysis(state, scope=scope, fingerprints=first)
    assert fail_analysis(state, scope=scope, run_id=run_id, error="engine failure")
    assert get_current_state(state, scope)["status"] == AnalysisStatus.FAILED
    assert get_current_result(state, scope=scope, fingerprints=_fingerprints(rules="rules-v2")) is None


def test_live_snapshot_ignores_viewer_controls_and_isolates_photo_scope():
    project = ProjectContext(
        "p1", "b1", image_records=(
            ImageRecord("photo-1", content_sha256="a"),
            ImageRecord("photo-2", content_sha256="b"),
        ),
    )
    state = {
        "circuitcare_v2_active_photo_id": "photo-1",
        "circuitcare_v2_photo_choice": "photo-1",
    }
    st = type("FakeStreamlit", (), {"session_state": state})()
    first = current_analysis_snapshot(st, project)
    run_id = begin_analysis(state, scope=first["scope"], fingerprints=first["fingerprints"])
    assert commit_analysis_result(state, scope=first["scope"], run_id=run_id,
                                  fingerprints=first["fingerprints"], result={"status": "OK"})

    state["circuitcare_v2_view_zoom"] = 1.8
    state["circuitcare_v2_selection"] = SelectionState("p1", "b1", selected_component="D9")
    same_scope = current_analysis_snapshot(st, project)
    assert same_scope["result"] == {"status": "OK"}

    state["cc_symptom_diagnosis_p1_description"] = "전원이 불안정함"
    changed_symptom = current_analysis_snapshot(st, project)
    assert changed_symptom["result"] is None
    assert changed_symptom["record"]["status"] == AnalysisStatus.STALE

    state["circuitcare_v2_active_photo_id"] = "photo-2"
    state["circuitcare_v2_photo_choice"] = "photo-2"
    other_photo = current_analysis_snapshot(st, project)
    assert other_photo["scope"].target_id == "photo-2"
    assert other_photo["result"] is None


def test_simple_symptom_prompt_and_text_invalidate_current_analysis(monkeypatch):
    project = ProjectContext("simple-p", "simple-b")
    state = {f"{symptoms._project_key(project)}_simple_initial_symptom": "기타·모르겠음"}
    st = type("FakeStreamlit", (), {"session_state": state})()
    monkeypatch.setattr(symptoms, "_auto_qwen_model", lambda _state: "qwen3.5:9b")
    first = current_analysis_snapshot(st, project, auto_qwen=True)
    run_id = begin_analysis(state, scope=first["scope"], fingerprints=first["fingerprints"])
    assert commit_analysis_result(state, scope=first["scope"], run_id=run_id,
                                  fingerprints=first["fingerprints"], result={"status": "OK"})

    state[f"{symptoms._project_key(project)}_simple_initial_symptom"] = "전원 안 켜짐"
    changed = current_analysis_snapshot(st, project, auto_qwen=True)
    assert changed["result"] is None
    assert changed["record"]["status"] == AnalysisStatus.STALE


def test_board_analysis_scope_survives_photo_switch_and_stales_on_registry_change():
    project = ProjectContext("p-board", "b-board", image_records=(
        ImageRecord("front"), ImageRecord("back"),
    ))
    snapshots = {
        "front": {"image_fingerprint": "front-v1", "review_version": "rv1",
                  "components": (SimpleNamespace(
                      component_id="D1", component_type="diode", bbox_xyxy=(1, 2, 5, 7),
                      coordinate_frame="SOURCE_PHOTO_PIXEL", identification_status="USER_CONFIRMED",
                      presence_status="CONFIRMED", component_type_status="USER_CONFIRMED",
                      ocr_marking=None, possible_part_number=None, component_info=None,
                      confirmed_functional_roles=(), evidence=("front review",),
                      overlay_source="yolo", ocr_candidates=(),
                  ),)},
        "back": {"image_fingerprint": "back-v1", "review_version": "rv2", "components": ()},
    }
    state = {"circuitcare_v2_active_photo_id": "front",
             "circuitcare_v2_photo_choice": "front"}
    registry = build_board_component_registry(
        state, project_id=project.project_id, board_id=project.board_id,
        photo_snapshots=snapshots,
    )
    components = diagnosis_components(registry)
    inventory = {"__board_registry__": {"registry_version": registry["version"]}}
    aliases = measurement_component_aliases(registry)
    st = type("FakeStreamlit", (), {"session_state": state})()
    scope_id = "board:registry:v1"
    first = current_analysis_snapshot(st, project, components=components,
                                      component_inventory=inventory,
                                      analysis_target_id=scope_id,
                                      component_id_aliases=aliases)
    run_id = begin_analysis(state, scope=first["scope"], fingerprints=first["fingerprints"])
    assert commit_analysis_result(state, scope=first["scope"], run_id=run_id,
                                  fingerprints=first["fingerprints"], current_fingerprints=first["fingerprints"],
                                  current_scope=first["scope"], result={"status": "CURRENT"})

    state["circuitcare_v2_active_photo_id"] = "back"
    state["circuitcare_v2_photo_choice"] = "back"
    state["circuitcare_v2_view_zoom"] = 2.0
    state["circuitcare_v2_selection"] = SelectionState("p-board", "b-board", selected_component="D1")
    unchanged = current_analysis_snapshot(st, project, components=components,
                                          component_inventory=inventory,
                                          analysis_target_id=scope_id,
                                          component_id_aliases=aliases)
    assert unchanged["scope"].key == first["scope"].key
    assert unchanged["result"] == {"status": "CURRENT"}

    changed = {**snapshots, "close-up": {"image_fingerprint": "close-v1",
                                         "review_version": "rv3", "components": ()}}
    next_registry = build_board_component_registry(
        state, project_id=project.project_id, board_id=project.board_id,
        photo_snapshots=changed,
    )
    stale = current_analysis_snapshot(
        st, project, components=diagnosis_components(next_registry),
        component_inventory={"__board_registry__": {"registry_version": next_registry["version"]}},
        analysis_target_id=scope_id,
        component_id_aliases=measurement_component_aliases(next_registry),
    )
    assert stale["result"] is None
    assert stale["record"]["status"] == AnalysisStatus.STALE


def test_board_analysis_projects_verified_measurements_and_drops_replaced_photo_links():
    project = ProjectContext("p-measure", "b-measure", image_records=(ImageRecord("front"),))
    component = SimpleNamespace(
        component_id="R1", component_type="resistor", bbox_xyxy=(1, 2, 5, 7),
        coordinate_frame="SOURCE_PHOTO_PIXEL", identification_status="USER_CONFIRMED",
        presence_status="CONFIRMED", component_type_status="USER_CONFIRMED",
        ocr_marking="103", possible_part_number=None, component_info=None,
        confirmed_functional_roles=(), evidence=(), overlay_source="yolo", ocr_candidates=(),
    )
    snapshots = {"front": {"image_fingerprint": "front-v1", "components": (component,)}}
    state = {"circuitcare_v2_active_photo_id": "front", "circuitcare_v2_photo_choice": "front"}
    registry = build_board_component_registry(
        state, project_id=project.project_id, board_id=project.board_id, photo_snapshots=snapshots,
    )
    board_id = registry["source_to_board_id"]['["front","R1"]']
    measurement = create_measurement_record(
        project_id=project.project_id, board_id=project.board_id, component_id="R1",
        component_identity_status="USER_CONFIRMED", measurement_type="resistance",
        value="10", unit="Ω", source="user_measurement",
        component_map_version=registry["version"], measurement_id="m-r1",
        board_component_id=board_id, source_photo_id="front",
        source_photo_fingerprint="front-v1", source_component_id="R1",
        identity_resolution_status="USER_SELECTED_VERIFIED",
    )
    save_measurement(state, measurement)
    st = type("FakeStreamlit", (), {"session_state": state})()
    args = dict(
        components=diagnosis_components(registry),
        component_inventory={"__board_registry__": {"registry_version": registry["version"]}},
        analysis_target_id="board:registry:v1",
        component_id_aliases=measurement_component_aliases(registry),
        board_registry=registry,
    )
    current = current_analysis_snapshot(st, project, **args)
    assert current["measurements"] == []  # Legacy record has no source observation fingerprint.
    reconfirm_measurement_identity(
        state, measurement_id="m-r1", project_id=project.project_id, board_id=project.board_id,
        board_component_id=board_id, source_photo_id="front", source_component_id="R1",
        source_photo_fingerprint="front-v1",
        source_observation_fingerprint=current_source_observation_fingerprint(
            registry, photo_id="front", component_id="R1"),
        component_type="resistor", component_map_version=registry["version"],
    )
    current = current_analysis_snapshot(st, project, **args)
    assert len(current["measurements"]) == 1
    assert current["measurements"][0].component_id == board_id
    assert current["measurements"][0].component_map_version == registry["version"]

    replaced_snapshots = {"front": {"image_fingerprint": "front-v2", "components": (component,)}}
    replaced_registry = build_board_component_registry(
        state, project_id=project.project_id, board_id=project.board_id,
        photo_snapshots=replaced_snapshots,
    )
    args["components"] = diagnosis_components(replaced_registry)
    args["component_inventory"] = {"__board_registry__": {"registry_version": replaced_registry["version"]}}
    args["component_id_aliases"] = measurement_component_aliases(replaced_registry)
    args["board_registry"] = replaced_registry
    stale_measurement = current_analysis_snapshot(st, project, **args)
    assert stale_measurement["measurements"] == []


def test_late_commit_cannot_cancel_new_run_or_hide_its_result():
    state = {}
    scope = AnalysisScope("p-late", "b", "board")
    first, second = _fingerprints(), _fingerprints(symptoms="no_led")
    old_run = begin_analysis(state, scope=scope, fingerprints=first)
    new_run = begin_analysis(state, scope=scope, fingerprints=second)
    before = get_current_state(state, scope)
    assert not commit_analysis_result(state, scope=scope, run_id=old_run,
                                      fingerprints=first, result={"old": True})
    assert get_current_state(state, scope) == before
    assert commit_analysis_result(state, scope=scope, run_id=new_run,
                                  fingerprints=second, result={"new": True})
    committed = get_current_state(state, scope)
    assert not commit_analysis_result(state, scope=scope, run_id=new_run,
                                      fingerprints=second, result={"duplicate": True})
    assert get_current_state(state, scope) == committed
    assert get_current_result(state, scope=scope, fingerprints=second) == {"new": True}
