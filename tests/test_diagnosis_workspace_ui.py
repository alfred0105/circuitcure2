from types import SimpleNamespace
from pathlib import Path

import pytest

from circuitcare.dashboard.views import unified_workspace as workspace
from circuitcare.dashboard.views import symptom_diagnosis_panel as symptoms
from circuitcare.dashboard.views import inspection_panel as inspection
from circuitcare.frontend.models import ProjectContext
from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel
from circuitcare.frontend.workbench_state import (
    ACTIVE_PHOTO_KEY, SELECTION_KEY, WORKFLOW_REVIEW, WORKFLOW_QUESTIONS, WORKFLOW_INSPECTION, WORKFLOW_COMPLETE,
    get_workflow_state, transition_workflow, enter_workbench,
)
from circuitcare.dashboard.views.symptom_diagnosis_panel import candidate_focus_widget_key
from circuitcare.dashboard.views.symptom_diagnosis_panel import (
    _queue_measurement_change_rerun, _consume_measurement_change_rerun,
    _ensure_question_plan, _question_plan_fingerprint, _save_follow_up_answer,
)
from circuitcare.analysis.models import AnalysisStatus
from circuitcare.frontend.board_component_registry import build_board_component_registry, source_component_key
from test_board_component_registry import _snapshots


class _Context:
    def button(self, *_args, **_kwargs):
        return False

    def __enter__(self):
        return self

    def container(self):
        return _Context()

    def __exit__(self, *_args):
        return False

    def markdown(self, *_args, **_kwargs):
        pass

    def caption(self, *_args, **_kwargs):
        pass

    def empty(self):
        pass

    def status(self, *_args, **_kwargs):
        return self

    def write(self, *_args, **_kwargs):
        pass

    def update(self, *_args, **_kwargs):
        pass


class _Status:
    def __init__(self, label):
        self.label = label
        self.state = "running"
        self.messages = []

    def write(self, message):
        self.messages.append(message)

    def update(self, *, label=None, state=None, **_kwargs):
        if label:
            self.label = label
        if state:
            self.state = state


class _StreamlitStub:
    def __init__(self, *, state=None, clicked=()):
        self.session_state = state or {}
        self.clicked = set(clicked)
        self.radio_calls = []
        self.statuses = []

    def container(self, **_kwargs):
        return _Context()

    def empty(self):
        return _Context()

    def status(self, label, **_kwargs):
        status = _Status(label)
        self.statuses.append(status)
        return status

    def columns(self, specs, **_kwargs):
        count = specs if isinstance(specs, int) else len(specs)
        return [_Context() for _ in range(count)]

    def expander(self, *_args, **_kwargs):
        return _Context()

    def radio(self, label, options, *, key=None, **_kwargs):
        self.radio_calls.append((label, tuple(options), key))
        return self.session_state.get(key, options[0])

    def button(self, _label, *, key=None, **_kwargs):
        return key in self.clicked

    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


def test_analysis_workspace_only_renders_selected_panel_instead_of_hidden_tabs(monkeypatch):
    project = ProjectContext(project_id="project-1", board_id="board-1")
    state = {}
    enter_workbench(state, project)
    st = _StreamlitStub(state=state)
    calls = []
    monkeypatch.setattr(workspace, "render_symptom_diagnosis_panel", lambda *_args, **_kwargs: calls.append("symptom"))
    workspace._render_fault_analysis_workspace(st, project, preview=False, mode="증상")

    assert calls == ["symptom"]


def test_workflow_state_is_board_scoped_and_only_accepts_explicit_valid_transitions():
    first = ProjectContext("wf-project", "front-board")
    second = ProjectContext("wf-project", "other-board")
    state = {}
    workflow = get_workflow_state(state, first)
    assert workflow["stage"] == WORKFLOW_REVIEW
    with pytest.raises(ValueError, match="허용되지 않은"):
        transition_workflow(state, first, WORKFLOW_INSPECTION)
    transition_workflow(state, first, WORKFLOW_QUESTIONS, event="review_complete")
    assert get_workflow_state(state, first)["stage"] == WORKFLOW_QUESTIONS
    assert get_workflow_state(state, second)["stage"] == WORKFLOW_REVIEW
    assert workflow["stage_history"][-1]["event"] == "review_complete"
    # No valid physical target is an explicit, safe return path; collected work remains.
    workflow["question_answers"] = [{"test_id": "ASK_POWER_SOURCE", "answer": "예"}]
    workflow["measurement_ids"] = ["measurement-kept"]
    transition_workflow(state, first, WORKFLOW_INSPECTION, event="questions_complete")
    transition_workflow(state, first, WORKFLOW_REVIEW, event="no_valid_target",
                        reason="Registry에 검사 가능한 관측이 없음")
    assert get_workflow_state(state, first)["question_answers"] == [
        {"test_id": "ASK_POWER_SOURCE", "answer": "예"}
    ]
    assert get_workflow_state(state, first)["measurement_ids"] == ["measurement-kept"]
    assert get_workflow_state(state, first)["transition_reason"] == "Registry에 검사 가능한 관측이 없음"
    with pytest.raises(ValueError, match="허용되지 않은"):
        transition_workflow(state, first, WORKFLOW_COMPLETE)


def test_question_plan_is_catalog_bounded_unique_and_stable_across_reruns():
    state = {"answered_test_ids": ["Q0"]}
    catalog = {f"Q{index}": {
        "type": "question", "question": f"질문 {index}",
        "results": {"yes": {"observations": {f"OBS_{index}": True}}},
    } for index in range(8)}
    diagnosis = {
        "llm_follow_up_question": {"test_id": "Q7", "type": "question",
                                  "question": "LLM의 비신뢰 문장은 사용하지 않음"},
        "next_tests": ([{"test_id": "Q1", "type": "question"},
                        {"test_id": "Q1", "type": "question"},
                        {"test_id": "UNLISTED", "type": "question"},
                        *[{"test_id": f"Q{index}", "type": "question"} for index in range(2, 8)]]),
    }
    knowledge = {"test_catalog": {"tests": catalog}, "fault_modes": {"fault_modes": {}}}
    plan = _ensure_question_plan(state, diagnosis, knowledge=knowledge, plan_fingerprint="fp-1")
    ids = [row["test_id"] for row in plan["questions"]]
    assert len(ids) <= 5
    assert len(ids) == len(set(ids))
    assert set(ids) <= set(catalog)
    assert ids[0] == "Q7"
    question = plan["questions"][0]["question"]
    _save_follow_up_answer(state, question, "예")
    assert _ensure_question_plan(state, diagnosis, knowledge=knowledge,
                                 plan_fingerprint="fp-1") is plan
    assert plan["questions"][0]["status"] == "ANSWERED"


def test_question_plan_fingerprint_ignores_answers_and_view_state_but_tracks_symptom_and_registry():
    state = {"initial_symptom_answer": "전원 안 켜짐", "answered_test_ids": []}
    result = {"hypotheses": [{"fault_id": "F1"}]}
    knowledge = {"test_catalog": {"tests": {"Q1": {
        "type": "question", "question": "출력이 있나요?", "risk_level": "none",
    }}}}
    registry = {"components": [{"board_component_id": "BC1", "component_type": "resistor",
                                "presence_status": "CONFIRMED"}]}
    before = _question_plan_fingerprint(
        state, knowledge=knowledge, selected_codes=["NO_POWER"],
        symptom_text="전원이 안 켜짐", board_registry=registry, suspected_components=[],
    )
    state["answered_test_ids"] = ["Q1"]
    state["question_number"] = 4
    state["zoom"] = 2.0
    after = _question_plan_fingerprint(
        state, knowledge=knowledge, selected_codes=["NO_POWER"],
        symptom_text="전원이 안 켜짐", board_registry=registry, suspected_components=[],
    )
    assert before == after
    changed_symptom = _question_plan_fingerprint(
        state, knowledge=knowledge, selected_codes=["OUTPUT_FAILURE"],
        symptom_text="전원이 안 켜짐", board_registry=registry, suspected_components=[],
    )
    assert changed_symptom != before
    changed_registry = {"components": [{"board_component_id": "BC1", "component_type": "capacitor",
                                       "presence_status": "CONFIRMED"}]}
    assert _question_plan_fingerprint(
        state, knowledge=knowledge, selected_codes=["NO_POWER"],
        symptom_text="전원이 안 켜짐", board_registry=changed_registry, suspected_components=[],
    ) != before


def test_question_plan_does_not_pad_unrelated_catalog_questions_and_avoids_answered_on_replan():
    state = {"answered_test_ids": ["Q1"]}
    knowledge = {"test_catalog": {"tests": {
        "Q1": {"type": "question", "risk_level": "none", "question": "기존 질문"},
        "Q2": {"type": "question", "risk_level": "none", "question": "관련 질문"},
        "Q3": {"type": "question", "risk_level": "none", "question": "무관 질문"},
    }}}
    result = {"next_tests": [{"test_id": "Q2", "type": "question"}],
              "llm_follow_up_question": {"test_id": "Q3", "type": "question"}}
    plan = _ensure_question_plan(state, result, knowledge=knowledge, plan_fingerprint="v1")
    assert [row["test_id"] for row in plan["questions"]] == ["Q2"]
    changed = _ensure_question_plan(state, result, knowledge=knowledge, plan_fingerprint="v2")
    assert [row["test_id"] for row in changed["questions"]] == ["Q2"]


def test_simple_diagnosis_question_renderer_runs_with_real_result_and_no_suspects():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string('''
import streamlit as st
from circuitcare.dashboard.views.symptom_diagnosis_panel import _render_simple_diagnosis_result
from circuitcare.frontend.models import ProjectContext
project = ProjectContext("app-test-question", "board")
_render_simple_diagnosis_result(
    st, project, {"status": "OK", "candidates": [], "next_tests": []},
    state={"symptoms": []}, knowledge={"test_catalog": {"tests": {}}, "fault_modes": {"fault_modes": {}}},
    preview=False, board_registry={"components": []}, suspected_components=[],
)
''').run()
    assert not app.exception


def test_question_renderer_receives_only_registry_valid_suspects():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string('''
import streamlit as st
from circuitcare.dashboard.views.symptom_diagnosis_panel import _render_simple_diagnosis_result
from circuitcare.frontend.models import ProjectContext
project = ProjectContext("app-test-suspects", "board")
state = {"symptoms": []}
_render_simple_diagnosis_result(
    st, project, {"status": "OK", "candidates": [], "next_tests": []}, state=state,
    knowledge={"test_catalog": {"tests": {}}, "fault_modes": {"fault_modes": {}}},
    preview=False, board_registry={"components": []},
    suspected_components=[{"component_id": "BC1", "board_component_id": "BC1"}],
)
st.write(state["valid_suspected_components"])
''').run()
    assert not app.exception


def test_question_renderer_receives_prevalidated_suspects(monkeypatch):
    captured = {}
    monkeypatch.setattr(symptoms, "_render_simple_followup_question",
                        lambda _st, _project, _result, *, state, **_kwargs:
                        captured.update(suspects=state["valid_suspected_components"]))
    project = ProjectContext("question-suspects", "board")
    valid, stale = symptoms._registry_valid_suspects(
        [{"board_component_id": "BC1", "component_id": "local1", "photo_id": "front",
          "selection_version": "obs1"},
         {"board_component_id": "BC2", "component_id": "local2", "photo_id": "front",
          "selection_version": "obs2"}],
        {"photos": [{"photo_id": "front", "image_fingerprint": "hash"}],
         "components": [{"board_component_id": "BC1", "presence_status": "CONFIRMED",
             "source_components": [{"component_id": "local1", "photo_id": "front",
                 "review_status": "USER_CONFIRMED", "source_observation_fingerprint": "obs1"}]},
             {"board_component_id": "BC2", "presence_status": "EXCLUDED",
              "source_components": [{"component_id": "local2", "photo_id": "front",
                  "review_status": "EXCLUDED", "source_observation_fingerprint": "obs2"}]}]},
    )
    symptoms._render_simple_diagnosis_result(
        SimpleNamespace(session_state={}, caption=lambda *_args, **_kwargs: None), project,
        {"status": "OK", "candidates": [], "next_tests": []}, state={"symptoms": []},
        knowledge={"test_catalog": {"tests": {}}, "fault_modes": {"fault_modes": {}}},
        preview=False, board_registry=None, suspected_components=valid,
    )
    assert [row["board_component_id"] for row in captured["suspects"]] == ["BC1"]
    assert len(stale) == 1


@pytest.mark.parametrize(("base_class", "subtype", "expected_component", "expected_status"), [
    ("connector", "fpc_ffc", "connector", "USER_CONFIRMED"),
    ("other_component", "ic_socket", "unknown", "USER_CONFIRMED_EXTENSION"),
    ("board_feature", "test_point", "unknown", "USER_CONFIRMED_EXTENSION"),
])
def test_confirmed_qwen_types_preserve_yolo_and_keep_extensions_out_of_18_classes(
    base_class, subtype, expected_component, expected_status,
):
    from circuitcare.dashboard.views.inspection_panel import confirm_qwen_component_type

    row = {"component_id": "U1", "소자 종류": "ic"}
    confirm_qwen_component_type(row, base_class=base_class, subtype=subtype)
    assert row["detector_class"] == "ic"
    assert row["confirmed_base_class"] == base_class
    assert row["confirmed_subtype"] == subtype
    assert row["소자 종류"] == expected_component
    assert row["qwen_component_type_status"] == expected_status
    assert row["상태"] == "확인"


def test_inspection_targets_use_real_registry_ids_order_evidence_and_never_invent_boxes():
    project = ProjectContext("workflow-targets", "workflow-target-board")
    registry = build_board_component_registry(
        {}, project_id=project.project_id, board_id=project.board_id,
        photo_snapshots=_snapshots(),
    )
    front_id = registry["source_to_board_id"][source_component_key("front", "local-1")]
    close_id = registry["source_to_board_id"][source_component_key("close", "close-1")]
    suspects = [{"board_component_id": close_id}]
    selected = workspace._inspection_targets(
        registry,
        [{"board_component_id": front_id, "component_type": "resistor",
          "fault_hypotheses": [{"fault_id": "RESISTOR_OPEN"}]}],
        suspects,
    )
    assert [row["board_component_id"] for row in selected[:2]] == [
        front_id, close_id,
    ]
    assert len(selected) <= 3
    assert all(row["board_component_id"] in {item["board_component_id"] for item in registry["components"]}
               for row in selected)

    excluded_only = build_board_component_registry(
        {}, project_id=project.project_id, board_id=project.board_id,
        photo_snapshots={"front": {"components": (
            SimpleNamespace(component_id="X1", component_type="resistor", bbox_xyxy=(1, 2, 8, 9),
                            coordinate_frame="SOURCE_PHOTO_PIXEL", identification_status="EXCLUDED",
                            presence_status="EXCLUDED", component_type_status="UNCONFIRMED",
                            ocr_marking=None, possible_part_number=None, confirmed_functional_roles=(),
                            evidence=(), overlay_source="test"),),
            "image_fingerprint": "excluded", "review_version": "excluded"}},
    )
    assert workspace._inspection_targets(excluded_only, [], []) == []


def test_cause_candidate_selection_opens_same_pcb_viewer_and_updates_shared_selection():
    project = ProjectContext(project_id="project-2", board_id="board-2")
    state = {"circuitcare_v4_current_components": (SimpleNamespace(component_id="U3"),)}
    enter_workbench(state, project)
    button_key = candidate_focus_widget_key("analysis_dashboard", "project-2", "board-2", "U3", "")
    st = _StreamlitStub(state=state, clicked={button_key})
    result = {"component_candidates": [{
        "component_id": "U3", "component_type": "ic", "fault_title": "전원 입력 이상",
        "candidate_rank": 1, "supporting_evidence": ["출력이 없음"],
        "opposing_evidence": [],
    }]}

    workspace._render_cause_dashboard(st, project, result)

    assert state[SELECTION_KEY].selected_component == "U3"


def test_board_candidate_focus_changes_to_its_source_photo_and_photo_local_box():
    project = ProjectContext("project-board", "board-board")
    state = {}
    enter_workbench(state, project)
    registry = {"components": [{
        "board_component_id": "bc-1",
        "source_components": [{"photo_id": "close-2", "component_id": "det-17",
                               "coordinate_frame": "SOURCE_PHOTO_PIXEL", "bbox_xyxy": [1, 2, 8, 9]}],
    }], "photos": [{"photo_id": "close-2", "image_size_px": [20, 20]}]}
    st = _StreamlitStub(state=state)

    assert workspace._focus_board_registry_component(st, project, registry, "bc-1")
    assert state[ACTIVE_PHOTO_KEY] == "close-2"
    assert state[SELECTION_KEY].active_photo_id == "close-2"
    assert state[SELECTION_KEY].selected_component == "det-17"


def test_measurement_target_changes_shared_selection_only_when_callback_runs():
    project = ProjectContext("project-measure", "board-measure")
    state = {}
    enter_workbench(state, project)
    registry = {"components": [{
        "board_component_id": "bc-2",
        "presence_status": "UNCONFIRMED",
        "source_components": [{"photo_id": "back-photo", "component_id": "R8",
                               "coordinate_frame": "SOURCE_PHOTO_PIXEL", "bbox_xyxy": [1, 2, 8, 9],
                               "component_type": "resistor", "review_status": "검수 필요"}],
    }]}
    st = _StreamlitStub(state=state)
    target_key = "measurement-target"
    state[target_key] = "bc-2"

    workspace._handle_measurement_target_change(state, project, registry, target_key)

    assert state[workspace.MEASUREMENT_TARGET_BOARD_COMPONENT_KEY] == "bc-2"
    assert workspace.BOARD_COMPONENT_SELECTION_KEY not in state
    assert "circuitcare_v2_photo_choice" not in state
    assert state[SELECTION_KEY].selected_component is None


def test_fixed_viewer_shell_starts_at_component_review_with_one_current_stage(monkeypatch):
    project = ProjectContext("project-shell", "board-shell")
    registry = {"project_id": project.project_id, "board_id": project.board_id,
                "version": "board-registry:v1", "components": []}
    monkeypatch.setattr(workspace, "_style", lambda _st: None)
    monkeypatch.setattr(workspace, "_prepare_board_registry_snapshot", lambda *_args, **_kwargs: (registry, {}))
    monkeypatch.setattr(workspace, "current_analysis_snapshot", lambda *_args, **_kwargs: {
        "result": None, "record": {"status": "NOT_RUN"}})
    monkeypatch.setattr(workspace, "_diagnosis_status_label", lambda *_args, **_kwargs: "분석 전")
    calls = []
    monkeypatch.setattr(workspace, "_render_pcb", lambda *_args, **_kwargs: calls.append("viewer"))
    monkeypatch.setattr(workspace, "_render_component_inspector", lambda *_args, **_kwargs: calls.append("review"))

    state = {}
    enter_workbench(state, project)
    state[f"cc_workbench_panel_{project.project_id}"] = "원인"  # ignored legacy navigation state
    workspace.render_workbench_shell(_StreamlitStub(state=state), project)
    assert calls == ["viewer", "review"]


def test_left_transition_status_is_running_before_preparation_and_replaced_after_viewer(monkeypatch):
    project = ProjectContext("project-left-loader", "board-left-loader")
    registry = {"project_id": project.project_id, "board_id": project.board_id,
                "version": "registry-v1", "components": []}
    monkeypatch.setattr(workspace, "_style", lambda _st: None)
    monkeypatch.setattr(workspace, "diagnosis_components", lambda _registry: ())
    monkeypatch.setattr(workspace, "measurement_component_aliases", lambda _registry: {})
    monkeypatch.setattr(workspace, "board_suspects", lambda *_args: [])
    monkeypatch.setattr(workspace, "current_analysis_snapshot", lambda *_args, **_kwargs: {
        "result": None, "record": {"status": "NOT_RUN"}, "input_fingerprint": "fp",
    })
    monkeypatch.setattr(workspace, "_diagnosis_status_label", lambda *_args, **_kwargs: "분석 전")
    rendered = []
    monkeypatch.setattr(workspace, "_render_pcb", lambda *_args, **_kwargs: rendered.append("viewer"))
    monkeypatch.setattr(workspace, "render_symptom_diagnosis_panel", lambda *_args, **_kwargs: None)
    state = {}
    enter_workbench(state, project)
    transition_workflow(state, project, WORKFLOW_QUESTIONS, event="review_complete",
                        left_status="질문 계획 준비 중")
    st = _StreamlitStub(state=state)

    def prepare(*_args, **_kwargs):
        assert st.statuses and st.statuses[-1].state == "running"
        assert rendered == []
        return registry, {"components": ()}

    monkeypatch.setattr(workspace, "_prepare_board_registry_snapshot", prepare)
    workspace.render_workbench_shell(st, project)

    assert rendered == ["viewer"]
    assert st.statuses[0].state == "complete"
    assert state["circuitcare_v8_diagnostic_workflows"][f"{project.project_id}|{project.board_id}"][
        "transition_status"] == "READY"


def test_simple_candidate_list_aggregates_faults_by_physical_component():
    result = {"candidate_fusion": {"top_candidates": [
        {"board_component_id": "BC-1", "fault_id": "F-A", "score": 8},
        {"board_component_id": "BC-1", "fault_id": "F-B", "score": 7},
        {"board_component_id": "BC-2", "fault_id": "F-C", "score": 6},
        {"board_component_id": "BC-3", "fault_id": "F-D", "score": 5},
        {"board_component_id": "BC-4", "fault_id": "F-E", "score": 4},
    ]}}

    candidates = symptoms._simple_candidate_rows(result)

    assert [row["board_component_id"] for row in candidates] == ["BC-1", "BC-2", "BC-3"]
    assert [row["fault_id"] for row in candidates[0]["fault_hypotheses"]] == ["F-A", "F-B"]


def test_reassessment_preserves_refuted_conflicted_hypotheses_but_recommendations_filter_them():
    result = {"candidate_fusion": {"candidates": [
        {"board_component_id": "BC-1", "fault_id": "F-SUPPORTED",
         "measurement_verdict": "supports", "visual_status": "FAULT_SUPPORTED"},
        {"board_component_id": "BC-1", "fault_id": "F-REFUTED",
         "measurement_verdict": "refutes", "visual_status": "FAULT_REFUTED"},
        {"board_component_id": "BC-1", "fault_id": "F-CONFLICTED",
         "measurement_verdict": "conflicted", "visual_status": "CONFLICTED"},
    ]}}
    hypotheses = symptoms._simple_candidate_rows(result, limit=1000)[0]["fault_hypotheses"]
    assert {row["fault_id"] for row in hypotheses} == {"F-SUPPORTED", "F-REFUTED", "F-CONFLICTED"}
    active = workspace._inspection_candidate_rows(result)
    assert [row["fault_id"] for row in active[0]["fault_hypotheses"]] == ["F-SUPPORTED"]


def test_inspection_targets_prioritize_measurable_components_and_skip_completed_profiles():
    project = ProjectContext("target-priority", "target-priority-board")
    registry = build_board_component_registry(
        {}, project_id=project.project_id, board_id=project.board_id,
        photo_snapshots=_snapshots(),
    )
    resistor_id = next(row["board_component_id"] for row in registry["components"]
                       if row.get("component_type") == "resistor")
    candidates = [{"board_component_id": resistor_id, "component_type": "resistor",
                   "fault_id": "RESISTOR_FAILURE"}]
    targets = workspace._inspection_targets(registry, candidates, [])
    assert targets[0]["board_component_id"] == resistor_id
    assert targets[0]["inspection_kind"] == "MEASUREMENT"
    assert targets[0]["available_tests"]
    completed = [{"board_component_id": resistor_id, "fault_id": "RESISTOR_FAILURE",
                  "test_id": "isolated_resistance", "verification_status": "FAULT_REFUTED"}]
    after = workspace._inspection_targets(registry, candidates, [], inspection_history=completed)
    resistor = next(row for row in after if row["board_component_id"] == resistor_id)
    assert not any(item["test_id"] == "isolated_resistance" for item in resistor["available_tests"])
    assert resistor["completed_test_count"] >= 1
    retry = workspace._inspection_targets(registry, candidates, [], inspection_history=completed,
                                         allow_retest=True)
    retry_resistor = next(row for row in retry if row["board_component_id"] == resistor_id)
    assert any(item["test_id"] == "isolated_resistance" for item in retry_resistor["available_tests"])


def test_visual_inspection_is_scoped_to_compatible_component_fault_and_symptom():
    catalog = symptoms.load_knowledge()["test_catalog"]["tests"]
    resistor_checks = workspace._compatible_visual_checks(
        catalog, target_scope="component", component_type="resistor",
        fault_ids=["RESISTOR_FAILURE"], symptoms={"OVERHEATING"},
    )
    assert {row["test_id"] for row in resistor_checks} == {"INSPECT_LOCAL_HEATING"}
    connector_checks = workspace._compatible_visual_checks(
        catalog, target_scope="component", component_type="connector",
        fault_ids=["CONNECTOR_CONTACT_FAILURE"], symptoms={"NO_POWER"},
    )
    assert {row["test_id"] for row in connector_checks} == {"INSPECT_POWER_CONNECTOR"}
    board_checks = workspace._compatible_visual_checks(
        catalog, target_scope="board", symptoms={"INTERMITTENT_OPERATION"},
        fault_ids=["SOLDER_JOINT_FAILURE"],
    )
    assert {row["test_id"] for row in board_checks} == {"INSPECT_VISIBLE_SOLDER_AND_TRACES"}
    assert not workspace._compatible_visual_checks(
        catalog, target_scope="component", component_type="resistor",
        fault_ids=["CONNECTOR_CONTACT_FAILURE"], symptoms=set(),
    )


def test_visual_inspection_cannot_save_without_safety_confirmation_and_result_is_unselected():
    class VisualStub(_StreamlitStub):
        def __init__(self, *, state=None, clicked=()):
            super().__init__(state=state, clicked=clicked)
            self.radio_kwargs = []
            self.button_disabled = []

        def checkbox(self, _label, *, key=None, **_kwargs):
            return bool(self.session_state.get(key, False))

        def radio(self, _label, options, *, key=None, **kwargs):
            self.radio_kwargs.append(kwargs)
            return self.session_state.get(key)

        def button(self, _label, *, key=None, disabled=False, **_kwargs):
            self.button_disabled.append(disabled)
            return key in self.clicked and not disabled

        def warning(self, *_args, **_kwargs):
            pass

    project = ProjectContext("visual-safe", "visual-safe-board")
    test = symptoms.load_knowledge()["test_catalog"]["tests"]["INSPECT_POWER_CONNECTOR"]
    state = {"cc_wf_visual_result_visual-safe_BC1_cycle1_INSPECT_POWER_CONNECTOR": None}
    st = VisualStub(state=state, clicked={"cc_wf_visual_save_visual-safe_BC1_cycle1_INSPECT_POWER_CONNECTOR"})
    workflow = {"inspection_cycle": {"cycle_id": 1}, "inspection_history": []}
    workspace._render_visual_inspection(
        st, project, workflow, test={"test_id": "INSPECT_POWER_CONNECTOR", **test},
        target_id="BC1", analysis_fingerprint="fp-1", preview=False,
        widget_suffix="visual-safe_BC1",
    )
    assert st.radio_kwargs[0]["index"] is None
    assert st.button_disabled[-1] is True
    assert workflow["inspection_history"] == []


def test_streamlit_apptest_radio_index_none_starts_unselected():
    app_testing = pytest.importorskip("streamlit.testing.v1")
    app = app_testing.AppTest.from_string(
        "import streamlit as st\nst.radio('결과', ('이상 발견', '정상'), index=None)"
    ).run()
    assert not app.exception
    assert app.radio[0].value is None


def test_visual_inspection_unknown_is_history_only_and_not_damage_evidence():
    class VisualStub(_StreamlitStub):
        def checkbox(self, _label, *, key=None, **_kwargs):
            return True

        def radio(self, _label, _options, *, key=None, **_kwargs):
            return "확인 불가"

        def button(self, _label, *, key=None, disabled=False, **_kwargs):
            return "_save_" in key and not disabled

        def warning(self, *_args, **_kwargs):
            pass

        def info(self, *_args, **_kwargs):
            pass

        def rerun(self):
            pass

    project = ProjectContext("visual-unknown", "visual-unknown-board")
    catalog_test = symptoms.load_knowledge()["test_catalog"]["tests"]["INSPECT_POWER_CONNECTOR"]
    test = {"test_id": "INSPECT_POWER_CONNECTOR", **catalog_test}
    workflow = {"stage": WORKFLOW_INSPECTION, "inspection_cycle": {"cycle_id": 3},
                "inspection_history": [], "stage_completion": {}, "stage_history": []}
    state = {"circuitcare_v8_diagnostic_workflows": {
        f"{project.project_id}|{project.board_id}": workflow,
    }}
    workspace._render_visual_inspection(
        VisualStub(state=state), project, workflow, test=test, target_id="BC1",
        analysis_fingerprint="fp-1", preview=False, widget_suffix="visual-unknown_BC1",
    )
    assert workflow["inspection_history"][0]["verification_status"] == "INCONCLUSIVE"
    assert workflow["inspection_history"][0]["observations"] == {}
    assert state["cc_symptom_diagnosis_visual-unknown"].get("manual_observations", {}) == {}


def test_identical_visual_observation_is_not_saved_again_without_explicit_retest():
    class VisualStub(_StreamlitStub):
        def __init__(self, *, state=None):
            super().__init__(state=state)
            self.button_disabled = []

        def checkbox(self, _label, *, key=None, **_kwargs):
            default = key.startswith(("cc_wf_visual_power_off_", "cc_wf_visual_energy_safe_"))
            return bool(self.session_state.get(key, default))

        def radio(self, _label, _options, *, key=None, **_kwargs):
            return self.session_state.get(key, "이상 발견")

        def button(self, _label, *, disabled=False, **_kwargs):
            self.button_disabled.append(disabled)
            return False

        def warning(self, *_args, **_kwargs):
            pass

        def info(self, *_args, **_kwargs):
            pass

    project = ProjectContext("visual-duplicate", "visual-duplicate-board")
    catalog_test = symptoms.load_knowledge()["test_catalog"]["tests"]["INSPECT_POWER_CONNECTOR"]
    test = {"test_id": "INSPECT_POWER_CONNECTOR", **catalog_test}
    checks = {"power_disconnected": True, "stored_energy_safe": True}
    observation = {"VISIBLE_CONNECTOR_DAMAGE": True}
    fingerprint = workspace._visual_observation_fingerprint(
        board_id=project.board_id, test_id="INSPECT_POWER_CONNECTOR",
        target_scope="connector", target_id="BC1", result="이상 발견",
        observations=observation, safety_checks=checks,
    )
    workflow = {
        "inspection_cycle": {"cycle_id": 4},
        "inspection_history": [{
            "history_id": "old-inspection", "observation_fingerprint": fingerprint,
            "verification_status": "FAULT_SUPPORTED",
        }],
    }
    st = VisualStub(state={
        "cc_wf_visual_result_visual-duplicate_BC1_cycle4_INSPECT_POWER_CONNECTOR": "이상 발견",
    })
    workspace._render_visual_inspection(
        st, project, workflow, test=test, target_id="BC1", analysis_fingerprint="fp-2",
        preview=False, widget_suffix="visual-duplicate_BC1", fault_id="CONNECTOR_CONTACT_FAILURE",
    )
    assert st.button_disabled[-1] is True
    assert len(workflow["inspection_history"]) == 1


def test_inspection_target_count_tracks_only_real_active_components():
    project = ProjectContext("target-count", "target-count-board")
    registry = build_board_component_registry(
        {}, project_id=project.project_id, board_id=project.board_id,
        photo_snapshots=_snapshots(),
    )
    active = [row for row in registry["components"] if row.get("presence_status") != "EXCLUDED"]
    assert len(workspace._inspection_targets({**registry, "components": active[:1]}, [], [])) == 1
    assert len(workspace._inspection_targets({**registry, "components": active[:2]}, [], [])) == 2
    assert len(workspace._inspection_targets(registry, [], [])) == min(3, len(active))


def test_auto_qwen_prefers_installed_qwen35_9b_and_caches_lookup(monkeypatch):
    class OllamaStub:
        calls = 0

        def __init__(self, *, timeout_seconds):
            assert timeout_seconds == 2

        def list_models(self):
            OllamaStub.calls += 1
            return [{"name": "qwen2.5:14b"}, {"name": "qwen3.5:9b"}, {"name": "gemma4:12b"}]

    monkeypatch.setattr(symptoms, "OllamaClient", OllamaStub)
    state = {}

    assert symptoms._auto_qwen_model(state) == "qwen3.5:9b"
    assert symptoms._auto_qwen_model(state) == "qwen3.5:9b"
    assert OllamaStub.calls == 1


def test_compact_inspector_renders_selected_component_without_scope_error():
    class InspectorStub(_StreamlitStub):
        def selectbox(self, _label, options, *, key=None, **_kwargs):
            return self.session_state.get(key, options[0])

    project = ProjectContext("compact-p", "compact-b")
    component = SimpleNamespace(
        component_id="R1", board_component_id="BC-R1", component_type="resistor",
        reference_designators=("R1",), ocr_marking="103", bbox_xyxy=(1, 2, 8, 9),
        source_backend_photo_ids=("photo-1",),
    )
    state = {SELECTION_KEY: SimpleNamespace(project_id=project.project_id, selected_component="R1")}

    inspection.render_inspection_panel(
        InspectorStub(state=state), project, components=(component,),
        selected_component=component, compact=True,
    )

    assert state[inspection.USER_SUSPECTS_KEY][f"{project.project_id}|{project.board_id}"] == {}


def test_dismissed_board_suspect_is_not_restored_by_legacy_scope_migration():
    project = ProjectContext("suspect-p", "suspect-b")
    owner = f"{project.project_id}|{project.board_id}"
    old_owner = f"{owner}|photo-1"
    state = {
        inspection.USER_SUSPECTS_KEY: {
            old_owner: {"BC-R1": {"board_component_id": "BC-R1", "component_id": "R1"}},
        },
        inspection.USER_SUSPECTS_MIGRATION_KEY: {},
    }

    inspection.dismiss_board_suspect(state, project, "BC-R1")
    assert inspection.board_suspects(state, project) == []
    assert "BC-R1" in state[inspection.USER_SUSPECTS_DISMISSED_KEY][owner]


def test_integrated_workspace_uses_candidate_rows_and_guides_measurement(monkeypatch):
    project = ProjectContext("integrated-p", "integrated-b")
    component = SimpleNamespace(component_id="R1", board_component_id="BC-R1", component_type="resistor")
    result = {"candidate_fusion": {"candidates": [
        {"board_component_id": "BC-R1", "component_id": "R1", "fault_id": "RESISTOR_OPEN"},
    ]}}
    snapshot = {"result": result, "record": {"status": "CURRENT"}}
    monkeypatch.setattr(workspace, "render_symptom_diagnosis_panel", lambda *_a, **_k: None)
    monkeypatch.setattr(workspace, "current_analysis_snapshot", lambda *_a, **_k: snapshot)
    measured = []
    monkeypatch.setattr(workspace, "render_measurement_panel",
                        lambda _st, _project, **kwargs: measured.append(kwargs["selected_component"]))

    workspace._render_fault_analysis_workspace(
        _StreamlitStub(state={"circuitcare_v4_current_components": (component,)}), project,
        preview=False, mode="통합", analysis_snapshot=snapshot,
        photo_snapshot={"components": (component,)}, board_components=(component,),
    )

    # A board-level diagnosis DTO has no photo-local geometry and must not be
    # passed directly into a form that expects a viewer/measurement model.
    assert measured == [None]


def test_integrated_measurement_never_falls_back_to_unrelated_viewer_selection(monkeypatch):
    project = ProjectContext("integrated-target-p", "integrated-target-b")

    def component(local_id, board_id, kind):
        return ComponentOverlayViewModel(
            project_id=project.project_id, board_id=project.board_id, board_side="TOP",
            component_id=local_id, board_component_id=board_id, component_type=kind,
            display_label=local_id, canonical_frame_id="photo-1",
            coordinate_frame="SOURCE_PHOTO_PIXEL", bbox_xyxy=(1, 2, 8, 9), polygon_xy=(),
            source_backend_photo_ids=("photo-1",), observation_ids=(), detection_confidence=.9,
            identification_status="USER_CONFIRMED", identity_match_status="PHOTO_LOCAL",
            ocr_marking=None, possible_part_number=None, evidence=(), overlay_status="CANDIDATE",
        )

    candidate = component("R1", "BC-R1", "resistor")
    unrelated_selection = component("C2", "BC-C2", "capacitor")
    result = {"candidate_fusion": {"candidates": [
        {"board_component_id": "BC-R1", "component_id": "R1", "fault_id": "RESISTOR_OPEN"},
    ]}}
    snapshot = {"result": result, "record": {"status": "CURRENT"}}
    monkeypatch.setattr(workspace, "render_symptom_diagnosis_panel", lambda *_a, **_k: None)
    monkeypatch.setattr(workspace, "current_analysis_snapshot", lambda *_a, **_k: snapshot)
    measured = []
    monkeypatch.setattr(workspace, "render_measurement_panel",
                        lambda _st, _project, **kwargs: measured.append(kwargs["selected_component"]))
    state = {
        "circuitcare_v4_current_components": (candidate, unrelated_selection),
        SELECTION_KEY: SimpleNamespace(selected_component="C2"),
        workspace.BOARD_COMPONENT_SELECTION_KEY: "BC-C2",
    }
    workspace._render_fault_analysis_workspace(
        _StreamlitStub(state=state), project, preview=False, mode="통합", analysis_snapshot=snapshot,
        photo_snapshot={"components": (candidate, unrelated_selection)},
        board_components=(candidate, unrelated_selection),
    )

    assert measured == [candidate]
    assert state[SELECTION_KEY].selected_component == "C2"
    assert state[workspace.BOARD_COMPONENT_SELECTION_KEY] == "BC-C2"
    assert state[workspace.MEASUREMENT_TARGET_BOARD_COMPONENT_KEY] == "BC-R1"


def test_excluded_registry_component_cannot_be_focused_as_measurement_or_viewer_target():
    project = ProjectContext("excluded-focus-p", "excluded-focus-b")
    registry = {"components": [{
        "board_component_id": "bc-x", "presence_status": "EXCLUDED",
        "source_components": [{"photo_id": "p1", "component_id": "R1",
                               "coordinate_frame": "SOURCE_PHOTO_PIXEL", "bbox_xyxy": [1, 2, 8, 9]}],
    }]}
    state = {}
    enter_workbench(state, project)
    st = _StreamlitStub(state=state)
    assert workspace._focus_board_registry_component(st, project, registry, "bc-x") is False
    state["target"] = "bc-x"
    workspace._handle_measurement_target_change(state, project, registry, "target")
    assert workspace.MEASUREMENT_TARGET_BOARD_COMPONENT_KEY not in state
    assert workspace.BOARD_COMPONENT_SELECTION_KEY not in state


def test_measurement_change_queues_exactly_one_reanalysis_for_current_fingerprint():
    state = {"measurement_fingerprint": "m-v1", "run_after_measurement_change": None}
    _queue_measurement_change_rerun(
        state, previous_fingerprint="m-v1", current_fingerprint="m-v2",
        prior_analysis_exists=True, current_result=None, central_status=AnalysisStatus.STALE.value,
    )
    assert state["run_after_measurement_change"] == "m-v2"
    assert _consume_measurement_change_rerun(state, "m-v2") is True
    assert state.get("run_after_measurement_change") is None
    assert _consume_measurement_change_rerun(state, "m-v2") is False


def test_measurement_change_does_not_run_diagnosis_for_stale_or_obsolete_fingerprint():
    state = {}
    _queue_measurement_change_rerun(
        state, previous_fingerprint="m-v1", current_fingerprint="m-v2",
        prior_analysis_exists=True, current_result=None, central_status=AnalysisStatus.STALE.value,
    )
    assert _consume_measurement_change_rerun(state, "m-v3") is False
    assert state.get("run_after_measurement_change") is None
    _queue_measurement_change_rerun(
        state, previous_fingerprint="m-v1", current_fingerprint="m-v2",
        prior_analysis_exists=True, current_result={"status": "CURRENT"},
        central_status=AnalysisStatus.CURRENT.value,
    )
    assert state.get("run_after_measurement_change") is None


def test_streamlit_measurement_submission_automatically_reexecutes_diagnosis(monkeypatch):
    app_testing = pytest.importorskip("streamlit.testing.v1")
    calls = []

    def diagnose_stub(**kwargs):
        calls.append(list(kwargs.get("measurements", [])))
        return {
            "status": "OK", "message": "mock result", "symptom_codes": ["NO_POWER"],
            "component_candidates": [{
                "component_id": "R1", "board_component_id": "BC-R1",
                "component_type": "resistor", "fault_id": "RESISTOR_FAILURE",
                "candidate_rank": 1, "supporting_evidence": ["입력 증상"],
                "opposing_evidence": [], "recommended_tests": ["isolated_resistance"],
            }],
            "hypotheses": [], "next_tests": [],
        }

    class OllamaStub:
        def __init__(self, *, timeout_seconds):
            pass

        def list_models(self):
            return []

    monkeypatch.setattr(symptoms, "diagnose", diagnose_stub)
    monkeypatch.setattr(symptoms, "OllamaClient", OllamaStub)
    app_path = Path(__file__).parent / "e2e" / "f5_measurement_reanalysis_app.py"
    app = app_testing.AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
    next(button for button in app.button if button.label == "AI 진단 시작").click().run()
    assert not app.exception
    assert len(calls) == 1

    owner = "f5-reanalysis-apptest-f5-reanalysis-board-BC-R1-RESISTOR_FAILURE"
    submit_key = f"FormSubmitter:cc_v5_add_measurement_{owner}-측정 기록 저장"
    assert app.button(key=submit_key).proto.disabled is True
    app.selectbox(key=f"cc_v5_power_state_{owner}").select("POWER_OFF_CONFIRMED").run()
    app.selectbox(key=f"cc_v5_stored_energy_{owner}").select("SAFE_CONFIRMED").run()
    app.selectbox(key=f"cc_v5_isolation_{owner}").select("ISOLATED_CONFIRMED").run()
    app.checkbox(key=f"cc_v5_location_verified_{owner}").check().run()
    app.checkbox(key=f"cc_v5_reference_verified_{owner}").check().run()
    assert app.button(key=submit_key).proto.disabled is False

    app.text_input(key=f"cc_v5_value_{owner}").set_value("10")
    next(item for item in app.text_input if item.label == "측정 대상/단자").set_value("R1 양단")
    app.button(key=submit_key).click().run()

    assert not app.exception
    assert len(calls) == 2
    assert calls[0] == []
    assert len(calls[1]) == 1
    assert calls[1][0]["component_id"] == "R1"
    app.run()
    assert not app.exception
    assert len(calls) == 2


def test_real_streamlit_app_test_renders_workbench_without_runtime_errors():
    app_testing = pytest.importorskip("streamlit.testing.v1")
    from circuitcare.frontend.state import INTAKE_IMAGES_KEY, INTAKE_PROJECT_KEY
    from circuitcare.frontend.workbench_state import SCREEN_KEY

    app = app_testing.AppTest.from_file(str(Path(__file__).parents[1] / "src/circuitcare/ui.py"),
                                        default_timeout=30)
    project = ProjectContext("apptest-project", "apptest-board")
    app.session_state[INTAKE_PROJECT_KEY] = project
    app.session_state[INTAKE_IMAGES_KEY] = []
    app.session_state[SCREEN_KEY] = "workbench"
    app.session_state[f"cc_workbench_panel_{project.project_id}"] = "원인"
    app.run()

    assert not app.exception
    assert len(app.radio) == 0  # first workflow stage is component review
    assert not any(item.label == "작업 패널" for item in app.radio)
    assert not any(item.label == "후보 표시 전략" for item in app.selectbox)
    assert "진단 미실행" in " ".join(item.value for item in app.caption)


def test_apptest_review_next_button_transitions_only_right_panel_to_questions(monkeypatch):
    app_testing = pytest.importorskip("streamlit.testing.v1")
    from circuitcare.frontend.state import INTAKE_IMAGES_KEY, INTAKE_PROJECT_KEY
    from circuitcare.frontend.workbench_state import SCREEN_KEY, WORKFLOW_STATES_KEY

    project = ProjectContext("workflow-apptest", "workflow-board")
    component = SimpleNamespace(component_id="R1", board_component_id="BC-R1",
                                component_type="resistor", presence_status="UNCONFIRMED")
    registry = {"project_id": project.project_id, "board_id": project.board_id,
                "version": "registry-test", "components": [{
                    "board_component_id": "BC-R1", "component_type": "resistor",
                    "presence_status": "UNCONFIRMED", "source_components": [
                        {"photo_id": "photo-1", "component_id": "R1", "review_status": "UNCONFIRMED"},
                    ],
                }]}
    monkeypatch.setattr(workspace, "_prepare_board_registry_snapshot",
                        lambda *_args, **_kwargs: (registry, {"components": (component,)}))
    monkeypatch.setattr(workspace, "diagnosis_components", lambda _registry: (component,))
    monkeypatch.setattr(workspace, "_render_pcb", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workspace, "_render_component_inspector", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workspace, "_diagnosis_status_label", lambda *_args, **_kwargs: "진단 전")
    monkeypatch.setattr(workspace, "current_analysis_snapshot", lambda *_args, **_kwargs: {
        "result": None, "record": {"status": "NOT_RUN"}, "input_fingerprint": "mock-fp",
    })
    monkeypatch.setattr(workspace, "render_symptom_diagnosis_panel",
                        lambda st, *_args, **_kwargs: st.markdown("질문 단계 AppTest 확인"))
    app = app_testing.AppTest.from_file(str(Path(__file__).parents[1] / "src/circuitcare/ui.py"),
                                        default_timeout=30)
    app.session_state[INTAKE_PROJECT_KEY] = project
    app.session_state[INTAKE_IMAGES_KEY] = []
    app.session_state[SCREEN_KEY] = "workbench"
    app.run()
    assert not app.exception
    assert any(button.label == "검수 완료 · 고장 질문으로 진행" for button in app.button)

    next(button for button in app.button if button.label == "검수 완료 · 고장 질문으로 진행").click().run()

    assert not app.exception
    assert any("질문 단계 AppTest 확인" in item.value for item in app.markdown)
    workflow_store = app.session_state[WORKFLOW_STATES_KEY]
    assert workflow_store[f"{project.project_id}|{project.board_id}"]["stage"] == WORKFLOW_QUESTIONS


def test_apptest_questions_next_uses_current_result_and_real_registry_targets(monkeypatch):
    app_testing = pytest.importorskip("streamlit.testing.v1")
    from circuitcare.frontend.state import INTAKE_IMAGES_KEY, INTAKE_PROJECT_KEY
    from circuitcare.frontend.workbench_state import (
        SCREEN_KEY, WORKFLOW_STATES_KEY, transition_workflow,
    )

    project = ProjectContext("workflow-inspection-apptest", "workflow-inspection-board")
    registry = build_board_component_registry(
        {}, project_id=project.project_id, board_id=project.board_id,
        photo_snapshots=_snapshots(),
    )
    components = workspace.diagnosis_components(registry)
    snapshot = {"components": components}
    monkeypatch.setattr(workspace, "_prepare_board_registry_snapshot",
                        lambda *_args, **_kwargs: (registry, snapshot))
    monkeypatch.setattr(workspace, "diagnosis_components", lambda _registry: components)
    monkeypatch.setattr(workspace, "_render_pcb", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workspace, "_render_component_inspector", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workspace, "_diagnosis_status_label", lambda *_args, **_kwargs: "현재 분석")

    def current_snapshot(*_args, **_kwargs):
        return {
            "result": {"candidate_fusion": {"candidates": [{
                "board_component_id": registry["components"][2]["board_component_id"],
                "component_type": "resistor",
                "fault_id": "RESISTOR_FAILURE",
            }]}},
            "record": {"status": AnalysisStatus.CURRENT.value},
            "input_fingerprint": "current-workflow-fingerprint",
        }

    monkeypatch.setattr(workspace, "current_analysis_snapshot", current_snapshot)
    monkeypatch.setattr(workspace, "render_symptom_diagnosis_panel",
                        lambda *_args, **_kwargs: None)
    def render_mock_measurement(st, *_args, **_kwargs):
        if st.button("Mock: 새로운 측정 저장", key="workflow_mock_measurement_save"):
            st.session_state["workflow_mock_new_measurement"] = True
            st.rerun()

    monkeypatch.setattr(workspace, "render_measurement_panel", render_mock_measurement)
    monkeypatch.setattr(workspace, "load_knowledge", lambda: {"component_test_profiles": {"profiles": {
        "resistor": {"tests": {"isolated_resistance": {
            "measurement_type": "resistance", "title": "저항 측정",
            "fault_relations": {"RESISTOR_FAILURE": {"TEST_NORMAL": "refutes", "TEST_ABNORMAL": "supports"}},
        }}}
    }}})

    class MeasurementStub:
        def __init__(self, measurement_id, value):
            self.measurement_id = measurement_id
            self.board_component_id = registry["components"][2]["board_component_id"]
            self.component_id = "R1"
            self.fault_hypothesis = "RESISTOR_FAILURE"
            self.verification_status = "UNVERIFIED"
            self.test_id = "isolated_resistance"
            self.value = value
            self.unit = "Ω"

        def to_dict(self):
            return {"measurement_id": self.measurement_id,
                    "board_component_id": self.board_component_id,
                    "component_id": self.component_id,
                    "fault_hypothesis": self.fault_hypothesis,
                    "verification_status": self.verification_status,
                    "test_id": self.test_id, "value": self.value, "unit": self.unit}

    measurement = MeasurementStub("measurement-existing", 10)
    new_measurement = MeasurementStub("measurement-new", 22)

    def measurements_from_state(state, **_kwargs):
        return [measurement, *([new_measurement] if state.get("workflow_mock_new_measurement") else [])]

    monkeypatch.setattr(workspace, "get_measurements", measurements_from_state)

    app = app_testing.AppTest.from_file(str(Path(__file__).parents[1] / "src/circuitcare/ui.py"),
                                        default_timeout=30)
    app.session_state[INTAKE_PROJECT_KEY] = project
    app.session_state[INTAKE_IMAGES_KEY] = []
    app.session_state[SCREEN_KEY] = "workbench"
    workflow_store: dict = {}
    transition_workflow(workflow_store, project, WORKFLOW_QUESTIONS, event="review_complete")
    app.session_state[WORKFLOW_STATES_KEY] = workflow_store[WORKFLOW_STATES_KEY]
    app.run()
    assert not app.exception
    assert any(button.label == "검사 후보 선정" for button in app.button)

    next(button for button in app.button if button.label == "검사 후보 선정").click().run()

    assert not app.exception
    workflow_state = app.session_state[WORKFLOW_STATES_KEY][f"{project.project_id}|{project.board_id}"]
    assert workflow_state["stage"] == WORKFLOW_INSPECTION
    valid_ids = {row["board_component_id"] for row in registry["components"]
                 if row.get("presence_status") != "EXCLUDED"}
    assert workflow_state["inspection_targets"]
    assert {row["board_component_id"] for row in workflow_state["inspection_targets"]} <= valid_ids
    assert any(button.label == "이 소자 측정" for button in app.button)
    next(button for button in app.button if button.label == "이 소자 측정").click().run()
    assert not app.exception
    analysis_button = next(button for button in app.button if button.label == "측정 결과 분석")
    assert analysis_button.disabled  # a previous-cycle measurement cannot trigger reassessment
    next(button for button in app.button if button.label == "Mock: 새로운 측정 저장").click().run()
    assert not app.exception
    assert app.session_state["workflow_mock_new_measurement"] is True
    assert not next(button for button in app.button if button.label == "측정 결과 분석").disabled

    next(button for button in app.button if button.label == "측정 결과 분석").click().run()

    assert not app.exception
    workflow_state = app.session_state[WORKFLOW_STATES_KEY][f"{project.project_id}|{project.board_id}"]
    assert workflow_state["stage"] == "REASSESSMENT"
    assert workflow_state["reassessment_history"][-1]["measurement_ids"] == [
        "measurement-existing", "measurement-new",
    ]


def test_apptest_empty_inspection_queue_returns_to_review_without_losing_history(monkeypatch):
    app_testing = pytest.importorskip("streamlit.testing.v1")
    from circuitcare.frontend.state import INTAKE_IMAGES_KEY, INTAKE_PROJECT_KEY
    from circuitcare.frontend.workbench_state import SCREEN_KEY, WORKFLOW_STATES_KEY, transition_workflow

    project = ProjectContext("workflow-no-target-apptest", "workflow-no-target-board")
    registry = {"project_id": project.project_id, "board_id": project.board_id,
                "version": "empty-active-registry", "components": []}
    monkeypatch.setattr(workspace, "_prepare_board_registry_snapshot",
                        lambda *_args, **_kwargs: (registry, {"components": ()}))
    monkeypatch.setattr(workspace, "diagnosis_components", lambda _registry: ())
    monkeypatch.setattr(workspace, "_render_pcb", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workspace, "_render_component_inspector", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workspace, "_diagnosis_status_label", lambda *_args, **_kwargs: "판정 보류")
    monkeypatch.setattr(workspace, "current_analysis_snapshot", lambda *_args, **_kwargs: {
        "result": None, "record": {"status": "NOT_RUN"}, "input_fingerprint": "empty-fp",
    })
    monkeypatch.setattr(workspace, "render_symptom_diagnosis_panel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workspace, "get_measurements", lambda *_args, **_kwargs: [])

    app = app_testing.AppTest.from_file(str(Path(__file__).parents[1] / "src/circuitcare/ui.py"),
                                        default_timeout=30)
    app.session_state[INTAKE_PROJECT_KEY] = project
    app.session_state[INTAKE_IMAGES_KEY] = []
    app.session_state[SCREEN_KEY] = "workbench"
    saved: dict = {}
    transition_workflow(saved, project, WORKFLOW_QUESTIONS, event="review_complete")
    transition_workflow(saved, project, WORKFLOW_INSPECTION, event="questions_complete")
    owner = f"{project.project_id}|{project.board_id}"
    saved[WORKFLOW_STATES_KEY][owner]["question_answers"] = [{"test_id": "ASK_POWER_SOURCE", "answer": "예"}]
    saved[WORKFLOW_STATES_KEY][owner]["measurement_ids"] = ["archived-measurement"]
    app.session_state[WORKFLOW_STATES_KEY] = saved[WORKFLOW_STATES_KEY]
    app.run()
    assert not app.exception
    assert any(button.label == "소자 검수로 돌아가기" for button in app.button)

    next(button for button in app.button if button.label == "소자 검수로 돌아가기").click().run()

    assert not app.exception
    workflow = app.session_state[WORKFLOW_STATES_KEY][owner]
    assert workflow["stage"] == WORKFLOW_REVIEW
    assert workflow["question_answers"] == [{"test_id": "ASK_POWER_SOURCE", "answer": "예"}]
    assert workflow["measurement_ids"] == ["archived-measurement"]
    assert workflow["transition_reason"] == "현재 Registry에서 측정·검사 대상으로 해석 가능한 소자가 없습니다."


def test_apptest_current_reassessment_can_finish_to_report_stage(monkeypatch):
    app_testing = pytest.importorskip("streamlit.testing.v1")
    from circuitcare.frontend.state import INTAKE_IMAGES_KEY, INTAKE_PROJECT_KEY
    from circuitcare.frontend.workbench_state import (
        SCREEN_KEY, WORKFLOW_STATES_KEY, WORKFLOW_QUESTIONS, WORKFLOW_INSPECTION,
        WORKFLOW_REASSESSMENT, transition_workflow,
    )

    project = ProjectContext("workflow-finish-apptest", "workflow-finish-board")
    registry = {"project_id": project.project_id, "board_id": project.board_id,
                "version": "finish-registry", "components": []}
    monkeypatch.setattr(workspace, "_prepare_board_registry_snapshot",
                        lambda *_args, **_kwargs: (registry, {"components": ()}))
    monkeypatch.setattr(workspace, "diagnosis_components", lambda _registry: ())
    monkeypatch.setattr(workspace, "_render_pcb", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workspace, "_render_component_inspector", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workspace, "_diagnosis_status_label", lambda *_args, **_kwargs: "현재 결과")
    monkeypatch.setattr(workspace, "render_symptom_diagnosis_panel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workspace, "get_measurements", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(workspace, "current_analysis_snapshot", lambda *_args, **_kwargs: {
        "result": {"hypotheses": [], "component_candidates": []},
        "record": {"status": AnalysisStatus.CURRENT.value},
        "input_fingerprint": "finish-current-fingerprint",
    })
    monkeypatch.setattr(workspace, "_render_workflow_report", lambda *_args, **_kwargs: None)

    state: dict = {}
    transition_workflow(state, project, WORKFLOW_QUESTIONS, event="review_complete")
    transition_workflow(state, project, WORKFLOW_INSPECTION, event="questions_complete")
    transition_workflow(state, project, WORKFLOW_REASSESSMENT, event="measurement_saved")
    app = app_testing.AppTest.from_file(str(Path(__file__).parents[1] / "src/circuitcare/ui.py"),
                                        default_timeout=30)
    app.session_state[INTAKE_PROJECT_KEY] = project
    app.session_state[INTAKE_IMAGES_KEY] = []
    app.session_state[SCREEN_KEY] = "workbench"
    app.session_state[WORKFLOW_STATES_KEY] = state[WORKFLOW_STATES_KEY]
    app.run()
    assert not app.exception
    finish = next(button for button in app.button if button.label == "검사 종료 및 결과 보고서")
    assert not finish.disabled
    finish.click().run()
    assert not app.exception
    owner = f"{project.project_id}|{project.board_id}"
    assert app.session_state[WORKFLOW_STATES_KEY][owner]["stage"] == "COMPLETE"
