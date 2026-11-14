from circuitcare.frontend.models import ImageRecord, ProjectContext
from circuitcare.symptom_diagnosis import load_knowledge
from circuitcare.symptom_diagnosis.llm_assist import (
    select_follow_up_question, suggest_pcb_fault_candidates,
)
from circuitcare.symptom_diagnosis.candidate_fusion import fuse_inspection_candidates
from circuitcare.symptom_diagnosis.pcb_context import build_pcb_context
from circuitcare.dashboard.views.symptom_diagnosis_panel import (
    INITIAL_SYMPTOM_OPTIONS, INITIAL_SYMPTOM_QUESTION_ID, _combined_observations,
    _registry_valid_suspects, _safe_initial_followup, _save_follow_up_answer,
    _save_initial_symptom_answer, candidate_rows_for_strategy, _apply_qwen_pcb_assist,
    _apply_pending_initial_symptom_widget, candidate_focus_widget_key,
)


class FakeClient:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def chat_json(self, model, messages, *, output_schema):
        self.calls.append((model, messages, output_schema))
        return {"content": self.content, "model": model}


def test_context_preserves_ids_photo_frames_and_unverified_suspicions():
    rows = [
        {"component_id": "board-c1", "board_component_id": "board-c1", "photo_id": "front",
         "photo_local_component_id": "local-1", "coordinate_frame": "source_image_px",
         "bbox_xyxy": [10, 20, 30, 40], "component_type": "capacitor", "ocr_marking": "104",
         "presence_status": "CONFIRMED"},
        {"component_id": "board-c2", "photo_local_component_id": "local-1", "bbox_xyxy": [1, 2, 3, 4],
         "component_type": "resistor", "presence_status": "CONFIRMED"},
    ]
    context = build_pcb_context(
        components=rows, photo_records=[ImageRecord("front", image_dimensions=(100, 200)),
                                       ImageRecord("back", image_dimensions=(100, 200))],
        diagnosis={"functional_blocks": [], "component_candidates": [], "symptom_codes": ["NO_POWER"]},
        suspected_components=[{"board_component_id": "board-c1", "suspected_fault_mode": "CAPACITOR_SHORT",
                              "evidence_status": "UNVERIFIED"}],
        board_registry={"components": [{"board_component_id": "board-c1", "source_components": [{
            "component_id": "local-1", "photo_id": "front", "detection_id": "det-1",
            "coordinate_frame": "SOURCE_PHOTO_PIXEL", "bbox_xyxy": [10, 20, 30, 40],
            "component_type": "capacitor", "review_status": "CONFIRMED", "ocr_marking": "104",
            "source_observation_fingerprint": "obs-sha256",
        }]}]},
    )
    first, second = context["focus_components"]
    assert first["board_component_id"] == "board-c1"
    assert first["photo_local_component_id"] == "local-1"
    assert first["normalized_geometry"]["center_x"] == 0.2
    assert first["ocr_marking"] == "104"
    assert first["connectivity_status"] == "UNKNOWN"
    assert first["user_suspicion"]["evidence_status"] == "UNVERIFIED"
    assert first["source_observations"][0]["photo_local_component_id"] == "local-1"
    assert first["source_observations"][0]["normalized_geometry"]["center_x"] == 0.2
    assert first["source_observations"][0]["source_observation_fingerprint"] == "obs-sha256"
    assert second["photo_id"] is None and second["normalized_geometry"] is None
    assert context["symptom_codes"] == ["NO_POWER"]


def test_large_context_limits_focus_and_records_omissions():
    rows = [{"component_id": f"c{i}", "component_type": "resistor", "bbox_xyxy": [i, 1, i+1, 2],
             "presence_status": "CONFIRMED"} for i in range(180)]
    context = build_pcb_context(components=rows, photo_records=[], diagnosis={}, token_budget=600,
                                max_focus_components=10)
    assert len(context["focus_components"]) <= 10
    assert context["omitted_component_count"] >= 170
    assert context["summary"]["component_count"] == 180
    assert context["context_token_estimate"] <= context["token_budget"]


def test_qwen_candidate_schema_and_result_are_allowlisted_and_unverified():
    knowledge = load_knowledge()
    fault_id = next(key for key, row in knowledge["fault_modes"]["fault_modes"].items()
                    if "capacitor" in row.get("candidate_component_types", []))
    response = {"proposals": [{"component_id": "c1", "suspected_fault_id": fault_id,
        "functional_block": "UNKNOWN", "reason": "검토 후보", "supporting_evidence_ids": ["NO_POWER"],
        "opposing_evidence_ids": [], "missing_information": [], "recommended_next_action": "전원 차단 후 육안 확인",
        "uncertainty_status": "UNVERIFIED"}]}
    client = FakeClient(response)
    context = {"focus_components": [{"component_id": "c1", "type": "capacitor", "review_status": "CONFIRMED",
                                     "functional_roles": []}], "observations": {"NO_POWER": True}}
    result = suggest_pcb_fault_candidates(client, "qwen", pcb_context=context, diagnosis={}, knowledge=knowledge)
    assert result["proposals"][0]["component_id"] == "c1"
    assert result["proposals"][0]["evidence_status"] == "UNVERIFIED"
    assert result["proposals"][0]["rank_status"] == "NOT_MERGED"
    item_schema = client.calls[0][2]["properties"]["proposals"]["items"]["properties"]
    assert item_schema["component_id"]["enum"] == ["c1"]
    assert "FAULT_SUPPORTED" not in item_schema["uncertainty_status"]["enum"]


def test_qwen_block_proposals_and_evidence_ledger_ids_are_allowlisted():
    knowledge = load_knowledge()
    fault_id = next(key for key, row in knowledge["fault_modes"]["fault_modes"].items()
                    if "capacitor" in row.get("candidate_component_types", []))
    response = {"proposals": [{"component_id": "bc-c1", "suspected_fault_id": fault_id,
        "functional_block": "power", "functional_block_id": "block-abc123", "reason": "가설",
        "supporting_evidence_ids": ["ev-verified-1"], "opposing_evidence_ids": [],
        "missing_information": ["측정 기준"], "recommended_next_action": "기준 확인",
        "uncertainty_status": "UNVERIFIED"}]}
    client = FakeClient(response)
    context = {"focus_components": [{"component_id": "bc-c1", "type": "capacitor",
        "review_status": "USER_CONFIRMED", "component_type_status": "USER_CONFIRMED"}],
        "observations": {}, "functional_block_groups": [{"block_id": "block-abc123",
        "role": "power", "component_ids": ["bc-c1"], "connectivity_status": "NOT_INFERRED"}],
        "evidence_ledger": [{"evidence_id": "ev-verified-1", "status": "VERIFIED"}]}
    result = suggest_pcb_fault_candidates(client, "qwen", pcb_context=context,
                                          diagnosis={}, knowledge=knowledge)
    assert result["proposals"][0]["functional_block_id"] == "block-abc123"
    assert result["proposals"][0]["supporting_evidence_ids"] == ["ev-verified-1"]
    schema = client.calls[0][2]["properties"]["proposals"]["items"]["properties"]
    assert schema["functional_block_id"]["enum"] == ["block-abc123", "__none__"]


def test_validated_qwen_role_proposal_enters_structure_as_ai_suggested_only(monkeypatch):
    import circuitcare.dashboard.views.symptom_diagnosis_panel as panel
    proposal = {"component_id": "bc-c1", "board_component_id": "bc-c1",
        "suspected_fault_id": "CAPACITOR_SHORT", "functional_block": "power",
        "functional_block_id": None, "reason": "전원 입력 인접 정보",
        "supporting_evidence_ids": [], "opposing_evidence_ids": [],
        "missing_information": [], "recommended_next_action": "전원 기준 확인",
        "uncertainty_status": "UNVERIFIED", "evidence_status": "UNVERIFIED"}
    monkeypatch.setattr(panel, "build_pcb_context", lambda **_kwargs: {
        "summary": {}, "focus_components": [{"component_id": "bc-c1"}],
        "omitted_component_ids": [], "context_token_estimate": 20,
    })
    monkeypatch.setattr(panel, "suggest_pcb_fault_candidates", lambda *_args, **_kwargs: {
        "proposals": [proposal], "status": "LOCAL_LLM_PROPOSED_UNVERIFIED", "model": "qwen-test"})
    monkeypatch.setattr(panel, "OllamaClient", lambda **_kwargs: object())
    component = {"component_id": "bc-c1", "board_component_id": "bc-c1",
                 "component_type": "capacitor", "component_type_status": "USER_CONFIRMED"}
    result = {"functional_blocks": [{"component_id": "bc-c1", "candidate_blocks": [],
                "role_evidence": {}}], "functional_block_groups": [],
              "all_component_candidates": [], "component_candidates": [],
              "hypotheses": [], "next_tests": [], "evidence_ledger": []}
    panel._apply_qwen_pcb_assist(result, project=ProjectContext("p", "b"),
        components=[component], suspected_components=[], selected_model="qwen-test",
        knowledge=load_knowledge(), answered_test_ids=[])
    group = result["functional_block_groups"][0]
    assert group["component_ids"] == ["bc-c1"]
    assert group["role"] == "power" and group["verification_status"] == "AI_SUGGESTED"
    assert group["connectivity_status"] == "NOT_INFERRED"
    assert result["functional_blocks"][0]["ai_suggested_functional_roles"] == ["power"]
    assert any(row["kind"] == "component_functional_role" and row["status"] == "AI_SUGGESTED"
               for row in result["evidence_ledger"])
    assert any(row["kind"] == "functional_block_membership" and row["status"] == "AI_SUGGESTED"
               for row in result["evidence_ledger"])
    assert result["active_inspection_plan"]["status"] == "NO_HYPOTHESES"

    unknown_result = {"functional_blocks": [], "functional_block_groups": [],
                      "all_component_candidates": [], "component_candidates": [],
                      "hypotheses": [], "next_tests": [], "evidence_ledger": []}
    unknown = {"component_id": "bc-unknown", "board_component_id": "bc-unknown",
               "component_type": "unknown", "component_type_status": "USER_CONFIRMED"}
    monkeypatch.setattr(panel, "suggest_pcb_fault_candidates", lambda *_args, **_kwargs: {
        "proposals": [{**proposal, "component_id": "bc-unknown", "board_component_id": "bc-unknown"}],
        "status": "LOCAL_LLM_PROPOSED_UNVERIFIED", "model": "qwen-test"})
    panel._apply_qwen_pcb_assist(unknown_result, project=ProjectContext("p", "b"),
        components=[unknown], suspected_components=[], selected_model="qwen-test",
        knowledge=load_knowledge(), answered_test_ids=[])
    assert unknown_result["functional_block_groups"] == []
    assert unknown_result["candidate_fusion"]["top_candidates"] == []


def test_qwen_rejects_wrong_fault_component_pair_and_unconfirmed_component():
    knowledge = load_knowledge()
    resistor_fault = next(key for key, row in knowledge["fault_modes"]["fault_modes"].items()
                          if "resistor" in row.get("candidate_component_types", []))
    response = {"proposals": [{"component_id": "c1", "suspected_fault_id": resistor_fault,
        "functional_block": "UNKNOWN", "reason": "", "supporting_evidence_ids": [],
        "opposing_evidence_ids": [], "missing_information": [], "recommended_next_action": "",
        "uncertainty_status": "UNVERIFIED"}]}
    context = {"focus_components": [{"component_id": "c1", "type": "capacitor",
                                     "review_status": "CONFIRMED", "functional_roles": []}],
               "observations": {}}
    rejected = suggest_pcb_fault_candidates(FakeClient(response), "qwen", pcb_context=context,
                                            diagnosis={}, knowledge=knowledge)
    assert rejected["proposals"] == []

    # A correctly typed but unreviewed detector row is an inspection task, not evidence.
    context["focus_components"] = [{"component_id": "c1", "type": "resistor",
                                    "review_status": "UNCONFIRMED", "functional_roles": []}]
    accepted = suggest_pcb_fault_candidates(FakeClient(response), "qwen", pcb_context=context,
                                            diagnosis={}, knowledge=knowledge)
    assert len(accepted["proposals"]) == 1
    assert accepted["proposals"][0]["verification_status"] == "NEEDS_COMPONENT_REVIEW"


def test_qwen_can_propose_unreviewed_detection_as_review_task_only():
    knowledge = load_knowledge()
    capacitor_fault = next(key for key, row in knowledge["fault_modes"]["fault_modes"].items()
                           if "capacitor" in row.get("candidate_component_types", []))
    response = {"proposals": [{"component_id": "board-c1", "suspected_fault_id": capacitor_fault,
        "functional_block": "UNKNOWN", "reason": "육안 재검수 필요", "supporting_evidence_ids": [],
        "opposing_evidence_ids": [], "missing_information": ["소자 종류 확인"],
        "recommended_next_action": "박스 종류 확인", "uncertainty_status": "UNVERIFIED"}]}
    context = {"focus_components": [{"component_id": "board-c1", "type": "unknown",
        "candidate_component_types": ["capacitor"], "review_status": "DETECTED_UNVERIFIED",
        "component_type_status": "UNCONFIRMED"}], "observations": {}}
    result = suggest_pcb_fault_candidates(FakeClient(response), "qwen", pcb_context=context,
                                          diagnosis={}, knowledge=knowledge)
    assert len(result["proposals"]) == 1
    assert result["proposals"][0]["verification_status"] == "NEEDS_COMPONENT_REVIEW"
    assert result["proposals"][0]["evidence_status"] == "UNVERIFIED"


def test_candidate_fusion_merges_pair_without_upgrading_qwen_verdict_or_score():
    rule = {"component_id": "bc-1", "fault_hypothesis": "CAP_OPEN", "score": 1.5,
            "candidate_rank": 4, "visual_status": "SUSPECTED", "measurement_verdict": "non_diagnostic",
            "evidence_ids": ["OBS_A"], "fault_title": "커패시터 개방"}
    proposal = {"component_id": "bc-1", "suspected_fault_id": "CAP_OPEN",
                "component_type": "capacitor", "reason": "검토 이유", "supporting_evidence_ids": ["OBS_A"],
                "opposing_evidence_ids": [], "recommended_next_action": "소자 확인",
                "component_review_status": "USER_CONFIRMED"}
    qwen_only = {**proposal, "component_id": "bc-2", "suspected_fault_id": "CAP_SHORT",
                 "component_review_status": "DETECTED_UNVERIFIED"}
    result = fuse_inspection_candidates(
        rule_candidates=[rule], qwen_proposals=[proposal, qwen_only],
        components=[{"component_id": "bc-1", "component_type": "capacitor", "presence_status": "CONFIRMED"},
                    {"component_id": "bc-2", "component_type": "unknown", "presence_status": "UNCONFIRMED"}],
        suspected_components=[{"board_component_id": "bc-1", "suspected_fault_mode": "CAP_OPEN",
                               "evidence_status": "UNVERIFIED"}],
        allowed_fault_ids={"CAP_OPEN", "CAP_SHORT"},
    )
    assert len([row for row in result["candidates"] if row["component_id"] == "bc-1"]) == 1
    merged = result["candidates"][0]
    assert set(merged["candidate_source"]) == {"RULE_BASELINE", "QWEN_UNVERIFIED", "USER_SUSPICION"}
    assert merged["score"] == 1.5
    assert merged["visual_status"] == "SUSPECTED"
    unreviewed = next(row for row in result["candidates"] if row["component_id"] == "bc-2")
    assert unreviewed["verification_status"] == "NEEDS_COMPONENT_REVIEW"
    assert unreviewed["score"] is None


def test_question_answers_accumulate_unknown_is_not_evidence_and_cancel_is_local():
    state = {"manual_observations": {"VISIBLE_SOLDER_DAMAGE": True}, "answers": {}}
    q_a = {"test_id": "A", "type": "question", "question": "A?", "results": {
        "yes": {"observations": {"POWER_SOURCE_VERIFIED": True}},
        "no": {"observations": {"POWER_SOURCE_VERIFIED": False}},
    }}
    q_b = {"test_id": "B", "type": "question", "question": "B?", "results": {
        "yes": {"observations": {"BUZZER_VERIFIED_FAILED": True}},
        "no": {"observations": {"BUZZER_VERIFIED_FAILED": False}},
    }}
    _save_follow_up_answer(state, q_a, "예")
    _save_follow_up_answer(state, q_b, "아니오")
    rows = _combined_observations(state)
    assert {row["code"] for row in rows} >= {"POWER_SOURCE_VERIFIED", "BUZZER_VERIFIED_FAILED",
                                            "VISIBLE_SOLDER_DAMAGE"}
    _save_follow_up_answer(state, q_a, "모르겠음")
    rows = _combined_observations(state)
    assert not any(row.get("test_id") == "A" for row in rows)
    assert any(row.get("test_id") == "B" and row["value"] is False for row in rows)
    _save_follow_up_answer(state, q_a, "예")
    _save_follow_up_answer(state, q_a, "답변 취소")
    rows = _combined_observations(state)
    assert not any(row.get("test_id") == "A" for row in rows)
    assert any(row.get("test_id") == "B" for row in rows)
    assert any(row.get("source") == "user_report" and row["code"] == "VISIBLE_SOLDER_DAMAGE" for row in rows)


def test_spatial_relations_are_photo_local_and_never_claim_connectivity():
    rows = [
        {"component_id": "front-a", "photo_id": "front", "coordinate_frame": "source_image_px",
         "bbox_xyxy": [0, 0, 10, 10], "component_type": "resistor", "presence_status": "CONFIRMED"},
        {"component_id": "front-b", "photo_id": "front", "coordinate_frame": "source_image_px",
         "bbox_xyxy": [20, 0, 30, 10], "component_type": "capacitor", "presence_status": "CONFIRMED"},
        {"component_id": "back-c", "photo_id": "back", "coordinate_frame": "source_image_px",
         "bbox_xyxy": [0, 0, 10, 10], "component_type": "diode", "presence_status": "CONFIRMED"},
    ]
    context = build_pcb_context(components=rows,
        photo_records=[ImageRecord("front", image_dimensions=(100, 100)),
                       ImageRecord("back", image_dimensions=(100, 100))], diagnosis={})
    relations = context["focus_components"][0]["nearby_components"]
    assert [row["component_id"] for row in relations] == ["front-b"]
    assert relations[0]["relation_status"] == "SPATIAL_ONLY"
    assert context["connectivity_status"] == "UNKNOWN"


def test_registry_replacement_or_reobservation_invalidates_user_suspect_for_analysis():
    active = {"component_id": "R1", "board_component_id": "bc-1", "photo_id": "front",
              "photo_sha256": "old-photo", "selection_version": "old-observation",
              "evidence_status": "UNVERIFIED"}
    registry = {"photos": [{"photo_id": "front", "image_fingerprint": "new-photo"}],
                "components": [{"board_component_id": "bc-1", "presence_status": "CONFIRMED",
                                "source_components": [{"photo_id": "front", "component_id": "R1",
                                    "source_observation_fingerprint": "new-observation",
                                    "review_status": "USER_CONFIRMED"}]}]}
    current, stale = _registry_valid_suspects([active], registry)
    assert current == [] and stale == [active]


def test_follow_up_chooses_one_allowed_question_and_skips_answered():
    q1 = {"type": "question", "test_id": "ASK_POWER_SOURCE", "question": "전원 입력이 정상인가요?"}
    q2 = {"type": "question", "test_id": "ASK_OTHER_OUTPUTS", "question": "다른 출력도 동작하지 않나요?"}
    client = FakeClient({"test_id": q2["test_id"], "reason": "증상 범위 구분"})
    result = select_follow_up_question(client, "qwen", pcb_context={}, diagnosis={"next_tests": [q1, q2]},
                                       answered_test_ids=[])
    assert result["test_id"] == q2["test_id"]
    assert result["question_source"] == "QWEN_SELECTED_EXISTING_SAFE_QUESTION"
    assert select_follow_up_question(client, "qwen", pcb_context={}, diagnosis={"next_tests": [q1]},
                                     answered_test_ids=[q1["test_id"]]) is None


class MalformedClient:
    def __init__(self, error):
        self.error = error

    def chat_json(self, *args, **kwargs):
        raise self.error


def _candidate_response(component_id, fault_id):
    return {"component_id": component_id, "suspected_fault_id": fault_id,
            "functional_block": "UNKNOWN", "reason": "검토", "supporting_evidence_ids": [],
            "opposing_evidence_ids": [], "missing_information": [],
            "recommended_next_action": "소자 확인", "uncertainty_status": "UNVERIFIED"}


def test_qwen_invalid_ids_fault_types_and_null_shapes_are_rejected_without_crashing():
    knowledge = load_knowledge()
    capacitor_fault = next(key for key, row in knowledge["fault_modes"]["fault_modes"].items()
                           if "capacitor" in row.get("candidate_component_types", []))
    valid_context = {"focus_components": [{"component_id": "c1", "type": "capacitor",
                      "review_status": "USER_CONFIRMED"}], "observations": {}}
    for payload in [
        {"proposals": [_candidate_response("missing", capacitor_fault)]},
        {"proposals": [_candidate_response("c1", "NOT_A_FAULT")]},
        {"proposals": [None, "bad-row", _candidate_response("c1", capacitor_fault)]},
        {"proposals": None},
        {"proposals": "not-an-array"},
        None,
        "invalid-json-fallback-value",
    ]:
        result = suggest_pcb_fault_candidates(FakeClient(payload), "qwen", pcb_context=valid_context,
                                              diagnosis={}, knowledge=knowledge)
        if isinstance(payload, dict) and isinstance(payload.get("proposals"), list):
            assert len(result["proposals"]) == (1 if payload["proposals"][-1] == _candidate_response("c1", capacitor_fault) else 0)
        else:
            assert result["status"] == "FAILED"
            assert result["failure_category"] == "INVALID_SCHEMA"


def test_qwen_rejects_excluded_stale_and_registry_invalid_rows_but_accepts_review_task():
    knowledge = load_knowledge()
    capacitor_fault = next(key for key, row in knowledge["fault_modes"]["fault_modes"].items()
                           if "capacitor" in row.get("candidate_component_types", []))
    response = {"proposals": [_candidate_response("excluded", capacitor_fault),
                               _candidate_response("stale", capacitor_fault),
                               _candidate_response("tentative", capacitor_fault)]}
    context = {"focus_components": [
        {"component_id": "excluded", "type": "capacitor", "presence_status": "EXCLUDED"},
        {"component_id": "stale", "type": "capacitor", "identity_status": "STALE_OR_UNRESOLVED",
         "review_status": "USER_CONFIRMED"},
        {"component_id": "tentative", "type": "unknown", "candidate_component_types": ["capacitor"],
         "review_status": "DETECTED_UNVERIFIED"}], "observations": {}}
    result = suggest_pcb_fault_candidates(FakeClient(response), "qwen", pcb_context=context,
                                          diagnosis={}, knowledge=knowledge)
    assert [row["component_id"] for row in result["proposals"]] == ["tentative"]
    assert result["proposals"][0]["verification_status"] == "NEEDS_COMPONENT_REVIEW"
    assert result["rejected_proposal_count"] == 2


def test_candidate_fusion_keeps_measured_support_ahead_and_surfaces_qwen_only_outside_top3():
    rules = [{"component_id": f"r{i}", "fault_hypothesis": f"F{i}", "fault_title": f"F{i}",
              "score": 9-i, "measurement_verdict": "supports" if i == 4 else "non_diagnostic",
              "visual_status": "FAULT_SUPPORTED" if i == 4 else "SUSPECTED", "evidence_ids": []}
             for i in range(5)]
    qwen = [{"component_id": f"q{i}", "suspected_fault_id": f"Q{i}", "reason": "new check",
             "component_review_status": "USER_CONFIRMED"} for i in range(2)]
    components = ([{"component_id": f"r{i}", "presence_status": "CONFIRMED", "component_type": "resistor"}
                   for i in range(5)]
                  + [{"component_id": f"q{i}", "presence_status": "CONFIRMED", "component_type": "resistor"}
                     for i in range(2)])
    result = fuse_inspection_candidates(rule_candidates=rules, qwen_proposals=qwen, components=components,
                                        allowed_fault_ids={f"F{i}" for i in range(5)} | {"Q0", "Q1"})
    assert result["top_candidates"][0]["component_id"] == "r4"
    assert len(result["top_candidates"]) == 3
    assert {row["component_id"] for row in result["additional_qwen_candidates"]} == {"q0", "q1"}
    assert all(row["score"] is None for row in result["additional_qwen_candidates"])


def test_review_status_counts_confirmed_tentative_excluded_and_deduplicates_board_identity():
    rows = []
    for index in range(3):
        rows.append({"component_id": f"local-confirmed-{index}", "board_component_id": f"bc{index}",
                     "photo_id": "front", "photo_local_component_id": f"local-confirmed-{index}",
                     "component_type": "resistor", "presence_status": "CONFIRMED"})
    # Same board component from another photo stays one board-level component.
    rows.append({"component_id": "local-confirmed-0-back", "board_component_id": "bc0",
                 "photo_id": "back", "photo_local_component_id": "local-confirmed-0-back",
                 "component_type": "resistor", "presence_status": "CONFIRMED"})
    rows.extend({"component_id": f"u{i}", "board_component_id": f"bu{i}", "photo_id": "front",
                 "component_type": "unknown", "candidate_component_types": ["capacitor"],
                 "presence_status": "UNCONFIRMED"} for i in range(5))
    rows.extend({"component_id": f"x{i}", "board_component_id": f"bx{i}", "photo_id": "front",
                 "component_type": "resistor", "presence_status": status}
                for i, status in enumerate(("EXCLUDED", "STALE_OR_UNRESOLVED")))
    context = build_pcb_context(components=rows, photo_records=[], diagnosis={})
    summary = context["summary"]
    assert summary["component_count"] == 8
    assert summary["confirmed_component_count"] == 3
    assert summary["tentative_component_count"] == 5
    assert summary["excluded_or_stale_component_count"] == 2
    assert len({row["component_id"] for row in context["focus_components"]}) == 8
    assert "bx0" not in {row["component_id"] for row in context["focus_components"]}
    tentative = next(row for row in context["focus_components"] if row["component_id"] == "bu0")
    assert tentative["review_status"] == "DETECTED_UNVERIFIED"
    assert tentative["type_evidence_status"] == "TENTATIVE"


def test_initial_symptom_answers_use_existing_ontology_and_safe_followup_catalog():
    assert INITIAL_SYMPTOM_OPTIONS["전원 안 켜짐"] == "NO_POWER"
    assert INITIAL_SYMPTOM_OPTIONS["출력 작동 안 함"] == "OUTPUT_FAILURE"
    assert INITIAL_SYMPTOM_OPTIONS["기타·모르겠음"] is None
    state = {"question_answers": {}, "answered_test_ids": []}
    _save_initial_symptom_answer(state, "불안정하게 작동")
    rows = _combined_observations(state)
    assert any(row["code"] == "INTERMITTENT_OPERATION" and row["value"] is True for row in rows)
    assert INITIAL_SYMPTOM_QUESTION_ID in state["answered_test_ids"]
    _save_initial_symptom_answer(state, "기타·모르겠음")
    rows = _combined_observations(state)
    assert not any(row["test_id"] == INITIAL_SYMPTOM_QUESTION_ID for row in rows)
    question = _safe_initial_followup(load_knowledge(), {INITIAL_SYMPTOM_QUESTION_ID})
    assert question and question["type"] == "question"
    assert question["test_id"] in load_knowledge()["test_catalog"]["tests"]


def test_qwen_transport_failures_are_propagated_for_rule_engine_fallback():
    from circuitcare.local_llm import LocalLLMError
    knowledge = load_knowledge()
    fault_id = next(key for key, row in knowledge["fault_modes"]["fault_modes"].items()
                    if "capacitor" in row.get("candidate_component_types", []))
    context = {"focus_components": [{"component_id": "c1", "type": "capacitor",
                                     "review_status": "USER_CONFIRMED"}], "observations": {}}
    for error in (LocalLLMError("invalid JSON content"),
                  LocalLLMError("response truncated"), TimeoutError("timed out")):
        try:
            suggest_pcb_fault_candidates(MalformedClient(error), "qwen", pcb_context=context,
                                         diagnosis={}, knowledge=knowledge)
        except (LocalLLMError, TimeoutError):
            pass
        else:
            raise AssertionError("Qwen failure should be visible to the caller for fallback handling")
    # The deterministic diagnosis path remains independent of these model failures.
    from circuitcare.symptom_diagnosis import diagnose
    assert diagnose(symptoms=["NO_POWER"]).get("status") != "FAILED"



def test_catalog_yaml_boolean_yes_no_keys_produce_distinct_observations():
    knowledge = load_knowledge()
    question = {"test_id": "ASK_POWER_SOURCE", **knowledge["test_catalog"]["tests"]["ASK_POWER_SOURCE"]}
    state = {"question_answers": {}, "answered_test_ids": []}
    _save_follow_up_answer(state, question, "예")
    assert any(row["code"] == "POWER_SOURCE_VERIFIED" and row["value"] is True
               for row in _combined_observations(state))
    _save_follow_up_answer(state, question, "아니오")
    rows = [row for row in _combined_observations(state) if row["code"] == "POWER_SOURCE_VERIFIED"]
    assert len(rows) == 1 and rows[0]["value"] is False



def test_candidate_display_selects_exact_strategy_and_never_mixes_rows():
    baseline = [{"component_id": "base", "fault_id": "B"}]
    evidence = {"component_candidates": [{"component_id": "evidence", "fault_id": "E"}]}
    fusion = {"top_candidates": [{"component_id": "fusion", "fault_id": "F"}]}
    result = {"component_candidates": baseline, "evidence_v1": evidence,
              "candidate_fusion": fusion}
    assert candidate_rows_for_strategy(result, "baseline") == baseline
    assert candidate_rows_for_strategy(result, "evidence_v1") == evidence["component_candidates"]
    assert candidate_rows_for_strategy(result, "candidate_fusion_v1") == fusion["top_candidates"]



def test_local_component_id_collision_across_photos_is_not_merged_without_board_id():
    rows = [
        {"component_id": "det-1", "photo_id": "front", "component_type": "resistor",
         "presence_status": "CONFIRMED"},
        {"component_id": "det-1", "photo_id": "back", "component_type": "capacitor",
         "presence_status": "CONFIRMED"},
    ]
    context = build_pcb_context(components=rows, photo_records=[], diagnosis={})
    assert context["summary"]["component_count"] == 2
    assert {row["photo_id"] for row in context["focus_components"]} == {"front", "back"}
    assert len({row["component_id"] for row in context["focus_components"]}) == 2



def test_qwen_invalid_json_keeps_rule_candidates_and_failure_category(monkeypatch):
    import circuitcare.dashboard.views.symptom_diagnosis_panel as panel
    from circuitcare.local_llm import LocalLLMError

    class BrokenOllama:
        def __init__(self, **kwargs):
            pass
        def chat_json(self, *args, **kwargs):
            raise LocalLLMError("Ollama returned invalid JSON content")

    monkeypatch.setattr(panel, "OllamaClient", BrokenOllama)
    baseline = [{"component_id": "bc1", "fault_hypothesis": "CAPACITOR_SHORT", "score": 1.0,
                 "visual_status": "SUSPECTED", "measurement_verdict": "non_diagnostic",
                 "fault_title": "커패시터 단락"}]
    result = {"status": "OK", "component_candidates": baseline, "all_component_candidates": baseline,
              "next_tests": [], "hypotheses": []}
    components = [{"component_id": "bc1", "board_component_id": "bc1", "component_type": "capacitor",
                   "type": "capacitor", "presence_status": "CONFIRMED", "review_status": "USER_CONFIRMED"}]
    _apply_qwen_pcb_assist(result, project=ProjectContext("project-1", "board-1"),
                           components=components, suspected_components=[], selected_model="qwen3.5:9b",
                           knowledge=load_knowledge(), answered_test_ids=[])
    assert result["qwen_fault_proposals"]["failure_category"] == "INVALID_JSON"
    assert result["component_candidates"] == baseline
    assert result["candidate_fusion"]["top_candidates"][0]["component_id"] == "bc1"


def test_candidate_fusion_does_not_let_qwen_upgrade_registry_review_state():
    proposal = {"component_id": "bc-1", "suspected_fault_id": "CAP_OPEN",
                "component_review_status": "USER_CONFIRMED", "reason": "proposal"}
    result = fuse_inspection_candidates(
        rule_candidates=[], qwen_proposals=[proposal],
        components=[{"component_id": "bc-1", "type": "capacitor",
                     "presence_status": "UNCONFIRMED"}],
        allowed_fault_ids={"CAP_OPEN"},
    )
    assert result["candidates"][0]["verification_status"] == "NEEDS_COMPONENT_REVIEW"


def test_app_test_initial_symptom_submission_applies_multiselect_value_before_widget():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from streamlit.testing.v1 import AppTest

    script = """import streamlit as st
from circuitcare.dashboard.views.symptom_diagnosis_panel import _apply_pending_initial_symptom_widget
state = st.session_state.setdefault('diagnosis', {})
_apply_pending_initial_symptom_widget(st.session_state, state, 'symptom_codes')
choice = st.selectbox('초기 증상', ['전원 안 켜짐', '기타·모르겠음'], key='initial')
st.multiselect('구조화된 증상', ['NO_POWER'], key='symptom_codes')
if st.button('답변으로 분석 시작', key='submit'):
    state['_pending_initial_symptom_widget_value'] = ['NO_POWER'] if choice == '전원 안 켜짐' else []
    st.session_state['diagnosis'] = state
    st.rerun()
"""
    with TemporaryDirectory() as folder:
        app_file = Path(folder) / "app.py"
        app_file.write_text(script, encoding="utf-8")
        app = AppTest.from_file(str(app_file)).run()
        assert not app.exception
        app.selectbox(key="initial").set_value("전원 안 켜짐")
        app.button(key="submit").click().run()
        assert not app.exception
        assert app.multiselect(key="symptom_codes").value == ["NO_POWER"]
        assert app.session_state["diagnosis"]["run_after_initial_answer"] is True


def test_candidate_focus_keys_include_fault_pair_and_scope():
    first = candidate_focus_widget_key("analysis", "p", "b", "c1", "OPEN")
    second = candidate_focus_widget_key("analysis", "p", "b", "c1", "SHORT")
    third = candidate_focus_widget_key("symptom", "p", "b", "c1", "OPEN")
    assert len({first, second, third}) == 3


def test_legacy_suspect_migration_runs_once_and_delete_stays_deleted():
    from circuitcare.dashboard.views.inspection_panel import (
        USER_SUSPECTS_KEY, board_suspects,
    )
    project = ProjectContext("project", "board")
    owner = "project|board"
    state = {USER_SUSPECTS_KEY: {
        owner: {}, f"{owner}|old-photo-scope": {
            "local-1": {"board_component_id": "bc-1", "component_id": "bc-1",
                                  "evidence_status": "UNVERIFIED"},
        },
    }}
    assert [row["board_component_id"] for row in board_suspects(state, project)] == ["bc-1"]
    state[USER_SUSPECTS_KEY][owner].pop("bc-1")
    assert board_suspects(state, project) == []
    assert board_suspects(state, project) == []


def test_candidate_fusion_rejects_unknown_fault_even_when_component_is_valid():
    result = fuse_inspection_candidates(
        rule_candidates=[],
        qwen_proposals=[{"component_id": "bc-1", "suspected_fault_id": "MADE_UP"}],
        components=[{"component_id": "bc-1", "type": "capacitor", "presence_status": "CONFIRMED"}],
        allowed_fault_ids={"CAP_OPEN"},
    )
    assert result["candidates"] == []
