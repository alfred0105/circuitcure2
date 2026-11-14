from circuitcare.symptom_diagnosis.active_inspection_planner import plan_active_inspections
from circuitcare.symptom_diagnosis import diagnose
from circuitcare.frontend.measurement_models import create_measurement_record
from circuitcare.symptom_diagnosis.measurement_validation import measurement_verification_gaps


def _knowledge():
    return {
        "fault_modes": {"fault_modes": {
            "F1": {"recommended_tests": ["ASK_SPLIT"], "supporting_evidence": {"OBS_A": 1}},
            "F2": {"recommended_tests": ["ASK_SPLIT"], "opposing_evidence": {"OBS_A": 1}},
        }},
        "test_catalog": {"tests": {
            "ASK_SPLIT": {
                "name": "분리 질문", "type": "question", "risk_level": "none",
                "results": {
                    "yes": {"observations": {"OBS_A": True}},
                    "no": {"observations": {"OBS_A": False}},
                },
            },
            "VISUAL_SAFE": {
                "name": "전원 차단 후 외관 확인", "type": "visual_inspection",
                "risk_level": "low", "target_scope": "board",
                "requires_power_off": True, "requires_stored_energy_safe": True,
                "results": {},
            },
        }},
        "component_test_profiles": {"profiles": {}},
    }


def _plan(**overrides):
    payload = {
        "hypotheses": [{"fault_id": "F1", "score": 2}, {"fault_id": "F2", "score": 1}],
        "component_candidates": [],
        "functional_block_groups": [],
        "evidence_gaps": [],
        "measurements": [],
        "observations": {},
        "inspection_history": [],
        "knowledge": _knowledge(),
    }
    payload.update(overrides)
    return plan_active_inspections(**payload)


def test_planner_ranks_deterministic_question_by_hypothesis_separation_without_probability():
    result = _plan()
    assert result["status"] == "READY"
    item = result["candidates"][0]
    assert item["test_id"] == "ASK_SPLIT"
    assert item["separated_hypothesis_pairs"] == [["F1", "F2"]]
    assert item["score_is_probability"] is False
    assert "information_gain" not in item


def test_equal_separation_checks_use_declared_safety_then_duration_and_cost_order():
    knowledge = _knowledge()
    fast = dict(knowledge["test_catalog"]["tests"]["ASK_SPLIT"],
                name="빠른 구분 질문", estimated_duration_seconds=10, estimated_cost=1)
    slow = dict(knowledge["test_catalog"]["tests"]["ASK_SPLIT"],
                name="느린 구분 질문", estimated_duration_seconds=40, estimated_cost=0)
    knowledge["test_catalog"]["tests"]["FAST"] = fast
    knowledge["test_catalog"]["tests"]["SLOW"] = slow
    knowledge["fault_modes"]["fault_modes"]["F1"]["recommended_tests"] = ["SLOW", "FAST"]
    knowledge["fault_modes"]["fault_modes"]["F2"]["recommended_tests"] = ["SLOW", "FAST"]
    result = _plan(knowledge=knowledge)
    assert [row["test_id"] for row in result["candidates"][:2]] == ["FAST", "SLOW"]


def test_measurement_separation_requires_declared_relations_for_both_hypotheses():
    knowledge = _knowledge()
    knowledge["test_catalog"]["tests"]["OHMS"] = {
        "name": "저항 측정", "type": "measurement", "risk_level": "unknown", "results": {},
    }
    knowledge["fault_modes"]["fault_modes"]["F1"]["recommended_tests"] = ["OHMS"]
    knowledge["fault_modes"]["fault_modes"]["F2"]["recommended_tests"] = ["OHMS"]
    knowledge["component_test_profiles"]["profiles"] = {"resistor": {"tests": {"OHMS": {
        "fault_relations": {
            "F1": {"TEST_NORMAL": "supports", "TEST_ABNORMAL": "refutes"},
            "F2": {"TEST_NORMAL": "refutes", "TEST_ABNORMAL": "supports"},
        },
    }}}}
    candidates = [
        {"component_id": "R1", "component_type": "resistor", "fault_hypothesis": "F1"},
        {"component_id": "R1", "component_type": "resistor", "fault_hypothesis": "F2"},
    ]
    plan = _plan(knowledge=knowledge, component_candidates=candidates)
    measurement = next(item for item in plan["blocked"] if item["test_id"] == "OHMS")
    assert measurement["separated_hypothesis_pairs"] == [["F1", "F2"]]
    knowledge["component_test_profiles"]["profiles"]["resistor"]["tests"]["OHMS"]["fault_relations"].pop("F2")
    plan_without_second_relation = _plan(knowledge=knowledge, component_candidates=candidates)
    measurement = next(item for item in plan_without_second_relation["blocked"] if item["test_id"] == "OHMS")
    assert measurement["separated_hypothesis_pairs"] == []


def test_profiled_measurement_becomes_actionable_only_after_each_safety_gate_is_confirmed():
    knowledge = _knowledge()
    knowledge["test_catalog"]["tests"]["OHMS"] = {
        "name": "저항 측정", "type": "measurement", "risk_level": "unknown", "results": {},
    }
    knowledge["fault_modes"]["fault_modes"]["F1"]["recommended_tests"] = ["OHMS"]
    knowledge["component_test_profiles"]["profiles"] = {"resistor": {"tests": {"OHMS": {
        "safety_protocol": "DEENERGIZED_ISOLATED_PASSIVE", "risk_level": "low",
        "required_conditions": {"power_state": "POWER_OFF_CONFIRMED",
                                "stored_energy_status": "SAFE_CONFIRMED",
                                "component_isolation_status": "ISOLATED_CONFIRMED"},
        "fault_relations": {"F1": {"TEST_NORMAL": "supports", "TEST_ABNORMAL": "refutes"}},
    }}}}
    candidate = [{"board_component_id": "bc-r1", "component_type": "resistor", "fault_hypothesis": "F1"}]
    blocked = _plan(knowledge=knowledge, component_candidates=candidate)
    assert any(row["test_id"] == "OHMS" for row in blocked["blocked"])
    ready = _plan(knowledge=knowledge, component_candidates=candidate,
        inspection_conditions={"bc-r1|F1": {
            "power_state": "POWER_OFF_CONFIRMED",
            "stored_energy_status": "SAFE_CONFIRMED",
            "component_isolation_status": "ISOLATED_CONFIRMED",
            "measurement_location_verified": True,
            "reference_source_verified": True,
        }})
    item = next(row for row in ready["candidates"] if row["test_id"] == "OHMS")
    assert item["blocked_until_safe"] is False
    assert item["status"] == "READY_FOR_USER_REVIEW"


def test_profiled_measurement_stays_blocked_until_location_and_reference_are_confirmed():
    knowledge = _knowledge()
    knowledge["test_catalog"]["tests"]["OHMS"] = {
        "name": "저항 측정", "type": "measurement", "risk_level": "low", "results": {},
    }
    knowledge["fault_modes"]["fault_modes"]["F1"]["recommended_tests"] = ["OHMS"]
    knowledge["component_test_profiles"]["profiles"] = {"resistor": {"tests": {"OHMS": {
        "safety_protocol": "DEENERGIZED_ISOLATED_PASSIVE", "risk_level": "low",
        "required_conditions": {"power_state": "POWER_OFF_CONFIRMED"},
        "fault_relations": {"F1": {"TEST_NORMAL": "supports", "TEST_ABNORMAL": "refutes"}},
    }}}}
    candidate = [{"board_component_id": "bc-r1", "component_type": "resistor", "fault_hypothesis": "F1"}]
    checked_electrical = {"power_state": "POWER_OFF_CONFIRMED"}
    plan = _plan(knowledge=knowledge, component_candidates=candidate,
                 inspection_conditions={"bc-r1|F1": checked_electrical})
    blocked = next(row for row in plan["blocked"] if row["test_id"] == "OHMS")
    assert "측정 위치·단자를 먼저 확인해야 함" in blocked["prerequisites"]
    assert "정상 기준·출처를 먼저 확인해야 함" in blocked["prerequisites"]
    checked_electrical.update(measurement_location_verified=True, reference_source_verified=True)
    ready = _plan(knowledge=knowledge, component_candidates=candidate,
                  inspection_conditions={"bc-r1|F1": checked_electrical})
    assert any(row["test_id"] == "OHMS" for row in ready["candidates"])


def test_profile_blocker_names_only_the_unconfirmed_safety_condition():
    knowledge = _knowledge()
    knowledge["test_catalog"]["tests"]["OHMS"] = {
        "name": "저항 측정", "type": "measurement", "risk_level": "low", "results": {},
    }
    knowledge["fault_modes"]["fault_modes"]["F1"]["recommended_tests"] = ["OHMS"]
    knowledge["component_test_profiles"]["profiles"] = {"resistor": {"tests": {"OHMS": {
        "safety_protocol": "DEENERGIZED_ISOLATED_PASSIVE", "risk_level": "low",
        "required_conditions": {"power_state": "POWER_OFF_CONFIRMED"},
        "fault_relations": {"F1": {"TEST_NORMAL": "supports", "TEST_ABNORMAL": "refutes"}},
    }}}}
    plan = _plan(knowledge=knowledge, component_candidates=[{
        "board_component_id": "bc-r1", "component_type": "resistor", "fault_hypothesis": "F1",
    }], inspection_conditions={"bc-r1|F1": {
        "power_state": "POWER_OFF_CONFIRMED", "measurement_location_verified": True,
    }})
    blocked = next(row for row in plan["blocked"] if row["test_id"] == "OHMS")
    assert blocked["prerequisites"] == ["정상 기준·출처를 먼저 확인해야 함", "해당 소자별 안전 상태가 아직 확인되지 않음"]


def test_completed_inspection_is_not_recommended_again_without_retest():
    result = _plan(inspection_history=[{
        "board_component_id": None, "fault_id": "F1", "related_fault_ids": ["F1", "F2"],
        "test_id": "ASK_SPLIT", "result": "예", "verification_status": "INCONCLUSIVE",
    }])
    assert not result["candidates"]


def test_risky_inspection_is_blocked_until_power_and_stored_energy_are_confirmed():
    knowledge = _knowledge()
    knowledge["fault_modes"]["fault_modes"]["F1"]["recommended_tests"].append("VISUAL_SAFE")
    result = _plan(knowledge=knowledge)
    blocked = next(row for row in result["blocked"] if row["test_id"] == "VISUAL_SAFE")
    assert blocked["blocked_until_safe"] is True
    assert blocked["status"] == "NEEDS_SAFETY_OR_TEST_DETAILS"
    assert "잔류 에너지 안전 확인" in blocked["prerequisites"]
    ready = _plan(knowledge=knowledge, inspection_conditions={
        "board|F1|VISUAL_SAFE": {"power_state": "POWER_OFF_CONFIRMED",
                                 "stored_energy_status": "SAFE_CONFIRMED"},
    })
    visual = next(row for row in ready["candidates"] if row["test_id"] == "VISUAL_SAFE")
    assert visual["blocked_until_safe"] is False
    assert "잔류 에너지 안전 확인됨" in visual["prerequisites"]


def test_question_answer_history_does_not_recommend_the_same_completed_check():
    result = _plan(inspection_history=[{
        "target_id": None, "related_fault_ids": ["F1", "F2"],
        "test_id": "ASK_SPLIT", "result": "예", "verification_status": "COMPLETE",
    }])
    assert not result["candidates"]


def test_missing_hypotheses_abstain_from_planning():
    result = _plan(hypotheses=[])
    assert result["status"] == "NO_HYPOTHESES"
    assert result["candidates"] == []


def test_production_diagnose_returns_inspection_plan_alongside_existing_rule_candidates():
    result = diagnose(symptoms=["NO_POWER"], observations={"POWER_SOURCE_VERIFIED": False})
    assert "active_inspection_plan" in result
    assert result["active_inspection_plan"]["score_is_probability"] is False
    assert result["hypotheses"]


def test_resistor_fault_workflow_moves_from_hypothesis_to_safe_check_and_suppresses_completed_test():
    diagnosed = diagnose(symptoms=["LOCALIZED_HEATING"], components=[{
        "component_id": "R7", "board_component_id": "R7", "component_type": "resistor",
        "component_type_status": "USER_CONFIRMED", "presence_status": "CONFIRMED",
        "bbox_xyxy": [10, 10, 30, 20],
    }], confirmed_component_ids={"R7"})
    fault = next((row for row in diagnosed["hypotheses"] if row["fault_id"] == "RESISTOR_FAILURE"), None)
    assert fault is not None
    candidates = [row for row in diagnosed["all_component_candidates"]
                  if row.get("fault_hypothesis") == "RESISTOR_FAILURE"]
    knowledge = _knowledge()
    knowledge["fault_modes"]["fault_modes"]["RESISTOR_FAILURE"] = {
        "recommended_tests": ["isolated_resistance"], "supporting_evidence": {}, "opposing_evidence": {}}
    knowledge["test_catalog"]["tests"]["isolated_resistance"] = {
        "name": "저항 분리 측정", "type": "measurement", "risk_level": "unknown", "results": {}}
    knowledge["component_test_profiles"]["profiles"] = {"resistor": {"tests": {
        "isolated_resistance": {"measurement_type": "resistance", "target_keywords": ["terminal"],
            "required_conditions": {"power_state": "POWER_OFF_CONFIRMED",
                "stored_energy_status": "SAFE_CONFIRMED", "component_isolation_status": "ISOLATED_CONFIRMED",
                "measurement_location_verified": True, "reference_source_verified": True},
            "safety_protocol": "DEENERGIZED_ISOLATED_PASSIVE", "risk_level": "low",
            "fault_relations": {"RESISTOR_FAILURE": {"TEST_NORMAL": "refutes", "TEST_ABNORMAL": "supports"}}}
    }}}
    groups = diagnosed["functional_block_groups"]
    gaps = diagnosed["evidence_gaps"]
    blocked = plan_active_inspections(hypotheses=[fault], component_candidates=candidates,
        functional_block_groups=groups, evidence_gaps=gaps, measurements=[], observations={},
        inspection_history=[], knowledge=knowledge)
    assert any(row["test_id"] == "isolated_resistance" for row in blocked["blocked"])
    ready = plan_active_inspections(hypotheses=[fault], component_candidates=candidates,
        functional_block_groups=groups, evidence_gaps=gaps, measurements=[], observations={},
        inspection_history=[], knowledge=knowledge, inspection_conditions={"R7|RESISTOR_FAILURE": {
            "power_state": "POWER_OFF_CONFIRMED", "stored_energy_status": "SAFE_CONFIRMED",
            "component_isolation_status": "ISOLATED_CONFIRMED", "measurement_location_verified": True,
            "reference_source_verified": True}})
    assert any(row["test_id"] == "isolated_resistance" for row in ready["candidates"])
    measurement = create_measurement_record(project_id="p", board_id="b", component_id="R7",
        board_component_id="R7", component_identity_status="USER_CONFIRMED",
        identity_resolution_status="USER_RECONFIRMED_VERIFIED", component_type="resistor",
        measurement_type="resistance", value="220", unit="Ω", uncertainty="1",
        uncertainty_source="meter spec", source="user_measurement", test_id="isolated_resistance",
        fault_hypothesis="RESISTOR_FAILURE", measurement_target="R7 terminal A-B",
        measurement_location="R7 terminal A-B", location_verified=True,
        reference_min="215", reference_max="225", reference_source="board service specification",
        reference_source_id="board-spec-r7", reference_verification_status="BOARD_SPEC_VERIFIED",
        operating_conditions="deenergized isolated", reference_conditions="deenergized isolated",
        measurement_conditions={"power_state": "POWER_OFF_CONFIRMED",
            "stored_energy_status": "SAFE_CONFIRMED", "component_isolation_status": "ISOLATED_CONFIRMED",
            "measurement_location_verified": True, "reference_source_verified": True})
    row = measurement.to_dict()
    assert measurement_verification_gaps(row, profiles=knowledge["component_test_profiles"]) == []
    after = plan_active_inspections(hypotheses=[fault], component_candidates=candidates,
        functional_block_groups=groups, evidence_gaps=gaps, measurements=[row], observations={},
        inspection_history=[], knowledge=knowledge,
        inspection_conditions={"R7|RESISTOR_FAILURE": {
            "power_state": "POWER_OFF_CONFIRMED", "stored_energy_status": "SAFE_CONFIRMED",
            "component_isolation_status": "ISOLATED_CONFIRMED", "measurement_location_verified": True,
            "reference_source_verified": True}})
    assert not any(item["test_id"] == "isolated_resistance"
                   for item in [*after["candidates"], *after["blocked"]])
