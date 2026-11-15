from circuitcare.symptom_diagnosis import diagnose, infer_symptom_codes, load_knowledge
from circuitcare.symptom_diagnosis.component_fault_ranker import rank_component_candidates
from circuitcare.symptom_diagnosis.measurement_evaluator import (
    evaluate_component_measurements, measurement_assessment_key,
)
from circuitcare.symptom_diagnosis.symptom_target_resolver import (
    bind_target_observations, resolve_symptom_targets,
)


class ParserClient:
    def __init__(self, payload):
        self.payload = payload
        self.schema = None

    def chat_json(self, _model, _messages, *, output_schema, output_budget):
        self.schema = output_schema
        assert output_budget == 480
        return {"content": self.payload, "model": "test-model"}


class ProposalClient:
    def __init__(self, proposal):
        self.proposal = proposal

    def chat_json(self, *_args, **_kwargs):
        return {"content": {"proposals": [self.proposal]}, "model": "test-model"}


def _reviewed_registry(component_count=4):
    components = []
    visible = []
    for index in range(1, component_count + 1):
        board_id, local_id = f"BC-LED-{index}", f"LED-{index}"
        components.append({
            "board_component_id": board_id, "component_type": "led",
            "component_type_status": "USER_CONFIRMED", "presence_status": "CONFIRMED",
            "reference_designators": [f"D{index}"], "ocr_markings": [],
            "source_components": [{"component_id": local_id, "review_status": "USER_CONFIRMED"}],
        })
        visible.append({"component_id": local_id, "board_component_id": board_id})
    return {"components": components}, visible


def test_parser_keeps_led_target_unresolved_and_validates_quote_and_ordinal():
    from circuitcare.symptom_diagnosis.symptom_parser import parse_user_report

    text = "빨간 LED 4개 중 세 번째만 켜지지 않습니다. 나머지는 정상입니다."
    client = ParserClient({
        "observations": [{"code": "LED_NOT_LIT", "value": True,
                          "evidence_quote": "세 번째만 켜지지 않습니다", "target_reference": "led-group"}],
        "targets": [{"reference_id": "led-group", "component_type": "led", "color": "빨간",
                     "region": "UNSPECIFIED", "group_size": 4, "ordinal": 3,
                     "ordering": "LEFT_TO_RIGHT", "reference_designator": "",
                     "evidence_quote": "빨간 LED 4개 중 세 번째"}],
        "peer_observations": [{"code": "LED_WORKING", "value": True,
                               "target_reference": "led-group", "evidence_quote": "나머지는 정상"}],
        "uncertain_facts": [], "requires_clarification": False,
    })
    result = parse_user_report(client, "qwen-test", text)
    assert result["targets"][0]["resolution_status"] == "UNRESOLVED"
    assert result["targets"][0]["ordering"] == "UNSPECIFIED"
    assert result["targets"][0]["ordinal"] == 3
    assert result["observations"][0]["target_reference"] == "led-group"
    assert result["peer_observations"][0]["code"] == "LED_WORKING"


def test_target_resolver_requires_user_choice_for_same_type_ordinal_group():
    registry, visible = _reviewed_registry(4)
    target = {"reference_id": "led-group", "component_type": "led", "group_size": 4,
              "ordinal": 3, "ordering": "UNSPECIFIED"}
    unresolved = resolve_symptom_targets([target], registry=registry, component_rows=visible)[0]
    assert unresolved["resolution_status"] == "AMBIGUOUS"
    assert len(unresolved["candidate_components"]) == 4
    selected = resolve_symptom_targets(
        [target], registry=registry, component_rows=visible,
        selected_board_component_ids={"led-group": "BC-LED-3"},
    )[0]
    assert selected["resolution_status"] == "RESOLVED_USER_CONFIRMED"
    assert selected["component_id"] == "LED-3"


def test_target_resolver_auto_resolves_only_a_single_confirmed_non_group_component():
    registry, visible = _reviewed_registry(1)
    resolved = resolve_symptom_targets(
        [{"reference_id": "only-led", "component_type": "led"}],
        registry=registry, component_rows=visible,
    )[0]
    assert resolved["resolution_status"] == "RESOLVED_UNIQUE"
    assert resolved["board_component_id"] == "BC-LED-1"


def test_target_resolver_does_not_auto_resolve_when_reported_color_is_unverified_or_conflicts():
    registry, visible = _reviewed_registry(1)
    target = {"reference_id": "red-led", "component_type": "led", "color": "빨간"}
    unresolved = resolve_symptom_targets([target], registry=registry, component_rows=visible)[0]
    assert unresolved["resolution_status"] == "NEEDS_ATTRIBUTE_CONFIRMATION"
    assert unresolved["board_component_id"] is None

    registry["components"][0]["confirmed_color"] = "blue"
    conflict = resolve_symptom_targets([target], registry=registry, component_rows=visible)[0]
    assert conflict["resolution_status"] == "UNRESOLVED"
    assert conflict["candidate_components"] == []


def test_led_working_observation_is_not_copied_to_same_type_parts_without_verified_group():
    registry, visible = _reviewed_registry(4)
    target = {"reference_id": "leds", "component_type": "led", "group_size": 4}
    resolved = resolve_symptom_targets(
        [target], registry=registry, component_rows=visible,
        selected_board_component_ids={"leds": "BC-LED-3"},
    )
    parsed = {"observations": [], "peer_observations": [{
        "code": "LED_WORKING", "value": True, "target_reference": "leds",
        "evidence_quote": "나머지는 정상입니다.",
    }]}
    _, peers = bind_target_observations(parsed, resolved)
    assert peers == []


def test_led_symptom_requires_led_context_and_board_power_failure_is_not_led_failure():
    assert "LED_NOT_LIT" in infer_symptom_codes("빨간 LED 4개 중 세 번째만 켜지지 않습니다.")
    assert "LED_NOT_LIT" not in infer_symptom_codes("기판 전원이 켜지지 않습니다.")
    assert "NO_POWER" in infer_symptom_codes("기판 전원이 켜지지 않습니다.")


def test_led_fault_has_a_dedicated_measurement_but_planner_blocks_it_until_safety_details():
    result = diagnose(
        symptoms=["LED_NOT_LIT"],
        components=[{"component_id": "D3", "component_type": "led",
                     "component_type_status": "USER_CONFIRMED", "presence_status": "CONFIRMED"}],
        confirmed_component_ids={"D3"},
    )
    plan = result["active_inspection_plan"]
    led_test = next(row for row in plan["blocked"] if row["test_id"] == "isolated_forward_drop")
    assert led_test["status"] == "NEEDS_SAFETY_OR_TEST_DETAILS"
    assert "측정 대상 분리 확인" in " ".join(led_test["prerequisites"])


def test_qwen_fault_proposal_must_match_a_reported_symptom_or_observation():
    from circuitcare.symptom_diagnosis.llm_assist import suggest_pcb_fault_candidates

    knowledge = load_knowledge()
    proposal = {
        "component_id": "BC-LED-1", "suspected_fault_id": "LED_FAILURE",
        "functional_block": "output", "functional_block_id": "__none__",
        "reason": "unverified suggestion", "supporting_evidence_ids": [],
        "opposing_evidence_ids": [], "missing_information": [],
        "recommended_next_action": "inspect", "uncertainty_status": "UNVERIFIED",
    }
    context = {"focus_components": [{
        "component_id": "BC-LED-1", "board_component_id": "BC-LED-1",
        "type": "led", "presence_status": "CONFIRMED", "review_status": "USER_CONFIRMED",
        "component_type_status": "USER_CONFIRMED", "candidate_component_types": ["led"],
        "identity_status": "VERIFIED",
    }], "functional_block_groups": [], "observations": {},
        "measurement_evidence": [], "evidence_ledger": []}

    irrelevant = suggest_pcb_fault_candidates(
        ProposalClient(proposal), "qwen", pcb_context=context,
        diagnosis={"symptom_codes": ["NO_POWER"], "observations": {}}, knowledge=knowledge,
    )
    assert irrelevant["proposals"] == []
    assert irrelevant["rejected_proposal_count"] == 1

    relevant = suggest_pcb_fault_candidates(
        ProposalClient(proposal), "qwen", pcb_context=context,
        diagnosis={"symptom_codes": ["LED_NOT_LIT"], "observations": {}}, knowledge=knowledge,
    )
    assert len(relevant["proposals"]) == 1


def test_component_symptom_evidence_targets_only_the_reported_component():
    components = [
        {"component_id": f"LED-{index}", "component_type": "led",
         "component_type_status": "USER_CONFIRMED", "presence_status": "CONFIRMED"}
        for index in range(1, 5)
    ]
    ranked = rank_component_candidates(
        hypotheses=[{"fault_id": "LED_FAILURE", "title": "LED 이상", "score": 2.0}],
        components=components, functional_blocks=[],
        observations=[{"component_id": "LED-3", "target_reference": "led-group",
                       "code": "LED_NOT_LIT", "value": True, "status": "observed"}],
        measurements=[], knowledge=load_knowledge(),
        confirmed_component_ids={row["component_id"] for row in components},
    )
    assert [row["component_id"] for row in ranked] == ["LED-3"]
    assert ranked[0]["direct_symptom_evidence"] == ["LED_NOT_LIT"]
    assert ranked[0]["candidate_group_id"] is None


def test_measurement_evidence_uses_the_same_normalized_key_as_evaluator():
    row = {
        "measurement_id": "m-r1", "component_id": "R1", "board_component_id": "BC-R1",
        "component_type": "resistor", "component_type_status": "USER_CONFIRMED",
        "component_identity_status": "USER_CONFIRMED", "identity_resolution_status": "USER_SELECTED_VERIFIED",
        "source": "user_measurement", "verification_status": "USER_REPORTED",
        "fault_hypothesis": "RESISTOR_FAILURE", "test_id": "isolated_resistance",
        "measurement_type": "resistance", "measurement_target": "R1 양단 terminal A-B",
        "location_verified": True, "value": 1000.0, "unit": "Ω", "uncertainty": 0.5,
        "uncertainty_source": "meter specification", "reference_min": 990.0, "reference_max": 1010.0,
        "reference_source_id": "BOARD-R1", "reference_source": "verified board specification",
        "reference_verification_status": "BOARD_SPEC_VERIFIED",
        "measurement_conditions": {
            "power_state": "POWER_OFF_CONFIRMED",
            "stored_energy_status": "SAFE_CONFIRMED",
            "component_isolation_status": "ISOLATED_CONFIRMED",
            "measurement_location_verified": True, "reference_source_verified": True,
        },
    }
    row["reference_conditions"] = dict(row["measurement_conditions"])
    profiles = load_knowledge()["component_test_profiles"]
    assessments = evaluate_component_measurements([row], profiles=profiles)
    key = measurement_assessment_key(row)
    assert key in assessments

    result = diagnose(
        symptoms=["PARTIAL_FUNCTION_FAILURE"],
        components=[{"component_id": "R1", "board_component_id": "BC-R1",
                     "component_type": "resistor", "component_type_status": "USER_CONFIRMED",
                     "presence_status": "CONFIRMED"}],
        confirmed_component_ids={"R1"}, measurements=[row],
    )
    evidence = next(item for item in result["measurement_evidence"] if item["measurement_id"] == "m-r1")
    assert evidence["assessment"] == assessments[key]["status"]
    assert evidence["verdict"] == assessments[key]["verdict"]
    assert evidence["assessment_status"] == assessments[key]["assessment_status"]
