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


def test_qwen_semantic_symptom_is_not_discarded_when_fallback_keywords_do_not_match():
    from circuitcare.symptom_diagnosis.symptom_parser import parse_user_report

    text = "위쪽 붉은 LED 하나가 빛을 내지 않습니다."
    assert "LED_NOT_LIT" not in infer_symptom_codes(text)
    client = ParserClient({
        "observations": [{"code": "LED_NOT_LIT", "value": True,
                          "evidence_quote": text, "target_reference": "red-led"}],
        "targets": [{"reference_id": "red-led", "component_type": "led", "color": "붉은",
                     "region": "UPPER", "group_size": 1, "ordinal": 0,
                     "ordering": "UNSPECIFIED", "reference_designator": "",
                     "evidence_quote": text}],
        "peer_observations": [], "uncertain_facts": [], "requires_clarification": False,
    })

    result = parse_user_report(client, "qwen-test", text)

    assert result["parser_status"] == "PARSED"
    assert result["observations"][0]["code"] == "LED_NOT_LIT"
    assert result["observations"][0]["target_reference"] == "red-led"


def test_partial_qwen_led_parse_keeps_failure_and_resolves_reported_third_led():
    from circuitcare.symptom_diagnosis.symptom_parser import parse_user_report

    text = (
        "기판 상단의 빨간 LED 4개 중 왼쪽에서 세 번째 LED(박스21-led)만 켜지지 않습니다. "
        "나머지 LED는 정상적으로 켜집니다."
    )
    parser = ParserClient({
        # This reproduces the real Qwen 3.5 9B response observed during audit:
        # it reported working peers but omitted LED_NOT_LIT and the target.
        "observations": [{"code": "LED_WORKING", "value": True,
                          "evidence_quote": "나머지 LED는 정상적으로 켜집니다."}],
        "targets": [], "peer_observations": [], "uncertain_facts": [],
        "requires_clarification": False,
    })
    parsed = parse_user_report(parser, "qwen3.5:9b", text)
    assert "LED_NOT_LIT" in {row["code"] for row in parsed["observations"]}
    assert parsed["observations"][-1]["target_reference"] == "rule-target-1"
    assert parsed["targets"][0]["ordinal"] == 3
    assert parsed["targets"][0]["component_type"] == "led"

    registry = {"components": []}
    visible = []
    boxes = ([560, 64, 669, 328], [670, 61, 802, 325],
             [805, 51, 922, 325], [939, 58, 1019, 322])
    components = []
    for index, box in enumerate(boxes, start=1):
        board_id, local_id = f"BC-LED-{index}", f"LED-{index}"
        registry["components"].append({
            "board_component_id": board_id, "component_type": "led",
            "component_type_status": "USER_CONFIRMED", "presence_status": "CONFIRMED",
            "reference_designators": [f"D{index}"],
            "source_components": [{"component_id": local_id, "review_status": "USER_CONFIRMED"}],
        })
        visible.append({"component_id": local_id, "board_component_id": board_id,
                        "photo_id": "reviewed-led-board", "coordinate_frame": "SOURCE_PHOTO_PIXEL",
                        "bbox_xyxy": box})
        components.append({"component_id": local_id, "board_component_id": board_id,
                           "component_type": "led", "component_type_status": "USER_CONFIRMED",
                           "presence_status": "CONFIRMED", "bbox_xyxy": box})
    resolved = resolve_symptom_targets(parsed["targets"], registry=registry, component_rows=visible)
    assert resolved[0]["resolution_status"] == "RESOLVED_POSITIONAL"
    assert resolved[0]["component_id"] == "LED-3"
    observations, _peers = bind_target_observations(parsed, resolved)
    result = diagnose(symptoms=["LED_WORKING"], symptom_text=text,
                      observations=observations, components=components,
                      confirmed_component_ids={row["component_id"] for row in components})
    assert "LED_NOT_LIT" in result["symptom_codes"]
    assert any(row["component_id"] == "LED-3" and row["fault_id"] == "LED_FAILURE"
               for row in result["all_component_candidates"])


def test_qwen_missing_symptom_target_is_merged_with_literal_positional_fallback():
    from circuitcare.symptom_diagnosis.symptom_parser import parse_user_report

    text = (
        "전원을 켜면 기판은 정상적으로 동작합니다. 그런데 위쪽에 있는 빨간 LED 4개 중 "
        "세 번째만 켜지지 않습니다. 나머지 LED는 정상적으로 작동합니다."
    )
    parser = ParserClient({
        "observations": [
            {"code": "LED_NOT_LIT", "value": True,
             "evidence_quote": "세 번째만 켜지지 않습니다", "target_reference": "target-1"},
            {"code": "LED_WORKING", "value": True,
             "evidence_quote": "나머지 LED는 정상적으로 작동합니다", "target_reference": "target-2"},
        ],
        "targets": [{"reference_id": "target-2", "component_type": "led", "color": "",
                     "region": "UNSPECIFIED", "group_size": 0, "ordinal": 0,
                     "ordering": "UNSPECIFIED", "reference_designator": "",
                     "evidence_quote": "나머지 LED는 정상적으로 작동합니다"}],
        "peer_observations": [], "uncertain_facts": [], "requires_clarification": False,
    })

    parsed = parse_user_report(parser, "qwen3.5:9b", text)

    failures = [row for row in parsed["observations"] if row["code"] == "LED_NOT_LIT"]
    assert len(failures) == 1
    assert failures[0]["target_reference"] == "rule-target-2"

    boxes = ([560, 64, 669, 328], [670, 61, 802, 325],
             [805, 51, 922, 325], [939, 58, 1019, 322])
    registry = {"components": []}
    components = []
    for index, box in enumerate(boxes, start=1):
        local_id, board_id = f"LED-{index}", f"BC-LED-{index}"
        registry["components"].append({
            "board_component_id": board_id, "component_type": "led",
            "component_type_status": "USER_CONFIRMED", "presence_status": "CONFIRMED",
            "source_components": [{"component_id": local_id, "photo_id": "review-photo",
                                   "coordinate_frame": "SOURCE_PHOTO_PIXEL", "bbox_xyxy": box,
                                   "review_status": "USER_CONFIRMED"}],
        })
        components.append({"component_id": local_id, "board_component_id": board_id,
                           "component_type": "led", "component_type_status": "USER_CONFIRMED",
                           "presence_status": "CONFIRMED", "bbox_xyxy": box,
                           "photo_id": "review-photo", "coordinate_frame": "SOURCE_PHOTO_PIXEL"})
    resolved = resolve_symptom_targets(parsed["targets"], registry=registry,
                                       selected_photo_id="review-photo")
    assert next(row for row in resolved if row["reference_id"] == "rule-target-2")["board_component_id"] == "BC-LED-3"
    observations, _peers = bind_target_observations(parsed, resolved)
    bound = next(row for row in observations if row["code"] == "LED_NOT_LIT")
    assert bound["board_component_id"] == "BC-LED-3"
    assert bound["component_id"] == "LED-3"
    result = diagnose(symptom_text=text, observations=observations, components=components,
                      confirmed_component_ids={row["component_id"] for row in components})
    assert result["hypotheses"][0]["fault_id"] == "LED_FAILURE"
    assert [(row["component_id"], row["fault_hypothesis"])
            for row in result["all_component_candidates"]] == [("LED-3", "LED_FAILURE")]


def test_unlocalized_led_failure_keeps_all_four_leds_as_ambiguous_group():
    components = [{
        "component_id": f"LED-{index}", "board_component_id": f"LED-{index}",
        "component_type": "led", "component_type_status": "USER_CONFIRMED",
        "presence_status": "CONFIRMED",
    } for index in range(1, 5)]

    result = diagnose(
        symptom_text="빨간 LED 네 개 중 하나가 켜지지 않습니다. 나머지 LED는 정상입니다.",
        components=components,
        confirmed_component_ids={row["component_id"] for row in components},
        _simulate_questions=False,
    )

    assert "LED_NOT_LIT" in result["symptom_codes"]
    led_candidates = [row for row in result["all_component_candidates"]
                      if row.get("fault_hypothesis") == "LED_FAILURE"]
    assert {row["component_id"] for row in led_candidates} == {f"LED-{index}" for index in range(1, 5)}
    assert {row["candidate_group_status"] for row in led_candidates} == {"UNRESOLVED_AMBIGUITY"}
    assert len({row["candidate_group_id"] for row in led_candidates}) == 1
    assert result["component_candidate_groups"][0]["status"] == "UNRESOLVED_AMBIGUITY"


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


def test_explicit_left_to_right_ordinal_resolves_to_matching_photo_box():
    registry, visible = _reviewed_registry(4)
    positioned = [{**row, "photo_id": "photo-front", "coordinate_frame": "SOURCE_PHOTO_PIXEL",
                   "bbox_xyxy": [index * 20, 10, index * 20 + 10, 30]}
                  for index, row in enumerate(visible, start=1)]
    target = {"reference_id": "led-group", "component_type": "led", "group_size": 4,
              "ordinal": 3, "ordering": "LEFT_TO_RIGHT"}

    resolved = resolve_symptom_targets([target], registry=registry, component_rows=positioned)[0]

    assert resolved["resolution_status"] == "RESOLVED_POSITIONAL"
    assert resolved["board_component_id"] == "BC-LED-3"


def test_explicit_ordinal_stays_ambiguous_without_valid_photo_coordinates():
    registry, visible = _reviewed_registry(4)
    target = {"reference_id": "led-group", "component_type": "led", "group_size": 4,
              "ordinal": 3, "ordering": "LEFT_TO_RIGHT"}

    resolved = resolve_symptom_targets([target], registry=registry, component_rows=visible)[0]

    assert resolved["resolution_status"] == "AMBIGUOUS"
    assert resolved["board_component_id"] is None


def test_board_level_diagnosis_rows_recover_coordinates_from_registry_photo_observations():
    boxes = ([560, 64, 669, 328], [670, 61, 802, 325],
             [805, 51, 922, 325], [939, 58, 1019, 322])
    registry = {"components": []}
    diagnosis_rows = []  # BoardDiagnosisComponent projections have no photo-local bbox.
    for index, box in enumerate(boxes, start=1):
        board_id, local_id = f"BC-LED-{index}", f"LED-{index}"
        registry["components"].append({
            "board_component_id": board_id,
            "component_type": "led",
            "component_type_status": "USER_CONFIRMED",
            "presence_status": "CONFIRMED",
            "source_components": [{
                "component_id": local_id, "photo_id": "photo-selected",
                "coordinate_frame": "SOURCE_PHOTO_PIXEL", "bbox_xyxy": box,
                "review_status": "USER_CONFIRMED",
            }],
        })
        diagnosis_rows.append({"component_id": board_id, "board_component_id": board_id,
                               "component_type": "led", "bbox_xyxy": None})
    target = {"reference_id": "led-group", "component_type": "led", "group_size": 4,
              "ordinal": 3, "ordering": "LEFT_TO_RIGHT"}

    resolved = resolve_symptom_targets(
        [target], registry=registry, component_rows=diagnosis_rows,
        selected_photo_id="photo-selected",
    )[0]

    assert resolved["resolution_status"] == "RESOLVED_POSITIONAL"
    assert resolved["board_component_id"] == "BC-LED-3"
    assert resolved["component_id"] == "LED-3"


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
    assert "LED_NOT_LIT" in infer_symptom_codes(
        "기판 상단의 빨간 LED 4개 중 왼쪽에서 세 번째 LED(박스21-led)만 켜지지 않는다."
    )
    assert "LED_NOT_LIT" not in infer_symptom_codes("기판 전원이 켜지지 않습니다.")
    assert "NO_POWER" in infer_symptom_codes("기판 전원이 켜지지 않습니다.")


def test_single_dark_indicator_does_not_become_whole_board_no_power():
    for text in (
        "빨간 표시등 하나만 안 켜지는 것 같아요.",
        "전원 LED만 안 켜져요.",
        "LED 4개 중 3번째만 불이 안 들어옴. 나머지는 됨.",
    ):
        codes = infer_symptom_codes(text)
        assert "LED_NOT_LIT" in codes
        assert "NO_POWER" not in codes

    codes = infer_symptom_codes("기판 전원이 안 켜지고 전원 표시등도 점등하지 않습니다.")
    assert "NO_POWER" in codes
    assert "LED_NOT_LIT" in codes


def test_colloquial_symptom_stress_phrases_use_conservative_fallback_codes():
    assert "NO_POWER" in infer_symptom_codes("전원 꽂아도 보드가 전혀 안켜져요ㅠ")
    assert "PARTIAL_FUNCTION_FAILURE" in infer_symptom_codes(
        "기판은 켜져 있는데 특정 기능만 영 안 돼요. IC 쪽일 수도? 모르겠어요."
    )
    assert "PARTIAL_FUNCTION_FAILURE" in infer_symptom_codes(
        "전원이 완전 죽은 건 아니고요, 제어가 반쯤 먹통인 듯."
    )
    assert "INTERMITTENT_OPERATION" in infer_symptom_codes(
        "커패시터가 나갔다고 생각했는데 그건 추측이고, 실제론 제어 기능이 가끔 멈춰요."
    )
    assert "LED_NOT_LIT" in infer_symptom_codes("빨간 LED 4개중 3번째만 안들어옴, 나머진 정상.")
    assert "OUTPUT_FAILURE" in infer_symptom_codes(
        "기판은 도는 것 같은데 연결된 장치가 반응하지 않아요. 장치 고장인지는 확인 안했음."
    )
    # Hearsay and mutually uncertain statements must not become asserted symptoms.
    assert infer_symptom_codes("어떤 연결이 끊긴 것 같다던데, 본 건 아니고 정확히 뭐가 문젠지 모르겠어.") == []
    assert "SUPPLY_VOLTAGE_DROOP" not in infer_symptom_codes(
        "부하(?) 켜면 전압이 쭉 빠지는 거 같기도 하고 아닐지도.."
    )


def test_qwen_keeps_clear_symptom_separate_from_guessed_component_cause():
    from circuitcare.symptom_diagnosis.symptom_parser import parse_user_report

    text = "커패시터가 나갔다고 생각했는데 그건 추측이고, 실제론 제어 기능이 가끔 멈춰요."
    parsed = parse_user_report(ParserClient({
        "observations": [{"code": "INTERMITTENT_OPERATION", "value": True,
                          "evidence_quote": "제어 기능이 가끔 멈춰요."}],
        "uncertain_facts": ["커패시터 고장은 확인되지 않은 원인 추측"],
        "requires_clarification": True,
    }), "qwen-test", text)
    assert [row["code"] for row in parsed["observations"]] == ["INTERMITTENT_OPERATION"]
    assert "커패시터 고장은 확인되지 않은 원인 추측" in parsed["uncertain_facts"]


def test_qwen_semantic_guard_rejects_unpowered_and_intermittent_overreach():
    from circuitcare.symptom_diagnosis.symptom_parser import parse_user_report

    text = "전원은 들어오는데 제어 쪽은 먹통이에요. 기판은 도는 것 같은데 연결 장치는 반응하지 않아요."
    parsed = parse_user_report(ParserClient({
        "observations": [
            {"code": "NO_POWER", "value": False, "evidence_quote": "전원은 들어오는데"},
            {"code": "PARTIAL_FUNCTION_FAILURE", "value": True, "evidence_quote": "제어 쪽은 먹통이에요"},
            {"code": "INTERMITTENT_OPERATION", "value": True, "evidence_quote": "기판은 도는 것 같은데"},
            {"code": "OUTPUT_FAILURE", "value": True, "evidence_quote": "연결 장치는 반응하지 않아요"},
        ],
        "uncertain_facts": [], "requires_clarification": False,
    }), "qwen-test", text)
    codes = [row["code"] for row in parsed["observations"]]
    assert "PARTIAL_FUNCTION_FAILURE" in codes
    assert "OUTPUT_FAILURE" in codes
    assert "NO_POWER" not in codes
    assert "INTERMITTENT_OPERATION" not in codes


def test_incomplete_qwen_parse_falls_back_to_rule_symptom_candidates():
    text = (
        "기판 상단의 빨간 LED 4개 중 왼쪽에서 세 번째 LED(박스21-led)만 켜지지 않는다. "
        "나머지 LED 3개는 정상적으로 점등된다."
    )
    parsed = {"parser_status": "PARSED", "observations": []}
    parsed_symptom_facts = [row for row in parsed["observations"]
                            if isinstance(row, dict) and row.get("code") in load_knowledge()["symptom_ontology"]["symptoms"]]
    rule_text = "" if parsed["parser_status"] == "PARSED" and parsed_symptom_facts else text
    components = [{
        "component_id": f"BC-LED-{index}", "board_component_id": f"BC-LED-{index}",
        "component_type": "led", "component_type_status": "USER_CONFIRMED",
        "presence_status": "CONFIRMED", "bbox_xyxy": [index * 20, 10, index * 20 + 10, 30],
    } for index in range(1, 5)]

    result = diagnose(
        symptom_text=rule_text, components=components,
        confirmed_component_ids={row["component_id"] for row in components},
    )

    assert result["status"] == "CANDIDATES_AVAILABLE"
    assert "LED_NOT_LIT" in result["symptom_codes"]
    assert result["component_candidates"]
    assert all(row["fault_id"] == "LED_FAILURE" for row in result["component_candidates"])


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


def test_ordinal_led_target_uses_unambiguous_single_row_photo_coordinates():
    registry, visible = _reviewed_registry(4)
    boxes = ([560, 64, 669, 328], [670, 61, 802, 325],
             [805, 51, 922, 325], [939, 58, 1019, 322])
    positioned = [{**row, "photo_id": "photo-front", "coordinate_frame": "SOURCE_PHOTO_PIXEL",
                   "bbox_xyxy": box} for row, box in zip(visible, boxes)]
    target = {"reference_id": "led-group", "component_type": "led", "group_size": 4,
              "ordinal": 3, "ordering": "UNSPECIFIED", "region": "UPPER"}

    resolved = resolve_symptom_targets([target], registry=registry, component_rows=positioned)[0]

    assert resolved["resolution_status"] == "RESOLVED_POSITIONAL"
    assert resolved["component_id"] == "LED-3"


def test_ordinal_target_stays_ambiguous_for_a_two_row_grid():
    registry, visible = _reviewed_registry(4)
    boxes = ([100, 100, 150, 150], [220, 100, 270, 150],
             [100, 220, 150, 270], [220, 220, 270, 270])
    positioned = [{**row, "photo_id": "photo-front", "coordinate_frame": "SOURCE_PHOTO_PIXEL",
                   "bbox_xyxy": box} for row, box in zip(visible, boxes)]
    target = {"reference_id": "led-group", "component_type": "led", "group_size": 4,
              "ordinal": 3, "ordering": "UNSPECIFIED"}

    resolved = resolve_symptom_targets([target], registry=registry, component_rows=positioned)[0]

    assert resolved["resolution_status"] == "AMBIGUOUS"
    assert resolved["board_component_id"] is None


def test_resolved_natural_ordinal_restricts_fault_candidate_to_the_reported_led():
    registry, visible = _reviewed_registry(4)
    boxes = ([560, 64, 669, 328], [670, 61, 802, 325],
             [805, 51, 922, 325], [939, 58, 1019, 322])
    local_ids = ["manual-36f9b53409a14be4", "manual-ca3d33298040314a-ff0c0e63e324",
                 "manual-41876e30dccf798a", "manual-f0d07245faef6064"]
    for registry_row, visible_row, local_id in zip(registry["components"], visible, local_ids):
        board_id = f"BC-{local_id}"
        registry_row["board_component_id"] = board_id
        registry_row["source_components"] = [{"component_id": local_id, "review_status": "USER_CONFIRMED"}]
        visible_row.update(component_id=local_id, board_component_id=board_id)
    positioned = [{**row, "photo_id": "photo-front", "coordinate_frame": "SOURCE_PHOTO_PIXEL",
                   "bbox_xyxy": box} for row, box in zip(visible, boxes)]
    target = {"reference_id": "led-group", "component_type": "led", "group_size": 4,
              "ordinal": 3, "ordering": "UNSPECIFIED"}
    resolved = resolve_symptom_targets([target], registry=registry, component_rows=positioned)
    observations, _peers = bind_target_observations({"observations": [{
        "code": "LED_NOT_LIT", "value": True, "status": "observed",
        "source": "user_report", "target_reference": "led-group",
    }]}, resolved)
    components = [{"component_id": local_id, "board_component_id": f"BC-{local_id}",
                   "component_type": "led",
                   "component_type_status": "USER_CONFIRMED", "presence_status": "CONFIRMED",
                   "bbox_xyxy": box, "photo_id": "photo-front",
                   "coordinate_frame": "SOURCE_PHOTO_PIXEL"}
                  for local_id, box in zip(local_ids, boxes)]

    result = diagnose(symptoms=["LED_NOT_LIT"], components=components,
                      confirmed_component_ids={row["component_id"] for row in components},
                      observations=observations)

    assert resolved[0]["component_id"] == "manual-41876e30dccf798a"
    assert [row["component_id"] for row in result["component_candidates"]] == [
        "manual-41876e30dccf798a"
    ]


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
