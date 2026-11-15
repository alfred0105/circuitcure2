from circuitcare.symptom_diagnosis import diagnose, load_knowledge
from circuitcare.symptom_diagnosis.component_fault_ranker import rank_component_candidates
from circuitcare.symptom_diagnosis.measurement_evaluator import (
    evaluate_component_measurements, measurement_assessment_key,
)
from circuitcare.symptom_diagnosis.symptom_target_resolver import resolve_symptom_targets


class ParserClient:
    def __init__(self, payload):
        self.payload = payload
        self.schema = None

    def chat_json(self, _model, _messages, *, output_schema, output_budget):
        self.schema = output_schema
        assert output_budget == 480
        return {"content": self.payload, "model": "test-model"}


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
