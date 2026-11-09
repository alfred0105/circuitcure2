from circuitcare.conversation import validate_explanation
from circuitcare.llm_prompts import build_messages


def test_numeric_fact_requires_matching_component_and_cited_evidence():
    evidence = {
        "facts": [
            {"evidence_id": "E001", "component_ref": "C1", "metric": "voltage_stress", "value": 8.0, "unit": "V"},
            {"evidence_id": "E002", "component_ref": "C1", "metric": "voltage_rating", "value": 6.3, "unit": "V"},
        ],
        "unknowns": ["No component damage has been confirmed."],
    }
    response = {
        "answer": "C1의 근거를 확인했습니다 [E001].",
        "facts": [{"text": "C1의 모의 전압은 8 V, 설정 정격은 6.3 V입니다.", "evidence_ids": ["E001", "E002"]}],
        "assumptions": [],
        "recommendations": [],
        "unknowns": [{"text": "실제 손상은 확인되지 않았습니다.", "evidence_ids": []}],
    }
    response["facts"][0]["text"] = "U1_OUTPUT_8V 조건에서 C1의 모의 전압은 8 V, 설정 정격은 6.3 V입니다."
    assert validate_explanation(response, evidence, question="C1을 설명해") is None

    wrong_component = {**response, "facts": [{"text": "R1 전압은 8 V입니다.", "evidence_ids": ["E001"]}]}
    assert "different component" in validate_explanation(wrong_component, evidence, question="R1은?")

    unsupported_value = {**response, "facts": [{"text": "C1 전압은 7.9 V입니다.", "evidence_ids": ["E001"]}]}
    assert "not supported" in validate_explanation(unsupported_value, evidence, question="C1은?")


def test_llm_context_filters_unrelated_component_evidence():
    evidence = {
        "facts": [
            {"evidence_id": "E001", "component_ref": "C1", "metric": "voltage_stress", "value": 8.0},
            {"evidence_id": "E002", "component_ref": "RLOAD", "metric": "power_stress", "value": 0.016},
        ],
        "assumptions": [],
        "unknowns": [],
        "recommendations": [],
    }
    messages = build_messages("C1을 왜 먼저 점검해?", evidence, {"recommended_checks": []})
    assert "E001" in messages[1]["content"]
    assert "E002" not in messages[1]["content"]
    assert '"value":0.016' not in messages[1]["content"]
