import json

from circuitcare.datasheet.evaluation import classify_candidate, score_extractions
from circuitcare.fault_ai.evaluation import build_blind_diagnostic_input, score_fault_proposals, score_fault_rankings
from circuitcare.fault_ai.llm_hypothesis import propose_hypotheses


def _blind_inputs(case_id="SECRET_U1_RISE_8V", *, target="regulator_output_rise", parameter="8"):
    case = {"case_id": case_id, "llm_case_id": "case_0001",
            "ground_truth": [{"component_ref": "U1", "fault_type": target,
                              "parameters": {"output_voltage": parameter}}],
            "circuit": {"netlist": "/private/REGULATOR_U1_RISE_8V.xml"},
            "evaluation_note": "Known injected fault: regulator output rise to 8 V"}
    components = [{"ref": "U1", "value": "REGULATOR", "library_id": "Power:Linear",
                   "nets": ["VIN", "VOUT", "GND"], "mapping_available": True,
                   "model_type": "regulator_output", "debug_path": "/tmp/U1_rise_8V.cir"}]
    observations = [{"node": "VOUT", "value": 8.0, "unit": "V", "uncertainty": 0.2,
                     "measurement_mode": "final_sample", "source": "synthetic fault U1 rise 8V"}]
    catalog = [{"component_ref": "U1", "fault_type": "regulator_output_rise",
                "parameters": {"output_voltage": "8"}, "nets": ["VOUT"],
                "validation_status": "VALIDATED_EXECUTABLE", "debug_path": "truth_8V"}]
    return case, components, observations, catalog


def _payload():
    case, components, observations, catalog = _blind_inputs()
    return build_blind_diagnostic_input(case, components=components,
                                        observations=observations, executable_catalog=catalog)


def test_answer_bearing_internal_case_id_is_not_sent_to_prompt():
    prompt = json.dumps(_payload())
    assert "SECRET_U1_RISE_8V" not in prompt
    assert "case_0001" in prompt


def test_ground_truth_is_not_sent_to_prompt():
    prompt = json.dumps(_payload())
    assert "ground_truth" not in prompt
    assert "ground_truth" not in _payload()


def test_gold_fault_parameters_are_redacted_from_catalog():
    payload = _payload()
    assert all("parameters" not in candidate for candidate in payload["executable_catalog"])
    assert "8" not in json.dumps(payload["executable_catalog"])


def test_actual_observation_and_public_circuit_details_are_retained():
    payload = _payload()
    assert payload["observations"] == [{"node": "VOUT", "value": 8.0, "unit": "V",
                                        "uncertainty": 0.2, "measurement_mode": "final_sample"}]
    assert payload["components"][0]["ref"] == "U1"
    assert payload["components"][0]["model_type"] == "regulator_output"
    assert "Observed VOUT = 8.0 V" in payload["symptoms"]


def test_different_private_truth_does_not_change_same_public_prompt():
    first = _payload()
    case, components, observations, catalog = _blind_inputs("OTHER_PRIVATE_LABEL", target="resistor_short", parameter="0")
    # Internal IDs, private labels, and evaluation paths differ; the anonymous ID
    # and public diagnostic content are identical.
    second = build_blind_diagnostic_input(case, components=components,
                                         observations=observations, executable_catalog=catalog)
    assert first == second


def test_paths_descriptions_sources_and_debug_fields_do_not_leak():
    prompt = json.dumps(_payload())
    for blocked in ("REGULATOR_U1_RISE_8V", "/private/", "/tmp/", "Known injected fault",
                    "synthetic fault U1 rise 8V", "debug_path", "evaluation_note"):
        assert blocked not in prompt


def test_actual_proposal_client_receives_redacted_candidate_parameters():
    class CapturingClient:
        messages = None

        def chat_json(self, model, messages):
            self.messages = messages
            return {"content": {"hypotheses": []}, "model": model}

    case, components, observations, catalog = _blind_inputs()
    payload = build_blind_diagnostic_input(case, components=components,
                                           observations=observations, executable_catalog=catalog)
    client = CapturingClient()
    propose_hypotheses(client, "local", circuit_id=payload["circuit_id"],
                       components=payload["components"], observations=payload["observations"],
                       symptoms=payload["symptoms"], executable_catalog=catalog,
                       include_candidate_parameters=False)
    prompt = client.messages[1]["content"]
    assert "parameters" not in prompt
    assert "SECRET_U1_RISE_8V" not in prompt
    assert "ground_truth" not in prompt


def test_llm_only_ranking_does_not_inject_missing_truth_candidate():
    cases = [{"case_id": "private-id", "truth_source": "synthetic_fault", "ground_truth": [
        {"component_ref": "R1", "fault_type": "resistance_drift", "parameters": {"fault_resistance": 10000}}
    ]}]
    llm_only = {"private-id": [{"component_ref": "R1", "fault_type": "resistance_drift",
        "parameters": {"fault_resistance": 5000}, "evidence_rank": 1}]}
    assert score_fault_proposals(cases, llm_only)["candidate_recall"] == 0.0
    assert score_fault_rankings(cases, llm_only)["top_k_accuracy"]["1"] == 0.0


def test_empty_fault_evaluation_has_null_metrics_and_no_ranking_claim():
    proposal = score_fault_proposals([], {})
    ranking = score_fault_rankings([], {})
    assert proposal["status"] == "NOT_EVALUATED"
    assert proposal["candidate_precision"] is None
    assert ranking["ranking_coverage"] is None
    assert ranking["top_k_accuracy"]["1"] is None


def _truth():
    return {"document_id": "D1", "manufacturer": "Vishay", "mpn": "MCT 0603",
            "rating_key": "max_power_w", "value": 0.1, "unit": "W",
            "rating_kind": "continuous", "conditions": {"temperature_c": 70},
            "pdf_page_index": 0, "evidence_text": "Rated power 0.1 W at 70 C",
            "evaluation_scope": {"scope_type": "target_rating", "complete_target_scope": True,
                "complete_document_inventory": False, "source_anchor": "Rated power"}}


def _candidate(**kwargs):
    row = {"document_id": "D1", "manufacturer": "Vishay", "mpn": "MCT 0603",
           "rating_key": "max_power_w", "normalized_value": 0.1,
           "normalized_unit": "W", "rating_kind": "continuous",
           "conditions": {"temperature_c": 70}, "pdf_page_index": 0,
           "source_text": "Rated power 0.1 W at 70 C", "evidence_status": "SOURCE_MATCHED"}
    return {**row, **kwargs}


def test_unlabeled_other_rating_is_excluded_from_target_precision():
    unrelated = _candidate(rating_key="max_current_a", normalized_value=0.03, normalized_unit="A",
                           source_text="Current 30 mA", table_context=None)
    result = score_extractions([_truth()], [_candidate(), unrelated])
    target = result["target_rating_evaluation"]
    assert target["candidate_count_in_scope"] == 1
    assert target["precision"]["value"] == 1.0
    assert result["full_document_evaluation"]["status"] == "NOT_EVALUATED"
    assert result["full_document_evaluation"]["precision"]["value"] is None


def test_wrong_unit_and_wrong_model_variant_are_classified_not_called_hallucinations():
    truth = _truth()
    wrong_unit = _candidate(normalized_unit="mW", normalized_value=100)
    assert classify_candidate(wrong_unit, truth) == "INCORRECT_UNIT"
    variant_truth = {**truth, "evaluation_scope": {**truth["evaluation_scope"], "component_variant": "MCT 0603"}}
    other_variant = _candidate(table_context={"variant": "MCS 0402"})
    assert classify_candidate(other_variant, variant_truth) == "WRONG_COMPONENT_VARIANT"
    other_mpn = _candidate(mpn="MCT 0402")
    assert classify_candidate(other_mpn, truth) == "WRONG_COMPONENT_VARIANT"
    assert score_extractions([truth], [_candidate(evidence_status="SOURCE_NOT_FOUND")])["hallucination_rate"]["status"] == "NOT_EVALUATED"


def test_source_check_rejection_is_separate_from_hallucination():
    result = score_extractions([_truth()], [_candidate(evidence_status="SOURCE_NOT_FOUND")])
    assert result["source_verification_rejection_count"] == 1
    assert result["source_verification_rejection_rate"]["value"] == 1.0
    assert result["hallucination_rate"]["value"] is None


def test_hallucination_rate_requires_explicit_human_adjudication():
    candidate = _candidate(human_adjudication={"review_status": "reviewed",
                                               "error_class": "FABRICATED_SOURCE"})
    assert classify_candidate(candidate, _truth()) == "FABRICATED_SOURCE"
    result = score_extractions([_truth()], [candidate])
    assert result["hallucination_rate"]["status"] == "MEASURED_HUMAN_ADJUDICATION"
    assert result["hallucination_rate"]["value"] == 1.0


def test_complete_inventory_metrics_only_run_when_all_labels_declare_complete_inventory():
    truth = _truth()
    truth["evaluation_scope"]["complete_document_inventory"] = True
    result = score_extractions([truth], [_candidate(), _candidate(rating_key="max_current_a")],
                               full_document_ground_truth_complete=True)
    full = result["full_document_evaluation"]
    assert full["status"] == "MEASURED_COMPLETE_INVENTORY"
    assert full["true_positive_count"] == 1
    assert full["false_positive_count"] == 1
