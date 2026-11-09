import json
from pathlib import Path

from circuitcare.datasheet.evaluation import score_extractions
from circuitcare.fault_ai.evaluation import score_fault_cases, score_fault_proposals, score_fault_rankings
from circuitcare.fault_ai.fault_catalog import generate_rule_candidates
from circuitcare.fault_ai.hypothesis_ranking import conditional_bayesian_update, rank_hypotheses
from circuitcare.fault_ai.llm_hypothesis import propose_hypotheses
from circuitcare.fault_ai.observation_matcher import compare_observations
from circuitcare.fault_ai.hypothesis_validator import validate_hypotheses
from circuitcare.fault_ai.simulation_bridge import preflight_candidates, simulate_hypothesis
from circuitcare.local_llm import OllamaClient
from circuitcare.netlist import load_netlist

ROOT = Path(__file__).resolve().parents[1]


def _demo():
    circuit = load_netlist(ROOT / "tests/fixtures/fault_demo_netlist.xml")
    mapping = json.loads((ROOT / "data/circuits/fault_demo_mapping.json").read_text())
    spice = (ROOT / "data/circuits/fault_demo.cir").read_text()
    return circuit, mapping, spice


def test_catalog_only_generates_preflighted_types_for_mapped_refs():
    circuit, mapping, spice = _demo()
    raw = generate_rule_candidates(circuit, mapping, spice)
    rows = preflight_candidates(raw, component_refs=set(circuit.components), component_mapping=mapping, spice_text=spice)
    assert rows
    assert all(row["validation_status"] == "VALIDATED_EXECUTABLE" for row in rows)
    assert {row["fault_type"] for row in rows} >= {"resistor_open", "regulator_output_rise", "stuck_open"}
    assert "C1" not in {row["component_ref"] for row in rows}


def test_validator_rejects_hallucinated_reference_type_and_parameter():
    circuit, mapping, spice = _demo()
    catalog = preflight_candidates(generate_rule_candidates(circuit, mapping, spice),
        component_refs=set(circuit.components), component_mapping=mapping, spice_text=spice)
    hallucinated = [
        {"component_ref": "U999", "fault_type": "regulator_output_rise", "parameters": {"output_voltage": "8"}},
        {"component_ref": "C1", "fault_type": "capacitor_short", "parameters": {}},
        {"component_ref": "U1", "fault_type": "regulator_output_rise", "parameters": {"output_voltage": "999"}},
    ]
    rows = validate_hypotheses(hallucinated, component_refs=set(circuit.components), component_mapping=mapping,
                               spice_text=spice, allowed_catalog=catalog)
    assert [row["validation_status"] for row in rows] == ["INVALID_REFERENCE", "UNSUPPORTED_FAULT_TYPE", "INSUFFICIENT_EVIDENCE"]


def test_llm_output_is_limited_to_exact_catalog_combinations_and_deduplicated():
    circuit, mapping, spice = _demo()
    catalog = preflight_candidates(generate_rule_candidates(circuit, mapping, spice),
        component_refs=set(circuit.components), component_mapping=mapping, spice_text=spice)
    allowed = next(row for row in catalog if row["component_ref"] == "U1" and row["fault_type"] == "regulator_output_rise")

    class Fake:
        def chat_json(self, model, messages):
            prompt = messages[1]["content"]
            assert "executable_catalog" in prompt and "SPICE" in messages[0]["content"]
            return {"content": {"hypotheses": [
                {"component_ref": "U1", "fault_type": allowed["fault_type"], "parameters": {"output_voltage": float(allowed["parameters"]["output_voltage"])}, "reason": "VOUT symptom", "supporting_observations": ["VOUT"]},
                {"component_ref": "U999", "fault_type": "resistor_open", "parameters": {"fault_resistance": "1T"}, "reason": "invented"},
            ]}, "model": model, "model_info": {"name": model}}

    raw = propose_hypotheses(Fake(), "fake-local", circuit_id="demo", components=[], observations=[], symptoms="VOUT high", executable_catalog=catalog)
    rows = validate_hypotheses(raw["hypotheses"], component_refs=set(circuit.components), component_mapping=mapping,
                                spice_text=spice, allowed_catalog=catalog)
    assert rows[0]["validation_status"] == "VALIDATED_EXECUTABLE"
    assert rows[1]["validation_status"] == "INVALID_REFERENCE"


def test_observation_matcher_checks_unit_sigma_and_incomplete_nodes():
    result = compare_observations({"VOUT": 8.0}, [
        {"node": "VOUT", "value": 7.9, "unit": "V", "uncertainty": 0.2, "source": "actual_meter"},
        {"node": "MISSING", "value": 1, "unit": "V", "uncertainty": 0.1},
        {"node": "VOUT", "value": 1, "unit": "mV", "uncertainty": 0.1},
    ])
    assert result["status"] == "CONSISTENT_WITH_OBSERVATIONS"
    assert abs(result["comparisons"][0]["normalized_residual"] + 0.5) < 1e-9
    assert len(result["unsupported_measurements"]) == 2
    assert compare_observations({"VOUT": 1}, [{"node": "VOUT", "value": 1, "unit": "V"}])["status"] == "INSUFFICIENT_MEASUREMENTS"


def test_unrun_hypothesis_never_gets_observation_rank():
    rows = rank_hypotheses([
        {"hypothesis_id": "sim", "simulation_status": "SIMULATION_SUCCESS", "observation_match": {"max_absolute_normalized_residual": 0.5}},
        {"hypothesis_id": "not-run", "simulation_status": "SIMULATION_NOT_RUN"},
    ])
    assert rows[0]["hypothesis_id"] == "sim" and rows[0]["evidence_rank"] == 1
    assert rows[1]["observation_status"] == "NOT_SIMULATED" and rows[1]["evidence_rank"] is None


def test_bayesian_reuse_requires_two_simulations_and_explicit_uncertainty():
    predictions = [
        {"hypothesis_id": "h1", "component_ref": "U1", "fault_type": "rise", "parameters": {}, "simulation_status": "SIMULATION_SUCCESS", "predicted_measurements": {"VOUT": 8.0}},
        {"hypothesis_id": "h2", "component_ref": "U1", "fault_type": "drop", "parameters": {}, "simulation_status": "SIMULATION_SUCCESS", "predicted_measurements": {"VOUT": 3.0}},
    ]
    obs = [{"node": "VOUT", "value": 7.9, "unit": "V", "uncertainty": 0.2, "source": "actual_meter"}]
    result = conditional_bayesian_update(predictions, obs)
    assert result["status"] == "ACTIVE_CONDITIONAL"
    assert result["posterior"][0]["hypothesis_id"] == "h1"
    assert "not real-world" in result["probability_interpretation"]
    assert conditional_bayesian_update(predictions, [{**obs[0], "uncertainty": None}])["status"] == "INACTIVE"


def test_evaluation_metrics_do_not_turn_empty_set_into_perfect_accuracy():
    assert score_fault_proposals([], {})["candidate_precision"] is None
    assert score_fault_rankings([], {})["top_k_accuracy"]["1"] is None
    assert score_extractions([], [])["status"] == "NOT_EVALUATED"


def test_synthetic_fault_metric_reports_case_recall_and_top_k():
    cases = [{"case_id": "c1", "truth_source": "synthetic_fault", "ground_truth": [{"component_ref": "R1", "fault_type": "resistance_drift", "parameters": {"fault_resistance": "10000"}}]}]
    outputs = {"c1": [
        {"component_ref": "R1", "fault_type": "resistance_drift", "parameters": {"fault_resistance": 10000.0}, "validation_status": "VALIDATED_EXECUTABLE", "evidence_rank": 1},
    ]}
    proposal = score_fault_proposals(cases, outputs)
    ranking = score_fault_rankings(cases, outputs)
    assert proposal["status"] == "MEASURED_SYNTHETIC"
    assert proposal["candidate_precision"] == proposal["candidate_recall"] == 1.0
    assert ranking["ranking_coverage"] == 1.0
    assert ranking["top_k_accuracy"]["1"] == 1.0


def test_datasheet_metrics_score_units_conditions_and_evidence_separately():
    truth = [{"document_id": "D1", "rating_key": "max_power_w", "value": 0.1, "unit": "W", "rating_kind": "continuous",
              "conditions": {"temperature_c": 70}, "pdf_page_index": 0, "evidence_text": "Power 0.1 W",
              "evaluation_scope": {"scope_type": "target_rating", "complete_target_scope": True,
                                   "complete_document_inventory": False, "source_anchor": "Power"}}]
    candidate = {"document_id": "D1", "rating_key": "max_power_w", "normalized_value": 0.1, "normalized_unit": "W", "rating_kind": "continuous",
                 "conditions": {"temperature_c": 70}, "pdf_page_index": 0, "source_text": "Power 0.1 W", "evidence_status": "SOURCE_MATCHED"}
    metrics = score_extractions(truth, [candidate])
    target = metrics["target_rating_evaluation"]
    assert target["precision"]["value"] == target["recall"]["value"] == 1.0
    assert target["condition_accuracy"]["value"] == target["evidence_accuracy"]["value"] == 1.0
    assert target["table_context_accuracy"]["value"] is None
    assert metrics["full_document_evaluation"]["status"] == "NOT_EVALUATED"
    assert metrics["hallucination_rate"]["value"] is None


def test_datasheet_table_context_is_scored_separately_from_value_and_quote():
    truth = [{"document_id": "D1", "rating_key": "max_power_w", "value": 0.1, "unit": "W", "rating_kind": "continuous",
              "conditions": {"temperature_c": 70}, "pdf_page_index": 1, "evidence_text": "P70 MCT 0603 0.100 W",
              "evaluation_scope": {"scope_type": "target_rating", "complete_target_scope": True,
                                   "complete_document_inventory": False, "component_variant": "MCT 0603",
                                   "row_label": "Rated dissipation, P70", "column_header": "STANDARD", "allow_blank_column": True},
              "table_context": {"row_label": "Rated dissipation, P70", "variant": "MCT 0603",
                                "column_header": "STANDARD", "cell_text": "0.100 W"}}]
    candidate = {"document_id": "D1", "rating_key": "max_power_w", "normalized_value": 0.1, "normalized_unit": "W", "rating_kind": "continuous",
                 "conditions": {"temperature_c": 70}, "pdf_page_index": 1, "source_text": "P70 MCT 0603 0.100 W", "evidence_status": "SOURCE_MATCHED",
                 "table_context": {"row_label": "Rated dissipation, P70", "variant": "MCT 0603",
                                   "column_header": "STANDARD", "cell_text": "0.100 W"}}
    assert score_extractions(truth, [candidate])["target_rating_evaluation"]["table_context_accuracy"]["value"] == 1.0
    candidate["table_context"]["variant"] = "MCT 0204"
    res = score_extractions(truth, [candidate])
    assert res["wrong_component_variant_count"] == 1
    assert res["target_rating_evaluation"]["precision"]["value"] is None


def test_simulation_bridge_never_runs_unvalidated_hypothesis(tmp_path):
    row = simulate_hypothesis({"component_ref": "U999", "fault_type": "anything", "validation_status": "INVALID_REFERENCE"},
        base_netlist=tmp_path / "not-read.cir", component_mapping={}, probes={}, observations=[])
    assert row["simulation_status"] == "UNSUPPORTED_HYPOTHESIS"
