import copy
import json

import pytest

from circuitcare.datasheet.evidence_validator import validate_candidate
from circuitcare.datasheet.llm_extractor import extract_with_local_llm
from circuitcare.datasheet.pdf_loader import DatasheetError, load_pdf
from circuitcare.datasheet.rating_adapter import (
    RatingApplicationError,
    apply_approved_rating,
    reassess_existing_risk,
)
from circuitcare.datasheet.rating_extractor import find_rating_candidates
from circuitcare.datasheet.rating_review import find_application, load_reviews, save_application, save_review


def test_rule_extraction_keeps_power_and_voltage_categories_separate():
    pages = [{"page_index": 0, "text": "Rated Power\n125 mW at 70 C\nMaximum Continuous Current\n500 mA"}]
    rows = find_rating_candidates(pages)
    assert {(row["rating_key"], row["normalized_value"], row["normalized_unit"]) for row in rows} == {
        ("max_power_w", 0.125, "W"), ("max_current_a", 0.5, "A")
    }


def test_negative_voltage_sign_is_preserved_but_engine_limit_is_magnitude():
    row = find_rating_candidates([{"page_index": 0, "text": "Reverse Voltage\n-3.3 V continuous"}])[0]
    assert row["raw_value"] == "-3.3 V"
    assert row["normalized_value"] == 3.3
    assert row["conditions"]["polarity"] == "negative"


def test_evidence_validator_requires_cited_page_and_exact_number_unit():
    pages = [{"page_index": 0, "text": "YAGEO Rated Power 0.125 W"}, {"page_index": 1, "text": "YAGEO Maximum Power 0.25 W"}]
    candidate = {"rating_key": "max_power_w", "normalized_value": 0.125, "normalized_unit": "W",
                 "source_text": "Rated Power 0.125 W", "pdf_page_index": 0}
    assert validate_candidate(candidate, pages, manufacturer="YAGEO", mpn="PART-1")["evidence_status"] == "SOURCE_MATCHED"
    assert validate_candidate({**candidate, "pdf_page_index": 1}, pages, manufacturer="YAGEO", mpn="PART-1")["evidence_status"] == "SOURCE_NOT_FOUND"
    assert validate_candidate({**candidate, "normalized_value": 12.5}, pages, manufacturer="YAGEO", mpn="PART-1")["evidence_status"] == "SOURCE_MISMATCH"
    assert validate_candidate({**candidate, "pdf_page_index": 4}, pages, manufacturer="YAGEO", mpn="PART-1")["evidence_status"] == "SOURCE_MISMATCH"
    incomplete = validate_candidate({**candidate, "source_text": "Power 0.125 W"}, pages + [{"page_index": 2, "text": "YAGEO Power 0.125 W"}], manufacturer="YAGEO", mpn="PART-1")
    assert incomplete["evidence_status"] == "CONDITION_INCOMPLETE"
    ranged_pages = [{"page_index": 0, "text": "YAGEO Rated Voltage 10 V to 20 V continuous"}]
    ranged = validate_candidate({"rating_key": "max_abs_voltage_v", "normalized_value": 10, "normalized_unit": "V",
        "source_text": "Rated Voltage 10 V to 20 V continuous", "pdf_page_index": 0}, ranged_pages,
        manufacturer="YAGEO", mpn="PART-1")
    assert ranged["evidence_status"] == "SOURCE_AMBIGUOUS"


def test_llm_candidate_is_only_a_proposal_and_local_json_is_validated():
    class FakeClient:
        def chat_json(self, model, messages):
            assert "untrusted data" in messages[0]["content"]
            request = json.loads(messages[1]["content"])
            assert request["pages"][0]["tables"] == [{"page_index": 0, "rows": [["Power", "125 mW"]]}]
            return {"content": {"ratings": [{"rating_key": "max_power_w", "rating_type": "power_rating",
                "rating_kind": "continuous", "raw_value": "125 mW", "normalized_value": "0.125",
                "normalized_unit": "W", "pdf_page_index": "0", "source_text": "Rated Power 125 mW"}]},
                "model": model, "model_info": {"name": model}, "total_duration_ns": 123}
    result = extract_with_local_llm(FakeClient(), "local-test", manufacturer="YAGEO", mpn="MPN",
        candidates=[{"pdf_page_index": 0, "rating_key": "max_power_w"}],
        pages=[{"page_index": 0, "text": "Rated Power 125 mW"}],
        tables=[{"page_index": 0, "rows": [["Power", "125 mW"]]}])
    assert result["ratings"][0]["human_approved"] is False
    assert result["ratings"][0]["extraction_method"] == "local_ollama"


def _candidate(**overrides):
    return {"rating_key": "max_power_w", "rating_type": "power_rating", "rating_kind": "continuous",
        "raw_value": "0.125 W", "normalized_value": 0.125, "normalized_unit": "W",
        "evidence_status": "SOURCE_MATCHED", "source_document_sha256": "abc123", "source_text": "Rated Power 0.125 W",
        "pdf_page_index": 0, "manufacturer": "YAGEO", "mpn": "RC0805FR-071KL",
        "human_approved": True, "review_decision": "approved", **overrides}


def test_adapter_only_applies_approved_matched_conditioned_rating_and_preserves_other_source():
    original = {"components": {"RLOAD": {"ratings": {"max_power_w": 0.2}, "rating_types": {"max_power_w": "continuous"},
        "source": {"type": "demo_assumption", "note": "keep this"}}}}
    applied = apply_approved_rating(original, _candidate(), component_ref="RLOAD", selected_reference="RLOAD",
        mpn_match_status="MPN_CONFIRMED", conditions_compatible=True)
    assert original["components"]["RLOAD"]["ratings"]["max_power_w"] == 0.2
    assert applied["components"]["RLOAD"]["ratings"]["max_power_w"] == 0.125
    assert applied["components"]["RLOAD"]["source"]["type"] == "demo_assumption"
    assert applied["components"]["RLOAD"]["rating_sources"]["max_power_w"]["type"] == "datasheet_user_approved"
    assert applied["components"]["RLOAD"]["rating_types"]["max_power_w"] == "continuous"


@pytest.mark.parametrize("changes", [
    {"human_approved": False}, {"evidence_status": "SOURCE_NOT_FOUND"}, {"normalized_unit": "V"},
    {"mpn": ""}, {"rating_type": "voltage_rating"},
])
def test_adapter_rejects_unverified_or_unsupported_rating(changes):
    with pytest.raises(RatingApplicationError):
        apply_approved_rating({"components": {}}, _candidate(**changes), component_ref="RLOAD",
            selected_reference="RLOAD", mpn_match_status="MPN_CONFIRMED", conditions_compatible=True)


def test_adapter_requires_reference_and_condition_compatibility():
    with pytest.raises(RatingApplicationError, match="Reference"):
        apply_approved_rating({"components": {}}, _candidate(), component_ref="R1", selected_reference="RLOAD",
            mpn_match_status="MPN_CONFIRMED", conditions_compatible=True)
    with pytest.raises(RatingApplicationError, match="MISSING_CONDITION"):
        apply_approved_rating({"components": {}}, _candidate(), component_ref="RLOAD", selected_reference="RLOAD",
            mpn_match_status="MPN_CONFIRMED", conditions_compatible=False)
    with pytest.raises(RatingApplicationError, match="CONDITION_INCOMPLETE"):
        apply_approved_rating({"components": {}}, _candidate(evidence_status="CONDITION_INCOMPLETE"),
            component_ref="RLOAD", selected_reference="RLOAD", mpn_match_status="MPN_CONFIRMED", conditions_compatible=True)


def test_rating_only_reassessment_reuses_immutable_stress_and_changes_risk():
    risk = {"failed_component": "U1", "scenario_id": "s1", "secondary_candidates": [{"component_ref": "RLOAD",
        "risk_level": "EXCEEDED", "priority_score": 98.0, "priority_rank": 1, "dominant_metric": "power",
        "metrics": {}, "connectivity_paths": [{"hops": 1, "nets": ["LOAD"]}], "shared_nets": ["LOAD"]}]}
    stress = {"simulation_success": True, "components": [{"component_ref": "RLOAD", "status": "simulated",
        "fault": {"power_w": {"minimum": 0.02, "maximum": 0.02}},
        "baseline": {"power_w": {"minimum": 0.01, "maximum": 0.01}}}]}
    before = json.dumps(stress, sort_keys=True)
    specs = {"components": {"RLOAD": {"ratings": {"max_power_w": 0.125}, "rating_types": {"max_power_w": "continuous"},
        "rating_sources": {"max_power_w": {"type": "datasheet_user_approved", "document_sha256": "abc"}}}}}
    updated = reassess_existing_risk(risk, stress, specs, {"RLOAD": {"model_type": "resistor"}})
    assert updated["secondary_candidates"][0]["risk_level"] == "WITHIN_LIMIT"
    assert updated["secondary_candidates"][0]["metrics"]["max_power_w"]["rated_limit"] == 0.125
    assert updated["simulation_rerun"] is False
    assert json.dumps(stress, sort_keys=True) == before
    assert risk["secondary_candidates"][0]["risk_level"] == "EXCEEDED"


def test_absolute_maximum_is_not_collapsed_into_continuous_rating():
    applied = apply_approved_rating({"components": {}}, _candidate(rating_kind="absolute_max"),
        component_ref="RLOAD", selected_reference="RLOAD", mpn_match_status="MPN_CONFIRMED", conditions_compatible=True)
    assert applied["components"]["RLOAD"]["rating_types"]["max_power_w"] == "unknown"


def test_pdf_input_validation_rejects_empty_and_non_pdf():
    with pytest.raises(DatasheetError):
        load_pdf(b"", "empty.pdf")
    with pytest.raises(DatasheetError):
        load_pdf(b"not a pdf", "bad.pdf")


def test_pdf_page_limit_is_enforced_without_downloading_documents():
    pypdf = pytest.importorskip("pypdf")
    from io import BytesIO
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_blank_page(width=100, height=100)
    buffer = BytesIO()
    writer.write(buffer)
    with pytest.raises(DatasheetError, match="processing limit"):
        load_pdf(buffer.getvalue(), "large.pdf", max_pages=1)


def test_pdf_scanned_and_encrypted_states_are_explicit():
    pypdf = pytest.importorskip("pypdf")
    from io import BytesIO
    blank = pypdf.PdfWriter()
    blank.add_blank_page(width=100, height=100)
    buffer = BytesIO()
    blank.write(buffer)
    assert load_pdf(buffer.getvalue(), "scan-like.pdf")["text_status"] == "OCR_REQUIRED"

    locked = pypdf.PdfWriter()
    locked.add_blank_page(width=100, height=100)
    locked.encrypt("test-password")
    locked_buffer = BytesIO()
    locked.write(locked_buffer)
    with pytest.raises(DatasheetError, match="Encrypted"):
        load_pdf(locked_buffer.getvalue(), "locked.pdf")


def test_pdf_upload_size_limit_is_checked():
    with pytest.raises(DatasheetError, match="size limit"):
        load_pdf(b"%PDF-" + (b"x" * 20), "large.pdf", max_bytes=10)


def test_review_and_applied_rating_records_are_append_only_and_context_scoped(tmp_path):
    history = tmp_path / "review_history.json"
    first = save_review({**_candidate(), "decision": "approved"}, history)
    save_review({**_candidate(), "decision": "rejected"}, history)
    assert len(json.loads(history.read_text())["records"]) == 2
    assert first["review_source"] == "user_review"
    assert first["human_approved"] is True
    approved = save_review({**_candidate(), "component_reference": "RLOAD", "document_revision": "Rev A",
                            "original_candidate": {"rating_key": "max_power_w", "pdf_page_index": 0, "source_text": "Rated Power 0.125 W"},
                            "decision": "approved"}, history)
    assert load_reviews(history, sha256="abc123", mpn="RC0805FR-071KL", reference="RLOAD", revision="Rev A",
                        extraction_model=None, model_digest=None) == [approved]
    assert load_reviews(history, sha256="abc123", mpn="RC0805FR-071KL", reference="R1", revision="Rev A") == []

    applications = tmp_path / "applied.json"
    save_application({"record": {**first, "component_reference": "RLOAD"}, "component_ref": "RLOAD",
                      "circuit_path": "demo.xml", "scenario_id": "s1", "stress_signature": "abc"}, applications)
    assert find_application(applications, circuit_path="demo.xml", scenario_id="s1", stress_sha256="abc") is not None
    assert find_application(applications, circuit_path="other.xml", scenario_id="s1", stress_sha256="abc") is None
    save_application({"event": "revoked", "component_ref": "RLOAD", "circuit_path": "demo.xml",
                      "scenario_id": "s1", "stress_signature": "abc"}, applications)
    assert find_application(applications, circuit_path="demo.xml", scenario_id="s1", stress_sha256="abc") is None
