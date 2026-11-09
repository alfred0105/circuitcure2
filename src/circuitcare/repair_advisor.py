"""Deterministic, safety-conscious inspection guidance independent of an LLM."""

from typing import Any


def build_repair_advice(evidence: dict[str, Any], question: str) -> dict[str, Any]:
    context = evidence["analysis_context"]
    candidates = context.get("inspection_order", [])
    facts = evidence.get("facts", [])
    checks = []
    for item in candidates:
        reference = item["component_ref"]
        risk_level = item.get("risk_level", "UNKNOWN")
        evidence_ids = [fact["evidence_id"] for fact in facts if fact.get("component_ref") == reference]
        if risk_level == "EXCEEDED" or item.get("any_hypothesis_exceeded"):
            checks.append({
                "action": f"Review the scenario-specific stress evidence for {reference}, verify its actual manufacturer rating, and inspect it while de-energized after isolation and stored-energy checks.",
                "component_ref": reference,
                "action_type": "rating_verification_and_unpowered_inspection",
                "basis": "At least one configured fault hypothesis has simulated stress above its configured rating; scenario likelihood and actual damage are not confirmed.",
                "evidence_ids": evidence_ids,
            })
        elif risk_level == "NEAR_LIMIT":
            checks.append({
                "action": f"Verify {reference}'s real part number and rating, then inspect it de-energized.",
                "component_ref": reference,
                "action_type": "rating_verification_and_unpowered_inspection",
                "basis": "Simulated stress is near a configured limit; damage is not confirmed.",
                "evidence_ids": evidence_ids,
            })
        elif risk_level == "UNKNOWN":
            checks.append({
                "action": f"Identify {reference} and obtain its actual ratings before drawing a safety conclusion.",
                "component_ref": reference,
                "action_type": "identify_and_verify_rating",
                "basis": "The current data are insufficient to classify this component as within limits.",
                "evidence_ids": evidence_ids,
            })
    if not checks:
        checks.append({
            "action": "Review the simulation assumptions and verify the fitted component part numbers and ratings.",
            "component_ref": None,
            "action_type": "documentation_check",
            "basis": "No candidate has enough evidence for a specific condition claim.",
            "evidence_ids": [],
        })

    measurement = context.get("measurement_recommendation")
    if measurement:
        checks.append({
            "action": f"Consider {measurement['node']} as the next information-ranked measurement location only after a qualified safety review and controlled test setup.",
            "component_ref": None,
            "action_type": "measurement_candidate_not_safety_approval",
            "basis": measurement.get("reason"),
            "expected_information_gain_nats": measurement.get("expected_information_gain_nats"),
            "evidence_ids": [fact["evidence_id"] for fact in facts if fact.get("metric") == "next_measurement_expected_information_gain"],
        })

    return {
        "failed_component": context.get("failed_component"),
        "fault_type": context.get("fault_type"),
        "scenario_id": context.get("scenario_id"),
        "user_question": question,
        "assessment_type": "deterministic_evidence_based_repair_advice",
        "inspection_priority": candidates,
        "recommended_checks": checks,
        "initial_safety_checks": [
            "Disconnect and isolate all power sources before inspection.",
            "Allow stored energy to discharge and verify the de-energized state using appropriately rated procedures and equipment.",
            "Do not probe a damaged energized circuit based solely on this software recommendation; seek a qualified person if safe isolation or measurement conditions are uncertain.",
        ],
        "post_repair_checks": [
            "Verify replacement part number, rating, polarity, pinout, and circuit compatibility against the design and manufacturer documentation.",
            "With power isolated, check assembly, solder bridges, and relevant continuity before re-energizing.",
            "Any controlled re-energization and functional measurements require an appropriately qualified person and suitable current-limited lab procedure.",
        ],
        "unconfirmed": evidence.get("unknowns", []),
        "damage_confirmed": False,
        "damage_probability": None,
        "priority_score_interpretation": "Inspection order only; not a damage probability.",
        "llm_status": "not_run",
    }
