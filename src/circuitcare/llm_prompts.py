"""Constrained prompt construction for evidence-grounded local explanations."""

import json
import re
from typing import Any

SYSTEM_PROMPT = """You are CircuitCare AI, an explanation layer over deterministic circuit evidence. Answer in Korean.
Rules:
- Use only the supplied evidence. Never recalculate, alter, or invent voltage, current, power, ratings, posterior, risk score, or information gain.
- Distinguish measured/simulated facts, assumptions, recommendations, and unknowns.
- A Bayesian posterior is conditional on the configured hypothesis set and stated measurement uncertainty; it is never a real-world failure probability. If the hypotheses have equal posterior, do not present their display order as a winning diagnosis.
- A simulated_demo observation is not a real instrument reading. A model_mismatch means no configured scenario explains the measurements well; do not make a decisive diagnosis.
- A priority index is not a damage probability. Rating exceedance is a stress finding, not proof of damage. Never say a component is confirmed damaged or safe based only on this evidence.
- Explicitly state that damage is not confirmed. If a rating source is demo_assumption, explicitly call it a demonstration assumption and say it is not manufacturer-verified.
- When any cited fact uses a demo_assumption rating, include an assumptions item that says the rating is a demo assumption and is not manufacturer-verified (Korean: "데모 가정이며 제조사 확인이 되지 않은 값").
- Do not provide live probing instructions or claim that a measurement is safe. Recommend power isolation, stored-energy checks, and qualified review where appropriate.
- Cite evidence for each factual statement in the answer and facts array using bracketed IDs such as [E001]. Only use supplied evidence_id values. Do not invent IDs.
- Return only a JSON object with keys: answer (string), facts (array of {text, evidence_ids}), assumptions (array of {text, evidence_ids}), recommendations (array of {text, evidence_ids}), unknowns (array of {text, evidence_ids}). Every fact must cite at least one valid ID. Use [] when a general safety recommendation has no numeric evidence.
- Keep the answer brief (under 120 Korean words). Include only 2-4 key facts and at most 3 items in each remaining array; do not copy the evidence record.
Example answer format: {"answer":"C1 우선 확인 근거는 [E011][E012][E013]입니다.","facts":[{"text":"...","evidence_ids":["E011","E012","E013"]}],"assumptions":[],"recommendations":[],"unknowns":[]}
"""


def build_messages(question: str, evidence: dict[str, Any], repair_advice: dict[str, Any]) -> list[dict[str, str]]:
    references = set(re.findall(r"(?<![A-Z0-9])[A-Z][A-Z0-9]*\d+(?![A-Z0-9])", question.upper()))
    facts = evidence.get("facts", [])
    if references:
        facts = [item for item in facts if item.get("component_ref") in references]
        # Preserve the active hypothesis and measurement facts that explain the
        # selected component's scenario, while avoiding unrelated component rows.
        facts += [
            item for item in evidence.get("facts", [])
            if item.get("metric") in {"selected_fault_hypothesis", "conditional_posterior", "observed_voltage", "simulated_prediction_for_measurement"}
        ]
        facts = list({item.get("evidence_id"): item for item in facts}.values())
    else:
        facts = evidence.get("facts", [])[:32]
    checks = repair_advice.get("recommended_checks", [])
    if references:
        relevant_checks = [item for item in checks if item.get("component_ref") in references]
    elif any(word in question.casefold() for word in ("측정", "점검", "수리", "measure", "inspect", "repair")):
        relevant_checks = checks
    else:
        relevant_checks = []
    unknowns = evidence.get("unknowns", [])
    if references:
        unknowns = [item for item in unknowns if any(reference in item for reference in references)] + [
            item for item in unknowns if any(token in item.casefold() for token in ("damage", "probability", "model", "손상", "고장 확률", "불일치"))
        ]
        unknowns = list(dict.fromkeys(unknowns))
    context = evidence.get("analysis_context", {})
    inspection_order = context.get("inspection_order", [])
    if references:
        inspection_order = [item for item in inspection_order if item.get("component_ref") in references]
    else:
        inspection_order = inspection_order[:3]
    relevant_fact_ids = {item.get("evidence_id") for item in facts}
    assumptions = [
        item for item in evidence.get("assumptions", [])
        if not item.get("evidence_ids") or any(identifier in relevant_fact_ids for identifier in item.get("evidence_ids", []))
    ][:8]
    payload = {
        "user_question": question,
        "evidence": {
            **evidence,
            "facts": facts,
            "assumptions": assumptions,
            "recommendations": relevant_checks,
            "unknowns": unknowns,
            "analysis_context": {
                key: value for key, value in context.items()
                if key not in {"inspection_order", "source_files", "measurement_recommendation"}
            } | {"inspection_order": inspection_order, "measurement_recommendation": context.get("measurement_recommendation")},
        },
        "deterministic_repair_advice": {
            "failed_component": repair_advice.get("failed_component"),
            "recommended_checks": relevant_checks,
            "initial_safety_checks": repair_advice.get("initial_safety_checks", []),
        },
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]
