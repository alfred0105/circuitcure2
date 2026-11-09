"""CLI for deterministic repair advice with optional local Ollama explanation."""

import argparse
import json
from pathlib import Path

from .conversation import answer_question
from .evidence_builder import build_evidence, display_path, resolve_related_evidence
from .local_llm import LocalLLMError, OllamaClient
from .repair_advisor import build_repair_advice


def main() -> None:
    parser = argparse.ArgumentParser(description="CircuitCare local conversational repair advisor")
    parser.add_argument("--diagnosis", required=True, type=Path, help="Phase 07 diagnosis.json")
    parser.add_argument("--inspection-ranking", required=True, type=Path, help="Phase 07 inspection_ranking.json")
    parser.add_argument("--measurement-recommendations", required=True, type=Path, help="Phase 07 measurement_recommendations.json")
    parser.add_argument("--risk-assessment", type=Path, help="Optional selected scenario risk_assessment.json")
    parser.add_argument("--component-stress", type=Path, help="Optional selected scenario component_stress.json")
    parser.add_argument("--model", default="qwen3:4b-instruct", help="Installed local Ollama model name")
    parser.add_argument("--question", required=True, help="User's natural-language circuit question")
    parser.add_argument("--output", required=True, type=Path, help="Directory for repair advice and explanation JSON")
    parser.add_argument("--no-llm", action="store_true", help="Skip Ollama and write deterministic advice only")
    args = parser.parse_args()

    try:
        diagnosis = _load(args.diagnosis)
        inspection = _load(args.inspection_ranking)
        recommendations = _load(args.measurement_recommendations)
        risk, stress, risk_path, stress_path = resolve_related_evidence(
            args.diagnosis,
            str(args.risk_assessment) if args.risk_assessment else None,
            str(args.component_stress) if args.component_stress else None,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    evidence = build_evidence(
        diagnosis,
        inspection,
        recommendations,
        risk,
        stress,
        source_files={
            "diagnosis": str(args.diagnosis),
            "inspection_ranking": str(args.inspection_ranking),
            "measurement_recommendations": str(args.measurement_recommendations),
            "risk_assessment": display_path(risk_path),
            "component_stress": display_path(stress_path),
        },
    )
    advice = build_repair_advice(evidence, args.question)
    evidence["recommendations"] = advice["recommended_checks"]
    args.output.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output / "evidence.json"
    advice_path = args.output / "repair_advice.json"
    llm_path = args.output / "llm_explanation.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    if args.no_llm:
        llm_result = {
            "status": "skipped",
            "model": args.model,
            "reason": "The --no-llm option was selected.",
            "fallback_used": True,
            "fallback_explanation": _fallback_text(evidence),
        }
    else:
        try:
            client = OllamaClient()
            llm_result = answer_question(client, args.model, args.question, evidence, advice)
            llm_result["fallback_used"] = llm_result["status"] != "success"
            if llm_result["fallback_used"]:
                llm_result["fallback_explanation"] = _fallback_text(evidence)
        except (LocalLLMError, ValueError) as exc:
            llm_result = {
                "status": "failed",
                "model": args.model,
                "error": str(exc),
                "fallback_used": True,
                "fallback_explanation": _fallback_text(evidence),
            }
    advice["llm_status"] = llm_result["status"]
    advice["llm_explanation_file"] = str(llm_path)
    advice["evidence_file"] = str(evidence_path)
    advice_path.write_text(json.dumps(advice, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    llm_path.write_text(json.dumps(llm_result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "repair_advice_file": str(advice_path),
        "evidence_file": str(evidence_path),
        "llm_explanation_file": str(llm_path),
        "llm_status": llm_result["status"],
        "fallback_used": llm_result["fallback_used"],
        "model": llm_result.get("model", args.model),
    }, indent=2, ensure_ascii=False))


def _load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _fallback_text(evidence: dict) -> str:
    context = evidence["analysis_context"]
    order = context.get("inspection_order", [])
    top = order[0] if order else None
    if context.get("model_mismatch"):
        return "현재 측정값을 설정된 시뮬레이션 가설이 충분히 설명하지 못합니다. 최상위 가설만으로 결론을 내리지 말고 회로 모델과 입력값을 다시 확인하세요."
    if top:
        return f"결정론적 분석에서 {top['component_ref']}는 우선 점검 후보입니다({top.get('risk_level')}). 실제 손상 여부는 확인되지 않았으며, 수치와 가정은 evidence.json을 참조하세요."
    return "분석 결과만으로 특정 부품의 상태를 분류하기 어렵습니다. UNKNOWN 항목의 정격과 측정 근거를 먼저 확인하세요."


if __name__ == "__main__":
    main()
