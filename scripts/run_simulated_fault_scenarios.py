"""Run ten explicitly simulated fault scenarios against every labeled PCB holdout photo.

The diagnosis input contains only the photo, cached production detections,
simulated symptom text/codes, and a synthetic post-inspection confirmation
assumption. Synthetic fault labels live only in ``reference`` and are consumed
after inference by ``score_run``. This is a logic exercise, not a real-fault
accuracy benchmark.
"""

from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circuitcare.benchmark.pcb_validation import (  # noqa: E402
    _bbox_iou, _canonical_components, _ranked_unique, run_case, score_run,
)
from circuitcare.component_taxonomy import normalize_component_class  # noqa: E402
from circuitcare.frontend.board_component_registry import (  # noqa: E402
    build_board_component_registry, source_component_key,
)


DATA_ROOT = Path.home() / ".local/share/circuitcare-ai/external_datasets/pcb-dslr"
HOLDOUT = DATA_ROOT / "holdout-50.json"
DETECTIONS = DATA_ROOT / "hourly-runs/initial"
OUTPUT = DATA_ROOT / "fault-simulations"
PROJECT_ID = "simulated-fault-validation"

SCENARIOS = [
    ("IC_INTERNAL_FAILURE", "PARTIAL_FUNCTION_FAILURE", "전원은 켜지지만 기판의 일부 기능이 동작하지 않습니다.", True),
    ("IC_INTERNAL_FAILURE", "LOCALIZED_HEATING", "기판 일부 기능이 불안정하고 특정 위치가 평소보다 뜨겁습니다.", True),
    ("IC_INTERNAL_FAILURE", "NO_POWER", "전원을 연결해도 기판이 켜지지 않습니다.", True),
    ("IC_INTERNAL_FAILURE", "PARTIAL_FUNCTION_FAILURE", "기판의 제어 기능 일부가 응답하지 않습니다.", True),
    ("EXTERNAL_SUPPLY_FAILURE", "NO_POWER", "외부 전원을 연결해도 기판 전원이 들어오지 않습니다.", False),
    ("SOLDER_JOINT_FAILURE", "INTERMITTENT_OPERATION", "기판이 간헐적으로 동작하고 접촉 상태에 따라 증상이 달라집니다.", False),
    ("PCB_TRACE_OPEN", "PARTIAL_FUNCTION_FAILURE", "기판의 일부 출력 기능만 동작하지 않습니다.", False),
    ("PCB_TRACE_SHORT", "SUPPLY_VOLTAGE_DROOP", "전원 연결 후 공급 전압이 떨어지고 기능이 불안정합니다.", False),
    ("UNKNOWN_CONNECTION_FAILURE", "INTERMITTENT_OPERATION", "기판이 간헐적으로 동작하지 않으며 연결 경로 이상이 의심됩니다.", False),
    ("LOAD_FAILURE", "OUTPUT_FAILURE", "기판은 동작하는 것처럼 보이지만 연결된 출력 부하가 반응하지 않습니다.", False),
]

# Ten deliberately varied reports for the rule-based natural-language fallback.
# The hidden fault label remains in reference and is not included in inference.
LANGUAGE_STRESS_SCENARIOS = [
    ("IC_INTERNAL_FAILURE", "전원은 들어오는데 제어 쪽 몇 개가 먹통이에요. 칩이 나간 것 같긴 한데 확실하진 않음.", True, "colloquial_uncertain_component_guess"),
    ("IC_INTERNAL_FAILURE", "기판은 켜져 있는데 특정 기능만 영 안 돼요. IC 쪽일 수도? 모르겠어요.", True, "colloquial_vague"),
    ("IC_INTERNAL_FAILURE", "전원이 완전 죽은 건 아니고요, 제어가 반쯤 먹통인 듯.", True, "negated_whole_board_failure"),
    ("IC_INTERNAL_FAILURE", "커패시터가 나갔다고 생각했는데 그건 추측이고, 실제론 제어 기능이 가끔 멈춰요.", True, "wrong_component_attribution"),
    ("EXTERNAL_SUPPLY_FAILURE", "전원 꽂아도 보드가 전혀 안켜져요ㅠ", False, "typo_colloquial"),
    ("SOLDER_JOINT_FAILURE", "기판이 가끔 켜졌다 꺼졌다 함. 커넥터 살짝 건드리면 더 그런 거 같음.", False, "colloquial_multi_clue"),
    ("PCB_TRACE_OPEN", "빨간 LED 4개중 3번째만 안들어옴, 나머진 정상.", False, "abbreviated_localized_symptom"),
    ("PCB_TRACE_SHORT", "부하(?) 켜면 전압이 쭉 빠지는 거 같기도 하고 아닐지도..", False, "uncertain_voltage_report"),
    ("UNKNOWN_CONNECTION_FAILURE", "어떤 연결이 끊긴 것 같다던데, 본 건 아니고 정확히 뭐가 문젠지 모르겠어.", False, "hearsay_no_observed_symptom"),
    ("LOAD_FAILURE", "기판은 도는 것 같은데 연결된 장치가 반응하지 않아요. 장치 고장인지는 확인 안했음.", False, "uncertain_external_device_attribution"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ten explicitly simulated fault scenarios per PCB holdout photo.")
    parser.add_argument("--source-manifest", type=Path, default=HOLDOUT)
    parser.add_argument("--detection-runs", type=Path, default=DETECTIONS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--language-stress", action="store_true",
                        help="Use ten natural-language-only stress cases per photo.")
    parser.add_argument("--qwen-symptom-parser", action="store_true",
                        help="Run the production Qwen symptom parser once per distinct stress phrase, then feed its validated observations to the production diagnosis path.")
    parser.add_argument("--qwen-model", default="qwen3.5:9b")
    args = parser.parse_args()
    source_manifest = args.source_manifest.expanduser().resolve()
    detection_runs = args.detection_runs.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    cases = payload["cases"]
    scenario_specs = LANGUAGE_STRESS_SCENARIOS if args.language_stress else SCENARIOS
    if args.qwen_symptom_parser and not args.language_stress:
        parser.error("--qwen-symptom-parser requires --language-stress.")
    qwen_parses: dict[str, dict] = {}
    if args.qwen_symptom_parser:
        from circuitcare.local_llm import OllamaClient
        from circuitcare.symptom_diagnosis.symptom_parser import parse_user_report

        qwen_client = OllamaClient(timeout_seconds=180, num_predict=480, keep_alive="30m", auto_start=True)
        for index, (_fault, text, _target, stress_tag) in enumerate(scenario_specs, 1):
            parse_started = time.perf_counter()
            try:
                qwen_parses[text] = parse_user_report(qwen_client, args.qwen_model, text)
                qwen_parses[text]["_benchmark_duration_ms"] = round((time.perf_counter() - parse_started) * 1000, 1)
                parsed = qwen_parses[text]
                print(f"Qwen 증상 파싱 {index}/{len(scenario_specs)} · {stress_tag} · "
                      f"관측 {len(parsed.get('observations', []))}개 · "
                      f"확인 질문 {'필요' if parsed.get('requires_clarification') else '불필요'}", flush=True)
            except Exception as exc:
                qwen_parses[text] = {"observations": [], "uncertain_facts": [],
                                     "requires_clarification": True,
                                     "parser_status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                                     "_benchmark_duration_ms": round((time.perf_counter() - parse_started) * 1000, 1)}
                print(f"Qwen 증상 파싱 {index}/{len(scenario_specs)} 실패 · {type(exc).__name__}: {exc}", flush=True)
    scenarios: list[dict] = []
    detection_cache: dict[str, dict] = {}
    missing_runs = []
    for photo_case in cases:
        run_paths = sorted(detection_runs.glob(f"{photo_case['case_id']}-*.json"))
        if not run_paths:
            missing_runs.append(photo_case["case_id"])
            continue
        run = json.loads(run_paths[-1].read_text(encoding="utf-8"))
        photo_path = Path(photo_case["inference_input"]["photo_path"])
        image_sha = hashlib.sha256(photo_path.read_bytes()).hexdigest()
        photo_id = str(photo_case.get("inference_input", {}).get("photo_id") or "photo-1")
        detections = [row for row in run.get("detections", []) if isinstance(row, dict)]
        detection_cache[photo_case["case_id"]] = {
            "detections": detections, "image_fingerprint": image_sha,
            "model_provenance": run.get("model_provenance", {}),
        }

        # The simulation assumes a human completed stage-one verification for
        # every named detector result. Unknown/not-component boxes remain
        # excluded, following the user's existing inspection policy.
        reviewed = []
        for row in detections:
            kind = normalize_component_class(str(row.get("class_name") or row.get("component_type") or row.get("소자 종류") or "unknown"))
            excluded = kind in {"unknown", "not_component"}
            reviewed.append({
                **row,
                "component_type": kind,
                "소자 종류": kind,
                "identification_status": "EXCLUDED" if excluded else "USER_CONFIRMED",
                "presence_status": "EXCLUDED" if excluded else "CONFIRMED",
                "component_type_status": "EXCLUDED" if excluded else "USER_CONFIRMED",
                "상태": "제외" if excluded else "확인",
            })
        canonical = _canonical_components(reviewed)
        snapshot = {photo_id: {
            "version": "simulated-inspection-v1", "image_fingerprint": image_sha,
            "review_version": "simulated-inspection-v1", "registry_components": canonical,
        }}
        registry = build_board_component_registry(
            {}, project_id=PROJECT_ID, board_id=str(photo_case["board_id"]), photo_snapshots=snapshot,
        )
        local_to_board = {
            str(source.get("component_id")): str(registry["source_to_board_id"].get(
                source_component_key(photo_id, str(source.get("component_id")))))
            for component in registry.get("components", [])
            for source in component.get("source_components", [])
            if source.get("component_id")
        }

        # Match source IC boxes to model detections only to establish synthetic
        # scenario targets. The annotation and target ID never enter inference.
        labels = (photo_case.get("reference") or {}).get("source_label_proxy_components", [])
        ic_predictions = []
        for row in detections:
            kind = normalize_component_class(str(row.get("class_name") or row.get("component_type") or row.get("소자 종류") or "unknown"))
            box = row.get("bbox_xyxy") or [row.get(key) for key in ("x1", "y1", "x2", "y2")]
            if kind == "ic" and isinstance(box, (list, tuple)) and len(box) == 4:
                ic_predictions.append((row, box))
        used_predictions: set[int] = set()
        matched_ic_ids: list[str] = []
        for label in labels:
            if normalize_component_class(str(label.get("component_type") or "unknown")) != "ic":
                continue
            target_box = label.get("bbox_xyxy")
            choices = sorted(
                ((_bbox_iou(box, target_box), index, row) for index, (row, box) in enumerate(ic_predictions)
                 if index not in used_predictions), reverse=True,
            )
            if choices and choices[0][0] >= 0.5:
                _iou, index, row = choices[0]
                used_predictions.add(index)
                local_id = str(row.get("component_id") or row.get("detection_id") or "")
                if local_to_board.get(local_id):
                    matched_ic_ids.append(local_to_board[local_id])

        # The reference uses source-label locations as scenario targets. If
        # the detector missed a source IC, keep an impossible sentinel so that
        # the synthetic component hit is counted as a miss, never omitted.
        source_ic_count = sum(1 for label in labels
                              if normalize_component_class(str(label.get("component_type") or "unknown")) == "ic")
        target_count = max(1, source_ic_count)
        for scenario_index, spec in enumerate(scenario_specs):
            if args.language_stress:
                fault_id, symptom_text, target_component, language_condition = spec
                symptom_code = ""
            else:
                fault_id, symptom_code, symptom_text, target_component = spec
                language_condition = "structured_symptom_baseline"
            component_id = None
            if target_component:
                if matched_ic_ids:
                    component_id = matched_ic_ids[scenario_index % len(matched_ic_ids)]
                else:
                    component_id = f"missed-source-ic:{photo_case['case_id']}:{scenario_index}"
            case_id = f"sim-{photo_case['case_id']}-{scenario_index + 1:02d}"
            inference = {
                "project_id": PROJECT_ID,
                "photo_id": photo_id,
                "photo_path": str(photo_path),
                "symptoms": [] if args.language_stress else [symptom_code],
                "symptom_text": symptom_text,
                "observations": list(qwen_parses.get(symptom_text, {}).get("observations", [])),
                "skip_qwen_review": True,
                "skip_question_value_simulation": True,
                "cache_provenance": {"model_version": "candidate-yolo-ensemble-initial-2026-11-16"},
                "cached_detection_by_photo": {photo_id: {
                    "image_fingerprint": image_sha, "detections": detections,
                    "model_provenance": run.get("model_provenance", {}),
                }},
                "reviewed_components_by_photo": {photo_id: reviewed},
            }
            reference = {
                "adjudication_status": "CONFIRMED",
                "failure_category": None,
                "fault_type_ids": [fault_id],
                "failure_scenario_status": "SIMULATED_NOT_REAL",
                "synthetic_ground_truth": True,
                "evidence_sources": ["scenario generator; hypothetical fault, not an observed board repair"],
                "source_label_proxy_component_count": target_count,
                "source_label_proxy_matched_ic_count": len(matched_ic_ids),
            }
            if component_id:
                reference["faulty_board_component_ids"] = [component_id]
                reference["fault_pairs"] = [{"fault_type_id": fault_id, "board_component_id": component_id}]
            scenario = {
                "case_id": case_id, "board_id": str(photo_case["board_id"]),
                "physical_board_id": str(photo_case.get("source_board_id") or photo_case["board_id"]),
                "case_kind": "SIMULATED", "dataset_name": "CircuitCare generated fault scenarios",
                "dataset_version": "simulated-faults-v1", "split": "simulation",
                "source_photo_case_id": photo_case["case_id"],
                "scenario_number": scenario_index + 1,
                "language_condition": language_condition,
                "photo_scope": "BOARD_LEVEL", "inference_input": inference, "reference": reference,
            }
            scenarios.append(scenario)

    if missing_runs:
        raise RuntimeError(f"Missing cached YOLO runs for {len(missing_runs)} holdout photos: {missing_runs[:5]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if qwen_parses:
        qwen_parse_report = {
            "model": args.qwen_model, "execution_mode": "REAL_MODEL", "calls": len(qwen_parses),
            "successes": sum(row.get("parser_status") == "PARSED" for row in qwen_parses.values()),
            "failed": sum(row.get("parser_status") != "PARSED" for row in qwen_parses.values()),
            "accepted_observations": sum(len(row.get("observations", [])) for row in qwen_parses.values()),
            "clarification_requests": sum(bool(row.get("requires_clarification")) for row in qwen_parses.values()),
            "mean_parse_ms": round(sum(float(row.get("_benchmark_duration_ms", 0)) for row in qwen_parses.values()) /
                                    len(qwen_parses), 1),
            "phrases": [{"text": text, **row} for text, row in qwen_parses.items()],
        }
        (output_dir / "qwen-parser-results.json").write_text(
            json.dumps(qwen_parse_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = output_dir / "simulated-fault-cases.json"
    # Save a leakage-auditable compact manifest without repeated detector rows.
    compact_cases = []
    for scenario in scenarios:
        compact = {key: value for key, value in scenario.items() if key != "inference_input"}
        compact["inference_input"] = {key: value for key, value in scenario["inference_input"].items()
                                      if key not in {"cached_detection_by_photo", "reviewed_components_by_photo"}}
        compact_cases.append(compact)
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "status": "SIMULATED_ONLY_NOT_REAL_FAULT_ACCURACY",
        "source_manifest": str(source_manifest),
        "source_images": len(cases), "scenarios_per_photo": len(scenario_specs),
        "scenario_count": len(scenarios),
        "diagnostic_input_excludes_fault_truth": True,
        "scenario_mode": ("NATURAL_LANGUAGE_STRESS_QWEN_PLUS_RULES" if args.qwen_symptom_parser else
                          "NATURAL_LANGUAGE_STRESS_RULE_FALLBACK" if args.language_stress else "STRUCTURED_SYMPTOM_BASELINE"),
        "qwen_symptom_parser_called": bool(args.qwen_symptom_parser),
        "symptom_codes_supplied": not args.language_stress,
        "assumption": "All named non-unknown detector boxes are treated as user-confirmed for this simulation; unknown/not_component boxes stay excluded.",
        "cases": compact_cases,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    score_rows = []
    case_results = []
    duration_total = 0.0
    diagnosis_status_counts: dict[str, int] = {}
    started = time.perf_counter()
    for index, scenario in enumerate(scenarios, 1):
        output = run_case(scenario, strategy="baseline", execution_mode="CACHED")
        score = score_run(scenario, output)
        score_rows.append(score)
        duration_total += float(output.get("duration_ms") or 0)
        status = str(output.get("status") or "UNKNOWN")
        diagnosis_status_counts[status] = diagnosis_status_counts.get(status, 0) + 1
        case_results.append({
            "case_id": scenario["case_id"], "source_photo_case_id": scenario["source_photo_case_id"],
            "scenario_number": scenario["scenario_number"],
            "fault_type": scenario["reference"]["fault_type_ids"][0],
            "language_condition": scenario.get("language_condition"),
            "component_truth_status": "EVALUATED" if score.get("component_top3_hit") is not None else "NOT_EVALUATED",
            "status": score.get("status"), "diagnosis_status": status,
            "fault_type_top1_hit": score.get("fault_type_top1_hit"),
            "fault_type_top3_hit": score.get("fault_type_top3_hit"),
            "component_top3_hit": score.get("component_top3_hit"),
            # The UI's legacy list is grouped by fault scope. Summaries must
            # explicitly re-sort on relative evidence score to mean Top 3.
            "fault_top3": [identifier for identifier, _score in _ranked_unique(
                output.get("all_hypotheses", []), "fault_id")[:3]],
            "component_top3": [identifier for identifier, _score in _ranked_unique(
                output.get("all_component_candidates", []), "component_id")[:3]],
            "duration_ms": output.get("duration_ms"),
            "production_rule_diagnosis_ms": next((row.get("duration_ms") for row in output.get("stage_trace", [])
                if row.get("stage") == "production_rule_diagnosis"), None),
            "question_value_simulation_ms": next((row.get("output_summary", {}).get("processing_timings_ms", {}).get(
                "recommendation_generation") for row in output.get("stage_trace", [])
                if row.get("stage") == "production_rule_diagnosis"), None),
        })
        del output
        if index % 50 == 0 or index == len(scenarios):
            print(f"진단 {index}/{len(scenarios)} · {time.perf_counter() - started:.1f}초", flush=True)
            partial = {
                "status": "RUNNING", "photos": len(cases), "scenario_count": index,
                "total_scenarios": len(scenarios), "scenarios_per_photo": len(scenario_specs),
                "simulated_only": True, "case_results": case_results,
            }
            temp_result = output_dir / ".simulated-fault-results.tmp"
            temp_result.write_text(json.dumps(partial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temp_result.replace(output_dir / "simulated-fault-results.json")

    scored = [row for row in score_rows if row.get("status") == "SCORED"]
    def ratio(key: str) -> dict:
        values = [row.get(key) for row in scored if row.get(key) is not None]
        hits = sum(value is True for value in values)
        n = len(values)
        return {"status": "EVALUATED" if n else "NOT_EVALUATED", "value": hits / n if n else None,
                "correct": hits, "n": n, "not_evaluated": len(score_rows) - n}

    summary = {
        "status": "SIMULATED_ONLY_NOT_REAL_FAULT_ACCURACY",
        "photos": len(cases), "scenarios_per_photo": len(scenario_specs), "scenario_count": len(scenarios),
        "execution_mode": "CACHED_REAL_YOLO_DETECTIONS_PLUS_PRODUCTION_RULE_DIAGNOSIS",
        "scenario_mode": ("NATURAL_LANGUAGE_STRESS_QWEN_PLUS_RULES" if args.qwen_symptom_parser else
                          "NATURAL_LANGUAGE_STRESS_RULE_FALLBACK" if args.language_stress else "STRUCTURED_SYMPTOM_BASELINE"),
        "qwen_symptom_parser_called": bool(args.qwen_symptom_parser),
        "symptom_codes_supplied": not args.language_stress,
        "qwen_calls": len(qwen_parses),
        "synthetic_component_confirmation_assumption": True,
        "real_fault_ground_truth_count": 0,
        "simulated_fault_type_top1": ratio("fault_type_top1_hit"),
        "simulated_fault_type_top3": ratio("fault_type_top3_hit"),
        "simulated_component_top3": ratio("component_top3_hit"),
        "scored_scenarios": len(scored),
        "not_scored_scenarios": len(scenarios) - len(scored),
        "diagnosis_status_counts": diagnosis_status_counts,
        "mean_diagnosis_ms": duration_total / len(scenarios) if scenarios else None,
        "case_results": case_results,
        "run_id": f"simulated-faults-{int(time.time())}",
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    result_path = output_dir / "simulated-fault-results.json"
    temp_result = output_dir / ".simulated-fault-results.tmp"
    temp_result.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_result.replace(result_path)
    print(json.dumps({key: value for key, value in summary.items() if key != "case_results"},
                     ensure_ascii=False, indent=2))
    print(f"시나리오 목록: {manifest_path}")
    print(f"결과: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
