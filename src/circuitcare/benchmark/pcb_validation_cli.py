"""Command line entry point for registering and executing PCB validation cases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

from .pcb_validation import (
    evaluate_runs, production_qwen_reviewer, production_yolo_detector,
    register_cases, run_case, save_metrics_csv, save_run, _bbox_iou, _score_detection,
)
from circuitcare.component_taxonomy import normalize_component_class


def _source_label_proxy(cases, runs):
    """Score source-provided labels separately; never merge them into gold metrics."""
    run_by_id = {row.get("case_id"): row for row in runs}
    counts = {"tp": 0, "fp": 0, "fn": 0, "cases": 0, "excluded_unlabeled_class_predictions": 0}
    box_counts = {"matched": 0, "labelled": 0, "cases": 0}
    ic_box_counts = {"matched": 0, "predicted": 0, "false_positive": 0,
                     "duplicate": 0, "duplicate_candidates": 0, "cases": 0}
    duplicate_box_counts = {"duplicate": 0, "overlapping_predictions": 0, "cases": 0}
    class_counts_on_located = {"correct": 0, "matched": 0, "cases": 0}
    class_counts: dict[str, dict[str, int]] = {}
    for case in cases:
        reference = case.get("reference") or {}
        labels = reference.get("source_label_proxy_components")
        run = run_by_id.get(case.get("case_id"), {})
        if (not isinstance(labels, list) or not labels
                or run.get("execution_mode") not in {"REAL_MODEL", "CACHED"}
                or run.get("status") == "PIPELINE_FAILED"
                or any(stage.get("status") == "ERROR"
                       and str(stage.get("stage", "")).startswith("yolo_detection")
                       for stage in run.get("stage_trace", []) if isinstance(stage, dict))):
            continue
        labeled_classes = {normalize_component_class(str(row.get("component_type") or "unknown"))
                           for row in labels if isinstance(row, dict)}
        predictions = run.get("detections", [])
        ic_prediction_indexes = {
            index for index, row in enumerate(predictions)
            if normalize_component_class(str(row.get("class_name", row.get("component_type", row.get("소자 종류", "unknown"))))) == "ic"
        }
        geometric_pairs = []
        for prediction_index, prediction in enumerate(predictions):
            box = prediction.get("bbox_xyxy") or [prediction.get(key) for key in ("x1", "y1", "x2", "y2")]
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            for label_index, label in enumerate(labels):
                target = label.get("bbox_xyxy") if isinstance(label, dict) else None
                if isinstance(target, (list, tuple)) and len(target) == 4:
                    iou = _bbox_iou(box, target)
                    if iou >= 0.5:
                        geometric_pairs.append((iou, prediction_index, label_index))
        used_predictions, used_labels = set(), set()
        correct_types = 0
        for _iou, prediction_index, label_index in sorted(geometric_pairs, reverse=True):
            if prediction_index in used_predictions or label_index in used_labels:
                continue
            used_predictions.add(prediction_index)
            used_labels.add(label_index)
            predicted_type = normalize_component_class(str(predictions[prediction_index].get(
                "class_name", predictions[prediction_index].get("component_type",
                predictions[prediction_index].get("소자 종류", "unknown")))))
            expected_type = normalize_component_class(str(labels[label_index].get("component_type") or "unknown"))
            correct_types += predicted_type == expected_type
        duplicate_prediction_indexes = {
            prediction_index for _iou, prediction_index, label_index in geometric_pairs
            if prediction_index not in used_predictions and label_index in used_labels
        }
        duplicate_box_counts["duplicate"] += len(duplicate_prediction_indexes)
        duplicate_box_counts["overlapping_predictions"] += len(used_predictions) + len(duplicate_prediction_indexes)
        duplicate_box_counts["cases"] += 1
        ic_pairs = [(iou, prediction_index, label_index) for iou, prediction_index, label_index in geometric_pairs
                    if prediction_index in ic_prediction_indexes]
        used_ic_predictions, used_ic_labels = set(), set()
        for _iou, prediction_index, label_index in sorted(ic_pairs, reverse=True):
            if prediction_index in used_ic_predictions or label_index in used_ic_labels:
                continue
            used_ic_predictions.add(prediction_index)
            used_ic_labels.add(label_index)
        ic_duplicates = ic_prediction_indexes & duplicate_prediction_indexes
        ic_spurious = ic_prediction_indexes - used_ic_predictions - ic_duplicates
        ic_box_counts["matched"] += len(used_ic_predictions)
        ic_box_counts["predicted"] += len(ic_prediction_indexes)
        ic_box_counts["false_positive"] += len(ic_spurious)
        ic_box_counts["duplicate"] += len(ic_duplicates)
        ic_box_counts["duplicate_candidates"] += len(used_ic_predictions) + len(ic_spurious) + len(ic_duplicates)
        ic_box_counts["cases"] += 1
        box_counts["matched"] += len(used_labels)
        box_counts["labelled"] += len(labels)
        box_counts["cases"] += 1
        class_counts_on_located["correct"] += correct_types
        class_counts_on_located["matched"] += len(used_labels)
        class_counts_on_located["cases"] += 1
        comparable_predictions = [row for row in predictions if normalize_component_class(str(
            row.get("class_name", row.get("component_type", row.get("소자 종류", "unknown"))))) in labeled_classes]
        counts["excluded_unlabeled_class_predictions"] += len(predictions) - len(comparable_predictions)
        score = _score_detection({"gold_components_status": "CONFIRMED", "gold_components": labels},
                                 {**run, "detections": comparable_predictions})
        if score.get("status") != "EVALUATED":
            continue
        counts["tp"] += int(score.get("tp", 0))
        counts["fp"] += int(score.get("fp", 0))
        counts["fn"] += int(score.get("fn", 0))
        counts["cases"] += 1
        for component_class, class_score in score.get("classes", {}).items():
            bucket = class_counts.setdefault(component_class, {"tp": 0, "fp": 0, "fn": 0})
            for key in bucket:
                bucket[key] += int(class_score.get(key, 0))
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else None
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else None)
    classes = {}
    for name, row in class_counts.items():
        p = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else None
        r = row["tp"] / (row["tp"] + row["fn"]) if row["tp"] + row["fn"] else None
        class_f1 = 2 * p * r / (p + r) if p is not None and r is not None and p + r else None
        classes[name] = {**row, "precision": p, "recall": r, "f1": class_f1}
    return {"status": "SOURCE_LABEL_PROXY" if counts["cases"] else "NOT_EVALUATED",
            **counts, "precision": precision, "recall": recall, "f1": f1, "classes": classes,
            "component_detection_rate": {
                "value": box_counts["matched"] / box_counts["labelled"] if box_counts["labelled"] else None,
                "detected": box_counts["matched"], "labelled": box_counts["labelled"],
                "cases": box_counts["cases"],
            },
            "component_location_false_positive_rate": {
                "value": ic_box_counts["false_positive"] / (ic_box_counts["matched"] + ic_box_counts["false_positive"])
                    if ic_box_counts["matched"] + ic_box_counts["false_positive"] else None,
                "false_positive": ic_box_counts["false_positive"],
                "predicted_ic_boxes": ic_box_counts["predicted"],
                "matched_ic_boxes": ic_box_counts["matched"],
                "duplicate_ic_boxes_excluded": ic_box_counts["duplicate"],
                "cases": ic_box_counts["cases"],
                "note": "IC 예측 중 다른 정답 IC와 겹친 중복 박스를 제외한 위치 오탐 비율. 다른 소자 종류 라벨은 없는 자료이므로 전체 부품 오탐률이 아닙니다.",
            },
            "duplicate_detection_rate": {
                "value": duplicate_box_counts["duplicate"] / duplicate_box_counts["overlapping_predictions"]
                    if duplicate_box_counts["overlapping_predictions"] else None,
                "duplicate_boxes": duplicate_box_counts["duplicate"],
                "overlapping_predictions": duplicate_box_counts["overlapping_predictions"],
                "cases": duplicate_box_counts["cases"],
                "note": "같은 정답 IC 박스에 추가로 겹친 예측 박스 비율(IoU≥0.5); IC 라벨 기준입니다.",
            },
            "component_recognition_rate": {
                "value": class_counts_on_located["correct"] / class_counts_on_located["matched"]
                    if class_counts_on_located["matched"] else None,
                "correct": class_counts_on_located["correct"], "located": class_counts_on_located["matched"],
                "cases": class_counts_on_located["cases"],
            },
            "note": "원본 제공 라벨과의 참고 비교입니다. 독립 검수 정답이 아니므로 모델 정확도/공식 점수로 취급하지 않습니다. 이미지에서 라벨된 클래스만 precision 분모에 포함했습니다."}


def main() -> int:
    parser = argparse.ArgumentParser(prog="circuitcare-validate")
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register", help="validate and import a JSON or CSV case file")
    register.add_argument("source")
    register.add_argument("destination")
    run = commands.add_parser("run", help="execute registered cases and save immutable run outputs")
    run.add_argument("cases")
    run.add_argument("results")
    run.add_argument("--real-yolo", action="store_true", help="use local production YOLO weights")
    run.add_argument("--mode", choices=("REAL_MODEL", "CACHED", "MOCK", "SYNTHETIC"),
                     help="explicit execution mode; REAL_MODEL requires local weights")
    run.add_argument("--strategy", choices=("baseline", "evidence_v1", "active_inspection_v1"), default="baseline",
                     help="baseline ranking, experimental evidence reranking, or ranking-neutral active inspection planning")
    run.add_argument("--dashboard", action="store_true",
                     help="write dashboard-live.json after each case for the live HTML dashboard")
    run.add_argument("--dashboard-path", type=Path,
                     help="write live dashboard snapshots to an explicit path")
    run.add_argument("--baseline-dashboard", type=Path,
                     help="compare the completed run against a prior dashboard with the same case IDs")
    run.add_argument("--skip-qwen", action="store_true",
                     help="run YOLO and production rule diagnosis without the slow optional OCR/Qwen review stage")
    args = parser.parse_args()
    if args.command == "register":
        print(register_cases(args.source, args.destination))
        return 0
    payload = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    if args.real_yolo and args.mode not in {None, "REAL_MODEL"}:
        parser.error("--real-yolo can only be combined with --mode REAL_MODEL")
    execution_mode = "REAL_MODEL" if args.real_yolo else args.mode
    detector = production_yolo_detector if execution_mode == "REAL_MODEL" else None
    reviewer = production_qwen_reviewer if execution_mode == "REAL_MODEL" and not args.skip_qwen else None
    result_paths = []
    outputs = []
    cases = payload.get("cases", [])
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = (args.dashboard_path.expanduser().resolve() if args.dashboard_path
                      else results_dir / "dashboard-live.json")
    run_started_at = datetime.now(timezone.utc).isoformat()
    if args.dashboard:
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=dashboard_path.parent,
                                         prefix=".dashboard-live-", suffix=".tmp", delete=False) as stream:
            json.dump({"schema_version": 1, "run_state": "RUNNING", "run_started_at": run_started_at,
                       "last_updated": run_started_at, "completed_cases": 0, "total_cases": len(cases),
                       "runs": [], "case_sources": []}, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            temp_path = Path(stream.name)
        os.replace(temp_path, dashboard_path)
    for case in cases:
        case_input = case
        if args.skip_qwen:
            case_input = {**case, "inference_input": {
                **(case.get("inference_input") or {}), "skip_qwen_review": True,
            }}
        output = run_case(case_input, detector=detector, reviewer=reviewer,
                          strategy=args.strategy, execution_mode=execution_mode)
        outputs.append(output)
        target = results_dir / f"{case['case_id']}-{output['run_id'][:8]}.json"
        result_paths.append(str(save_run(target, output)))
        if args.dashboard:
            summary = evaluate_runs(cases, outputs)
            score_by_id = {str(row.get("case_id") or ""): row for row in summary.get("cases", [])}
            case_by_id = {str(row.get("case_id") or ""): row for row in cases}
            dashboard_payload = {
                "schema_version": 1,
                "run_state": "COMPLETE" if len(outputs) == len(cases) else "RUNNING",
                "run_started_at": run_started_at,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "completed_cases": len(outputs),
                "total_cases": len(cases),
                "runs": [{key: output.get(key) for key in (
                    "run_id", "case_id", "board_id", "case_kind", "execution_mode", "status",
                    "failure_category", "error", "duration_ms", "stage_trace", "model_provenance",
                )} | {
                    "components": [{
                        "component_id": str(item.get("board_component_id") or item.get("component_id") or item.get("detection_id") or ""),
                        "component_type": str(item.get("class_name") or item.get("component_type") or item.get("소자 종류") or "unknown"),
                        "confidence": item.get("confidence", item.get("score")),
                        "bbox_xyxy": item.get("bbox_xyxy") or [item.get(key) for key in ("x1", "y1", "x2", "y2")],
                    } for item in output.get("detections", []) if isinstance(item, dict)],
                    "fault_top3": [{
                        "fault_id": str(item.get("fault_id") or item.get("hypothesis_id") or ""),
                        "component_id": str(item.get("board_component_id") or item.get("component_id") or ""),
                        "score": item.get("score"),
                    } for item in output.get("all_hypotheses", output.get("hypotheses", []))[:3]
                      if isinstance(item, dict)],
                    "fault_top3_hit": score_by_id.get(str(output.get("case_id") or ""), {}).get("fault_type_top3_hit"),
                    "component_top3_hit": score_by_id.get(str(output.get("case_id") or ""), {}).get("component_top3_hit"),
                    "measurement_count": len((case_by_id.get(str(output.get("case_id") or ""), {}).get("inference_input") or {}).get("measurements", []) or []),
                    "diagnosis_status": output.get("status"),
                } for output in outputs],
                "case_sources": [{
                    "case_id": str(case.get("case_id") or ""),
                    "board_id": str(case.get("board_id") or ""),
                    "case_kind": str(case.get("case_kind") or ""),
                    "dataset_name": case.get("dataset_name") or case.get("source_dataset"),
                    "dataset_url": case.get("dataset_url") or case.get("source_url"),
                    "photo_source": case.get("photo_source") or case.get("source_description"),
                    "label_source": case.get("label_source") or case.get("adjudication_source"),
                    "adjudication_status": (case.get("reference") or {}).get("adjudication_status", "UNKNOWN"),
                    "evidence_sources": (case.get("reference") or {}).get("evidence_sources", []),
                    "photo_paths": case.get("photo_paths") or
                        (case.get("inference_input") or {}).get("photo_paths") or
                        [(case.get("inference_input") or {}).get("photo_path")]
                        if ((case.get("photo_paths") or (case.get("inference_input") or {}).get("photo_paths")
                             or (case.get("inference_input") or {}).get("photo_path"))) else [],
                } for case in cases],
                "source_label_proxy": _source_label_proxy(cases, outputs),
                "summary": summary,
            }
            if args.baseline_dashboard and len(outputs) == len(cases):
                try:
                    baseline = json.loads(args.baseline_dashboard.read_text(encoding="utf-8"))
                    baseline_ids = {str(row.get("case_id") or "") for row in baseline.get("case_sources", [])}
                    current_ids = {str(case.get("case_id") or "") for case in cases}
                    if baseline_ids == current_ids and len(current_ids) == len(cases):
                        before = baseline.get("source_label_proxy") or {}
                        after = dashboard_payload["source_label_proxy"]
                        comparison = {"status": "COMPARABLE_SOURCE_LABELS", "cases": len(cases),
                                      "baseline_label": "처음 기본 가중치", "trained_label": "검수 학습 가중치",
                                      "baseline_updated_at": baseline.get("last_updated"), "metrics": {}}
                        for key in ("component_detection_rate", "component_recognition_rate",
                                    "component_location_false_positive_rate", "duplicate_detection_rate"):
                            baseline_value = (before.get(key) or {}).get("value")
                            trained_value = (after.get(key) or {}).get("value")
                            comparison["metrics"][key] = {
                                "baseline": baseline_value, "trained": trained_value,
                                "change_percentage_points": ((trained_value - baseline_value) * 100
                                    if baseline_value is not None and trained_value is not None else None),
                            }
                        comparison["note"] = "같은 사진 ID를 사용한 원본 라벨 참고 비교입니다. 독립 정답 기반 공식 정확도는 아닙니다."
                        dashboard_payload["model_comparison"] = comparison
                    else:
                        dashboard_payload["model_comparison"] = {"status": "NOT_COMPARABLE",
                            "note": "초기 실행과 같은 사진 ID 집합이 아니어서 개선량을 계산하지 않았습니다."}
                except (OSError, json.JSONDecodeError, TypeError):
                    dashboard_payload["model_comparison"] = {"status": "NOT_COMPARABLE",
                        "note": "초기 대시보드 데이터를 읽지 못해 개선량을 계산하지 않았습니다."}
            dashboard_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=dashboard_path.parent,
                                             prefix=".dashboard-live-", suffix=".tmp", delete=False) as stream:
                json.dump(dashboard_payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                temp_path = Path(stream.name)
            os.replace(temp_path, dashboard_path)
    summary = evaluate_runs(cases, outputs)
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    csv_path = save_metrics_csv(results_dir / "summary.csv", summary)
    print(json.dumps({"saved_runs": result_paths, "metrics_report": str(summary_path),
                      "metrics_csv": str(csv_path),
                      "dashboard_live_json": str(dashboard_path) if args.dashboard else None,
                      "source_label_proxy": _source_label_proxy(cases, outputs)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
