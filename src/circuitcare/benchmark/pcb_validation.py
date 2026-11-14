"""Independent real-PCB case registry, production-engine harness, and metrics.

Gold answers live under ``reference`` and are only read by ``score_run``;
``run_case`` receives the case's ``inference_input`` object only.
"""

from __future__ import annotations

import csv
from contextlib import nullcontext
import hashlib
import json
import os
import platform
import sys
import time
import uuid
import subprocess
from pathlib import Path
from typing import Any, Callable

from circuitcare.frontend.board_component_registry import (
    build_board_component_registry, diagnosis_components, project_measurements_to_registry,
)
from circuitcare.component_taxonomy import normalize_component_class
from circuitcare.symptom_diagnosis import load_knowledge
from circuitcare.symptom_diagnosis.diagnosis_state import build_overlay_state
from circuitcare.symptom_diagnosis.engine import diagnose


CASE_KINDS = {"REAL_CONFIRMED", "REAL_HEALTHY", "REAL_UNKNOWN", "REAL_MULTIPLE_FAULTS",
              "REAL_NON_COMPONENT", "SYNTHETIC", "SIMULATED", "MOCK"}
GOLD_STATUSES = {"CONFIRMED", "UNCONFIRMED", "UNKNOWN"}


def validate_case(case: dict[str, Any]) -> None:
    if not isinstance(case, dict) or not str(case.get("case_id") or "").strip():
        raise ValueError("PCB validation case requires case_id.")
    if case.get("case_kind") not in CASE_KINDS:
        raise ValueError("case_kind must explicitly distinguish real, synthetic, simulated, or mock data.")
    inference = case.get("inference_input")
    reference = case.get("reference")
    if not isinstance(inference, dict) or not isinstance(reference, dict):
        raise ValueError("inference_input and reference sections are required.")
    _validate_inference_input(case)
    if not case.get("board_id"):
        raise ValueError("board_id is required.")
    if reference.get("adjudication_status", "UNKNOWN") not in GOLD_STATUSES:
        raise ValueError("Invalid gold adjudication_status.")
    if reference.get("failure_category") not in {None, "DETECTION_ERROR", "COMPONENT_IDENTITY_ERROR",
            "REGISTRY_ERROR", "SYMPTOM_INTERPRETATION_ERROR", "RULE_INFERENCE_ERROR",
            "MEASUREMENT_VALIDATION_ERROR", "UI_STATE_ERROR"}:
        raise ValueError("Unknown failure_category.")
    if case["case_kind"].startswith("REAL_"):
        if not (case.get("photo_paths") or inference.get("photo_paths") or inference.get("photo_path")):
            raise ValueError("Real cases require source photo paths.")
        if reference.get("adjudication_status") == "CONFIRMED" and not reference.get("evidence_sources"):
            raise ValueError("Confirmed real fault labels require evidence_sources.")


def _validate_inference_input(case: dict[str, Any]) -> dict[str, Any]:
    """Validate the inference-only portion without reading adjudicated answers."""
    if not isinstance(case, dict) or not str(case.get("case_id") or "").strip():
        raise ValueError("PCB validation case requires case_id.")
    if case.get("case_kind") not in CASE_KINDS:
        raise ValueError("case_kind must explicitly distinguish real, synthetic, simulated, or mock data.")
    inference = case.get("inference_input")
    if not isinstance(inference, dict) or not case.get("board_id"):
        raise ValueError("inference_input and board_id are required.")
    forbidden = {"ground_truth", "ground_truth_component_id", "fault_type", "fault_type_ids",
                 "fault_parameter", "injected_fault", "reference", "root_cause_ids",
                 "faulty_board_component_ids", "faulty_component_ids", "fault_pairs",
                 "gold_components", "gold_faults", "expected_fault_ids", "adjudication_status"}
    def contains_forbidden(value: Any) -> bool:
        if isinstance(value, dict):
            return bool(forbidden.intersection(value)) or any(contains_forbidden(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_forbidden(item) for item in value)
        return False
    if contains_forbidden(inference):
        raise ValueError("Gold-answer fields must not appear in inference_input.")
    return inference


def register_cases(source: str | Path, destination: str | Path) -> Path:
    """Validate and copy case JSON, or import CSV rows with JSON in case_json."""
    source, destination = Path(source), Path(destination)
    if source.suffix.casefold() == ".csv":
        cases = []
        with source.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                payload = json.loads(row.get("case_json") or "{}")
                cases.append(payload)
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        cases = payload.get("cases", [payload])
    ids: set[str] = set()
    boards_by_split: dict[str, set[str]] = {}
    images_by_split: dict[str, set[str]] = {}
    for case in cases:
        validate_case(case)
        if case["case_id"] in ids:
            raise ValueError(f"Duplicate PCB case ID: {case['case_id']}")
        ids.add(case["case_id"])
        if case["case_kind"].startswith("REAL_"):
            specs = normalize_case_photos(case, base_dir=source.parent, require_exists=True)
            case["photo_paths"] = [{"photo_id": spec["photo_id"], "path": spec["path"]} for spec in specs]
            if len({spec["sha256"] for spec in specs}) != len(specs):
                raise ValueError(f"Duplicate image within PCB case {case['case_id']}.")
            split = str(case.get("split") or "unassigned")
            board = str(case.get("physical_board_id") or case.get("pcb_group_id") or case.get("board_id"))
            boards_by_split.setdefault(board, set()).add(split)
            for spec in specs:
                images_by_split.setdefault(spec["sha256"], set()).add(split)
    if any(len(splits) > 1 for splits in boards_by_split.values()):
        raise ValueError("Same PCB board_id appears in more than one data split.")
    if any(len(splits) > 1 for splits in images_by_split.values()):
        raise ValueError("Identical PCB image appears in more than one data split.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"schema_version": 1, "cases": cases}, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    return destination


def normalize_case_photos(case: dict[str, Any], *, base_dir: str | Path | None = None,
                          require_exists: bool = True) -> list[dict[str, str]]:
    """Normalize path spellings and reject missing/duplicate image assets."""
    inputs = case.get("inference_input", {})
    raw = case.get("photo_paths") or inputs.get("photo_paths") or []
    single = inputs.get("photo_path")
    if single:
        raw_list = list(raw) if isinstance(raw, list) else ([raw] if raw else [])
        def raw_path(value: Any) -> str:
            return str(value.get("path") or value.get("photo_path") or "") if isinstance(value, dict) else str(value)
        base = Path(base_dir).expanduser().resolve() if base_dir else None
        single_path = Path(str(single)).expanduser()
        if not single_path.is_absolute() and base:
            single_path = base / single_path
        if str(single_path.resolve()) not in {
            str((Path(raw_path(item)).expanduser() if Path(raw_path(item)).expanduser().is_absolute() or not base
                 else base / Path(raw_path(item)).expanduser()).resolve()) for item in raw_list if raw_path(item)
        }:
            raw_list.append(single)
        raw = raw_list
    if isinstance(raw, (str, Path, dict)):
        raw = [raw]
    result = []
    seen_paths: set[str] = set()
    seen_photo_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for index, item in enumerate(raw):
        if isinstance(item, dict):
            value = item.get("path") or item.get("photo_path")
            photo_id = str(item.get("photo_id") or f"photo-{index + 1}")
        else:
            value, photo_id = item, (str(inputs.get("photo_id")) if index == 0 and inputs.get("photo_id")
                                     else f"photo-{index + 1}")
        if not photo_id or photo_id in seen_photo_ids:
            raise ValueError(f"Missing or duplicate PCB photo_id: {photo_id!r}")
        seen_photo_ids.add(photo_id)
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if not path.is_absolute() and base_dir:
            path = Path(base_dir) / path
        path = path.resolve()
        normalized = str(path)
        if normalized in seen_paths:
            raise ValueError(f"Duplicate PCB photo path: {normalized}")
        seen_paths.add(normalized)
        if require_exists and not path.is_file():
            raise ValueError(f"PCB photo file not found: {normalized}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
        if digest and digest in seen_hashes:
            raise ValueError(f"Duplicate PCB image content within case: {normalized}")
        if digest:
            seen_hashes.add(digest)
        result.append({"photo_id": photo_id, "path": normalized, "sha256": digest})
    if case.get("case_kind", "").startswith("REAL_") and not result:
        raise ValueError("Real PCB case has no readable photo path.")
    return result


def run_case(case: dict[str, Any], *, detector: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
             reviewer: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
             llm: Callable[[dict[str, Any]], Any] | None = None,
             strategy: str = "baseline", execution_mode: str | None = None) -> dict[str, Any]:
    """Run injected model stages and the production Registry/diagnose/evaluator path."""
    inputs = _validate_inference_input(case)
    if strategy not in {"baseline", "evidence_v1", "active_inspection_v1"}:
        raise ValueError(f"Unsupported diagnosis strategy: {strategy}")
    started = time.perf_counter()
    stages: list[dict[str, Any]] = []
    photo_specs = normalize_case_photos(case, base_dir=inputs.get("case_root"), require_exists=False)
    if execution_mode is None:
        execution_mode = ("REAL_MODEL" if detector is production_yolo_detector else
                          "MOCK" if case["case_kind"] == "MOCK" else
                          "SYNTHETIC" if case["case_kind"] in {"SYNTHETIC", "SIMULATED"} else
                          "CACHED" if inputs.get("cached_detection") and inputs.get("cache_provenance") else
                          "NOT_RUN")
    execution_mode = str(execution_mode).upper()
    if execution_mode not in {"REAL_MODEL", "CACHED", "MOCK", "SYNTHETIC", "NOT_RUN"}:
        raise ValueError(f"Unsupported execution mode: {execution_mode}")
    if execution_mode in {"REAL_MODEL", "CACHED"}:
        photo_specs = normalize_case_photos(case, base_dir=inputs.get("case_root"), require_exists=True)

    def stage(name: str, operation: Callable[[], Any], *, input_summary: dict[str, Any] | None = None) -> Any:
        begin = time.perf_counter()
        try:
            value = operation()
            stages.append({"stage": name, "status": "OK", "duration_ms": round((time.perf_counter() - begin) * 1000, 3),
                           "input_summary": input_summary or {}, "output_summary": _summarize(value)})
            return value
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            stages.append({"stage": name, "status": "ERROR", "duration_ms": round((time.perf_counter() - begin) * 1000, 3),
                           "input_summary": input_summary or {}, "error": error})
            return {"_stage_error": error}

    if execution_mode == "NOT_RUN" and case["case_kind"].startswith("REAL_"):
        return {"run_id": str(uuid.uuid4()), "case_id": case["case_id"], "case_kind": case["case_kind"],
                "execution_mode": execution_mode, "status": "MODEL_NOT_RUN",
                "error": "Real PCB performance evaluation requires an explicit REAL_MODEL or CACHED run.",
                "stage_trace": [], "metrics": "NOT_EVALUATED", "runtime_metadata": _runtime_metadata(case, inputs, execution_mode)}
    detection_results: list[tuple[dict[str, str] | None, dict[str, Any]]] = []
    if execution_mode == "REAL_MODEL" and detector is None:
        return {"run_id": str(uuid.uuid4()), "case_id": case["case_id"], "case_kind": case["case_kind"],
                "execution_mode": execution_mode, "status": "PIPELINE_FAILED", "failure_category": "DETECTION_ERROR",
                "error": "REAL_MODEL mode requires the production detector callback.", "stage_trace": stages,
                "metrics": "NOT_EVALUATED", "runtime_metadata": _runtime_metadata(case, inputs, execution_mode)}
    if execution_mode == "CACHED":
        cached = inputs.get("cached_detection")
        cached_by_photo = inputs.get("cached_detection_by_photo")
        provenance = inputs.get("cache_provenance")
        if not isinstance(provenance, dict) or not provenance.get("model_version"):
            raise ValueError("CACHED mode requires cache_provenance.model_version.")
        if isinstance(cached_by_photo, dict) and photo_specs:
            for photo in photo_specs:
                cached_row = cached_by_photo.get(photo["photo_id"])
                if not isinstance(cached_row, dict) or cached_row.get("image_fingerprint") != photo.get("sha256"):
                    raise ValueError(f"CACHED detection missing or stale for photo {photo['photo_id']}.")
                detection_results.append((photo, {**cached_row, "execution_mode": "CACHED"}))
        else:
            if not isinstance(cached, dict):
                raise ValueError("CACHED mode requires cached_detection or cached_detection_by_photo.")
            if len(photo_specs) > 1:
                raise ValueError("Multiple photos require cached_detection_by_photo keyed by photo_id.")
            photo = photo_specs[0] if photo_specs else None
            cached_fingerprint = cached.get("image_fingerprint") or provenance.get("image_fingerprint")
            if photo and cached_fingerprint != photo.get("sha256"):
                raise ValueError("Cached detection image fingerprint does not match the source photo.")
            detection_results.append((photo, {**cached, "execution_mode": "CACHED"}))
    elif execution_mode in {"MOCK", "SYNTHETIC"} and detector is None:
        detection_results.append((None, {"status": f"{execution_mode}_INPUT",
            "detections": inputs.get("detections", [])}))
        stages.append({"stage": "yolo_detection", "status": "MOCK_INPUT" if execution_mode == "MOCK" else "SYNTHETIC_INPUT",
                       "duration_ms": 0.0, "input_summary": {"execution_mode": execution_mode},
                       "output_summary": {"detections_count": len(inputs.get("detections", []))}})
    else:
        detector_specs = photo_specs or [None]
        for index, photo in enumerate(detector_specs):
            photo_input = {**inputs}
            if photo:
                photo_input.update(photo_id=photo["photo_id"], photo_path=photo["path"], image_fingerprint=photo.get("sha256"))
            detection = stage(f"yolo_detection:{photo['photo_id'] if photo else index}",
                lambda source=photo_input: detector(source) if detector else {"detections": []},
                input_summary={"execution_mode": execution_mode, "photo_id": photo.get("photo_id") if photo else inputs.get("photo_id"),
                               "photo_path_present": bool(photo or inputs.get("photo_path"))})
            if detection.get("_stage_error"):
                return {"run_id": str(uuid.uuid4()), "case_id": case["case_id"], "case_kind": case["case_kind"],
                        "execution_mode": execution_mode, "status": "PIPELINE_FAILED", "failure_category": "DETECTION_ERROR",
                        "error": detection["_stage_error"], "stage_trace": stages,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 3), "metrics": "NOT_EVALUATED",
                        "runtime_metadata": _runtime_metadata(case, inputs, execution_mode)}
            detection_results.append((photo, detection))

    review_results: list[tuple[dict[str, str] | None, dict[str, Any]]] = []
    for index, (photo, detection) in enumerate(detection_results):
        photo_input = {**inputs}
        if photo:
            photo_input.update(photo_id=photo["photo_id"], photo_path=photo["path"], image_fingerprint=photo.get("sha256"))
        if reviewer and execution_mode == "REAL_MODEL":
            reviewed = stage(f"ocr_qwen_review:{photo['photo_id'] if photo else index}",
                lambda source=photo_input, detected=detection: reviewer(source, detected),
                input_summary={"execution_mode": execution_mode, "detection_count": len(detection.get("detections", []))})
        else:
            reviewed_by_photo = inputs.get("reviewed_components_by_photo", {})
            pre_review = (reviewed_by_photo[photo["photo_id"]]
                          if photo and photo["photo_id"] in reviewed_by_photo else None)
            components_for_photo = pre_review if pre_review is not None else inputs.get("components", detection.get("detections", []))
            reviewed = {"components": components_for_photo, "status": "PRE_REVIEWED" if pre_review is not None else execution_mode}
            stages.append({"stage": f"component_review:{photo['photo_id'] if photo else index}",
                           "status": "CACHED_REVIEW" if pre_review is not None else execution_mode,
                           "duration_ms": 0.0, "input_summary": {"detection_count": len(detection.get("detections", []))},
                           "output_summary": {"components_count": len(components_for_photo)}})
        if reviewed.get("_stage_error"):
            # OCR/Qwen review is an assistive stage; preserve detector boxes and
            # let the production rules engine continue if it is unavailable.
            reviewed = {"components": detection.get("detections", []),
                        "status": "OCR_QWEN_FAILED_DETECTIONS_RETAINED",
                        "review_error": reviewed["_stage_error"]}
        review_results.append((photo, reviewed))
    detections = []
    reviewed_components = []
    for photo, detection in detection_results:
        for row in detection.get("detections", []):
            if isinstance(row, dict):
                detections.append({**row, **({"photo_id": photo["photo_id"]} if photo else {})})
    for photo, reviewed in review_results:
        for row in reviewed.get("components", []):
            if isinstance(row, dict):
                reviewed_components.append({**row, **({"photo_id": photo["photo_id"]} if photo else {})})
    snapshots = dict(inputs.get("photo_snapshots", {})) if execution_mode != "REAL_MODEL" else {}
    if execution_mode == "REAL_MODEL" or (not snapshots and any(photo for photo, _ in detection_results)):
        for photo, detection in detection_results:
            if photo is None:
                continue
            reviewed_for_photo = next((value for review_photo, value in review_results
                                       if review_photo and review_photo["photo_id"] == photo["photo_id"]), {})
            snapshots[photo["photo_id"]] = {
                "version": str(inputs.get("component_map_version", "validation-photo-v1")),
                "image_fingerprint": str(detection.get("image_fingerprint") or photo.get("sha256") or ""),
                "review_version": str(inputs.get("review_version", "validation-review-v1")),
                "registry_components": _canonical_components(reviewed_for_photo.get("components", [])),
            }
    elif not snapshots and inputs.get("photo_id") and reviewed_components:
        one_photo = str(inputs["photo_id"])
        snapshots = {one_photo: {
            "version": str(inputs.get("component_map_version", "validation-photo-v1")),
            "image_fingerprint": str(inputs.get("image_fingerprint") or ""),
            "review_version": str(inputs.get("review_version", "validation-review-v1")),
            "registry_components": _canonical_components(reviewed_components),
        }}
    if snapshots:
        registry = stage("board_registry", lambda: build_board_component_registry(
            {}, project_id=str(inputs.get("project_id", "validation")),
            board_id=str(case["board_id"]), photo_snapshots=snapshots),
            input_summary={"photo_count": len(snapshots), "component_count": sum(len(item.get("registry_components", item.get("components", []))) for item in snapshots.values())})
        if registry.get("_stage_error"):
            return {"run_id": str(uuid.uuid4()), "case_id": case["case_id"], "case_kind": case["case_kind"],
                    "execution_mode": execution_mode, "status": "PIPELINE_FAILED", "failure_category": "REGISTRY_ERROR",
                    "error": registry["_stage_error"], "stage_trace": stages,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3), "metrics": "NOT_EVALUATED",
                    "runtime_metadata": _runtime_metadata(case, inputs, execution_mode, stages=stages)}
        components = [item.__dict__ for item in diagnosis_components(registry)]
        confirmed = [row.component_id for row in diagnosis_components(registry) if row.presence_status == "CONFIRMED"]
        original_measurements = inputs.get("measurements", [])
        _history_measurements, current_measurements = stage("measurement_registry_projection",
            lambda: project_measurements_to_registry(original_measurements, registry),
            input_summary={"measurement_records": len(original_measurements),
                           "registry_version": registry.get("version")})
    else:
        components = _canonical_components(reviewed_components)
        registry = {"components": [], "photos": []}
        confirmed = [str(item.get("component_id")) for item in components
                     if item.get("presence_status") == "CONFIRMED" or item.get("identification_status") == "USER_CONFIRMED"]
        current_measurements = inputs.get("measurements", [])
    diagnosis_inputs = {key: inputs.get(key) for key in (
        "symptoms", "symptom_text", "observations", "inventory_complete",
        "current_context", "silkscreen_labels")}
    current_context = dict(diagnosis_inputs.get("current_context") or {})
    current_context.setdefault("project_id", str(inputs.get("project_id", "validation")))
    current_context.setdefault("board_id", str(case["board_id"]))
    current_context.setdefault("component_identity_statuses", {
        str(row.get("component_id")): str(row.get("identification_status", "UNCONFIRMED"))
        for row in components})
    if registry.get("version"):
        current_context["component_map_version"] = registry["version"]
        current_context.setdefault("component_identity_statuses", {
            str(row.get("component_id")): str(row.get("identification_status", "UNCONFIRMED"))
            for row in components})
    diagnosis_inputs.update(measurements=current_measurements, current_context=current_context)
    diagnosis_inputs.update(components=components, confirmed_component_ids=confirmed)
    baseline_result = stage("production_rule_diagnosis", lambda: diagnose(**diagnosis_inputs), input_summary={
        "symptoms": diagnosis_inputs.get("symptoms"), "component_count": len(components),
        "confirmed_component_count": len(confirmed), "measurement_count": len(diagnosis_inputs.get("measurements") or [])})
    if baseline_result.get("_stage_error"):
        return {"run_id": str(uuid.uuid4()), "case_id": case["case_id"], "case_kind": case["case_kind"],
                "status": "PIPELINE_FAILED", "error": baseline_result["_stage_error"], "stage_trace": stages,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3), "metrics": "NOT_EVALUATED",
                "runtime_metadata": _runtime_metadata(case, inputs, execution_mode)}
    phase_timings = baseline_result.get("processing_timings_ms", {})
    for phase, output_key in (("measurement_validation", "measurement_evidence"),
                              ("candidate_ranking", "component_candidates"),
                              ("recommendation_generation", "next_tests")):
        rows = baseline_result.get(output_key, [])
        stages.append({"stage": phase, "status": "OK",
                       "duration_ms": float(phase_timings.get(phase, 0.0)),
                       "input_summary": {"measurement_count": len(diagnosis_inputs.get("measurements") or []),
                                         "component_count": len(components)},
                       "output_summary": {f"{output_key}_count": len(rows),
                                          "diagnosis_status": baseline_result.get("status")}})
    result = baseline_result
    strategy_comparison = None
    if strategy == "active_inspection_v1":
        strategy_result = {**baseline_result, "strategy": "active_inspection_v1",
                           "ranking_strategy": "unchanged_rule_baseline_with_active_inspection_plan"}
        strategy_comparison = {
            "status": "COMPARABLE", "baseline": baseline_result,
            "active_inspection_v1": strategy_result,
            "ranking_changed": False,
            "note": "Phase 4 planner changes inspection order only; it does not modify fault scores or measurement verdicts.",
        }
        result = strategy_result
    elif strategy == "evidence_v1":
        from circuitcare.symptom_diagnosis.evidence_strategy import rank_by_verified_evidence, optimize_safe_tests
        strategy_result = stage("evidence_v1_rerank", lambda: rank_by_verified_evidence(
            baseline_result, components=components, measurements=diagnosis_inputs.get("measurements", []),
            observations=diagnosis_inputs.get("observations")), input_summary={
                "candidate_count": len(baseline_result.get("component_candidates", [])),
                "measurement_evidence_count": len(baseline_result.get("measurement_evidence", []))})
        if strategy_result.get("_stage_error"):
            # Experimental strategy failures never replace the production baseline.
            strategy_comparison = {"status": "FAILED_BASELINE_RETAINED", "error": strategy_result["_stage_error"]}
        else:
            optimized_tests = optimize_safe_tests(strategy_result.get("next_tests", []),
                observations=diagnosis_inputs.get("observations"), measurements=diagnosis_inputs.get("measurements", []))
            strategy_result["next_tests"] = optimized_tests
            strategy_comparison = {"status": "COMPARABLE", "baseline": baseline_result,
                                   "evidence_v1": strategy_result}
            result = strategy_result
    llm_result = None
    if llm is not None:
        llm_result = stage("llm_assist", lambda: llm(baseline_result))
        if llm_result.get("_stage_error"):
            # Rule-based diagnosis remains the valid production output.
            llm_result = {"status": "FAILED_RULES_RETAINED", "error": llm_result["_stage_error"]}
    measurement_comparison = None
    if "measurements_before" in inputs:
        before_inputs = dict(diagnosis_inputs)
        if registry.get("version"):
            _before_history, before_inputs["measurements"] = project_measurements_to_registry(
                inputs.get("measurements_before") or [], registry)
        else:
            before_inputs["measurements"] = inputs.get("measurements_before") or []
        before_result = stage("pre_measurement_rule_diagnosis", lambda: diagnose(**before_inputs), input_summary={
            "measurement_count": len(before_inputs["measurements"]), "component_count": len(components)})
        if not before_result.get("_stage_error"):
            before_rank = [item.get("fault_id") for item in before_result.get("hypotheses", [])]
            after_rank = [item.get("fault_id") for item in result.get("hypotheses", [])]
            measurement_comparison = {"before_rank": before_rank, "after_rank": after_rank,
                                      "rank_changed": before_rank != after_rank,
                                      "reassessment_duration_ms": round(float(
                                          phase_timings.get("total", 0.0)), 3)}
    overlays = stage("current_result_overlay", lambda: build_overlay_state(
        project_id=str(inputs.get("project_id", "validation")), board_id=str(case["board_id"]),
        component_map_version=str(registry.get("version", "validation")), result=result),
        input_summary={"diagnosis_status": result.get("status"), "candidate_count": len(result.get("component_candidates", []))})
    qwen_results = [reviewed.get("qwen_review") for _photo, reviewed in review_results if reviewed.get("qwen_review")]
    ocr_results = [record for _photo, reviewed in review_results
                   for record in (reviewed.get("ocr_result") or {}).get("records", [])]
    return {
        "run_id": str(uuid.uuid4()), "case_id": case["case_id"], "case_kind": case["case_kind"],
        "execution_mode": execution_mode, "detections": detections,
        "reviewed_components": reviewed_components,
        "ocr_evidence": ocr_results,
        "qwen_review": qwen_results,
        "review_errors": [row.get("review_error") for _photo, row in review_results if row.get("review_error")],
        "registry_component_count": len(registry.get("components", [])),
        "status": result.get("status"), "top_candidates": result.get("component_candidates", [])[:3],
        "all_component_candidates": result.get("all_component_candidates", result.get("component_candidates", [])),
        "hypotheses": result.get("hypotheses", [])[:3],
        "all_hypotheses": result.get("hypotheses", []),
        "functional_block_hypotheses": result.get("hierarchical_hypotheses", {}).get("functional_blocks", []),
        "functional_block_groups": result.get("functional_block_groups", []),
        "joint_candidates": result.get("joint_candidates", []),
        "next_tests": result.get("next_tests", []),
        "active_inspection_plan": result.get("active_inspection_plan", {}),
        "inspection_history": inputs.get("inspection_history", []),
        "measurement_evidence": result.get("measurement_evidence", []), "llm": llm_result,
        "qwen_call_count": sum(int(reviewed.get("qwen_call_count", 0) or 0)
                                for _photo, reviewed in review_results),
        "strategy": strategy, "baseline_result": baseline_result,
        "strategy_comparison": strategy_comparison,
        "measurement_comparison": measurement_comparison,
        "overlay": overlays, "stage_trace": stages,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "runtime_metadata": _runtime_metadata(case, inputs, execution_mode,
                                               stages=stages, duration_ms=(time.perf_counter() - started) * 1000),
        "metrics": "NOT_EVALUATED",
    }


def _summarize(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"type": type(value).__name__}
    summary: dict[str, Any] = {}
    for key in ("status", "verification_status", "component_count", "image_width", "image_height"):
        if key in value and isinstance(value[key], (str, int, float, bool, type(None))):
            summary[key] = value[key]
    for key in ("detections", "components", "class_reviews", "duplicate_reviews", "group_reviews",
                "missed_component_candidates", "hypotheses", "component_candidates", "next_tests", "measurement_evidence"):
        if isinstance(value.get(key), (list, tuple)):
            summary[f"{key}_count"] = len(value[key])
    if "error" in value:
        summary["error"] = str(value["error"])
    if isinstance(value.get("processing_timings_ms"), dict):
        summary["processing_timings_ms"] = value["processing_timings_ms"]
    return summary


def _runtime_metadata(case: dict[str, Any], inputs: dict[str, Any], execution_mode: str, *,
                      stages: list[dict[str, Any]] | None = None,
                      duration_ms: float | None = None) -> dict[str, Any]:
    """Capture enough provenance to reproduce and interpret a validation run."""
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True,
                                  text=True, timeout=2).stdout.strip()
    except Exception:
        revision = os.environ.get("CIRCUITCARE_CODE_VERSION", "UNKNOWN")
    try:
        import resource
        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            peak_rss *= 1024
    except Exception:
        peak_rss = None
    return {
        "code_version": revision,
        "case_dataset_version": case.get("dataset_version", "UNSPECIFIED"),
        "execution_mode": execution_mode,
        "model_versions": inputs.get("model_versions", inputs.get("cache_provenance", {})),
        "configuration": inputs.get("inference_config", {}),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "duration_ms": round(duration_ms, 3) if duration_ms is not None else None,
        "peak_rss_bytes": peak_rss,
        "stage_error_count": sum(row.get("status") == "ERROR" for row in stages or []),
        "stage_count": len(stages or []),
        "evaluation_case_count": 1,
    }


def production_yolo_detector(inference_input: dict[str, Any]) -> dict[str, Any]:
    """Invoke the app's cached first-pass YOLO path for real-weight validation runs."""
    import io
    from PIL import Image
    from circuitcare.vision.component_recognition_panel import ensure_component_detection

    photo_path = inference_input.get("photo_path")
    if not photo_path:
        raise ValueError("inference_input.photo_path is required for --real-yolo.")
    image_bytes = Path(photo_path).read_bytes()
    import hashlib
    with Image.open(io.BytesIO(image_bytes)) as image:
        size = image.size

    state: dict[str, Any] = {}
    app_api = type("ValidationStreamlitAdapter", (), {
        "session_state": state, "spinner": lambda self, _message: nullcontext(),
    })()
    result, error = ensure_component_detection(
        app_api, image_bytes=image_bytes,
        project_id=str(inference_input.get("project_id", "validation")),
        photo_id=str(inference_input.get("photo_id", Path(photo_path).stem)),
        review_image_size=size,
    )
    if error or result is None:
        raise RuntimeError(error or "Production YOLO returned no result.")
    scope = next((key.removeprefix("cc_component_recognition_") for key in state
                  if key.startswith("cc_component_recognition_")), None)
    return {**result, "detections": state.get(f"cc_component_review_rows_{scope}", result.get("detections", [])),
            "review_scope": scope, "_app_state": state, "_image_bytes": image_bytes,
            "image_fingerprint": hashlib.sha256(image_bytes).hexdigest()}


def production_qwen_reviewer(inference_input: dict[str, Any], detection: dict[str, Any]) -> dict[str, Any]:
    """Use the production whole-photo OCR association and selected-box Qwen path."""
    from contextlib import nullcontext
    import io
    from PIL import Image
    from circuitcare.vision.component_recognition_panel import _run_ai_review_workflow, review_aux_keys, review_keys
    from circuitcare.vision.ocr_backend import run_local_ocr
    from circuitcare.vision.box_reconciliation import associate_ocr_with_components

    scope = detection.get("review_scope")
    state = detection.get("_app_state")
    image_bytes = detection.get("_image_bytes")
    if not scope or not isinstance(state, dict) or not image_bytes:
        raise ValueError("Production YOLO review session is incomplete.")
    records_key = review_keys(scope)[1]
    ocr_key, qwen_key = review_aux_keys(scope)
    rows = state.get(records_key, [])
    with Image.open(io.BytesIO(image_bytes)) as source_image:
        image = source_image.convert("RGB")
        image_fingerprint = hashlib.sha256(image_bytes).hexdigest()
        ocr_result = run_local_ocr(image)
    ocr_result["source_image_fingerprint"] = image_fingerprint
    bbox_rows = [
        {**row, "bbox_xyxy": [row["x1"], row["y1"], row["x2"], row["y2"]]}
        for row in rows if isinstance(row, dict) and all(key in row for key in ("x1", "y1", "x2", "y2"))
    ]
    associated = associate_ocr_with_components(bbox_rows, ocr_result.get("records", []))
    row_by_id = {str(row.get("component_id")): row for row in rows
                 if isinstance(row, dict) and row.get("component_id")}
    active_rows = [row for row in rows if isinstance(row, dict) and row.get("상태") != "제외"]
    for source, evidence in zip(bbox_rows, associated):
        row = row_by_id.get(str(source.get("component_id")))
        if row is None:
            continue
        for key in ("ocr_records", "ocr_candidates", "reference_designators",
                    "ocr_class_hints", "ocr_association_status"):
            row[key] = evidence.get(key)
    state[records_key] = rows
    state[ocr_key] = ocr_result
    api = type("ValidationStreamlitAdapter", (), {
        "session_state": state, "spinner": lambda self, _message: nullcontext(),
        "warning": lambda self, _message: None,
    })()
    selected_ids = [str(row.get("component_id")) for row in active_rows if row.get("component_id")]
    if selected_ids:
        _run_ai_review_workflow(api, image_bytes=image_bytes, scope=scope,
            rows=state.get(records_key, []), records_key=records_key,
            ocr_key=ocr_key, qwen_key=qwen_key, selected_component_ids=selected_ids)
    review = state.get(qwen_key) if isinstance(state.get(qwen_key), dict) else {}
    return {"components": state.get(records_key, []), "qwen_review": state.get(qwen_key),
            "ocr_result": state.get(ocr_key), "status": "PRODUCTION_OCR_QWEN",
            "qwen_call_count": int(review.get("performance", {}).get("class_review_requests", 0) or 0)}


def _canonical_components(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = row.get("component_type", row.get("class_name", row.get("소자 종류", "unknown")))
        status = row.get("identification_status", row.get("상태"))
        confirmed = status in {"USER_CONFIRMED", "확인"} or row.get("presence_status") == "CONFIRMED"
        box = row.get("bbox_xyxy") or [row.get(key) for key in ("x1", "y1", "x2", "y2")]
        result.append({**row, "component_id": str(row.get("component_id") or ""),
            "component_type": normalize_component_class(str(label)),
            "bbox_xyxy": box, "identification_status": "USER_CONFIRMED" if confirmed else "UNCONFIRMED",
            "presence_status": "CONFIRMED" if confirmed else "UNCONFIRMED",
            "component_type_status": "USER_CONFIRMED" if confirmed else "UNCONFIRMED"})
    return result


def score_run(case: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """Score only adjudicated labels; unresolved/multi-fault labels are explicit."""
    validate_case(case)
    truth = case["reference"]
    detection = _score_detection(truth, run)
    if case["case_kind"].startswith("REAL_") and run.get("execution_mode") not in {"REAL_MODEL", "CACHED"}:
        return {"case_id": case["case_id"], "status": "NOT_EVALUATED",
                "reason": "real case was not run with REAL_MODEL or provenance-validated CACHED inputs",
                "execution_mode": run.get("execution_mode"), "detection": detection}
    if truth.get("adjudication_status") != "CONFIRMED":
        return {"case_id": case["case_id"], "status": "NOT_EVALUATED", "reason": "gold truth not confirmed",
                "detection": detection}
    fault_truth = set(truth.get("fault_type_ids", truth.get("root_cause_ids", [])))
    component_truth = set(truth.get("faulty_board_component_ids", []))
    pair_truth = {(str(item.get("fault_type_id") or item.get("fault_id") or ""),
                   str(item.get("board_component_id") or item.get("component_id") or ""))
                  for item in truth.get("fault_pairs", []) if isinstance(item, dict)}
    pair_truth.discard(("", ""))
    fault_truth.update(fault for fault, _component in pair_truth if fault)
    component_truth.update(component for _fault, component in pair_truth if component)
    evaluation_hypotheses = run.get("all_hypotheses", run.get("hypotheses", []))
    evaluation_components = run.get("all_component_candidates", run.get("top_candidates", []))
    fault_top1, fault_top3 = _topk_ranked_ids(evaluation_hypotheses, "fault_id", 1), _topk_ranked_ids(evaluation_hypotheses, "fault_id", 3)
    comp_top1, comp_top3 = _topk_ranked_ids(evaluation_components, "component_id", 1), _topk_ranked_ids(evaluation_components, "component_id", 3)
    joint_top1 = _topk_pairs(evaluation_components, 1)
    joint_top3 = _topk_pairs(evaluation_components, 3)
    block_truth = {str(value) for value in truth.get("faulty_functional_block_ids", []) if value}
    block_top3 = _topk_ranked_ids(run.get("functional_block_hypotheses", []), "functional_block_id", 3)
    plan = run.get("active_inspection_plan") if isinstance(run.get("active_inspection_plan"), dict) else {}
    planned_checks = plan.get("candidates", []) if isinstance(plan.get("candidates"), list) else []
    history = run.get("inspection_history", []) if isinstance(run.get("inspection_history"), list) else []
    measurement_records = case.get("inference_input", {}).get("measurements", [])
    executed_check_count = sum(1 for row in [*history, *(measurement_records if isinstance(measurement_records, list) else [])]
                               if isinstance(row, dict) and row.get("test_id"))
    if case.get("case_kind") == "REAL_HEALTHY" and truth.get("adjudication_status") == "CONFIRMED":
        predicted_fault = bool(run.get("hypotheses"))
        return {"case_id": case["case_id"], "status": "SCORED", "healthy_case": True,
                "healthy_correct": not predicted_fault, "false_fault_candidate": predicted_fault,
                "component_metrics_status": "NOT_APPLICABLE", "joint_metrics_status": "NOT_APPLICABLE",
                "abstained": str(run.get("status", "")).startswith("ABSTAIN"),
                "detection": detection, "failure_category": truth.get("failure_category", "UNCLASSIFIED")}
    if not fault_truth and not component_truth and not pair_truth:
        return {"case_id": case["case_id"], "status": "NOT_EVALUATED", "reason": "confirmed reference has no fault/component labels",
                "detection": detection}
    component_applicable = case.get("case_kind") != "REAL_NON_COMPONENT"
    fault_evaluable = bool(fault_truth)
    component_evaluable = bool(component_truth) and component_applicable
    pair_evaluable = bool(pair_truth) and component_applicable
    return {"case_id": case["case_id"], "status": "SCORED", "failure_category": truth.get("failure_category", "UNCLASSIFIED"),
            "fault_type_count": len(fault_truth), "fault_component_count": len(component_truth),
            "fault_type_top1_hit": (bool(fault_top1 & fault_truth) if fault_evaluable else None),
            "fault_type_top3_hit": (bool(fault_top3 & fault_truth) if fault_evaluable else None),
            "fault_type_all_top3_hit": (fault_truth.issubset(fault_top3) if fault_evaluable else None),
            "component_top1_hit": (bool(comp_top1 & component_truth) if component_evaluable else None),
            "component_top3_hit": (bool(comp_top3 & component_truth) if component_evaluable else None),
            "component_all_top3_hit": (component_truth.issubset(comp_top3) if component_evaluable else None),
            "joint_top1_hit": (bool(joint_top1 & pair_truth) if pair_evaluable else None),
            "joint_top3_hit": (bool(joint_top3 & pair_truth) if pair_evaluable else None),
            "block_recall_at_3": (len(block_top3 & block_truth) / len(block_truth) if block_truth else None),
            "block_any_hit_at_3": (bool(block_top3 & block_truth) if block_truth else None),
            "block_recall_status": "EVALUATED" if block_truth else "NOT_EVALUATED",
            "planned_inspection_count": len(planned_checks),
            "executed_inspection_count": executed_check_count,
            "measurement_reassessment_rank_changed": (
                bool(run.get("measurement_comparison", {}).get("rank_changed"))
                if isinstance(run.get("measurement_comparison"), dict) else None),
            "active_inspection_plan": plan,
            "reassessment_duration_ms": (
                float(run.get("measurement_comparison", {}).get("reassessment_duration_ms"))
                if isinstance(run.get("measurement_comparison"), dict)
                and isinstance(run.get("measurement_comparison", {}).get("reassessment_duration_ms"), (int, float))
                else None),
            "qwen_call_count": (int(run["qwen_call_count"]) if isinstance(run.get("qwen_call_count"), int) else None),
            "joint_all_top3_hit": (pair_truth.issubset(joint_top3) if pair_evaluable else None),
            "fault_type_metrics_status": "EVALUATED" if fault_evaluable else "NOT_EVALUATED",
            "component_metrics_status": ("NOT_APPLICABLE" if not component_applicable else
                                          "EVALUATED" if component_evaluable else "NOT_EVALUATED"),
            "joint_metrics_status": ("NOT_APPLICABLE" if not component_applicable else
                                     "EVALUATED" if pair_evaluable else "NOT_EVALUATED"),
            "top1_hit": bool(fault_top1 & fault_truth) if fault_evaluable else None,
            "top3_hit": bool(fault_top3 & fault_truth) if fault_evaluable else None,
            "abstained": str(run.get("status", "")).startswith("ABSTAIN"),
            "unsupported_specific_attribution": any(
                not (row.get("evidence") or row.get("supporting_evidence") or row.get("opposing_evidence")
                     or row.get("evidence_ids") or row.get("measurement_verdict") not in {None, "non_diagnostic"})
                for row in run.get("top_candidates", [])
            ),
            "useful_test_recommended": (bool(set(truth.get("useful_test_ids", [])) &
                {str(row.get("test_id")) for row in (
                    planned_checks if run.get("strategy") == "active_inspection_v1" else
                    (run.get("next_tests", []) or planned_checks))})
                if truth.get("recommendation_labels_status") == "CONFIRMED" else None),
            "verdict_matches": _score_verdicts(truth, run),
            "rank_improvement": _rank_improvement(sorted(fault_truth), run.get("measurement_comparison")),
            "detection": detection}


def _ranked_unique(rows: list[dict[str, Any]], id_key: str) -> list[tuple[str, float]]:
    best: dict[str, float] = {}
    for index, row in enumerate(rows):
        identifier = str(row.get(id_key) or "").strip()
        if id_key == "component_id":
            identifier = str(row.get("board_component_id") or identifier).strip()
        if id_key in {"fault_id", "fault_type_id"}:
            identifier = identifier or str(row.get("fault_hypothesis") or "").strip()
        if not identifier:
            continue
        try:
            score = float(row["score"]) if row.get("score") is not None else -float(index)
        except (TypeError, ValueError):
            score = 0.0
        best[identifier] = max(score, best.get(identifier, float("-inf")))
    return sorted(best.items(), key=lambda item: (-item[1], item[0]))


def _topk_ranked_ids(rows: list[dict[str, Any]], id_key: str, k: int) -> set[str]:
    ranked = _ranked_unique(rows, id_key)
    if not ranked:
        return set()
    cutoff = ranked[min(k, len(ranked)) - 1][1]
    return {identifier for identifier, score in ranked if score >= cutoff}


def _ranked_unique_pairs(rows: list[dict[str, Any]]) -> list[tuple[tuple[str, str], float]]:
    best: dict[tuple[str, str], float] = {}
    for index, row in enumerate(rows):
        fault = str(row.get("fault_hypothesis") or row.get("fault_id") or "").strip()
        component = str(row.get("board_component_id") or row.get("component_id") or "").strip()
        if not fault or not component:
            continue
        try:
            score = float(row["score"]) if row.get("score") is not None else -float(index)
        except (TypeError, ValueError):
            score = 0.0
        key = (fault, component)
        best[key] = max(score, best.get(key, float("-inf")))
    return sorted(best.items(), key=lambda item: (-item[1], item[0]))


def _topk_pairs(rows: list[dict[str, Any]], k: int | None) -> set[tuple[str, str]]:
    ranked = _ranked_unique_pairs(rows)
    if not ranked:
        return set()
    cutoff = ranked[-1][1] if k is None else ranked[min(k, len(ranked)) - 1][1]
    return {pair for pair, score in ranked if score >= cutoff}


def _score_verdicts(reference: dict[str, Any], run: dict[str, Any]) -> list[bool] | None:
    if reference.get("measurement_verdict_labels_status") != "CONFIRMED":
        return None
    actual = {str(row.get("measurement_id")): str(row.get("verdict"))
              for row in run.get("measurement_evidence", [])}
    expected = {str(row.get("measurement_id")): str(row.get("expected_verdict"))
                for row in reference.get("measurement_verdict_labels", [])}
    return [actual.get(key) == value for key, value in expected.items()]


def _rank_improvement(roots: list[str], comparison: Any) -> int | None:
    if not isinstance(comparison, dict) or not roots:
        return None
    def best_rank(rows: list[str]) -> int | None:
        positions = [rows.index(root) for root in roots if root in rows]
        return min(positions) if positions else None
    before, after = best_rank(comparison.get("before_rank", [])), best_rank(comparison.get("after_rank", []))
    if before is None or after is None:
        return None
    return 1 if after < before else (-1 if after > before else 0)


def _score_detection(reference: dict[str, Any], run: dict[str, Any], *, iou_threshold: float = 0.5) -> dict[str, Any]:
    if reference.get("gold_components_status") != "CONFIRMED":
        return {"status": "NOT_EVALUATED", "tp": 0, "fp": 0, "fn": 0}
    gold = reference.get("gold_components", [])
    predicted = run.get("detections", [])
    candidates = []
    for pi, prediction in enumerate(predicted):
        box = prediction.get("bbox_xyxy") or [prediction.get(key) for key in ("x1", "y1", "x2", "y2")]
        for gi, label in enumerate(gold):
            target = label.get("bbox_xyxy")
            if not box or not target or len(box) != 4 or len(target) != 4:
                continue
            iou = _bbox_iou(box, target)
            predicted_class = normalize_component_class(str(prediction.get(
                "class_name", prediction.get("component_type", prediction.get("소자 종류", "unknown")))))
            target_class = normalize_component_class(str(label.get("component_type", "unknown")))
            if iou >= iou_threshold and predicted_class == target_class:
                candidates.append((iou, pi, gi, predicted_class))
    candidates.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    tp = 0
    matches: list[tuple[int, int, str]] = []
    for _iou, pi, gi, component_class in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        tp += 1
        matches.append((pi, gi, component_class))
    predicted_classes = [normalize_component_class(str(row.get("class_name", row.get("component_type", row.get("소자 종류", "unknown"))))) for row in predicted]
    gold_classes = [normalize_component_class(str(row.get("component_type", "unknown"))) for row in gold]
    class_metrics = {}
    for component_class in sorted(set(predicted_classes) | set(gold_classes)):
        class_tp = sum(1 for _pi, _gi, matched_class in matches if matched_class == component_class)
        class_fp = predicted_classes.count(component_class) - class_tp
        class_fn = gold_classes.count(component_class) - class_tp
        class_precision = class_tp / (class_tp + class_fp) if class_tp + class_fp else None
        class_recall = class_tp / (class_tp + class_fn) if class_tp + class_fn else None
        class_f1 = (2 * class_precision * class_recall / (class_precision + class_recall)
                    if class_precision is not None and class_recall is not None and class_precision + class_recall else None)
        class_metrics[component_class] = {"tp": class_tp, "fp": class_fp, "fn": class_fn,
            "precision": class_precision, "recall": class_recall, "f1": class_f1}
    return {"status": "EVALUATED", "tp": tp, "fp": len(predicted) - tp,
            "fn": len(gold) - tp, "classes": class_metrics, "iou_threshold": iou_threshold,
            "predicted_count": len(predicted), "gold_count": len(gold)}


def _bbox_iou(left: Any, right: Any) -> float:
    try:
        ax1, ay1, ax2, ay2 = map(float, left)
        bx1, by1, bx2, by2 = map(float, right)
    except (TypeError, ValueError):
        return 0.0
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def evaluate_runs(cases: list[dict[str, Any]], runs: list[dict[str, Any]], *,
                  _include_dataset_breakdown: bool = True) -> dict[str, Any]:
    """Return metrics with explicit denominators; missing/unknown labels stay unscored."""
    by_id = {run.get("case_id"): run for run in runs}
    scores = [score_run(case, by_id[case["case_id"]]) for case in cases if case["case_id"] in by_id]
    scored = [item for item in scores if item["status"] == "SCORED"]
    root_scored = [item for item in scored if not item.get("healthy_case")]
    abstention = [item for item in scores if item["case_id"] in by_id]
    detection_scored = [item["detection"] for item in scores if item.get("detection", {}).get("status") == "EVALUATED"]
    recommendation_rows = [item["useful_test_recommended"] for item in scored if item.get("useful_test_recommended") is not None]
    verdict_rows = [value for item in scored for value in (item.get("verdict_matches") or [])]
    ranking_rows = [item["rank_improvement"] for item in scored if item.get("rank_improvement") is not None]
    separation_rows = [row for item in scored for row in (
        ((item.get("active_inspection_plan") or {}).get("candidates", []))
        if isinstance(item.get("active_inspection_plan"), dict) else []) if isinstance(row, dict)]
    reassessment_rows = [item for item in scored if item.get("reassessment_duration_ms") is not None]
    qwen_call_rows = [item for item in scored if item.get("qwen_call_count") is not None]
    block_recall_rows = [item["block_recall_at_3"] for item in scored
                         if item.get("block_recall_status") == "EVALUATED"]
    attribution_rows = [item for item in root_scored if "unsupported_specific_attribution" in item]
    tp = sum(item["tp"] for item in detection_scored)
    fp = sum(item["fp"] for item in detection_scored)
    fn = sum(item["fn"] for item in detection_scored)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    class_names = sorted({name for item in detection_scored for name in item.get("classes", {})})
    class_detection = {}
    for name in class_names:
        class_tp = sum(item.get("classes", {}).get(name, {}).get("tp", 0) for item in detection_scored)
        class_fp = sum(item.get("classes", {}).get(name, {}).get("fp", 0) for item in detection_scored)
        class_fn = sum(item.get("classes", {}).get(name, {}).get("fn", 0) for item in detection_scored)
        class_p = class_tp / (class_tp + class_fp) if class_tp + class_fp else None
        class_r = class_tp / (class_tp + class_fn) if class_tp + class_fn else None
        class_f1 = 2 * class_p * class_r / (class_p + class_r) if class_p is not None and class_r is not None and class_p + class_r else None
        class_detection[name] = {"tp": class_tp, "fp": class_fp, "fn": class_fn,
                                 "precision": class_p, "recall": class_r, "f1": class_f1}
    def ratio(key: str) -> dict[str, Any]:
        eligible = [row for row in root_scored if row.get(key) is not None]
        n = len(eligible)
        hits = sum(bool(row.get(key)) for row in eligible)
        value = hits / n if n else None
        # Wilson 95% interval; shown with denominator, including at small n.
        if n:
            from math import sqrt
            z = 1.959963984540054
            denom = 1 + z * z / n
            center = (value + z * z / (2 * n)) / denom
            margin = z * sqrt(value * (1 - value) / n + z * z / (4 * n * n)) / denom
            interval = [max(0.0, center - margin), min(1.0, center + margin)]
        else:
            interval = None
        return {"value": value, "n": n, "correct": hits, "confidence_interval_95_wilson": interval,
                "status": "EVALUATED" if n else "NOT_EVALUATED"}
    category_counts: dict[str, int] = {}
    for case in cases:
        category = case.get("reference", {}).get("failure_category")
        if category:
            category_counts[category] = category_counts.get(category, 0) + 1
    paired_baseline_scores = []
    paired_active_scores = []
    for case in cases:
        run = by_id.get(case["case_id"])
        baseline = (run.get("baseline_result") if isinstance(run, dict)
                    and run.get("strategy") in {"evidence_v1", "active_inspection_v1"} else None)
        if not isinstance(baseline, dict):
            continue
        baseline_run = {**run, "hypotheses": baseline.get("hypotheses", []),
                        "all_hypotheses": baseline.get("hypotheses", []),
                        "top_candidates": baseline.get("component_candidates", []),
                        "all_component_candidates": baseline.get("all_component_candidates",
                                                                  baseline.get("component_candidates", [])),
                        "next_tests": baseline.get("next_tests", []),
                        "measurement_evidence": baseline.get("measurement_evidence", []),
                        "active_inspection_plan": baseline.get("active_inspection_plan", {})}
        if run.get("strategy") == "evidence_v1":
            paired_baseline_scores.append(score_run(case, baseline_run))
        elif run.get("strategy") == "active_inspection_v1":
            paired_active_scores.append(score_run(case, baseline_run))
    baseline_fault1 = [item for item in paired_baseline_scores if item.get("fault_type_top1_hit") is not None]
    baseline_fault3 = [item for item in paired_baseline_scores if item.get("fault_type_top3_hit") is not None]
    runtime_rows = [run for run in runs if run.get("execution_mode") in {"REAL_MODEL", "CACHED", "MOCK", "SYNTHETIC"}]
    durations = [float(run.get("duration_ms")) for run in runtime_rows if isinstance(run.get("duration_ms"), (int, float))]
    stage_rows = [stage for run in runtime_rows for stage in run.get("stage_trace", [])]
    model_modes: dict[str, int] = {}
    for run in runs:
        mode = str(run.get("execution_mode", "UNKNOWN"))
        model_modes[mode] = model_modes.get(mode, 0) + 1
    report = {
        "fault_type_top1": ratio("fault_type_top1_hit"),
        "fault_type_top3": ratio("fault_type_top3_hit"),
        "component_top1": ratio("component_top1_hit"),
        "component_top3": ratio("component_top3_hit"),
        "block_recall_at_3": {
            "status": "EVALUATED" if block_recall_rows else "NOT_EVALUATED",
            "value": sum(block_recall_rows) / len(block_recall_rows) if block_recall_rows else None,
            "n": len(block_recall_rows),
            "reason": None if block_recall_rows else "no independently adjudicated functional block IDs"},
        "joint_top1": ratio("joint_top1_hit"), "joint_top3": ratio("joint_top3_hit"),
        "top1_accuracy": ratio("fault_type_top1_hit"), "top3_hit_rate": ratio("fault_type_top3_hit"),
        "healthy_board_specificity": {"value": (sum(bool(row.get("healthy_correct")) for row in scored if row.get("healthy_case")) /
                sum(bool(row.get("healthy_case")) for row in scored)) if any(row.get("healthy_case") for row in scored) else None,
            "n": sum(bool(row.get("healthy_case")) for row in scored),
            "status": "EVALUATED" if any(row.get("healthy_case") for row in scored) else "NOT_EVALUATED"},
        "diagnostic_abstention_rate": {
            "value": sum(bool(row.get("abstained")) for row in abstention) / len(abstention) if abstention else None,
            "n": len(abstention), "status": "EVALUATED" if abstention else "NOT_EVALUATED"},
        "unsupported_specific_attribution_rate": {"status": "EVALUATED" if attribution_rows else "NOT_EVALUATED",
            "n": len(attribution_rows),
            "value": sum(bool(item.get("unsupported_specific_attribution")) for item in attribution_rows) / len(attribution_rows) if attribution_rows else None},
        "detection_precision_recall_f1": {"status": "EVALUATED" if detection_scored else "NOT_EVALUATED",
            "n_cases": len(detection_scored), "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "reason": None if detection_scored else "no independently adjudicated detection boxes",
            "classes": class_detection},
        "measurement_recommendation_utility": {"status": "EVALUATED" if recommendation_rows else "NOT_EVALUATED",
            "n": len(recommendation_rows), "useful_recommendation_rate": sum(recommendation_rows) / len(recommendation_rows) if recommendation_rows else None,
            "reason": None if recommendation_rows else "no independently verified case-level useful-test labels"},
        "active_inspection_planning": {
            "status": "EVALUATED" if separation_rows else "NOT_EVALUATED",
            "planned_check_count": len(separation_rows),
            "checks_with_deterministic_hypothesis_separation": sum(
                int(row.get("separation_count", 0) > 0) for row in separation_rows),
            "checks_with_separation_rate": (
                sum(int(row.get("separation_count", 0) > 0) for row in separation_rows) / len(separation_rows)
                if separation_rows else None),
            "reason": None if separation_rows else "no recorded active-inspection plans"},
        "inspection_execution": {
            "status": "EVALUATED" if any(item.get("executed_inspection_count") is not None for item in scored)
            else "NOT_EVALUATED",
            "planned_count": sum(int(item.get("planned_inspection_count", 0) or 0) for item in scored),
            "executed_count": sum(int(item.get("executed_inspection_count", 0) or 0) for item in scored),
            "case_count": sum(item.get("executed_inspection_count") is not None for item in scored)},
        "verdict_agreement": {"status": "EVALUATED" if verdict_rows else "NOT_EVALUATED", "n": len(verdict_rows),
            "agreement": sum(verdict_rows) / len(verdict_rows) if verdict_rows else None,
            "reason": None if verdict_rows else "no adjudicated per-measurement verdict labels"},
        "measurement_rank_improvement": {"status": "EVALUATED" if ranking_rows else "NOT_EVALUATED", "n": len(ranking_rows),
            "improved": sum(value > 0 for value in ranking_rows), "unchanged": sum(value == 0 for value in ranking_rows),
            "worsened": sum(value < 0 for value in ranking_rows),
            "reason": None if ranking_rows else "no paired adjudicated before/after measurements"},
        "reassessment_runtime": {
            "status": "EVALUATED" if reassessment_rows else "NOT_EVALUATED",
            "n": len(reassessment_rows),
            "duration_ms_mean": (sum(float(row["reassessment_duration_ms"]) for row in reassessment_rows) /
                                 len(reassessment_rows) if reassessment_rows else None)},
        "qwen_calls": {
            "status": "EVALUATED" if qwen_call_rows else "NOT_EVALUATED",
            "n": len(qwen_call_rows),
            "total": sum(int(row["qwen_call_count"]) for row in qwen_call_rows) if qwen_call_rows else None},
        "model_cost": {"status": "NOT_EVALUATED", "n": 0,
            "reason": "provider billing metadata unavailable"},
        "baseline_vs_evidence_v1": {"status": "EVALUATED" if paired_baseline_scores else "NOT_EVALUATED",
            "paired_n": len(paired_baseline_scores),
            "baseline_fault_type_top1": (sum(bool(row.get("fault_type_top1_hit")) for row in baseline_fault1) / len(baseline_fault1) if baseline_fault1 else None),
            "evidence_v1_fault_type_top1": ratio("fault_type_top1_hit")["value"],
            "baseline_fault_type_top3": (sum(bool(row.get("fault_type_top3_hit")) for row in baseline_fault3) / len(baseline_fault3) if baseline_fault3 else None),
            "evidence_v1_fault_type_top3": ratio("fault_type_top3_hit")["value"],
            "reason": None if paired_baseline_scores else "runs do not contain an opt-in evidence_v1 comparison"},
        "baseline_vs_active_inspection_v1": {
            "status": "EVALUATED" if paired_active_scores else "NOT_EVALUATED",
            "paired_n": len(paired_active_scores),
            "baseline_fault_type_top1": (
                sum(bool(row.get("fault_type_top1_hit")) for row in paired_active_scores
                    if row.get("fault_type_top1_hit") is not None) /
                sum(row.get("fault_type_top1_hit") is not None for row in paired_active_scores)
                if any(row.get("fault_type_top1_hit") is not None for row in paired_active_scores) else None),
            "active_inspection_fault_type_top1": ratio("fault_type_top1_hit")["value"],
            "baseline_fault_type_top3": (
                sum(bool(row.get("fault_type_top3_hit")) for row in paired_active_scores
                    if row.get("fault_type_top3_hit") is not None) /
                sum(row.get("fault_type_top3_hit") is not None for row in paired_active_scores)
                if any(row.get("fault_type_top3_hit") is not None for row in paired_active_scores) else None),
            "active_inspection_fault_type_top3": ratio("fault_type_top3_hit")["value"],
            "fault_ranking_changed": False,
            "note": "The active inspection planner is intentionally ranking-neutral; compare next-check utility and execution metrics separately."},
        "runtime": {"runs": len(runtime_rows), "execution_modes": model_modes,
            "duration_ms_mean": sum(durations) / len(durations) if durations else None,
            "duration_ms_n": len(durations),
            "stage_error_count": sum(stage.get("status") == "ERROR" for stage in stage_rows),
            "stage_count": len(stage_rows),
            "failure_rate": sum(run.get("status") == "PIPELINE_FAILED" for run in runs) / len(runs) if runs else None,
            "memory_metrics": "see per-run runtime_metadata"},
        "cases": scores,
        "failure_categories": category_counts,
        "improvement_priorities": sorted(category_counts, key=lambda key: (-category_counts[key], key)),
    }
    if _include_dataset_breakdown:
        for kind in sorted({str(case.get("case_kind") or "UNKNOWN") for case in cases}):
            kind_cases = [case for case in cases if str(case.get("case_kind") or "UNKNOWN") == kind]
            kind_ids = {case.get("case_id") for case in kind_cases}
            report.setdefault("dataset_breakdown", {})[kind] = evaluate_runs(
                kind_cases, [run for run in runs if run.get("case_id") in kind_ids],
                _include_dataset_breakdown=False,
            )
    return report


def compare_before_after_measurement(case: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Run identical production logic without and with recorded measurements."""
    from copy import deepcopy
    baseline_case = deepcopy(case)
    baseline_case["inference_input"]["measurements"] = []
    before = run_case(baseline_case, **kwargs)
    after = run_case(case, **kwargs)
    before_rank = [row.get("fault_id") for row in before.get("all_hypotheses", before.get("hypotheses", []))]
    after_rank = [row.get("fault_id") for row in after.get("all_hypotheses", after.get("hypotheses", []))]
    return {"before": before, "after": after, "before_rank": before_rank, "after_rank": after_rank,
            "rank_changed": before_rank != after_rank}


def save_run(path: str | Path, run: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite validation run: {target}")
    target.write_text(json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return target


def save_metrics_csv(path: str | Path, report: dict[str, Any]) -> Path:
    """Save the aggregate metric values and denominators as a flat CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metric", "status", "value", "n", "details"))
        writer.writeheader()
        for key, value in report.items():
            if not isinstance(value, dict) or "status" not in value:
                continue
            writer.writerow({"metric": key, "status": value.get("status"), "value": value.get("value"),
                             "n": value.get("n", value.get("n_cases", value.get("paired_n"))),
                             "details": json.dumps(value, ensure_ascii=False, sort_keys=True)})
    return target
