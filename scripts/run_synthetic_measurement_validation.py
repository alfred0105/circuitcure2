"""Exercise production measurement reassessment on cached full-PCB detections.

All assignments and readings produced here are synthetic. Cached YOLO boxes
provide spatial distractors only; they are not component or fault ground truth.
The script never writes to the source dataset, Registry store, or measurement
history. Its only output is the requested JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circuitcare.benchmark.pcb_validation import run_case  # noqa: E402
from circuitcare.component_taxonomy import normalize_component_class  # noqa: E402
from circuitcare.frontend.board_component_registry import (  # noqa: E402
    build_board_component_registry,
    current_source_observation_fingerprint,
    source_component_key,
)


FAULT_TESTS = {
    "resistor": ("RESISTOR_FAILURE", "isolated_resistance", "resistance", "저항 양단 단자",
                 10.0, 1.0, 2.0, ["LOCALIZED_HEATING"], "저항 주변이 뜨겁고 일부 기능이 불안정합니다."),
    "fuse": ("FUSE_OPEN", "continuity", "resistance", "퓨즈 양단 단자",
             100.0, 0.0, 1.0, ["NO_POWER"], "전원을 연결해도 기판이 켜지지 않습니다."),
    "switch": ("SWITCH_CONTACT_FAILURE", "contact_resistance", "resistance", "스위치 접점 단자",
               100.0, 0.0, 0.5, ["SWITCH_NO_RESPONSE"], "스위치를 조작해도 기대한 반응이 없습니다."),
    "diode": ("DIODE_SHORT_FAILURE", "forward_drop", "voltage", "다이오드 애노드 캐소드 순방향",
              0.01, 0.5, 0.8, ["LOCALIZED_HEATING"], "다이오드 부근이 뜨겁고 전원이 불안정합니다."),
    "led": ("LED_FAILURE", "isolated_forward_drop", "voltage", "LED 양단 애노드 캐소드",
            9.0, 1.5, 2.5, ["LED_NOT_LIT"], "전원은 들어오지만 표시 LED 하나만 켜지지 않습니다."),
}


def _readings_for(
    *, selected: dict[str, dict[str, Any]], photo_id: str, photo_sha: str,
    fingerprints: dict[str, str], board_id: str, project_id: str,
) -> dict[str, dict[str, Any]]:
    readings = {}
    for component_type, component in selected.items():
        fault, test_id, measurement_type, target, value, low, high, _symptoms, _text = FAULT_TESTS[component_type]
        local_id = str(component["component_id"])
        readings[component_type] = {
            "measurement_id": f"synthetic-{component_type}-{photo_id}",
            "project_id": project_id, "board_id": board_id,
            "component_id": local_id, "source_component_id": local_id,
            "source_photo_id": photo_id, "source_photo_fingerprint": photo_sha,
            "source_observation_fingerprint": fingerprints[local_id],
            "component_map_version": "synthetic-review-v1",
            "component_identity_status": "USER_CONFIRMED",
            "identity_resolution_status": "USER_RECONFIRMED_VERIFIED",
            "fault_hypothesis": fault, "test_id": test_id,
            "measurement_target": target, "measurement_location": target,
            "operating_conditions": "POWER_OFF_CONFIRMED",
            "reference_conditions": "POWER_OFF_CONFIRMED",
            "measurement_conditions": {
                "power_state": "POWER_OFF_CONFIRMED",
                "stored_energy_status": "SAFE_CONFIRMED",
                "component_isolation_status": "ISOLATED_CONFIRMED",
                "measurement_location_verified": True,
                "reference_source_verified": True,
            },
            "location_verified": True, "source": "user_measurement",
            "verification_status": "USER_REPORTED", "uncertainty": 0.01,
            "uncertainty_source": "SYNTHETIC_FIXTURE_ONLY",
            "reference_min": low, "reference_max": high,
            "reference_source": "SYNTHETIC_FIXTURE_ONLY",
            "reference_source_id": "SYNTHETIC_FIXTURE_ONLY",
            "reference_verification_status": "DATASHEET_VERIFIED",
            "measurement_type": measurement_type,
            "unit": "V" if measurement_type == "voltage" else "ohm",
            "value": value,
        }
    return readings


def main() -> int:
    default_root = Path.home() / ".local/share/circuitcare-ai/external_datasets/epid/validation"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=default_root / "registered-cases.json")
    parser.add_argument("--detection-runs", type=Path, default=default_root / "candidate/runs")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/synthetic-measurement-validation.json")
    args = parser.parse_args()

    cases = json.loads(args.source_manifest.expanduser().read_text(encoding="utf-8"))["cases"]
    detections_dir = args.detection_runs.expanduser()
    output_rows: list[dict[str, Any]] = []
    multifault_rows: list[dict[str, Any]] = []
    missing_runs: list[str] = []
    started = time.perf_counter()

    for photo_case in cases:
        case_id = str(photo_case["case_id"])
        run_paths = sorted(detections_dir.glob(f"{case_id}-*.json"))
        if not run_paths:
            missing_runs.append(case_id)
            continue
        detection_run = json.loads(run_paths[-1].read_text(encoding="utf-8"))
        photo_path = Path(photo_case["inference_input"]["photo_path"]).expanduser()
        photo_id = photo_path.name
        photo_sha = hashlib.sha256(photo_path.read_bytes()).hexdigest()
        components: dict[str, dict[str, Any]] = {}
        targets: dict[str, dict[str, Any]] = {}
        for row in detection_run.get("detections", []):
            if not isinstance(row, dict):
                continue
            kind = normalize_component_class(str(
                row.get("class_name") or row.get("component_type") or row.get("소자 종류") or "unknown"))
            local_id = str(row.get("component_id") or row.get("detection_id") or "")
            if not local_id or local_id in components or kind in {"unknown", "not_component"}:
                continue
            raw_box = row.get("bbox_xyxy") or [row.get(key) for key in ("x1", "y1", "x2", "y2")]
            if not isinstance(raw_box, (tuple, list)) or len(raw_box) != 4 or any(value is None for value in raw_box):
                continue
            item = {
                "component_id": local_id, "component_type": kind,
                "bbox_xyxy": [float(value) for value in raw_box],
                # Test-only assumption. These are model labels, not human-verified ground truth.
                "identification_status": "USER_CONFIRMED",
                "presence_status": "CONFIRMED", "component_type_status": "USER_CONFIRMED",
                "detection_id": local_id, "source": "synthetic_confirmation_assumption",
            }
            components[local_id] = item
            if kind in FAULT_TESTS:
                targets.setdefault(kind, item)
        if not components:
            continue

        project_id = "synthetic-measurement-validation"
        board_id = f"synthetic-board-{case_id}"
        snapshot = {
            "version": "synthetic-review-v1", "image_fingerprint": photo_sha,
            "review_version": "synthetic-review-v1",
            "registry_components": list(components.values()),
        }
        registry = build_board_component_registry(
            {}, project_id=project_id, board_id=board_id, photo_snapshots={photo_id: snapshot})
        fingerprints = {
            local_id: current_source_observation_fingerprint(registry, photo_id=photo_id, component_id=local_id)
            for local_id in components
        }
        board_ids = {
            local_id: registry["source_to_board_id"][source_component_key(photo_id, local_id)]
            for local_id in components
            if source_component_key(photo_id, local_id) in registry["source_to_board_id"]
        }

        for component_type, target in targets.items():
            fault_id, _test, _measure_type, _target, _value, _low, _high, symptoms, symptom_text = FAULT_TESTS[component_type]
            base_case = {
                "case_id": f"synthetic-{case_id}-{component_type}",
                "board_id": board_id, "case_kind": "SYNTHETIC", "photo_paths": [],
                "inference_input": {
                    "project_id": project_id, "photo_id": photo_id,
                    "photo_snapshots": {photo_id: snapshot},
                    "symptoms": symptoms, "symptom_text": symptom_text,
                    "observations": [], "measurements": [],
                    "skip_question_value_simulation": True,
                },
            }
            before_started = time.perf_counter()
            before = run_case(base_case, execution_mode="SYNTHETIC")
            before_run_ms = round((time.perf_counter() - before_started) * 1000, 3)
            measurement = _readings_for(
                selected={component_type: target}, photo_id=photo_id, photo_sha=photo_sha,
                fingerprints=fingerprints, board_id=board_id, project_id=project_id,
            )[component_type]
            after_case = {**base_case, "inference_input": {
                **base_case["inference_input"], "measurements": [measurement]}}
            after_started = time.perf_counter()
            after = run_case(after_case, execution_mode="SYNTHETIC")
            after_run_ms = round((time.perf_counter() - after_started) * 1000, 3)
            expected_board_id = board_ids.get(str(target["component_id"]))

            def target_candidate(result: dict[str, Any]) -> dict[str, Any] | None:
                return next((row for row in result.get("all_component_candidates", [])
                             if row.get("component_id") == expected_board_id
                             and row.get("fault_hypothesis") == fault_id), None)

            before_row, after_row = target_candidate(before), target_candidate(after)
            before_rank = next((index for index, row in enumerate(sorted(
                before.get("all_component_candidates", []),
                key=lambda item: (-float(item.get("score") or 0), str(item.get("component_id") or ""),
                                  str(item.get("fault_hypothesis") or "")),
            ), 1) if row.get("component_id") == expected_board_id and row.get("fault_hypothesis") == fault_id), None)
            after_rank = next((index for index, row in enumerate(sorted(
                after.get("all_component_candidates", []),
                key=lambda item: (-float(item.get("score") or 0), str(item.get("component_id") or ""),
                                  str(item.get("fault_hypothesis") or "")),
            ), 1) if row.get("component_id") == expected_board_id and row.get("fault_hypothesis") == fault_id), None)
            evidence = next((row for row in after.get("measurement_evidence", [])
                             if row.get("measurement_id") == measurement["measurement_id"]), {})
            output_rows.append({
                "photo_case_id": case_id, "photo_id": photo_id, "component_type": component_type,
                "board_component_id": expected_board_id, "synthetic_fault_hypothesis": fault_id,
                "same_type_distractor_count": max(0, sum(item["component_type"] == component_type
                                                          for item in components.values()) - 1),
                "before_measurement": {
                    "candidate_rank": before_rank,
                    "status": before_row.get("visual_status") if before_row else "NOT_CANDIDATE",
                    "run_ms": before_run_ms,
                },
                "after_one_synthetic_measurement": {
                    "candidate_rank": after_rank,
                    "status": after_row.get("visual_status") if after_row else "NOT_CANDIDATE",
                    "assessment": evidence.get("assessment"), "verdict": evidence.get("verdict"),
                    "run_ms": after_run_ms,
                },
            })

        # Paired-fault sweep: keep every detected box as a distractor and assign
        # one hypothetical resistor fault plus one hypothetical LED fault.
        if "resistor" in targets and "led" in targets:
            selected = {"resistor": targets["resistor"], "led": targets["led"]}
            readings = _readings_for(
                selected=selected, photo_id=photo_id, photo_sha=photo_sha,
                fingerprints=fingerprints, board_id=board_id, project_id=project_id,
            )
            base_case = {
                "case_id": f"synthetic-multifault-{case_id}",
                "board_id": board_id, "case_kind": "SYNTHETIC", "photo_paths": [],
                "inference_input": {
                    "project_id": project_id, "photo_id": photo_id,
                    "photo_snapshots": {photo_id: snapshot},
                    "symptoms": ["LOCALIZED_HEATING", "LED_NOT_LIT", "OTHER_FUNCTIONS_WORKING"],
                    "symptom_text": "기판은 동작하지만 LED 하나가 켜지지 않고 저항 근처가 뜨겁습니다.",
                    "observations": [], "measurements": [],
                    "skip_question_value_simulation": True,
                },
            }
            initial_started = time.perf_counter()
            initial = run_case(base_case, execution_mode="SYNTHETIC")
            initial_run_ms = round((time.perf_counter() - initial_started) * 1000, 3)
            one_case = {**base_case, "inference_input": {
                **base_case["inference_input"], "measurements": [readings["resistor"]]}}
            after_resistor_started = time.perf_counter()
            after_one = run_case(one_case, execution_mode="SYNTHETIC")
            after_resistor_run_ms = round((time.perf_counter() - after_resistor_started) * 1000, 3)
            both_case = {**base_case, "inference_input": {
                **base_case["inference_input"], "measurements": [readings["resistor"], readings["led"]]}}
            after_led_started = time.perf_counter()
            after_both = run_case(both_case, execution_mode="SYNTHETIC")
            after_led_run_ms = round((time.perf_counter() - after_led_started) * 1000, 3)

            def state_for(result: dict[str, Any], component_type: str) -> str:
                fault_id = FAULT_TESTS[component_type][0]
                board_component_id = board_ids.get(str(targets[component_type]["component_id"]))
                row = next((item for item in result.get("all_component_candidates", [])
                            if item.get("component_id") == board_component_id
                            and item.get("fault_hypothesis") == fault_id), None)
                return str(row.get("visual_status") if row else "NOT_CANDIDATE")

            multifault_rows.append({
                "photo_case_id": case_id, "photo_id": photo_id,
                "component_count": len(components),
                "resistor_same_type_distractors": max(0, sum(item["component_type"] == "resistor"
                                                              for item in components.values()) - 1),
                "led_same_type_distractors": max(0, sum(item["component_type"] == "led"
                                                        for item in components.values()) - 1),
                "initial": {"resistor": state_for(initial, "resistor"), "led": state_for(initial, "led")},
                "after_resistor_reading": {"resistor": state_for(after_one, "resistor"),
                                           "led": state_for(after_one, "led")},
                "after_led_reading": {"resistor": state_for(after_both, "resistor"),
                                      "led": state_for(after_both, "led")},
                "cycles_to_support_both": 2,
                "diagnosis_run_ms": {
                    "initial": initial_run_ms,
                    "after_resistor": after_resistor_run_ms,
                    "after_led": after_led_run_ms,
                },
                "unrelated_resistor_supported_after_one": sum(
                    row.get("component_type") == "resistor" and row.get("component_id") != board_ids.get(str(targets["resistor"]["component_id"]))
                    and row.get("visual_status") == "FAULT_SUPPORTED"
                    for row in after_one.get("all_component_candidates", [])),
                "unrelated_led_supported_after_resistor_reading": sum(
                    row.get("component_type") == "led" and row.get("component_id") != board_ids.get(str(targets["led"]["component_id"]))
                    and row.get("visual_status") == "FAULT_SUPPORTED"
                    for row in after_one.get("all_component_candidates", [])),
            })
        print(f"측정 시뮬레이션 {case_id} · 소자별 {sum(row['photo_case_id'] == case_id for row in output_rows)}건 · "
              f"복수 고장 {'실행' if 'resistor' in targets and 'led' in targets else '대상 소자 없음'}", flush=True)

    if missing_runs:
        raise FileNotFoundError(f"Missing cached detection runs for {len(missing_runs)} photos: {missing_runs[:5]}")

    component_summary: dict[str, dict[str, Any]] = {}
    all_run_ms = [timing for row in output_rows for timing in (
        row["before_measurement"]["run_ms"], row["after_one_synthetic_measurement"]["run_ms"])]
    all_run_ms.extend(timing for row in multifault_rows for timing in row["diagnosis_run_ms"].values())
    sorted_run_ms = sorted(all_run_ms)
    for component_type in FAULT_TESTS:
        group = [row for row in output_rows if row["component_type"] == component_type]
        before_top1 = sum(row["before_measurement"]["candidate_rank"] == 1 for row in group)
        before_top3 = sum(row["before_measurement"]["candidate_rank"] is not None
                          and row["before_measurement"]["candidate_rank"] <= 3 for row in group)
        after_top1 = sum(row["after_one_synthetic_measurement"]["candidate_rank"] == 1 for row in group)
        after_top3 = sum(row["after_one_synthetic_measurement"]["candidate_rank"] is not None
                         and row["after_one_synthetic_measurement"]["candidate_rank"] <= 3 for row in group)
        supported = sum(row["after_one_synthetic_measurement"]["verdict"] == "supports" for row in group)
        component_summary[component_type] = {
            "cases": len(group),
            "candidate_before": sum(row["before_measurement"]["candidate_rank"] is not None for row in group),
            "top1_before": {"hits": before_top1, "count": len(group)},
            "top3_before": {"hits": before_top3, "count": len(group)},
            "top1_after": {"hits": after_top1, "count": len(group)},
            "top3_after": {"hits": after_top3, "count": len(group)},
            "measurement_supported": supported,
            "mean_rank_before": statistics.mean(
                row["before_measurement"]["candidate_rank"] for row in group
                if row["before_measurement"]["candidate_rank"] is not None) if any(
                    row["before_measurement"]["candidate_rank"] is not None for row in group) else None,
            "mean_rank_after": statistics.mean(
                row["after_one_synthetic_measurement"]["candidate_rank"] for row in group
                if row["after_one_synthetic_measurement"]["candidate_rank"] is not None) if any(
                    row["after_one_synthetic_measurement"]["candidate_rank"] is not None for row in group) else None,
            "mean_diagnosis_run_ms": round(statistics.mean(
                [row["before_measurement"]["run_ms"] for row in group]
                + [row["after_one_synthetic_measurement"]["run_ms"] for row in group]), 3),
        }
    multifault_both = sum(row["after_led_reading"]["resistor"] == "FAULT_SUPPORTED"
                          and row["after_led_reading"]["led"] == "FAULT_SUPPORTED" for row in multifault_rows)
    multifault_one_scope = sum(row["after_resistor_reading"]["resistor"] == "FAULT_SUPPORTED"
                               and row["after_resistor_reading"]["led"] == "SUSPECTED" for row in multifault_rows)
    report = {
        "status": "SYNTHETIC_ONLY_NOT_REAL_FAULT_ACCURACY",
        "execution_mode": "PRODUCTION_RUN_CASE_SYNTHETIC_WITH_CACHED_YOLO_BOXES",
        "source_manifest": str(args.source_manifest.expanduser().resolve()),
        "detection_runs": str(detections_dir.resolve()),
        "source_photos_with_component_tests": len({row["photo_id"] for row in output_rows}),
        "component_case_count": len(output_rows),
        "component_type_summary": component_summary,
        "multifault_case_count": len(multifault_rows),
        "diagnosis_run_time_ms": {
            "count": len(sorted_run_ms),
            "mean": round(statistics.mean(sorted_run_ms), 3) if sorted_run_ms else None,
            "p50": round(statistics.median(sorted_run_ms), 3) if sorted_run_ms else None,
            "p95": (sorted_run_ms[min(len(sorted_run_ms) - 1, int((len(sorted_run_ms) - 1) * 0.95))]
                    if sorted_run_ms else None),
        },
        "multifault_measurement_cycles": {
            "min": 2 if multifault_rows else None,
            "median": 2 if multifault_rows else None,
            "max": 2 if multifault_rows else None,
        },
        "first_measurement_scope_expected_cases": multifault_one_scope,
        "both_faults_supported_after_second_reading_cases": multifault_both,
        "unrelated_fault_support_count": sum(
            row["unrelated_resistor_supported_after_one"] + row["unrelated_led_supported_after_resistor_reading"]
            for row in multifault_rows),
        "real_confirmed_fault_ground_truth_count": 0,
        "synthetic_fixture_ranges": "Invented only to exercise profile validation/ranking; not board specifications or field advice.",
        "assumptions": [
            "Cached YOLO boxes provide spatial inputs and same-type distractors; they are not gold labels.",
            "Named detector classes are marked user-confirmed only inside the synthetic input fixture.",
            "A configured abnormal reading is injected into each assigned target; this is not real fault verification.",
            "Every reported status is a fault-hypothesis status, not a component-wide fault confirmation.",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "component_cases": output_rows,
        "multifault_cases": multifault_rows,
    }
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"component_cases", "multifault_cases"}}, ensure_ascii=False, indent=2))
    print(f"결과 저장 · {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
