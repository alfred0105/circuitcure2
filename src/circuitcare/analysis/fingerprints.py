"""Canonical, content-based fingerprints for PCB diagnosis inputs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from circuitcare.analysis.models import AnalysisFingerprints


def canonical_json(value: Any) -> bytes:
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str,
                      allow_nan=False).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def build_input_fingerprint(payload: dict[str, Any]) -> str:
    return sha256_json(payload)


def build_symptom_input_fingerprint(
    *, project: Any, components: Iterable[Any], symptoms: list[str], symptom_text: str,
    observations: dict[str, Any], answers: dict[str, Any], silkscreen_labels: list[dict[str, Any]],
    component_inventory: dict[str, dict[str, Any]] | None, confirmed_component_ids: list[str] | None,
    inventory_complete: bool, target_id: str, target_photo_sha256: str | None = None,
    additional_inputs: dict[str, Any] | None = None,
) -> str:
    """Build the content fingerprint for one selected PCB photo and its diagnosis inputs."""
    rows = [{
        "id": getattr(row, "component_id", None),
        "type": getattr(row, "component_type", None),
        "bbox": getattr(row, "bbox_xyxy", None),
        "marking": getattr(row, "ocr_marking", None),
        "part_number": getattr(row, "possible_part_number", None),
        "info": getattr(row, "component_info", None),
        "review_status": getattr(row, "identification_status", None),
    } for row in components]
    photo = next((row for row in getattr(project, "image_records", ())
                  if str(getattr(row, "photo_id", getattr(row, "image_id", ""))) == str(target_id)), None)
    payload = {
        "project_id": project.project_id,
        "board_id": project.board_id,
        "analysis_target_id": str(target_id),
        "target_photo": ({
            "content_sha256": target_photo_sha256 or getattr(photo, "content_sha256", None),
            "dimensions": getattr(photo, "image_dimensions", None),
            "side": getattr(photo, "side", None),
        } if photo else None),
        "symptom_category": project.symptom_category,
        "symptom_description": project.symptom_description,
        "symptom_text": symptom_text,
        "symptoms": sorted(symptoms),
        "observations": observations,
        "answers": answers,
        "components": rows,
        "confirmed_component_ids": sorted(confirmed_component_ids or []),
        "component_inventory_complete": bool(inventory_complete),
        "component_inventory": component_inventory or {},
        "silkscreen_labels": silkscreen_labels,
        "additional_inputs": additional_inputs or {},
    }
    return build_input_fingerprint(payload)


def _measurement_payload(record: Any) -> dict[str, Any]:
    row = record.to_dict() if hasattr(record, "to_dict") else dict(record)
    # Record IDs, timestamps and measurement-set IDs identify history entries,
    # not the measured facts. Editing an entry without changing its facts keeps
    # the analysis current.
    for key in ("measurement_id", "timestamp", "changed_at", "measurement_set_id"):
        row.pop(key, None)
    return row


def build_measurement_fingerprint(records: Iterable[Any]) -> str:
    rows = [_measurement_payload(record) for record in records]
    rows.sort(key=canonical_json)
    return sha256_json(rows)


def build_rules_fingerprint(*, extra_payload: dict[str, Any] | None = None) -> str:
    """Fingerprint the active deterministic engine and bundled YAML rules."""
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "symptom_diagnosis" / "engine.py",
        root / "symptom_diagnosis" / "component_fault_ranker.py",
        root / "symptom_diagnosis" / "measurement_evaluator.py",
        root / "symptom_diagnosis" / "functional_blocks.py",
        root / "symptom_diagnosis" / "symptom_parser.py",
        root / "symptom_diagnosis" / "fault_hypothesis_validator.py",
        root / "symptom_diagnosis" / "schemas.py",
        *sorted((root / "symptom_diagnosis" / "knowledge").glob("*.yaml")),
    ]
    files: dict[str, str] = {}
    for path in paths:
        if path.is_file():
            files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sha256_json({"files": files, "extra": extra_payload or {}})


def build_analysis_fingerprints(
    *, input_payload: dict[str, Any], measurements: Iterable[Any],
    extra_rules_payload: dict[str, Any] | None = None,
) -> AnalysisFingerprints:
    input_hash = build_input_fingerprint(input_payload)
    measurement_hash = build_measurement_fingerprint(measurements)
    rules_hash = build_rules_fingerprint(extra_payload=extra_rules_payload)
    final_hash = sha256_json({"input": input_hash, "measurement": measurement_hash,
                              "rules": rules_hash})
    return AnalysisFingerprints(input_hash, measurement_hash, rules_hash, final_hash)
