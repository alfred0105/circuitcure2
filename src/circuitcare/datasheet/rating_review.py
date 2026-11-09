"""Append-only user review records for candidate datasheet ratings."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from circuitcare.workspace.atomic_json import write_json_atomic


def applied_rating_session_key(context_key: str, stress_signature: str | None) -> str:
    """Return the one session cache key for a circuit/scenario/stress rating application."""
    payload = json.dumps({"context": context_key, "stress_signature": stress_signature},
                         sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"datasheet_applied_{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def save_review(record: dict[str, Any], path: str | Path) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        history = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        history = {"records": []}
    if not isinstance(history, dict) or not isinstance(history.get("records"), list):
        raise ValueError("Review history has an invalid structure; refusing to overwrite it.")  # noqa: TRY004 - preserve public validation contract
    saved = {**record, "reviewed_at": datetime.now(UTC).isoformat(),
             "review_source": "user_review", "human_approved": record.get("decision") == "approved"}
    history["records"].append(saved)
    write_json_atomic(target, history)
    return saved


def review_matches_context(record: dict[str, Any], *, sha256: str, mpn: str, reference: str) -> bool:
    return record.get("source_document_sha256") == sha256 and record.get("mpn", "").casefold() == mpn.casefold() and record.get("component_reference") == reference


def load_reviews(path: str | Path, *, sha256: str, mpn: str, reference: str, revision: str,
                 extraction_model: str | None = None, model_digest: str | None = None) -> list[dict[str, Any]]:
    try:
        records = json.loads(Path(path).read_text(encoding="utf-8")).get("records", [])
    except (OSError, json.JSONDecodeError, AttributeError):
        return []
    latest: dict[tuple, dict[str, Any]] = {}
    for record in records:
        if (record.get("source_document_sha256") != sha256
                or record.get("mpn", "").casefold() != mpn.casefold()
                or record.get("component_reference") != reference
                or record.get("document_revision", "") != revision
                or record.get("llm_model") != extraction_model
                or record.get("llm_model_digest") != model_digest):
            continue
        original = record.get("original_candidate") or record
        key = (original.get("rating_key"), original.get("pdf_page_index"), original.get("source_text"))
        latest[key] = record
    return list(latest.values())


def save_application(record: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        data = {"applications": []}
    if not isinstance(data, dict) or not isinstance(data.get("applications"), list):
        raise ValueError("Applied-rating registry is invalid; refusing to overwrite it.")  # noqa: TRY004 - preserve public validation contract
    data["applications"].append({"event": "applied", **record, "applied_at": datetime.now(UTC).isoformat()})
    write_json_atomic(target, data)


def find_applications(path: str | Path, *, circuit_path: str, scenario_id: str, stress_sha256: str) -> list[dict[str, Any]]:
    try:
        rows = json.loads(Path(path).read_text(encoding="utf-8")).get("applications", [])
    except (OSError, json.JSONDecodeError, AttributeError):
        return []
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if (row.get("circuit_path") == circuit_path and row.get("scenario_id") == scenario_id
                and row.get("stress_signature") == stress_sha256):
            reference = row.get("component_ref") or (row.get("record") or {}).get("component_reference")
            if reference:
                latest[reference] = row
    return [row for row in latest.values() if row.get("event") != "revoked"]


def find_application(path: str | Path, *, circuit_path: str, scenario_id: str, stress_sha256: str) -> dict[str, Any] | None:
    """Compatibility helper for callers expecting one active application."""
    rows = find_applications(path, circuit_path=circuit_path, scenario_id=scenario_id, stress_sha256=stress_sha256)
    return rows[-1] if rows else None
