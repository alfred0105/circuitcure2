"""Load and classify one simulation result snapshot for dashboard views."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from circuitcare.dashboard.context import _simulation_status
from circuitcare.workspace.simulation_context import verify_simulation_manifest


def json_object_if_exists(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


@dataclass(frozen=True)
class AnalysisSnapshot:
    result_dir: Path
    comparison: dict
    risk: dict
    stress: dict
    manifest: dict
    integrity: str
    csv_integrity: str
    csv_sha256: str | None
    status: str


def load_analysis_snapshot(*, result_dir: Path, current_fingerprint: dict,
                           live_used: bool, live_error: str | None,
                           cached_selection: bool) -> AnalysisSnapshot:
    """Read files and require verified provenance for every successful result."""
    comparison = json_object_if_exists(result_dir / "comparison.json")
    risk = json_object_if_exists(result_dir / "risk_assessment.json")
    stress = json_object_if_exists(result_dir / "component_stress.json")
    manifest = json_object_if_exists(result_dir / "simulation_context.json")
    integrity, _ = verify_simulation_manifest(result_dir, manifest, current_fingerprint)
    result_hashes = manifest.get("result_sha256") if isinstance(manifest, dict) else None
    required_results = {"comparison.json", "risk_assessment.json", "component_stress.json"}
    if integrity == "VERIFIED" and not required_results.issubset(result_hashes or {}):
        integrity = "PROVENANCE_UNVERIFIED"
    csv_path = result_dir / "comparison.csv"
    csv_integrity = "NOT_PRESENT"
    if csv_path.exists():
        if isinstance(result_hashes, dict) and "comparison.csv" in result_hashes:
            csv_integrity = integrity
        else:
            csv_integrity = "PROVENANCE_UNVERIFIED"
            if integrity == "VERIFIED":
                integrity = "PROVENANCE_UNVERIFIED"
    elif isinstance(result_hashes, dict) and "comparison.csv" in result_hashes:
        csv_integrity = "STALE"
        integrity = "STALE"
    status = _simulation_status(live_used, live_error, cached_selection)
    if status in {"SUCCESS", "CACHED"}:
        status = ("SUCCESS" if live_used else "CACHED_SIMULATION") if integrity == "VERIFIED" else integrity
    if status in {"STALE", "PROVENANCE_UNVERIFIED"} or (
        status == "FAILED" and cached_selection and integrity != "VERIFIED"
    ):
        comparison, risk, stress = {}, {}, {}
    csv_sha256 = result_hashes.get("comparison.csv") if isinstance(result_hashes, dict) else None
    return AnalysisSnapshot(result_dir, comparison, risk, stress, manifest, integrity,
                            csv_integrity, csv_sha256, status)
