"""Input fingerprints for simulations and verification of cached result manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from circuitcare.workspace.context import file_sha256


def build_simulation_fingerprint(*, netlist: Path, spice_deck: Path,
                                 component_mapping: Path, probes: Path,
                                 fault: dict[str, Any], settings: dict[str, Any] | None = None,
                                 relevant_models: list[Path] | None = None,
                                 ratings: Path | None = None) -> dict[str, Any]:
    model_file_hashes: dict[str, str | None] = {
        _portable_model_name(path, spice_deck.parent): file_sha256(path)
        for path in sorted(relevant_models or [])
    }
    inputs: dict[str, Any] = {
        "netlist_sha256": file_sha256(netlist),
        "spice_deck_sha256": file_sha256(spice_deck),
        "mapping_sha256": file_sha256(component_mapping),
        "probes_sha256": file_sha256(probes),
        "ratings_sha256": file_sha256(ratings),
        "model_files_sha256": model_file_hashes,
        "fault_scenario": fault,
        "simulation_settings": settings or {},
    }
    if any(inputs[key] is None for key in ("netlist_sha256", "spice_deck_sha256", "mapping_sha256", "probes_sha256")) \
            or any(value is None for value in inputs["model_files_sha256"].values()):
        return {**inputs, "fingerprint_sha256": None}
    serialized = json.dumps(inputs, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return {**inputs, "fingerprint_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest()}


def _portable_model_name(path: Path, deck_directory: Path) -> str:
    try:
        return path.resolve().relative_to(deck_directory.resolve()).as_posix()
    except ValueError:
        return f"external/{path.name}"


def verify_cached_fingerprint(cached: dict[str, Any] | None,
                              current: dict[str, Any]) -> tuple[str, list[str]]:
    cached_digest = cached.get("fingerprint_sha256") if isinstance(cached, dict) else None
    current_digest = current.get("fingerprint_sha256") if isinstance(current, dict) else None
    if (not isinstance(cached_digest, str) or len(cached_digest) != 64
            or not isinstance(current_digest, str) or len(current_digest) != 64):
        return "PROVENANCE_UNVERIFIED", ["UNVERIFIED_SIMULATION_CACHE"]
    if cached_digest != current_digest:
        return "STALE", ["UNVERIFIED_SIMULATION_CACHE"]
    return "VERIFIED", []


def verify_simulation_manifest(result_dir: Path, manifest: dict[str, Any] | None,
                               current: dict[str, Any]) -> tuple[str, list[str]]:
    if not isinstance(manifest, dict):
        return "PROVENANCE_UNVERIFIED", ["UNVERIFIED_SIMULATION_CACHE"]
    status, errors = verify_cached_fingerprint(manifest.get("fingerprint"), current)
    if status != "VERIFIED":
        return status, errors
    expected = manifest.get("result_sha256")
    if not isinstance(expected, dict) or not expected:
        return "PROVENANCE_UNVERIFIED", ["UNVERIFIED_SIMULATION_CACHE"]
    for name, digest in expected.items():
        if Path(name).name != name:
            return "PROVENANCE_UNVERIFIED", ["UNVERIFIED_SIMULATION_CACHE"]
        actual = file_sha256(result_dir / name)
        if actual != digest:
            return "STALE", ["UNVERIFIED_SIMULATION_CACHE"]
    return "VERIFIED", []
