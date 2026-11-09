"""Validation for report provenance and exported bundle manifests."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from circuitcare.workspace.measurement_provenance import measurement_set_sha256

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def validate_report(report: dict[str, Any], *, expected_circuit_hash: str | None = None) -> list[str]:
    errors = []
    circuit = report.get("circuit", {})
    if expected_circuit_hash and circuit.get("hash") != expected_circuit_hash:
        errors.append("STALE_CIRCUIT_CONTEXT")
    evidence_ids = {row.get("evidence_id") for row in report.get("evidence", [])}
    current_hash = circuit.get("hash")
    for evidence in report.get("evidence", []):
        evidence_hash = evidence.get("circuit_hash")
        if evidence_hash and current_hash and evidence_hash != current_hash:
            errors.append(f"CROSS_CIRCUIT_EVIDENCE:{evidence.get('evidence_id')}")
    scenario_ids = {row.get("scenario_id") for row in report.get("evidence", []) if row.get("scenario_id")}
    report_scenario = report.get("spice_verification", {}).get("scenario_id")
    if len(scenario_ids) > 1 or (report_scenario and scenario_ids and scenario_ids != {report_scenario}):
        errors.append("CROSS_SCENARIO_EVIDENCE")
    for finding in report.get("findings", []):
        if any(source not in evidence_ids for source in finding.get("source_evidence_ids", [])):
            errors.append(f"MISSING_FINDING_EVIDENCE:{finding.get('finding_id')}")
    if report.get("secondary_damage_risk", {}).get("damage_confirmed") is True:
        errors.append("UNSUPPORTED_DAMAGE_CONFIRMATION")
    measurement_hash = report.get("measurement_context", {}).get("measurement_set_sha256")
    if "measurement_context" in report:
        try:
            observed_hash = measurement_set_sha256(report.get("fault_or_symptom", {}).get("observations", []))
            if measurement_hash != observed_hash:
                errors.append("MEASUREMENT_DIAGNOSIS_MISMATCH")
        except (TypeError, ValueError):
            errors.append("MEASUREMENT_DIAGNOSIS_MISMATCH")
    diagnosis_context = report.get("diagnosis_context", {})
    diagnosis = report.get("diagnosis", {})
    if diagnosis_context.get("validation_status") == "MEASUREMENT_DIAGNOSIS_MISMATCH":
        errors.append("MEASUREMENT_DIAGNOSIS_MISMATCH")
    hypothesis_context = report.get("hypothesis_context", {})
    if hypothesis_context and hypothesis_context.get("validation_status") != "MATCHED":
        errors.append("HYPOTHESIS_CONTEXT_MISMATCH")
    elif hypothesis_context:
        hypothesis_rows = report.get("fault_influence", {}).get("hypotheses", [])
        actual_hypothesis_hash = __import__("hashlib").sha256(
            __import__("json").dumps(hypothesis_rows, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        if hypothesis_context.get("hypothesis_set_sha256") != actual_hypothesis_hash:
            errors.append("HYPOTHESIS_CONTEXT_MISMATCH")
        if hypothesis_context.get("measurement_set_sha256") != measurement_hash:
            errors.append("HYPOTHESIS_CONTEXT_MISMATCH")
    if diagnosis.get("status") not in {None, "NOT_EVALUATED"}:
        if diagnosis_context.get("measurement_set_sha256") != measurement_hash:
            errors.append("MEASUREMENT_DIAGNOSIS_MISMATCH")
        if diagnosis_context.get("circuit_hash") and current_hash and diagnosis_context["circuit_hash"] != current_hash:
            errors.append("STALE_CIRCUIT_CONTEXT")
    comparison = report.get("spice_verification", {}).get("comparison") or {}
    risk = report.get("secondary_damage_risk", {})
    comparison_scenario = comparison.get("scenario_id") or report_scenario
    risk_scenario = risk.get("scenario_id")
    if comparison and risk_scenario and comparison_scenario != risk_scenario:
        errors.append("CROSS_SCENARIO_EVIDENCE")
    fault = report.get("fault_or_symptom", {})
    for source, reference in ((comparison, "failed_component"), (risk, "failed_component")):
        if source.get(reference) and fault.get("component_ref") and source[reference] != fault["component_ref"]:
            errors.append("SIMULATION_CONTEXT_MISMATCH")
    for source in (comparison, risk):
        source_fault = source.get("fault_type")
        if source_fault and fault.get("fault_type") and source_fault != fault["fault_type"]:
            errors.append("SIMULATION_CONTEXT_MISMATCH")
    for key in ("fault_parameters", "fault_parameter", "parameter", "input_fingerprint_sha256"):
        if comparison.get(key) is not None and risk.get(key) is not None and comparison[key] != risk[key]:
            errors.append("SIMULATION_CONTEXT_MISMATCH")
    simulation_context = report.get("simulation_context")
    if report.get("spice_verification", {}).get("source") == "CACHED_SIMULATION" and not (
            simulation_context and simulation_context.get("status") == "VERIFIED"
            and SHA256_RE.fullmatch(simulation_context.get("fingerprint", {}).get("fingerprint_sha256", ""))
            and simulation_context.get("result_sha256")):
        errors.append("UNVERIFIED_SIMULATION_CACHE")
    if report.get("preview_context") and report["preview_context"].get("status") != "PREVIEW_ONLY":
        errors.append("WHAT_IF_PREVIEW_NOT_ISOLATED")
    if report.get("preview_context"):
        try:
            preview_hash = measurement_set_sha256(report.get("preview_measurements", []))
            if preview_hash != report["preview_context"].get("measurement_set_sha256"):
                errors.append("WHAT_IF_PREVIEW_CONTEXT_MISMATCH")
            if any(row.get("source") != "simulated_demo" for row in report.get("preview_measurements", [])):
                errors.append("WHAT_IF_PREVIEW_SOURCE_MISMATCH")
        except (TypeError, ValueError):
            errors.append("WHAT_IF_PREVIEW_CONTEXT_MISMATCH")
    return errors


def verify_bundle(bundle: bytes) -> dict[str, Any]:
    if len(bundle) > 50 * 1024 * 1024:
        raise ValueError("Evidence bundle exceeds the compressed size limit.")
    with zipfile.ZipFile(__import__("io").BytesIO(bundle)) as archive:
        names = archive.namelist()
        if len(names) > 64 or sum(info.file_size for info in archive.infolist()) > 50 * 1024 * 1024:
            raise ValueError("Evidence bundle exceeds the file count or expanded size limit.")
        if any(info.file_size > 25 * 1024 * 1024 for info in archive.infolist()):
            raise ValueError("Evidence bundle contains an oversized file.")
        if len(names) != len(set(names)):
            raise ValueError("Duplicate bundle paths are not allowed.")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or re.match(r"^[A-Za-z]:", name):
                raise ValueError("Unsafe path in evidence bundle.")
        manifest = json.loads(archive.read("manifest.json"))
        listed = set(manifest.get("artifacts", {}))
        if listed != (set(names) - {"manifest.json"}):
            raise ValueError("Bundle manifest and archive file list do not match.")
        for name, expected in manifest.get("artifacts", {}).items():
            if name not in names:
                raise ValueError(f"Manifest artifact missing: {name}")
            actual = hashlib.sha256(archive.read(name)).hexdigest()
            if not SHA256_RE.fullmatch(expected) or actual != expected:
                raise ValueError(f"Artifact hash mismatch: {name}")
        report = json.loads(archive.read("diagnosis.json"))
        if report.get("circuit", {}).get("hash") != manifest.get("circuit_hash"):
            raise ValueError("Bundle report and manifest circuit hashes differ.")
        if report.get("spice_verification", {}).get("scenario_id") != manifest.get("scenario_id"):
            raise ValueError("Bundle report and manifest scenario IDs differ.")
        bundled = set(listed)
        for evidence in report.get("evidence", []):
            artifact = evidence.get("source_artifact")
            if evidence.get("included_in_bundle"):
                bundled_digest = manifest["artifacts"].get(artifact)
                if artifact not in bundled or evidence.get("bundled_artifact_sha256") != bundled_digest:
                    raise ValueError(f"Report evidence reference does not match the bundle: {evidence.get('evidence_id')}")
                expected_digest = evidence.get("artifact_sha256")
                if evidence.get("bundle_verification_status") == "VERIFIED_ARTIFACT" and expected_digest != bundled_digest:
                    raise ValueError("Report marks an evidence artifact verified despite a source hash mismatch.")
                if expected_digest and expected_digest != bundled_digest and evidence.get("bundle_verification_status") != "ARTIFACT_HASH_MISMATCH":
                    raise ValueError("Evidence source hash mismatch is not clearly reported.")
            elif evidence.get("bundle_verification_status") == "VERIFIED_ARTIFACT":
                raise ValueError("Report marks an artifact verified that is absent from the bundle.")
        return manifest
