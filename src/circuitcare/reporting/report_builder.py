"""Build reports from supplied snapshots; this module never runs analysis."""

from __future__ import annotations

import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from circuitcare.workspace.measurement_provenance import (
    measurement_set_sha256,
    measurement_source_class,
    normalize_measurements,
)


def _git_sha(root: Path) -> str | None:
    """Read the checkout revision without spawning a child during UI reruns."""
    try:
        git_dir = root / ".git"
        if git_dir.is_file():
            pointer = git_dir.read_text(encoding="utf-8").strip()
            if not pointer.startswith("gitdir:"):
                return None
            git_dir = (root / pointer.partition(":")[2].strip()).resolve()
        head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
        if re.fullmatch(r"[0-9a-fA-F]{40,64}", head):
            return head
        if not head.startswith("ref: "):
            return None
        ref = head[5:]
        ref_path = git_dir / ref
        if ref_path.is_file():
            sha = ref_path.read_text(encoding="ascii").strip()
            return sha if re.fullmatch(r"[0-9a-fA-F]{40,64}", sha) else None
        for line in (git_dir / "packed-refs").read_text(encoding="ascii").splitlines():
            if line and not line.startswith(("#", "^")):
                sha, packed_ref = line.split(" ", 1)
                if packed_ref == ref and re.fullmatch(r"[0-9a-fA-F]{40,64}", sha):
                    return sha
    except (OSError, ValueError):
        return None
    return None


def _portable_artifact(value: Any, key: str = "") -> Any:
    """Remove machine-specific absolute paths from exported result payloads."""
    if isinstance(value, dict):
        omitted = {"private_notes", "created_by", "source_case_id", "local_paths", "home_directory",
                   "database_path", "image_path", "pdf_path", "absolute_path"}
        return {name: _portable_artifact(item, name) for name, item in value.items()
                if name.casefold() not in omitted}
    if isinstance(value, list):
        return [_portable_artifact(item, key) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return Path(value).name
    if isinstance(value, str):
        value = re.sub(r"(?:/Users/[^\s]+|/home/[^\s]+|/tmp/[^\s]+|[A-Za-z]:\\Users\\[^\s]+)",
                       "[local path redacted]", value)
    return value


def classify_parameter_grid_coverage(*, component_ref: str, fault_type: str,
                                     fault_parameter: str | None,
                                     rule_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare a supplied fault parameter to already-generated catalog candidates."""
    if not fault_parameter or not rule_candidates:
        return {"status": "NOT_EVALUATED", "candidate_count": 0}
    parameter_name = "fault_resistance" if fault_type == "resistance_drift" else (
        "output_voltage" if fault_type in {"regulator_output_rise", "regulator_output_drop"} else None)
    if parameter_name is None:
        return {"status": "UNSUPPORTED_FAULT_MODEL", "candidate_count": 0}
    rows = [row for row in rule_candidates
            if row.get("component_ref") == component_ref and row.get("fault_type") == fault_type
            and parameter_name in row.get("parameters", {})]
    if not rows:
        return {"status": "NOT_EVALUATED", "candidate_count": 0}
    from circuitcare.fault_ai.fault_catalog import _spice_number
    try:
        target = _spice_number(str(fault_parameter))
        values = [_spice_number(str(row["parameters"][parameter_name])) for row in rows]
    except (ValueError, TypeError, KeyError):
        return {"status": "NOT_EVALUATED", "candidate_count": len(rows)}
    matched = any(abs(value - target) <= max(abs(target), 1.0) * 1e-12 for value in values)
    return {"status": "CATALOG_CONTAINS_EXACT_PARAMETER" if matched else "EXACT_FAULT_PARAMETER_NOT_IDENTIFIED",
            "candidate_count": len(rows), "target_parameter": fault_parameter,
            "catalog_parameters": sorted(set(values)),
            "interpretation": "Candidate coverage only; presence does not mean a diagnosis system selected the parameter."}


def build_diagnostic_snapshot(*, root: Path, workspace: dict[str, Any], circuit: Any,
                              fault: dict[str, Any], simulation_status: str,
                              comparison: dict[str, Any] | None = None,
                              risk_assessment: dict[str, Any] | None = None,
                              observations: list[dict[str, Any]] | None = None,
                              diagnosis: dict[str, Any] | None = None,
                              inspection_recommendations: dict[str, Any] | None = None,
                              parameter_identification: dict[str, Any] | None = None,
                              active_fault_candidates: dict[str, Any] | None = None,
                              diagnosis_context: dict[str, Any] | None = None,
                              simulation_context: dict[str, Any] | None = None,
                              preview_context: dict[str, Any] | None = None,
                              preview_measurements: list[dict[str, Any]] | None = None,
                              preview_diagnosis: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect currently available values; never reads private data or reruns engines."""
    components = getattr(circuit, "components", {})
    connections = getattr(circuit, "connections", [])
    components_dict = {key: vars(value) if hasattr(value, "__dict__") else value
                       for key, value in components.items()}
    unique_nets = sorted({str(getattr(connection, "net", "")) for connection in connections})
    now = datetime.now(timezone.utc).isoformat()
    effective_simulation_status = simulation_status
    simulation_context = dict(simulation_context or {})
    cached_fingerprint = simulation_context.get("fingerprint", {})
    cached_verified = (simulation_context.get("status") == "VERIFIED"
                       and isinstance(cached_fingerprint, dict)
                       and len(cached_fingerprint.get("fingerprint_sha256", "")) == 64
                       and bool(simulation_context.get("result_sha256")))
    if simulation_status in {"CACHED", "CACHED_SIMULATION"} and not cached_verified:
        comparison, risk_assessment = None, None
        effective_simulation_status = "PROVENANCE_UNVERIFIED"
    elif simulation_status == "STALE":
        comparison, risk_assessment = None, None
    comparison = _portable_artifact(comparison) if comparison else None
    risk_assessment = _portable_artifact(risk_assessment) if risk_assessment else None
    scenario_id = (risk_assessment or {}).get("scenario_id") or (comparison or {}).get("scenario_id")
    simulation_source = ("LIVE_SIMULATION" if simulation_status == "SUCCESS" else
                         "CACHED_SIMULATION" if comparison and cached_verified else
                         "PROVENANCE_UNVERIFIED" if effective_simulation_status == "PROVENANCE_UNVERIFIED" else
                         "NOT_EVALUATED")
    measurements = normalize_measurements(observations or [])
    measurements = _portable_artifact(measurements)
    preview_measurements = normalize_measurements(preview_measurements or [])
    preview_measurements = _portable_artifact(preview_measurements)
    measurement_hash = measurement_set_sha256(measurements)
    diagnosis_context = dict(diagnosis_context or {})
    diagnosis_mismatch = bool(diagnosis and diagnosis_context.get("measurement_set_sha256")
                              and diagnosis_context["measurement_set_sha256"] != measurement_hash)
    if diagnosis_mismatch:
        diagnosis = {"status": "NOT_EVALUATED", "reason": "MEASUREMENT_DIAGNOSIS_MISMATCH"}
    diagnosis = _portable_artifact(diagnosis or {"status": "NOT_EVALUATED"})
    inspection_recommendations = _portable_artifact(inspection_recommendations or {"status": "NOT_EVALUATED"})
    raw_candidate_rows = (active_fault_candidates or {}).get("hypotheses", [])
    source_hypothesis_hash = hashlib.sha256(json.dumps(raw_candidate_rows, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    active_fault_candidates = _portable_artifact(active_fault_candidates or {"status": "NOT_EVALUATED"})
    fault = _portable_artifact(fault)
    workspace = _portable_artifact(workspace)
    candidate_measurement_hash = active_fault_candidates.get("measurement_set_sha256")
    candidate_rows = active_fault_candidates.get("hypotheses", [])
    actual_hypothesis_hash = hashlib.sha256(json.dumps(candidate_rows, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    candidate_context_status = "MATCHED"
    if candidate_measurement_hash and candidate_measurement_hash != measurement_hash:
        candidate_context_status = "MEASUREMENT_HYPOTHESIS_MISMATCH"
    if active_fault_candidates.get("hypothesis_set_sha256") and active_fault_candidates["hypothesis_set_sha256"] != source_hypothesis_hash:
        candidate_context_status = "HYPOTHESIS_SET_MISMATCH"
    if candidate_context_status != "MATCHED":
        active_fault_candidates["hypotheses"] = []
        active_fault_candidates["rule_candidates"] = []
    result_hashes = simulation_context.get("result_sha256", {}) if cached_verified else {}
    comparison_hash = result_hashes.get("comparison.json") if comparison else None
    risk_hash = result_hashes.get("risk_assessment.json") if risk_assessment else None
    evidence = [{"evidence_id": "EV-SPICE", "evidence_type": "SPICE_SIMULATION",
                 "source_artifact": "comparison.json" if comparison else None,
                 "artifact_sha256": comparison_hash, "verification_status": "RECORDED" if comparison else "NOT_EVALUATED",
                 "circuit_hash": workspace.get("circuit_hash"), "scenario_id": scenario_id}]
    evidence.append({"evidence_id": "EV-GRAPH", "evidence_type": "HEURISTIC_ESTIMATE",
                     "source_artifact": "fault influence graph", "verification_status": "CONNECTIVITY_ONLY",
                     "circuit_hash": workspace.get("circuit_hash"), "scenario_id": scenario_id})
    if measurements:
        source_class = measurement_source_class(measurements)
        observation_type = {"simulated_demo": "SYNTHETIC_DEMO", "user_measurement": "USER_REPORTED_OBSERVATION",
                            "simulation": "SPICE_SIMULATION", "imported_measurement": "USER_REPORTED_OBSERVATION",
                            "MIXED_SOURCES": "MIXED_SOURCES", "unknown": "NOT_EVALUATED"}.get(source_class, "NOT_EVALUATED")
        evidence.append({"evidence_id": "EV-OBS", "evidence_type": observation_type,
                         "source_artifact": None, "verification_status": "SIMULATED" if observation_type == "SYNTHETIC_DEMO" else
                         "USER_REPORTED" if observation_type == "USER_REPORTED_OBSERVATION" else
                         "MODEL_OUTPUT" if observation_type == "SPICE_SIMULATION" else "UNVERIFIED",
                         "circuit_hash": workspace.get("circuit_hash"), "scenario_id": scenario_id})
    candidates = (active_fault_candidates or {}).get("hypotheses", [])
    rule_candidates = (active_fault_candidates or {}).get("rule_candidates", [])
    if candidates or rule_candidates:
        generation_methods = [row.get("generation_method", "") for row in candidates]
        has_llm_proposal = any("llm" in str(method).casefold() for method in generation_methods)
        evidence.append({"evidence_id": "EV-HYP", "evidence_type": "LLM_PROPOSAL" if has_llm_proposal else "HEURISTIC_ESTIMATE",
                         "source_artifact": "fault_hypotheses", "verification_status": "PROPOSAL",
                         "circuit_hash": workspace.get("circuit_hash"), "scenario_id": scenario_id})
    findings = []
    findings.append({"finding_id": "F-GRAPH", "finding_type": "connectivity_candidate_map",
                     "status": "CONNECTIVITY_ONLY", "source_evidence_ids": ["EV-GRAPH"],
                     "summary": "Shared nets identify candidate paths only; they do not confirm damaging electrical propagation."})
    if comparison:
        findings.append({"finding_id": "F-SPICE", "finding_type": "simulation_comparison",
                         "status": "SIMULATED", "source_evidence_ids": ["EV-SPICE"],
                         "summary": "SPICE comparison is a model result, not physical damage confirmation."})
    if candidates or rule_candidates:
        findings.append({"finding_id": "F-HYP", "finding_type": "fault_hypothesis_candidates",
                         "status": "PROPOSAL", "source_evidence_ids": ["EV-HYP"],
                         "summary": "Fault AI entries are candidate hypotheses and retain their individual validation/simulation states."})
    if risk_assessment:
        findings.append({"finding_id": "F-RISK", "finding_type": "rating_stress_review",
                         "status": "ASSESSED_WITH_CONFIGURED_RATINGS", "source_evidence_ids": ["EV-SPICE"],
                         "summary": "Risk levels compare simulated stress with configured rating records."})
    return {
        "schema_version": "circuitcare-diagnosis-report-v1",
        "generated_at": now,
        "workspace": dict(workspace),
        "circuit": {"name": workspace.get("circuit_id"), "hash": workspace.get("circuit_hash"),
                    "netlist_path": workspace.get("netlist_path"),
                    "component_count": len(components_dict), "net_count": len(unique_nets),
                    "components": components_dict, "nets": unique_nets},
        "fault_or_symptom": {"mode": "UNKNOWN_FAULT_MODE" if (active_fault_candidates or {}).get("hypothesis_symptom") else "KNOWN_FAULT",
                             "symptom": (active_fault_candidates or {}).get("hypothesis_symptom"), **fault,
                             "observations": measurements,
                             "observation_classification": measurement_source_class(measurements)},
        "measurement_context": {"measurement_set_sha256": measurement_hash,
                                 "source_classification": measurement_source_class(measurements),
                                 "verification_status": "USER_REPORTED" if measurement_source_class(measurements) == "user_measurement" else "NOT_PHYSICALLY_VERIFIED"},
        "diagnosis_context": {**diagnosis_context,
                              "measurement_set_sha256": diagnosis_context.get("measurement_set_sha256", measurement_hash),
                              "circuit_hash": diagnosis_context.get("circuit_hash", workspace.get("circuit_hash")),
                              "validation_status": "MEASUREMENT_DIAGNOSIS_MISMATCH" if diagnosis_mismatch else "MATCHED"},
        "hypothesis_context": {"measurement_set_sha256": candidate_measurement_hash or measurement_hash,
                               "hypothesis_set_sha256": actual_hypothesis_hash,
                               "validation_status": candidate_context_status},
        "preview_context": ({**preview_context, "mode": "WHAT_IF_PREVIEW", "status": "PREVIEW_ONLY"}
                            if preview_context and preview_diagnosis else None),
        "preview_measurements": preview_measurements if preview_context and preview_diagnosis else [],
        "preview_diagnosis": preview_diagnosis if preview_context and preview_diagnosis else None,
        "parameter_identification": parameter_identification or {"status": "NOT_EVALUATED"},
        "analysis_status": {"circuit_validation": workspace.get("validation_status", "READY"),
                            "graph_analysis": "COMPLETE", "fault_hypotheses": "PARTIAL" if candidates or rule_candidates else "NOT_EVALUATED",
                            "spice_verification": effective_simulation_status,
                            "risk_assessment": "COMPLETE" if risk_assessment else "NOT_EVALUATED"},
        "fault_influence": active_fault_candidates or {"status": "NOT_EVALUATED", "note": "Connectivity candidates are not damage evidence."},
        "spice_verification": {"status": effective_simulation_status, "source": simulation_source,
                               "scenario_id": scenario_id, "comparison": comparison},
        "simulation_context": simulation_context or None,
        "diagnosis": diagnosis or {"status": "NOT_EVALUATED"},
        "secondary_damage_risk": {**(risk_assessment or {"status": "NOT_EVALUATED"}),
                                  "artifact_sha256": risk_hash, "damage_confirmed": False},
        "recommended_inspection": inspection_recommendations or {"status": "NOT_EVALUATED", "note": "Review connectivity and measured stress before selecting an inspection."},
        "evidence": evidence,
        "findings": findings,
        "limitations": ["Synthetic or simulated evidence is not a physical repair result.",
                        "A priority score or conditional posterior is not a failure probability.",
                        "Unsupported fault models and unapproved ratings remain unevaluated.",
                        "This report collects the current snapshot and does not rerun analysis."],
        "reproducibility": {"git_sha": _git_sha(root), "python": platform.python_version(),
                            "platform": platform.platform(), "workspace_id": workspace.get("workspace_id"),
                            "circuit_hash": workspace.get("circuit_hash"),
                            "input_hashes": workspace.get("input_hashes", {}),
                            "analysis_timestamp": now, "ngspice_version": None,
                            "model_version": None, "prompt_version": None},
    }
