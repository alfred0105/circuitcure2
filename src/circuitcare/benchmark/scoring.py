"""Hierarchical diagnosis metrics, schema phase17-metrics-v2."""

from __future__ import annotations

import math
import re
from typing import Any

LEVELS = ("component", "fault_mode", "exact_hypothesis")


def canonical_hypothesis_identity(hypothesis: dict[str, Any], level: str) -> tuple:
    if level not in LEVELS:
        raise ValueError(f"Unknown hypothesis level: {level}")
    identity = (str(hypothesis.get("component_ref", "")),)
    if level == "component":
        return identity
    identity += (str(hypothesis.get("fault_type", "")),)
    if level == "fault_mode":
        return identity
    params = tuple(sorted((str(key), _normal(value)) for key, value in hypothesis.get("parameters", {}).items()))
    return identity + (params,)


def score_hypothesis_levels(cases: list[dict[str, Any]], truths: dict[str, list[dict[str, Any]]],
                            predictions: dict[str, list[dict[str, Any]]], *,
                            parameter_tolerance: float = 1e-9) -> dict[str, Any]:
    """Deduplicate predictions at each level; retain raw-candidate scores and counts."""
    if parameter_tolerance < 0 or not math.isfinite(parameter_tolerance):
        raise ValueError("parameter_tolerance must be finite and non-negative")
    counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in LEVELS}
    raw_counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in LEVELS}
    unique_totals = {name: 0 for name in LEVELS}
    duplicate_totals = {name: 0 for name in LEVELS}
    per_case, oracle_errors, selected_errors, legacy_errors = [], [], [], []
    raw_total = supported = executable = invalid_refs = 0

    for case in cases:
        cid = str(case["case_id"])
        expected_raw, proposals_raw = truths.get(cid, []), predictions.get(cid, [])
        if not case.get("ground_truth_supported", True):
            per_case.append({"case_id": cid, "status": "UNSUPPORTED_GROUND_TRUTH"})
            continue
        supported += 1
        raw_total += len(proposals_raw)
        executable += sum(row.get("validation_status") == "VALIDATED_EXECUTABLE" for row in proposals_raw)
        invalid_refs += sum(row.get("validation_status") == "INVALID_REFERENCE" for row in proposals_raw)
        correct, unique = {}, {}
        for level in LEVELS:
            expected = _dedupe(expected_raw, level, parameter_tolerance)
            proposed = _dedupe(proposals_raw, level, parameter_tolerance)
            tp = _match_count(expected, proposed, level, parameter_tolerance)
            raw_tp = _match_count(expected, proposals_raw, level, parameter_tolerance)
            counts[level]["tp"] += tp
            counts[level]["fp"] += len(proposed) - tp
            counts[level]["fn"] += len(expected) - tp
            raw_counts[level]["tp"] += raw_tp
            raw_counts[level]["fp"] += max(0, len(proposals_raw) - raw_tp)
            raw_counts[level]["fn"] += max(0, len(expected) - raw_tp)
            unique_totals[level] += len(proposed)
            duplicate_totals[level] += len(proposals_raw) - len(proposed)
            correct[level] = tp
            unique[level] = {"prediction_count": len(proposed), "truth_count": len(expected),
                             "duplicates_removed": len(proposals_raw) - len(proposed)}

        oracle = _parameter_errors(expected_raw, proposals_raw, parameter_tolerance, mode="oracle_best_candidate")
        selected_rows = [row for row in proposals_raw if row.get("selected_for_diagnosis") is True]
        selected = _parameter_errors(expected_raw, selected_rows, parameter_tolerance, mode="selected_candidate")
        legacy = _parameter_errors(expected_raw, proposals_raw, parameter_tolerance, mode="legacy_first_candidate")
        oracle_errors.extend(oracle)
        selected_errors.extend(selected)
        legacy_errors.extend(legacy)
        per_case.append({"case_id": cid, "status": "SCORED", "raw_candidate_count": len(proposals_raw),
                         "unique_counts": unique, "correct": correct,
                         "oracle_best_candidate_parameter_error": oracle,
                         "selected_candidate_parameter_error": selected,
                         "legacy_first_candidate_parameter_error": legacy})

    legacy_summary = _summarize([row["relative_error"] for row in legacy_errors])
    return {
        "status": "MEASURED" if supported else "NOT_EVALUATED",
        "metric_schema_version": "phase17-metrics-v2",
        "case_count": len(cases), "supported_case_count": supported,
        "raw_candidate_count": raw_total, "candidate_count": raw_total,
        "unique_component_count": unique_totals["component"],
        "unique_fault_mode_count": unique_totals["fault_mode"],
        "unique_exact_hypothesis_count": unique_totals["exact_hypothesis"],
        "duplicate_component_candidate_count": duplicate_totals["component"],
        "duplicate_fault_mode_candidate_count": duplicate_totals["fault_mode"],
        "duplicate_exact_candidate_count": duplicate_totals["exact_hypothesis"],
        "executable_candidate_count": executable,
        "unsupported_candidate_rate": (raw_total - executable) / raw_total if raw_total else None,
        "invalid_reference_rate": invalid_refs / raw_total if raw_total else None,
        "duplicate_hypothesis_rate": duplicate_totals["exact_hypothesis"] / raw_total if raw_total else None,
        "levels": {name: _rates(value) for name, value in counts.items()},
        "raw_candidate_levels": {name: _rates(value) for name, value in raw_counts.items()},
        "parameter_error": {
            **legacy_summary,
            "selected_candidate": {"status": "MEASURED" if selected_errors else "NOT_EVALUATED",
                                   **_summarize([row["relative_error"] for row in selected_errors])},
            "oracle_best_candidate": {"status": "MEASURED" if oracle_errors else "NOT_EVALUATED",
                                      **_summarize([row["relative_error"] for row in oracle_errors])},
            "legacy_first_candidate": {"status": "MEASURED" if legacy_errors else "NOT_EVALUATED",
                                       **_summarize([row["relative_error"] for row in legacy_errors])},
            "selection_method": "selected_for_diagnosis flag; absent selection is NOT_EVALUATED",
            "oracle_interpretation": "Post-hoc optimistic bound, not system selection performance.",
        },
        "per_case": per_case,
    }


def score_ranked_hypotheses(cases: list[dict[str, Any]], truths: dict[str, list[dict[str, Any]]],
                            ranked: dict[str, list[dict[str, Any]]], *, ks=(1, 3),
                            residual_tie_tolerance: float = 0.1) -> dict[str, Any]:
    if not math.isfinite(residual_tie_tolerance) or residual_tie_tolerance < 0:
        raise ValueError("residual_tie_tolerance must be finite and non-negative")
    if not cases:
        return {"status": "NOT_EVALUATED", "case_count": 0, "top_k_accuracy": {str(k): None for k in ks},
                "mean_reciprocal_rank": None, "coverage": None, "strict_top1": None, "tie_aware_top1": None}
    counts, reciprocal, covered, strict, tie_aware, tie_cases, per_case = {k: 0 for k in ks}, [], 0, 0, 0, 0, []
    for case in cases:
        cid = str(case["case_id"])
        if not case.get("ground_truth_supported", True):
            per_case.append({"case_id": cid, "status": "UNSUPPORTED_GROUND_TRUTH"})
            continue
        truth, rows = truths.get(cid, []), ranked.get(cid, [])
        remaining, ranks = list(rows), []
        for label in truth:
            index = next((i for i, row in enumerate(remaining) if _matches(label, row, "exact_hypothesis", 1e-9)), None)
            if index is not None:
                ranks.append(index + 1)
                remaining.pop(index)
        rank = min(ranks) if ranks else None
        successful = [row for row in rows if row.get("simulation_status") == "SIMULATION_SUCCESS"
                      and isinstance(row.get("observation_match", {}).get("max_absolute_normalized_residual"), (int, float))]
        best_residual = min((float(row["observation_match"]["max_absolute_normalized_residual"])
                             for row in successful), default=None)
        top = [row for row in successful if best_residual is not None and
               float(row["observation_match"]["max_absolute_normalized_residual"]) <= best_residual + residual_tie_tolerance]
        truth_present = any(_matches(label, row, "exact_hypothesis", 1e-9) for label in truth for row in rows)
        truth_simulated = [row for row in successful if any(_matches(label, row, "exact_hypothesis", 1e-9) for label in truth)]
        truth_residual = min((float(row["observation_match"]["max_absolute_normalized_residual"])
                              for row in truth_simulated), default=None)
        near = [row for row in successful if truth_residual is not None and
                abs(float(row["observation_match"]["max_absolute_normalized_residual"]) - truth_residual) <= residual_tie_tolerance]
        tied_truth = any(any(_matches(label, row, "exact_hypothesis", 1e-9) for label in truth) for row in top)
        is_strict, is_tied = rank == 1, bool(tied_truth)
        strict += int(is_strict)
        tie_aware += int(is_tied)
        tie_cases += int(len(top) > 1)
        covered += int(bool(rows))
        if rank is not None:
            reciprocal.append(1 / rank)
        for k in ks:
            counts[k] += int(rank is not None and rank <= k)
        identifiability = ("UNSUPPORTED_TRUTH" if not truth else
                           "CORRECT_CANDIDATE_NOT_SIMULATED" if not truth_simulated else
                           "AMBIGUOUS" if len(near) > 1 else "IDENTIFIABLE_BY_RESIDUAL")
        per_case.append({"case_id": cid, "status": "SCORED", "ranked_count": len(rows),
                         "correct_hypothesis_present": truth_present,
                         "correct_hypothesis_simulated": bool(truth_simulated),
                         "correct_hypothesis_rank": rank, "correct_hypothesis_residual": truth_residual,
                         "top_hypothesis_residual": best_residual, "near_tied_candidates": len(near),
                         "strict_top1": is_strict, "tie_aware_top1": is_tied, "tie_candidate_count": len(top),
                         "diagnostic_identifiability": identifiability})
    n = sum(row["status"] == "SCORED" for row in per_case)
    return {"status": "MEASURED" if n else "NOT_EVALUATED", "case_count": n, "independent_case_count": n,
            "top_k_accuracy": {str(k): counts[k] / n if n else None for k in ks},
            "mean_reciprocal_rank": sum(reciprocal) / n if n else None,
            "coverage": covered / n if n else None, "strict_top1": strict / n if n else None,
            "tie_aware_top1": tie_aware / n if n else None, "tie_case_count": tie_cases,
            "residual_tie_tolerance": residual_tie_tolerance,
            "ambiguous_case_count": sum(row.get("diagnostic_identifiability") == "AMBIGUOUS" for row in per_case),
            "per_case": per_case}


def _dedupe(rows, level, tolerance):
    unique = []
    for row in rows:
        if not any(_matches(row, old, level, tolerance) for old in unique):
            unique.append(row)
    return unique


def _match_count(expected, predicted, level, tolerance):
    remaining, hits = list(predicted), 0
    for truth in expected:
        index = next((i for i, row in enumerate(remaining) if _matches(truth, row, level, tolerance)), None)
        if index is not None:
            hits += 1
            remaining.pop(index)
    return hits


def _matches(truth, prediction, level, tolerance):
    if str(truth.get("component_ref", "")) != str(prediction.get("component_ref", "")):
        return False
    if level == "component":
        return True
    if str(truth.get("fault_type", "")) != str(prediction.get("fault_type", "")):
        return False
    if level == "fault_mode":
        return True
    expected, found = truth.get("parameters", {}), prediction.get("parameters", {})
    if set(expected) != set(found):
        return False
    for key, value in expected.items():
        a, b = _numeric(value), _numeric(found[key])
        if a is not None and b is not None:
            if abs(a - b) > tolerance * max(abs(a), abs(b), 1.0):
                return False
        elif str(value).strip().casefold() != str(found[key]).strip().casefold():
            return False
    return True


def _parameter_errors(expected, proposals, tolerance, *, mode):
    errors = []
    for truth in expected:
        candidates = [row for row in proposals if str(row.get("component_ref", "")) == str(truth.get("component_ref", ""))
                      and str(row.get("fault_type", "")) == str(truth.get("fault_type", ""))]
        values = []
        for row in candidates:
            params, current = row.get("parameters", {}), {}
            for key, target in truth.get("parameters", {}).items():
                actual, desired = _numeric(params.get(key)), _numeric(target)
                if actual is not None and desired is not None:
                    current[key] = {"actual": actual, "expected": desired,
                                    "relative_error": abs(actual - desired) / max(abs(desired), tolerance, 1e-15)}
            if current and len(current) == len(truth.get("parameters", {})):
                average = sum(item["relative_error"] for item in current.values()) / len(current)
                values.append((average, current))
        if not values:
            continue
        if mode in {"selected_candidate", "legacy_first_candidate"}:
            _, chosen = values[0]
        else:
            _, chosen = min(values, key=lambda item: item[0])
        errors.extend({"component_ref": truth.get("component_ref"), "fault_type": truth.get("fault_type"),
                       "parameter": key, **item, "selection_mode": mode} for key, item in chosen.items())
    return errors


def _numeric(value):
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]*)\s*", str(value))
    if not match:
        return None
    scale = {"": 1.0, "T": 1e12, "G": 1e9, "MEG": 1e6, "K": 1e3, "M": 1e-3, "U": 1e-6,
             "N": 1e-9, "P": 1e-12, "F": 1e-15}
    suffix = match.group(2).upper()
    if suffix not in scale:
        return None
    number = float(match.group(1)) * scale[suffix]
    return number if math.isfinite(number) else None


def _normal(value):
    number = _numeric(value)
    return f"{number:.12g}" if number is not None else str(value).strip().casefold()


def _rates(value):
    precision = value["tp"] / (value["tp"] + value["fp"]) if value["tp"] + value["fp"] else None
    recall = value["tp"] / (value["tp"] + value["fn"]) if value["tp"] + value["fn"] else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else 0.0 if precision is not None and recall is not None else None
    return {**value, "precision": precision, "recall": recall, "f1": f1}


def _summarize(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "mae": None}
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {"count": len(values), "mean": sum(values) / len(values), "median": median,
            "mae": sum(values) / len(values)}
