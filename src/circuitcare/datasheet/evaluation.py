"""Scope-aware datasheet metrics that do not assume incomplete labels are exhaustive."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

_HALLUCINATION_CLASSES = {"UNSUPPORTED_NUMERIC_CLAIM", "FABRICATED_SOURCE"}
_KNOWN_ERROR_CLASSES = {
    "SUPPORTED_CORRECT", "SUPPORTED_WRONG_KIND", "SUPPORTED_WRONG_CONDITION",
    "WRONG_COMPONENT_VARIANT", "INCORRECT_UNIT", "UNSUPPORTED_NUMERIC_CLAIM",
    "FABRICATED_SOURCE", "SOURCE_AMBIGUOUS", "UNVERIFIED", "OUT_OF_SCOPE",
}


def score_extractions(ground_truth: list[dict[str, Any]], predictions: list[dict[str, Any]], *,
                      latency_seconds: float | None = None,
                      full_document_ground_truth_complete: bool = False) -> dict[str, Any]:
    """Score explicitly scoped target ratings; full-document metrics need exhaustive labels."""
    if not ground_truth:
        return {"status": "NOT_EVALUATED", "target_rating_evaluation": _empty_target(),
                "full_document_evaluation": _not_evaluated("No verified target-rating labels were supplied."),
                **_error_metrics(predictions), "latency_seconds": latency_seconds}

    per_target = []
    scored_candidates: set[int] = set()
    matched = []
    scope_complete = True
    for truth in ground_truth:
        scope = truth.get("evaluation_scope") or {}
        if scope.get("scope_type") != "target_rating" or scope.get("complete_target_scope") is not True:
            scope_complete = False
        indexed = [(index, row) for index, row in enumerate(predictions)
                   if _same_document(truth, row) and row.get("rating_key") == truth.get("rating_key")]
        scoped = []
        class_counts: Counter[str] = Counter()
        for index, row in indexed:
            scope_status = _scope_status(row, truth)
            classification = classify_candidate(row, truth)
            class_counts[classification] += 1
            if scope_status == "IN_SCOPE":
                scoped.append((index, row))
                scored_candidates.add(index)
        best = max(scoped, key=lambda pair: _match_score(truth, pair[1]), default=None)
        if best:
            matched.append((truth, best[1]))
            is_correct = _target_correct(truth, best[1])
        else:
            is_correct = False
        tp = int(is_correct)
        fp = max(0, len(scoped) - tp) if scope.get("complete_target_scope") is True else None
        fn = 1 - tp
        per_target.append({"document_id": truth.get("document_id"), "rating_key": truth.get("rating_key"),
            "target_scope": scope,
            "status": "MEASURED_DEVELOPMENT_PILOT" if scope.get("complete_target_scope") else "NOT_EVALUATED",
            "candidate_count_in_scope": len(scoped), "true_positive_count": tp,
            "false_positive_count": fp, "false_negative_count": fn,
            "target_precision": tp / (tp + fp) if fp is not None and tp + fp else None,
            "target_recall": tp / (tp + fn) if tp + fn else None,
            "scope_class_counts": dict(class_counts)})

    correct_count = sum(row["true_positive_count"] for row in per_target)
    scoped_count = sum(row["candidate_count_in_scope"] for row in per_target)
    complete_target_count = sum(row.get("status") == "MEASURED_DEVELOPMENT_PILOT" for row in per_target)
    target_precision = correct_count / scoped_count if scope_complete and scoped_count else None
    target_recall = correct_count / len(ground_truth) if scope_complete else None
    target_status = "MEASURED_DEVELOPMENT_PILOT" if scope_complete else "PARTIALLY_EVALUATED"

    def ratio(key: str) -> dict[str, Any]:
        values = [_metric_for_match(truth, prediction).get(key) for truth, prediction in matched]
        values = [value for value in values if value is not None]
        return _metric(sum(bool(value) for value in values) / len(values) if values else None,
                       "MEASURED_DEVELOPMENT_PILOT" if values else "NOT_EVALUATED",
                       denominator=len(values),
                       reason=None if values else "No matched target predictions with this property.")

    target_metrics = {
        "status": target_status, "target_count": len(ground_truth),
        "complete_target_scope_count": complete_target_count,
        "candidate_count_in_scope": scoped_count, "correct_target_count": correct_count,
        "false_positive_count": sum(row["false_positive_count"] or 0 for row in per_target),
        "false_negative_count": len(ground_truth) - correct_count,
        "precision": _metric(target_precision, target_status, denominator=scoped_count,
                             reason=None if target_precision is not None else "Target scope is incomplete or has no in-scope candidates."),
        "recall": _metric(target_recall, target_status, denominator=len(ground_truth),
                          reason=None if target_recall is not None else "At least one target scope is incomplete."),
        "f1": _metric(_f1(target_precision, target_recall), target_status, denominator=len(ground_truth),
                      reason=None if target_precision is not None and target_recall is not None else "Precision or recall is not evaluable."),
        "value_accuracy": ratio("value"), "unit_accuracy": ratio("unit"),
        "rating_kind_accuracy": ratio("kind"), "condition_accuracy": ratio("condition"),
        "evidence_accuracy": ratio("evidence"), "table_context_accuracy": ratio("table_context"),
        "per_target": per_target,
    }
    full_doc = (_complete_inventory_metrics(ground_truth, predictions)
                if full_document_ground_truth_complete else
                _not_evaluated("Ground truth labels only selected target ratings; the complete document inventory is not verified."))
    return {"status": "MEASURED" if full_doc.get("status") == "MEASURED_COMPLETE_INVENTORY" else "PARTIALLY_EVALUATED",
            "evaluation_scope": "TARGET_RATING_ONLY", "target_rating_evaluation": target_metrics,
            "full_document_evaluation": full_doc, "candidate_count_all_scopes": len(predictions),
            "scored_candidate_count": len(scored_candidates),
            "excluded_candidate_count": len(predictions) - len(scored_candidates),
            **_error_metrics(predictions, ground_truth), "latency_seconds": latency_seconds}


def classify_candidate(candidate: dict[str, Any], truth: dict[str, Any]) -> str:
    """Return a conservative scope/evidence class; hallucination requires review."""
    review = candidate.get("human_adjudication")
    if isinstance(review, dict) and review.get("review_status") == "reviewed":
        explicit = review.get("error_class")
        if explicit in _KNOWN_ERROR_CLASSES:
            return explicit
    scope_status = _scope_status(candidate, truth)
    if scope_status != "IN_SCOPE":
        return scope_status
    evidence_status = candidate.get("evidence_status")
    if evidence_status == "SOURCE_AMBIGUOUS":
        return "SOURCE_AMBIGUOUS"
    if evidence_status in {"SOURCE_MISMATCH", "SOURCE_NOT_FOUND", None, "PENDING_REVIEW"}:
        return "UNVERIFIED"
    if _target_correct(truth, candidate):
        return "SUPPORTED_CORRECT"
    return classify_in_scope_error(truth, candidate)


def classify_in_scope_error(truth: dict[str, Any], candidate: dict[str, Any]) -> str:
    if candidate.get("evidence_status") == "SOURCE_AMBIGUOUS":
        return "SOURCE_AMBIGUOUS"
    if candidate.get("evidence_status") in {"SOURCE_MISMATCH", "SOURCE_NOT_FOUND", None, "PENDING_REVIEW"}:
        return "UNVERIFIED"
    if truth.get("rating_kind") not in (None, "unknown") and candidate.get("rating_kind") != truth.get("rating_kind"):
        return "SUPPORTED_WRONG_KIND"
    if truth.get("unit") != candidate.get("normalized_unit"):
        return "INCORRECT_UNIT"
    if _number_equal(truth.get("value"), candidate.get("normalized_value")) and not _conditions_equal(truth.get("conditions", {}), candidate.get("conditions", {})):
        return "SUPPORTED_WRONG_CONDITION"
    return "UNVERIFIED"


def _scope_status(candidate: dict[str, Any], truth: dict[str, Any]) -> str:
    if not _same_document(truth, candidate) or candidate.get("rating_key") != truth.get("rating_key"):
        return "OUT_OF_SCOPE"
    for key in ("manufacturer", "mpn"):
        expected_identity, actual_identity = truth.get(key), candidate.get(key)
        if expected_identity and actual_identity and _norm(expected_identity) != _norm(actual_identity):
            return "WRONG_COMPONENT_VARIANT"
    status = candidate.get("evidence_status")
    if status == "SOURCE_AMBIGUOUS":
        return "SOURCE_AMBIGUOUS"
    if status in {"SOURCE_MISMATCH", "SOURCE_NOT_FOUND", None, "PENDING_REVIEW"}:
        return "UNVERIFIED"
    scope = truth.get("evaluation_scope") or {}
    context = candidate.get("table_context") if isinstance(candidate.get("table_context"), dict) else {}
    expected_variant, actual_variant = scope.get("component_variant"), context.get("variant")
    if expected_variant:
        if not actual_variant:
            return "UNVERIFIED"
        if _norm(actual_variant) != _norm(expected_variant):
            return "WRONG_COMPONENT_VARIANT"
    expected_row = scope.get("row_label")
    if expected_row:
        actual_row = context.get("row_label")
        if actual_row:
            if not _row_matches(expected_row, actual_row):
                return "OUT_OF_SCOPE"
        elif not _contains(candidate.get("source_text", ""), scope.get("source_anchor", expected_row)):
            return "UNVERIFIED"
    expected_column = scope.get("column_header")
    if expected_column:
        actual_column = context.get("column_header")
        if actual_column and _norm(actual_column) != _norm(expected_column):
            return "OUT_OF_SCOPE"
        if not actual_column and not scope.get("allow_blank_column", False):
            return "UNVERIFIED"
    source_anchor = scope.get("source_anchor")
    if source_anchor and not _contains(candidate.get("source_text", ""), source_anchor):
        if context.get("row_label") and expected_row and _row_matches(expected_row, context.get("row_label", "")):
            pass
        else:
            return "OUT_OF_SCOPE"
    return "IN_SCOPE"


def _metric_for_match(truth: dict[str, Any], candidate: dict[str, Any]) -> dict[str, bool | None]:
    return {"value": _number_equal(truth.get("value"), candidate.get("normalized_value")),
            "unit": truth.get("unit") == candidate.get("normalized_unit"),
            "kind": None if truth.get("rating_kind") in (None, "unknown") else truth.get("rating_kind") == candidate.get("rating_kind"),
            "condition": _conditions_equal(truth.get("conditions", {}), candidate.get("conditions", {})),
            "evidence": _evidence_matches(truth, candidate),
            "table_context": _table_context_equal(truth.get("table_context"), candidate.get("table_context"))}


def _target_correct(truth: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if not _number_equal(truth.get("value"), candidate.get("normalized_value")):
        return False
    if truth.get("unit") != candidate.get("normalized_unit"):
        return False
    expected_kind = truth.get("rating_kind")
    if expected_kind not in (None, "unknown") and expected_kind != candidate.get("rating_kind"):
        return False
    return candidate.get("evidence_status") in {"SOURCE_MATCHED", "CONDITION_INCOMPLETE"}


def _error_metrics(predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    ground_truth = ground_truth or []
    classes = []
    for row in predictions:
        matching_targets = [label for label in ground_truth if _same_document(label, row)
                            and label.get("rating_key") == row.get("rating_key")]
        if matching_targets:
            classes.append(classify_candidate(row, matching_targets[0]))
        elif ground_truth:
            classes.append("OUT_OF_SCOPE")
        else:
            classes.append(classify_candidate_loose(row))
    rejected = sum(row.get("evidence_status") in {"SOURCE_MISMATCH", "SOURCE_NOT_FOUND", "SOURCE_AMBIGUOUS"}
                   for row in predictions)
    adjudicated = [row for row in predictions if isinstance(row.get("human_adjudication"), dict)
                   and row["human_adjudication"].get("review_status") == "reviewed"
                   and row["human_adjudication"].get("error_class") in _KNOWN_ERROR_CLASSES]
    hallucinations = sum(row["human_adjudication"]["error_class"] in _HALLUCINATION_CLASSES for row in adjudicated)
    return {
        "source_verification_rejection_rate": _metric(rejected / len(predictions) if predictions else None,
            "MEASURED_AUTOMATED_CHECK" if predictions else "NOT_EVALUATED", denominator=len(predictions),
            reason=None if predictions else "No extracted candidates."),
        "source_verification_rejection_count": rejected,
        "unverified_candidate_count": classes.count("UNVERIFIED") + classes.count("SOURCE_AMBIGUOUS"),
        "out_of_scope_candidate_count": classes.count("OUT_OF_SCOPE"),
        "wrong_component_variant_count": classes.count("WRONG_COMPONENT_VARIANT"),
        "candidate_error_class_counts": dict(Counter(classes)),
        "unsupported_claim_rate": _metric(None, "NOT_EVALUATED", denominator=len(adjudicated),
            reason="No candidate-level human adjudication records are included."),
        "hallucination_rate": _metric(hallucinations / len(adjudicated) if adjudicated else None,
            "MEASURED_HUMAN_ADJUDICATION" if adjudicated else "NOT_EVALUATED", denominator=len(adjudicated),
            reason=None if adjudicated else "Hallucination requires candidate-level human confirmation; source-check failure alone is insufficient."),
    }


def classify_candidate_loose(candidate: dict[str, Any]) -> str:
    """Classify without a gold target; missing labels never imply hallucination."""
    review = candidate.get("human_adjudication")
    if isinstance(review, dict) and review.get("review_status") == "reviewed" and review.get("error_class") in _KNOWN_ERROR_CLASSES:
        return review["error_class"]
    if candidate.get("evidence_status") == "SOURCE_AMBIGUOUS":
        return "SOURCE_AMBIGUOUS"
    if candidate.get("evidence_status") in {"SOURCE_MISMATCH", "SOURCE_NOT_FOUND", None, "PENDING_REVIEW", "CONDITION_INCOMPLETE"}:
        return "UNVERIFIED"
    return "UNVERIFIED"


def _complete_inventory_metrics(ground_truth: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not ground_truth or not all((row.get("evaluation_scope") or {}).get("complete_document_inventory") is True for row in ground_truth):
        return _not_evaluated("No exhaustive, human-verified full-document rating inventory is present.")
    relevant_predictions = [row for row in predictions if any(_same_document(label, row) for label in ground_truth)]
    unmatched = set(range(len(relevant_predictions)))
    true_positive = 0
    for truth in ground_truth:
        candidates = [index for index in unmatched if relevant_predictions[index].get("rating_key") == truth.get("rating_key")
                     and _same_document(truth, relevant_predictions[index])]
        if not candidates:
            continue
        index = max(candidates, key=lambda value: _match_score(truth, relevant_predictions[value]))
        candidate = relevant_predictions[index]
        if _full_document_correct(truth, candidate):
            true_positive += 1
        unmatched.remove(index)
    false_positive = len(relevant_predictions) - true_positive
    false_negative = len(ground_truth) - true_positive
    precision = true_positive / len(relevant_predictions) if relevant_predictions else None
    recall = true_positive / len(ground_truth) if ground_truth else None
    status = "MEASURED_COMPLETE_INVENTORY"
    return {"status": status, "document_count": len({row.get("document_id") for row in ground_truth}),
            "ground_truth_count": len(ground_truth), "candidate_count": len(relevant_predictions),
            "true_positive_count": true_positive, "false_positive_count": false_positive,
            "false_negative_count": false_negative,
            "precision": _metric(precision, status, denominator=len(relevant_predictions), reason=None if precision is not None else "No candidates were produced."),
            "recall": _metric(recall, status, denominator=len(ground_truth), reason=None),
            "f1": _metric(_f1(precision, recall), status, denominator=len(ground_truth), reason=None)}


def _full_document_correct(truth: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if not _target_correct(truth, candidate):
        return False
    context = truth.get("table_context")
    if context is not None and _table_context_equal(context, candidate.get("table_context")) is not True:
        return False
    conditions = _conditions_equal(truth.get("conditions", {}), candidate.get("conditions", {}))
    return conditions is not False


def _same_document(truth, prediction) -> bool:
    return bool(truth.get("document_id") and prediction.get("document_id") == truth.get("document_id"))


def _match_score(truth: dict[str, Any], candidate: dict[str, Any]) -> tuple[int, ...]:
    metrics = _metric_for_match(truth, candidate)
    return tuple(int(metrics[key] is True) for key in ("value", "unit", "kind", "condition", "evidence", "table_context"))


def _number_equal(expected: Any, actual: Any) -> bool:
    try:
        value, candidate = float(expected), float(actual)
        return abs(value - candidate) <= max(1e-9, abs(value) * 1e-6)
    except (TypeError, ValueError):
        return False


def _conditions_equal(expected: dict[str, Any], actual: dict[str, Any]) -> bool | None:
    keys = [key for key in ("temperature_c", "duration", "polarity") if expected.get(key) is not None]
    if not keys:
        return None
    return all(expected.get(key) == actual.get(key) for key in keys)


def _evidence_matches(truth: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if truth.get("pdf_page_index") != candidate.get("pdf_page_index"):
        return False
    quoted, label_quote = _norm(candidate.get("source_text", "")), _norm(truth.get("evidence_text", ""))
    return bool(quoted and label_quote and (quoted in label_quote or label_quote in quoted))


def _table_context_equal(expected: Any, actual: Any) -> bool | None:
    if expected is None:
        return None
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return False
    fields = ("row_label", "variant", "column_header", "cell_text")
    present = [key for key in fields if expected.get(key) not in (None, "")]
    if not present:
        return None
    return all(_norm(expected[key]) == _norm(actual.get(key, "")) for key in present)


def _row_matches(expected: str, actual: str) -> bool:
    def tokens(value: str) -> set[str]:
        value = re.sub(r"\bp\s*(?:\(\d+\)\s*)?70\b", "p70", value.casefold())
        return set(re.findall(r"[a-z0-9]+", value))
    expected_tokens = tokens(expected)
    actual_tokens = tokens(actual)
    return bool(expected_tokens) and expected_tokens <= actual_tokens


def _contains(text: str, anchor: str) -> bool:
    return bool(anchor and _norm(anchor) in _norm(text))


def _norm(value: Any) -> str:
    text = "".join(char for char in str(value) if unicodedata.category(char) != "Cf")
    return re.sub(r"\s+", "", text).casefold().replace("−", "-")


def _metric(value: float | None, status: str, *, denominator: int, reason: str | None) -> dict[str, Any]:
    return {"value": value, "status": status, "denominator": denominator, "reason": reason}


def _not_evaluated(reason: str) -> dict[str, Any]:
    return {"status": "NOT_EVALUATED", "reason": reason,
            "precision": _metric(None, "NOT_EVALUATED", denominator=0, reason=reason),
            "recall": _metric(None, "NOT_EVALUATED", denominator=0, reason=reason),
            "f1": _metric(None, "NOT_EVALUATED", denominator=0, reason=reason)}


def _empty_target() -> dict[str, Any]:
    reason = "No verified target-rating labels were supplied."
    return {"status": "NOT_EVALUATED", "target_count": 0,
            "precision": _metric(None, "NOT_EVALUATED", denominator=0, reason=reason),
            "recall": _metric(None, "NOT_EVALUATED", denominator=0, reason=reason),
            "f1": _metric(None, "NOT_EVALUATED", denominator=0, reason=reason)}


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
