"""Single source of truth for versioned analysis state and result history."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from circuitcare.analysis.models import AnalysisFingerprints, AnalysisScope, AnalysisStatus

ANALYSIS_STATES_KEY = "circuitcare_v6_analysis_states"
ANALYSIS_HISTORY_LIMIT = 100


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _store(state: Any) -> dict[str, dict[str, Any]]:
    value = state.get(ANALYSIS_STATES_KEY)
    if not isinstance(value, dict):
        value = {}
        state[ANALYSIS_STATES_KEY] = value
    return value


def _archive_results(record: dict[str, Any], *, reason: str) -> None:
    results = record.get("results")
    if not isinstance(results, dict) or not results:
        return
    history = list(record.get("history", []))
    history.append({"results": deepcopy(results),
                    "fingerprints": deepcopy(record.get("fingerprints")),
                    "scope": deepcopy(record.get("scope")), "reason": reason,
                    "archived_at": _now()})
    record["history"] = history[-ANALYSIS_HISTORY_LIMIT:]
    record["results"] = {}


def _record(state: Any, scope: AnalysisScope) -> dict[str, Any]:
    store = _store(state)
    record = store.get(scope.key)
    if not isinstance(record, dict):
        record = {"scope": scope.to_dict(), "status": AnalysisStatus.NOT_RUN.value,
                  "fingerprints": None, "results": {}, "history": [],
                  "generation": 0, "reason": None, "updated_at": _now()}
        store[scope.key] = record
    return record


def get_current_state(
    state: Any, scope: AnalysisScope,
    fingerprints: AnalysisFingerprints | dict[str, str] | None = None,
) -> dict[str, Any]:
    """Read state and mark it stale when the content version changed."""
    record = _record(state, scope)
    if fingerprints is not None:
        expected = fingerprints.to_dict() if isinstance(fingerprints, AnalysisFingerprints) else dict(fingerprints)
        current = record.get("fingerprints")
        if current and current.get("final_sha256") != expected.get("final_sha256"):
            _archive_results(record, reason="analysis fingerprint changed")
            record["status"] = AnalysisStatus.STALE.value
            record["generation"] = int(record.get("generation", 0)) + 1
            record["run_id"] = None
            record["reason"] = "분석 입력 또는 규칙이 변경되었습니다."
        if record.get("status") != AnalysisStatus.RUNNING.value:
            record["fingerprints"] = expected
        record["updated_at"] = _now()
        _store(state)[scope.key] = record
    return deepcopy(record)


def get_current_state_for_scope(state: Any, scope: AnalysisScope) -> dict[str, Any]:
    """Read a scope without changing its version (useful for compatibility UI)."""
    return deepcopy(_record(state, scope))


def invalidate_analysis(
    state: Any, *, scope: AnalysisScope | None = None,
    project_id: str | None = None, board_id: str | None = None,
    reason: str = "분석 입력이 변경되었습니다.",
) -> int:
    """Invalidate one target or every photo target on a project/board."""
    store = _store(state)
    selected = []
    if scope is not None:
        record = _record(state, scope)
        selected = [(scope.key, record)]
    else:
        selected = [(key, value) for key, value in store.items()
                    if isinstance(value, dict)
                    and value.get("scope", {}).get("project_id") == project_id
                    and value.get("scope", {}).get("board_id") == (board_id or "")]
        if not selected and project_id:
            fallback = AnalysisScope(project_id, board_id or "", "legacy:default")
            record = _record(state, fallback)
            selected = [(fallback.key, record)]
    count = 0
    for key, record in selected:
        if record.get("status") == AnalysisStatus.STALE.value and not record.get("results") and record.get("reason") == reason:
            continue
        _archive_results(record, reason=reason)
        record["status"] = AnalysisStatus.STALE.value
        record["generation"] = int(record.get("generation", 0)) + 1
        record["run_id"] = None
        record["reason"] = reason
        record["updated_at"] = _now()
        store[key] = record
        count += 1
    return count


def begin_analysis(
    state: Any, *, scope: AnalysisScope,
    fingerprints: AnalysisFingerprints | dict[str, str], result_kind: str = "symptom",
) -> str:
    expected = fingerprints.to_dict() if isinstance(fingerprints, AnalysisFingerprints) else dict(fingerprints)
    record = _record(state, scope)
    if record.get("fingerprints") and record["fingerprints"].get("final_sha256") != expected.get("final_sha256"):
        _archive_results(record, reason="analysis inputs changed before rerun")
    record["generation"] = int(record.get("generation", 0)) + 1
    run_id = str(uuid4())
    record.update({"status": AnalysisStatus.RUNNING.value, "run_id": run_id,
                   "run_result_kind": result_kind, "fingerprints": expected,
                   "reason": None, "updated_at": _now()})
    _store(state)[scope.key] = record
    return run_id


def commit_analysis_result(
    state: Any, *, scope: AnalysisScope, run_id: str,
    fingerprints: AnalysisFingerprints | dict[str, str], result: dict[str, Any],
    result_kind: str = "symptom", current_fingerprints: AnalysisFingerprints | dict[str, str] | None = None,
    current_scope: AnalysisScope | None = None,
) -> bool:
    expected = fingerprints.to_dict() if isinstance(fingerprints, AnalysisFingerprints) else dict(fingerprints)
    live = (current_fingerprints.to_dict() if isinstance(current_fingerprints, AnalysisFingerprints)
            else dict(current_fingerprints) if current_fingerprints is not None else expected)
    record = _record(state, scope)
    # A late/duplicate completion does not own the active record. Reject it
    # without archiving a newer result or cancelling a newer run.
    if record.get("status") != AnalysisStatus.RUNNING.value or record.get("run_id") != run_id:
        return False
    if (record.get("scope") != scope.to_dict()
            or record.get("run_result_kind") != result_kind
            or (current_scope is not None and current_scope.key != scope.key)
            or record.get("fingerprints", {}).get("final_sha256") != expected.get("final_sha256")
            or live.get("final_sha256") != expected.get("final_sha256")):
        _archive_results(record, reason="analysis inputs changed while analysis was running")
        record.update({"status": AnalysisStatus.STALE.value, "run_id": None,
                       "reason": "분석 중 입력이 변경되어 오래된 결과를 저장하지 않았습니다.",
                       "generation": int(record.get("generation", 0)) + 1,
                       "updated_at": _now()})
        _store(state)[scope.key] = record
        return False
    results = dict(record.get("results", {}))
    results[result_kind] = deepcopy(result)
    record.update({"status": AnalysisStatus.CURRENT.value, "run_id": None,
                   "results": results, "fingerprints": expected,
                   "reason": None, "updated_at": _now()})
    _store(state)[scope.key] = record
    return True


def fail_analysis(state: Any, *, scope: AnalysisScope, run_id: str, error: str) -> bool:
    record = _record(state, scope)
    if record.get("run_id") != run_id or record.get("status") != AnalysisStatus.RUNNING.value:
        return False
    record.update({"status": AnalysisStatus.FAILED.value, "run_id": None,
                   "reason": str(error), "updated_at": _now()})
    _store(state)[scope.key] = record
    return True


def get_current_result(
    state: Any, *, scope: AnalysisScope,
    fingerprints: AnalysisFingerprints | dict[str, str], result_kind: str = "symptom",
) -> dict[str, Any] | None:
    expected = fingerprints.to_dict() if isinstance(fingerprints, AnalysisFingerprints) else dict(fingerprints)
    record = get_current_state(state, scope, expected)
    if (record.get("status") != AnalysisStatus.CURRENT.value
            or record.get("fingerprints", {}).get("final_sha256") != expected.get("final_sha256")):
        return None
    result = record.get("results", {}).get(result_kind)
    return deepcopy(result) if isinstance(result, dict) else None


def migrate_legacy_result(
    state: Any, *, scope: AnalysisScope, fingerprints: AnalysisFingerprints,
    result: dict[str, Any], legacy_fingerprint: str | None,
) -> bool:
    """One-time compatibility bridge for an existing session result."""
    if not isinstance(result, dict) or not legacy_fingerprint:
        return False
    record = _record(state, scope)
    if record.get("fingerprints") or record.get("results"):
        return False
    copy = deepcopy(result)
    copy.setdefault("legacy_input_fingerprint", legacy_fingerprint)
    record.update({"status": AnalysisStatus.CURRENT.value,
                   "fingerprints": fingerprints.to_dict(), "results": {"symptom": copy},
                   "reason": None, "updated_at": _now()})
    _store(state)[scope.key] = record
    return True
