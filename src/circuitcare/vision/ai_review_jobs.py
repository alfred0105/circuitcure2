"""Bounded, session-isolated background execution for PCB AI review.

Workers receive immutable input snapshots and never call Streamlit APIs. Results
are read and committed by the owning Streamlit session after fingerprint checks.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from collections import OrderedDict
import threading
import time
from typing import Any
import re


JOB_CHANGED = threading.Condition()
JOB_CHANGED.version = 0


def _notify_progress(*_args):
    with JOB_CHANGED:
        JOB_CHANGED.version += 1
        JOB_CHANGED.notify_all()


@dataclass
class _Job:
    owner: str
    job_id: str
    scope: str
    fingerprint: str
    future: Future
    progress: "_Progress"
    request_key: str = ""
    attempt: int = 1
    started_at: float = field(default_factory=time.perf_counter)
    created_at: float = field(default_factory=time.time)


class _Progress:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.messages: list[str] = []
        self.stage = "PENDING"
        self.message = "AI 검수 대기 중"
        self.completed_batches = 0
        self.total_batches = 0
        self.current_batch = 0
        self.errors: list[str] = []
        self.updated_at = time.time()

    def set_stage(self, stage: str, message: str, *, completed_batches: int | None = None,
                  total_batches: int | None = None, current_batch: int | None = None,
                  error: str | None = None) -> None:
        with self._lock:
            self.stage = str(stage)
            self.message = str(message)[:240]
            if completed_batches is not None:
                self.completed_batches = max(0, int(completed_batches))
            if total_batches is not None:
                self.total_batches = max(0, int(total_batches))
            if current_batch is not None:
                self.current_batch = max(0, int(current_batch))
            if error:
                self.errors.append(str(error)[:400])
                del self.errors[:-8]
            self.updated_at = time.time()
        _notify_progress()

    def write(self, message: str) -> None:
        value = str(message)[:180]
        with self._lock:
            self.messages.append(value)
            del self.messages[:-8]
        lower = value.lower()
        if "실패" in lower or "오류" in lower:
            self.set_stage("FAILED", value, error=value)
        elif "pcb ocr" in lower:
            self.set_stage("OCR_BOARD", value)
        elif "각인 ocr" in lower:
            self.set_stage("OCR_MARKINGS", value)
        elif "qwen 소자 검수" in lower and "배치" in lower:
            match = re.search(r"(\d+)\s*/\s*(\d+)", value)
            if match:
                current, total = map(int, match.groups())
                self.set_stage("QWEN_CLASS_REVIEW", value,
                               completed_batches=current - 1,
                               total_batches=total, current_batch=current)
        elif "완료" in lower:
            self.set_stage("RESULT_VALIDATION", value)
        _notify_progress()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"stage": self.stage, "message": self.message,
                    "completed_batches": self.completed_batches,
                    "total_batches": self.total_batches, "current_batch": self.current_batch,
                    "last_event": self.messages[-1] if self.messages else self.message,
                    "messages": list(self.messages), "errors": list(self.errors),
                    "updated_at": self.updated_at}

    def update(self, **_kwargs: Any) -> None:
        return None


class _WorkerAPI:
    """Small Streamlit-free adapter for the existing review workflow."""
    def __init__(self, state: dict[str, Any]) -> None:
        self.session_state = state

    def spinner(self, *_args: Any, **_kwargs: Any) -> Any:
        from contextlib import nullcontext
        return nullcontext()

    def warning(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def status(self, *_args: Any, **_kwargs: Any) -> _Progress:
        return _Progress()


def _unload_review_model(state: dict[str, Any], scope: str, progress: _Progress) -> bool | None:
    if not state.pop(f"cc_ai_review_model_used_{scope}", False):
        return None
    try:
        from circuitcare.vision.component_recognition_panel import _ollama_review_client
        from circuitcare.vision.ai_review_service import DEFAULT_REVIEW_MODEL
        _ollama_review_client().unload_model(DEFAULT_REVIEW_MODEL)
        return True
    except Exception as unload_error:
        progress.set_stage("MODEL_UNLOAD_WARNING", "검수는 끝났지만 Qwen3.5 9B 메모리 해제를 확인하지 못했습니다.",
                           error=str(unload_error)[:300])
        return False


_LOCK = threading.RLock()
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="circuitcare-ai-review")
_JOBS: OrderedDict[tuple[str, str], _Job] = OrderedDict()
_LATEST_REQUEST: dict[tuple[str, str], str] = {}
_RETRY_COUNTS: dict[tuple[str, str], int] = {}
_MAX_JOBS = 32
_MAX_PENDING = 2
_JOB_TTL_SECONDS = 60 * 60


def _result_status(job: _Job) -> str:
    if not job.future.done():
        progress = job.progress.snapshot()
        return "PENDING" if progress.get("stage") == "PENDING" else "RUNNING"
    try:
        result = job.future.result()
    except Exception:
        return "FAILED"
    if not isinstance(result, dict):
        return "FAILED"
    status = str(result.get("status", "FAILED")).upper()
    return status if status in {"COMPLETE", "FAILED", "UNAVAILABLE", "AI_REVIEW_PARTIAL", "STALE"} else "FAILED"


def _prune_jobs(now: float | None = None) -> None:
    """Bound retained job metadata while never evicting a running worker."""
    now = time.time() if now is None else now
    for key, job in list(_JOBS.items()):
        if job.future.done() and now - job.created_at > _JOB_TTL_SECONDS:
            _JOBS.pop(key, None)
            request = (job.owner, job.request_key)
            if _LATEST_REQUEST.get(request) == job.job_id:
                _LATEST_REQUEST.pop(request, None)
                _RETRY_COUNTS.pop(request, None)
    while len(_JOBS) > _MAX_JOBS:
        oldest_done = next((key for key, job in _JOBS.items() if job.future.done()), None)
        if oldest_done is None:
            break
        removed = _JOBS.pop(oldest_done)
        request = (removed.owner, removed.request_key)
        if _LATEST_REQUEST.get(request) == removed.job_id:
            _LATEST_REQUEST.pop(request, None)
            _RETRY_COUNTS.pop(request, None)


def _run(snapshot: dict[str, Any], progress: _Progress | None = None) -> dict[str, Any]:
    from circuitcare.vision.component_recognition_panel import (
        ensure_ai_component_review,
    )

    started = time.perf_counter()
    state = deepcopy(snapshot["state"])
    progress = progress or _Progress()
    progress.set_stage("QWEN_CLASS_REVIEW" if snapshot.get("selected_component_ids") else "OCR_BOARD",
                       "선택 박스 분류 시작" if snapshot.get("selected_component_ids")
                       else "OCR·Qwen 검수 작업 시작")
    api = _WorkerAPI(state)
    try:
        result = ensure_ai_component_review(
            api, image_bytes=snapshot["image_bytes"], scope=snapshot["scope"],
            rows=deepcopy(snapshot["rows"]), records_key=snapshot["records_key"],
            progress=progress, selected_component_ids=snapshot.get("selected_component_ids"),
        )
        review_key = f"cc_component_qwen_review_{snapshot['scope']}"
        review = state.get(review_key) if isinstance(state.get(review_key), dict) else {}
        model_unloaded = _unload_review_model(state, snapshot["scope"], progress)
        result_status = ("AI_REVIEW_PARTIAL" if review.get("verification_status") == "AI_REVIEW_PARTIAL"
                         else "FAILED" if result.upper() in {"FAILED", "UNAVAILABLE"}
                         else "COMPLETE")
        final_stage = "FAILED" if result_status == "FAILED" else "COMPLETE"
        failures = review.get("failed_batches") or []
        error = next((str(item.get("error")) for item in failures
                      if isinstance(item, dict) and item.get("error")), None)
        if not error:
            diagnostics = review.get("review_diagnostics") or {}
            error = next((str(diagnostics.get(stage, {}).get("error"))
                          for stage in ("qwen", "candidate_search")
                          if isinstance(diagnostics.get(stage), dict)
                          and diagnostics[stage].get("error")), None)
        final_message = ((f"AI 검수 실패 · {error}" if error else "AI 검수 작업 실패")
                         if final_stage == "FAILED" else
                         "AI 검수 완료 · Qwen3.5 9B 메모리 해제" if model_unloaded is True else
                         "AI 검수 완료 · 모델 메모리 해제 확인 실패" if model_unloaded is False else
                         "AI 검수 작업 완료")
        progress.set_stage(final_stage, final_message)
        return {"status": result_status, "state": state, "messages": progress.snapshot()["messages"],
                "error": error,
                "progress": progress.snapshot(),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "fingerprint": snapshot["fingerprint"], "scope": snapshot["scope"],
                "user_version": snapshot.get("user_version"),
                "review_fingerprint": snapshot.get("review_fingerprint"),
                "selected_component_ids": snapshot.get("selected_component_ids")}
    except Exception as exc:  # Keep worker failures isolated from Streamlit's script thread.
        _unload_review_model(state, snapshot["scope"], progress)
        error = f"{type(exc).__name__}: {str(exc)[:400]}"
        progress.set_stage("FAILED", error, error=error)
        return {"status": "FAILED", "error": error,
                "messages": progress.snapshot()["messages"], "progress": progress.snapshot(),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "fingerprint": snapshot["fingerprint"], "scope": snapshot["scope"],
                "user_version": snapshot.get("user_version"),
                "review_fingerprint": snapshot.get("review_fingerprint"),
                "selected_component_ids": snapshot.get("selected_component_ids")}


def submit_ai_review(*, owner: str, scope: str, fingerprint: str, image_bytes: bytes,
                     rows: list[dict[str, Any]], records_key: str,
                     state_snapshot: dict[str, Any], user_version: str | None = None,
                     review_fingerprint: str | None = None,
                     selected_component_ids: list[str] | None = None,
                     retry_failed: bool = False) -> tuple[str | None, str]:
    """Submit or reuse one review job; never queue an unbounded model backlog."""
    selected_ids = sorted({str(value) for value in (selected_component_ids or [])})
    request_key = hashlib.sha256(json.dumps(
        [scope, fingerprint, "selected_class_only" if selected_ids else "legacy_full_review", selected_ids],
        separators=(",", ":"),
    ).encode()).hexdigest()[:24]
    request = (owner, request_key)
    with _LOCK:
        _prune_jobs()
        latest_id = _LATEST_REQUEST.get(request)
        existing = _JOBS.get((owner, latest_id)) if latest_id else None
        if existing:
            existing_status = _result_status(existing)
            if existing_status in {"PENDING", "RUNNING"}:
                return existing.job_id, existing_status
            if existing_status == "COMPLETE":
                return existing.job_id, existing_status
            if existing_status == "AI_REVIEW_PARTIAL" and not retry_failed:
                return existing.job_id, existing_status
            if not retry_failed:
                return existing.job_id, existing_status
        attempt = _RETRY_COUNTS.get(request, 0) + 1
        if existing and retry_failed:
            attempt = max(attempt, existing.attempt + 1)
        job_id = hashlib.sha256(f"{owner}|{request_key}|attempt:{attempt}".encode()).hexdigest()[:24]
        key = (owner, job_id)
        active = [job for job in _JOBS.values() if not job.future.done()]
        if len(active) >= _MAX_PENDING + 1:
            return None, "QUEUE_FULL"
        _RETRY_COUNTS[request] = attempt
        snapshot = {"state": deepcopy(state_snapshot), "image_bytes": bytes(image_bytes),
                    "scope": scope, "rows": deepcopy(rows), "records_key": records_key,
                    "fingerprint": fingerprint, "user_version": user_version,
                    "review_fingerprint": review_fingerprint,
                    "selected_component_ids": selected_ids or None}
        progress = _Progress()
        progress.set_stage("PENDING", "AI 검수 대기 중")
        future = _EXECUTOR.submit(lambda queued_snapshot: _run(queued_snapshot, progress), snapshot)
        future.add_done_callback(_notify_progress)
        _JOBS[key] = _Job(owner, job_id, scope, fingerprint, future, progress,
                          request_key=request_key, attempt=attempt)
        _JOBS.move_to_end(key)
        _LATEST_REQUEST[request] = job_id
        _prune_jobs()
        return job_id, "PENDING"


def get_ai_review_job(*, owner: str, job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get((owner, job_id))
        if not job:
            return None
        if not job.future.done():
            progress = job.progress.snapshot()
            return {"status": _result_status(job),
                    "scope": job.scope, "fingerprint": job.fingerprint,
                    "elapsed_ms": round((time.perf_counter() - job.started_at) * 1000, 2),
                    "progress": progress, "job_id": job.job_id, "attempt": job.attempt}
        try:
            result = job.future.result()
            if isinstance(result, dict):
                return {**result, "job_id": job.job_id, "attempt": job.attempt}
            return {"status": "FAILED", "scope": job.scope, "fingerprint": job.fingerprint,
                    "error": "AI review worker returned an invalid result.",
                    "job_id": job.job_id, "attempt": job.attempt}
        except Exception as exc:
            return {"status": "FAILED", "scope": job.scope, "fingerprint": job.fingerprint,
                    "error": f"{type(exc).__name__}: {str(exc)[:400]}",
                    "job_id": job.job_id, "attempt": job.attempt}

