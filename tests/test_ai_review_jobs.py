from __future__ import annotations

from threading import current_thread

from circuitcare.vision import ai_review_jobs


def test_ai_review_worker_uses_isolated_state_adapter_and_returns_result(monkeypatch):
    from circuitcare.vision import component_recognition_panel

    observed = {}

    def fake_review(api, *, image_bytes, scope, rows, records_key, progress, selected_component_ids=None):
        observed["thread"] = current_thread().name
        observed["api_type"] = type(api).__name__
        observed["image_bytes"] = image_bytes
        observed["scope"] = scope
        api.session_state[records_key] = [{**rows[0], "ocr_candidates": ["R1"]}]
        progress.write("review complete")
        return "complete"

    monkeypatch.setattr(component_recognition_panel, "ensure_ai_component_review", fake_review)
    result = ai_review_jobs._run({
        "state": {"cc_rows": [{"component_id": "c1"}]}, "image_bytes": b"image",
        "scope": "scope", "rows": [{"component_id": "c1"}],
        "records_key": "cc_rows", "fingerprint": "fp", "selected_component_ids": ["c1"],
    })
    assert observed == {"thread": current_thread().name, "api_type": "_WorkerAPI",
                        "image_bytes": b"image", "scope": "scope"}
    assert result["status"] == "COMPLETE"
    assert result["state"]["cc_rows"][0]["ocr_candidates"] == ["R1"]
    assert result["messages"] == ["review complete"]
    assert result["fingerprint"] == "fp"


def test_ai_review_job_results_are_owner_scoped_and_duplicate_jobs_reused(monkeypatch):
    from concurrent.futures import Future

    class _ImmediateExecutor:
        def __init__(self):
            self.calls = 0

        def submit(self, fn, snapshot):
            self.calls += 1
            future = Future()
            future.set_result({"status": "COMPLETE", "scope": snapshot["scope"],
                               "fingerprint": snapshot["fingerprint"]})
            return future

    executor = _ImmediateExecutor()
    monkeypatch.setattr(ai_review_jobs, "_EXECUTOR", executor)
    monkeypatch.setattr(ai_review_jobs, "_JOBS", ai_review_jobs.OrderedDict())
    monkeypatch.setattr(ai_review_jobs, "_LATEST_REQUEST", {})
    monkeypatch.setattr(ai_review_jobs, "_RETRY_COUNTS", {})
    args = {"scope": "photo-a", "fingerprint": "f1", "image_bytes": b"x",
            "rows": [], "records_key": "rows", "state_snapshot": {}}
    job_id, status = ai_review_jobs.submit_ai_review(owner="session-a", **args)
    same_job, reused_status = ai_review_jobs.submit_ai_review(owner="session-a", **args)
    assert status == "PENDING"
    assert reused_status == "COMPLETE"
    assert job_id == same_job
    assert executor.calls == 1
    assert ai_review_jobs.get_ai_review_job(owner="session-b", job_id=job_id) is None
    assert ai_review_jobs.get_ai_review_job(owner="session-a", job_id=job_id)["fingerprint"] == "f1"


def test_failed_job_is_not_reported_complete_and_explicit_retry_gets_new_id(monkeypatch):
    from concurrent.futures import Future

    class _FailThenPassExecutor:
        def __init__(self):
            self.calls = 0

        def submit(self, fn, snapshot):
            self.calls += 1
            future = Future()
            future.set_result({"status": "FAILED" if self.calls == 1 else "COMPLETE",
                               "scope": snapshot["scope"], "fingerprint": snapshot["fingerprint"]})
            return future

    executor = _FailThenPassExecutor()
    monkeypatch.setattr(ai_review_jobs, "_EXECUTOR", executor)
    monkeypatch.setattr(ai_review_jobs, "_JOBS", ai_review_jobs.OrderedDict())
    monkeypatch.setattr(ai_review_jobs, "_LATEST_REQUEST", {})
    monkeypatch.setattr(ai_review_jobs, "_RETRY_COUNTS", {})
    args = {"owner": "session", "scope": "photo", "fingerprint": "input", "image_bytes": b"x",
            "rows": [], "records_key": "rows", "state_snapshot": {}}
    failed_id, first_status = ai_review_jobs.submit_ai_review(**args)
    duplicate_id, duplicate_status = ai_review_jobs.submit_ai_review(**args)
    assert first_status == "PENDING"
    assert duplicate_status == "FAILED"
    assert duplicate_id == failed_id
    assert ai_review_jobs.get_ai_review_job(owner="session", job_id=failed_id)["status"] == "FAILED"
    retry_id, retry_status = ai_review_jobs.submit_ai_review(**args, retry_failed=True)
    assert retry_status == "PENDING"
    assert retry_id != failed_id
    assert ai_review_jobs.get_ai_review_job(owner="session", job_id=retry_id)["status"] == "COMPLETE"
    assert executor.calls == 2


def test_prune_jobs_removes_old_terminal_jobs_but_keeps_running_jobs(monkeypatch):
    from concurrent.futures import Future

    done = Future()
    done.set_result({"status": "COMPLETE"})
    running = Future()
    terminal = ai_review_jobs._Job("owner", "done", "scope", "fp", done,
                                   ai_review_jobs._Progress(), request_key="req", created_at=0)
    active = ai_review_jobs._Job("owner", "active", "scope", "fp", running,
                                 ai_review_jobs._Progress(), request_key="req2", created_at=0)
    monkeypatch.setattr(ai_review_jobs, "_JOBS", ai_review_jobs.OrderedDict({
        ("owner", "done"): terminal, ("owner", "active"): active,
    }))
    monkeypatch.setattr(ai_review_jobs, "_LATEST_REQUEST", {("owner", "req"): "done"})
    ai_review_jobs._prune_jobs(now=ai_review_jobs._JOB_TTL_SECONDS + 1)
    assert ("owner", "done") not in ai_review_jobs._JOBS
    assert ("owner", "active") in ai_review_jobs._JOBS


def test_commit_merges_ai_evidence_after_user_edit_without_overwriting_review_or_measurement(monkeypatch):
    from circuitcare.dashboard.views import unified_workspace
    from circuitcare.vision.component_recognition_panel import (
        _observation_fingerprint, _review_fingerprint, _review_user_version,
        _ocr_evidence_fingerprint, review_aux_keys, review_keys,
    )

    scope = "photo-scope"
    records_key = review_keys(scope)[1]
    current = [{"component_id": "r1", "detector_class": "resistor", "소자 종류": "capacitor",
                "상태": "제외", "user_edited": True, "confirmed_part_number": "C22",
                "x1": 1, "y1": 2, "x2": 10, "y2": 20, "measurement_id": "m-7"}]
    worker_rows = [{**current[0], "소자 종류": "resistor", "상태": "검수 필요",
                    "confirmed_part_number": "WRONG", "ocr_candidates": ["C22"]}]
    job_id = "job-1"
    state = {records_key: current,
             f"cc_ai_review_job_{scope}": job_id,
             f"cc_ai_review_job_owner_{scope}": "owner",
             f"cc_ai_review_job_fingerprint_{scope}": _observation_fingerprint(current),
             f"cc_ai_review_job_review_fingerprint_{scope}": _review_fingerprint(current, None),
             f"cc_ai_review_job_ocr_fingerprint_{scope}": _ocr_evidence_fingerprint(None)}
    fingerprint = state[f"cc_ai_review_job_fingerprint_{scope}"]
    job = {"status": "COMPLETE", "job_id": job_id, "fingerprint": fingerprint,
           "review_fingerprint": state[f"cc_ai_review_job_review_fingerprint_{scope}"],
           "user_version": _review_user_version(worker_rows), "duration_ms": 12,
           "selected_component_ids": ["r1"],
           "state": {records_key: worker_rows,
                     review_aux_keys(scope)[1]: {"source_fingerprint": fingerprint,
                                                   "review_mode": "selected_class_only",
                                                   "selected_component_ids": ["r1"],
                                                   "verification_status": "AI_SUGGESTIONS_ONLY"}}}
    monkeypatch.setattr(unified_workspace, "get_ai_review_job", lambda **_kwargs: job)
    assert unified_workspace._commit_review_job(type("St", (), {"session_state": state})(), scope=scope)
    saved = state[records_key][0]
    assert saved["상태"] == "제외" and saved["user_edited"] is True
    assert saved["소자 종류"] == "capacitor"
    assert saved["confirmed_part_number"] == "C22"
    assert saved["measurement_id"] == "m-7"
    assert saved["ocr_candidates"] == ["C22"]


def test_commit_accepts_ocr_evidence_created_inside_the_same_review_job(monkeypatch):
    from circuitcare.dashboard.views import unified_workspace
    from circuitcare.vision.component_recognition_panel import (
        _observation_fingerprint, _review_fingerprint, _review_user_version,
        _ocr_evidence_fingerprint, review_aux_keys, review_keys,
    )

    scope = "p-photo-native"
    records_key = review_keys(scope)[1]
    ocr_key, qwen_key = review_aux_keys(scope)
    rows = [{"component_id": "r1", "detector_class": "resistor", "소자 종류": "resistor",
             "상태": "검수 필요", "x1": 1, "y1": 2, "x2": 10, "y2": 20}]
    fingerprint = _observation_fingerprint(rows)
    review_fingerprint = _review_fingerprint(rows, None)
    ocr_result = {"source_image_fingerprint": "sha256:photo", "engine": "RapidOCR",
                  "records": [{"text": "R1", "bbox_xyxy": [1, 1, 3, 3]}]}
    worker_rows = [{**rows[0], "ocr_candidates": ["R1"], "reference_designators": ["R1"]}]
    state = {records_key: rows,
             f"cc_ai_review_job_{scope}": "job-ocr",
             f"cc_ai_review_job_owner_{scope}": "owner",
             f"cc_ai_review_job_fingerprint_{scope}": fingerprint,
             f"cc_ai_review_job_review_fingerprint_{scope}": review_fingerprint,
             f"cc_ai_review_job_ocr_fingerprint_{scope}": _ocr_evidence_fingerprint(None)}
    job = {"status": "COMPLETE", "job_id": "job-ocr", "fingerprint": fingerprint,
           "review_fingerprint": review_fingerprint, "user_version": _review_user_version(rows),
           "duration_ms": 20, "selected_component_ids": ["r1"],
           "state": {records_key: worker_rows, ocr_key: ocr_result,
                     qwen_key: {"verification_status": "AI_SUGGESTIONS_ONLY",
                                "review_mode": "selected_class_only",
                                "selected_component_ids": ["r1"]}}}
    monkeypatch.setattr(unified_workspace, "get_ai_review_job", lambda **_kwargs: job)
    assert unified_workspace._commit_review_job(type("St", (), {"session_state": state})(), scope=scope)
    assert state[f"cc_ai_review_job_status_{scope}"] == "COMPLETE"
    assert state[ocr_key] == ocr_result
    assert state[records_key][0]["ocr_candidates"] == ["R1"]


def test_review_progress_tracks_stage_and_batch_counts_without_streamlit(monkeypatch):
    progress = ai_review_jobs._Progress()
    progress.write("PCB OCR · 완료 · 8개 문자 후보")
    assert progress.snapshot()["stage"] == "OCR_BOARD"
    progress.write("Qwen 소자 검수 · 3/10 배치 진행 중")
    state = progress.snapshot()
    assert state["stage"] == "QWEN_CLASS_REVIEW"
    assert state["current_batch"] == 3
    assert state["completed_batches"] == 2
    assert state["total_batches"] == 10


def test_running_job_exposes_thread_safe_progress_snapshot(monkeypatch):
    from concurrent.futures import Future

    class _DelayedExecutor:
        def submit(self, fn, snapshot):
            self.future = Future()
            self.snapshot = snapshot
            return self.future

    executor = _DelayedExecutor()
    monkeypatch.setattr(ai_review_jobs, "_EXECUTOR", executor)
    monkeypatch.setattr(ai_review_jobs, "_JOBS", ai_review_jobs.OrderedDict())
    monkeypatch.setattr(ai_review_jobs, "_LATEST_REQUEST", {})
    monkeypatch.setattr(ai_review_jobs, "_RETRY_COUNTS", {})
    job_id, _status = ai_review_jobs.submit_ai_review(
        owner="owner", scope="scope", fingerprint="f", image_bytes=b"x",
        rows=[], records_key="rows", state_snapshot={},
    )
    job = ai_review_jobs._JOBS[("owner", job_id)]
    job.progress.set_stage("OCR_MARKINGS", "소자 각인 OCR 처리 중")
    running = ai_review_jobs.get_ai_review_job(owner="owner", job_id=job_id)
    assert running["status"] == "RUNNING"
    assert running["progress"]["stage"] == "OCR_MARKINGS"
    assert running["progress"]["message"] == "소자 각인 OCR 처리 중"


def test_worker_failure_returns_original_batch_error(monkeypatch):
    from circuitcare.vision import component_recognition_panel as panel
    def failed(api, **kwargs):
        api.session_state['cc_component_qwen_review_s'] = {
            'verification_status': 'AI_REVIEW_FAILED',
            'failed_batches': [{'error': 'Ollama request timed out'}]}
        return 'failed'
    monkeypatch.setattr(panel, 'ensure_ai_component_review', failed)
    result = ai_review_jobs._run({'state': {}, 'scope': 's', 'rows': [],
        'records_key': 'rows', 'image_bytes': b'photo', 'fingerprint': 'v1'})
    assert result['status'] == 'FAILED'
    assert result['error'] == 'Ollama request timed out'


def test_initial_review_runs_class_then_duplicate_checks_and_unloads_qwen(monkeypatch):
    from circuitcare.vision import component_recognition_panel as panel

    calls = []
    def classify(api, *, image_bytes, scope, rows, records_key, progress, selected_component_ids=None):
        calls.append("class")
        api.session_state["cc_ai_review_model_used_scope"] = True
        api.session_state["cc_component_qwen_review_scope"] = {
            "model": "qwen3.5:9b", "verification_status": "AI_SUGGESTIONS_ONLY",
            "class_reviews": [{"detection_id": item, "proposed_class": "unknown"}
                               for item in selected_component_ids],
        }
        return "complete"

    def duplicates(api, **kwargs):
        calls.append("duplicates")
        review = api.session_state["cc_component_qwen_review_scope"]
        review["duplicate_reviews"] = [{"detection_ids": ["a", "b"], "decision": "uncertain"}]
        return {"total": 1, "reviewed": 1, "pending": 0, "deferred": 0, "failed": 0}

    class Client:
        def unload_model(self, model):
            calls.append(f"unload:{model}")
            return True

    monkeypatch.setattr(panel, "ensure_ai_component_review", classify)
    monkeypatch.setattr(panel, "continue_duplicate_pair_review", duplicates)
    monkeypatch.setattr(panel, "_ollama_review_client", lambda: Client())
    result = ai_review_jobs._run({
        "state": {"rows": [
            {"component_id": "a", "소자 종류": "resistor", "상태": "검수 필요",
             "x1": 1, "y1": 1, "x2": 20, "y2": 20},
            {"component_id": "b", "소자 종류": "resistor", "상태": "검수 필요",
             "x1": 3, "y1": 3, "x2": 22, "y2": 22},
        ]}, "image_bytes": b"image", "scope": "scope",
        "rows": [{"component_id": "a", "x1": 1, "y1": 1, "x2": 20, "y2": 20},
                 {"component_id": "b", "x1": 3, "y1": 3, "x2": 22, "y2": 22}],
        "records_key": "rows", "fingerprint": "fp", "selected_component_ids": ["a", "b"],
    })
    assert calls == ["class", "duplicates", "unload:qwen3.5:9b"]
    assert result["status"] == "COMPLETE"
    assert result["state"]["cc_component_qwen_review_scope"]["duplicate_reviews"][0]["detection_ids"] == ["a", "b"]
