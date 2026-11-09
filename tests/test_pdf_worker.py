from __future__ import annotations

import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import pytest

from circuitcare.datasheet.pdf_loader import (
    DatasheetError,
    clear_pdf_cache,
    extract_pdf_tables,
    load_pdf,
)
from circuitcare.datasheet.pdf_supervisor import run_pdf_worker


def _pdf(page_count: int = 1, *, text: str | None = None, table: bool = False) -> bytes:
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = pypdf.PdfWriter()
    for index in range(page_count):
        page = writer.add_blank_page(width=612, height=792)
        if (text or table) and index == 0:
            font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                     NameObject("/Subtype"): NameObject("/Type1"),
                                     NameObject("/BaseFont"): NameObject("/Helvetica")})
            fonts = DictionaryObject({NameObject("/F1"): writer._add_object(font)})
            page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): fonts})
            commands = []
            if text:
                escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                commands.append(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET")
            if table:
                commands.extend(("0 0 0 RG 1 w", "72 700 m 172 700 l S", "72 650 m 172 650 l S",
                                 "72 600 m 172 600 l S", "72 700 m 72 600 l S", "122 700 m 122 600 l S",
                                 "172 700 m 172 600 l S", "BT /F1 10 Tf 80 670 Td (Part) Tj ET",
                                 "BT /F1 10 Tf 130 670 Td (Value) Tj ET",
                                 "BT /F1 10 Tf 80 620 Td (R1) Tj ET",
                                 "BT /F1 10 Tf 130 620 Td (10k) Tj ET"))
            stream = DecodedStreamObject()
            stream.set_data("\n".join(commands).encode())
            page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.mark.integration
@pytest.mark.parametrize("pages", [1, 50])
def test_pdf_worker_parses_small_and_medium_documents(pages: int) -> None:
    document = load_pdf(_pdf(pages), f"{pages}-page.pdf")
    assert document["page_count"] == pages
    assert document["worker"]["result_status"] == "PASSED"
    assert document["worker"]["worker_pid"] != document["worker"]["parent_pid"]
    assert len(document["sha256"]) == 64
    assert document["worker"]["rss_measurement_status"] in {"AVAILABLE", "PARTIAL"}
    assert document["worker"]["rss_sample_count"] > 0
    assert document["worker"]["reader_threads_stopped"] is True


@pytest.mark.integration
def test_pdf_worker_extracts_text_and_tables() -> None:
    text = load_pdf(_pdf(text="Hello CircuitCare"), "text.pdf")
    assert "Hello CircuitCare" in text["pages"][0]["text"]
    tables = extract_pdf_tables(_pdf(table=True))
    assert tables and tables[0]["rows"][0] == ["Part", "Value"]
    assert tables[0]["rows"][1] == ["R1", "10k"]


@pytest.mark.integration
def test_pdf_worker_rejects_malformed_input_without_terminating_parent() -> None:
    with pytest.raises(DatasheetError):
        load_pdf(b"%PDF-1.7 damaged structure", "damaged.pdf")


@pytest.mark.integration
def test_pdf_worker_timeout_and_memory_limit_are_supervised() -> None:
    data = _pdf()
    with pytest.raises(DatasheetError, match="TIMEOUT"):
        run_pdf_worker(data, "document", timeout_seconds=0.001)
    with pytest.raises(DatasheetError, match="MEMORY_LIMIT"):
        run_pdf_worker(data, "document", max_rss_bytes=1)


@pytest.mark.integration
def test_pdf_worker_records_signal_termination_without_crashing_parent(monkeypatch) -> None:
    import circuitcare.datasheet.pdf_supervisor as supervisor

    records = []

    class SignaledWorker:
        pid = 12345
        stdout = BytesIO(b"")
        stderr = BytesIO(b"isolated worker signal")

        @staticmethod
        def poll():
            return -11

        @staticmethod
        def wait():
            return -11

    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: SignaledWorker())
    monkeypatch.setattr(supervisor, "_save_diagnostic", records.append)
    with pytest.raises(DatasheetError, match="exit -11"):
        run_pdf_worker(_pdf(), "document")
    assert records[0]["signal"] == "SIGSEGV"
    assert records[0]["result_status"] == "FAILED"
    assert records[0]["failure_reason"] == "WORKER_EXIT"


@pytest.mark.integration
def test_unavailable_rss_measurement_is_bounded_and_cleans_worker(tmp_path: Path, monkeypatch) -> None:
    import circuitcare.datasheet.pdf_supervisor as supervisor

    monkeypatch.setattr(supervisor, "_rss_bytes", lambda _pid: None)
    monkeypatch.setattr(supervisor, "RSS_UNAVAILABLE_GRACE_SECONDS", 0.05)
    monkeypatch.setenv("CIRCUITCARE_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))

    class RunningWorker:
        pid = 12345
        stdout = BytesIO()
        stderr = BytesIO()

        def __init__(self, *_args, **_kwargs): self.exit_code = None
        def poll(self): return self.exit_code
        def terminate(self): self.exit_code = -15
        def kill(self): self.exit_code = -9
        def wait(self, timeout=None): return self.exit_code

    monkeypatch.setattr(supervisor.subprocess, "Popen", RunningWorker)
    with pytest.raises(DatasheetError, match="RSS_UNAVAILABLE"):
        run_pdf_worker(_pdf(), "document", timeout_seconds=5)
    records = list((tmp_path / "diagnostics").glob("pdf-worker-*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["rss_measurement_status"] == "UNAVAILABLE_LIMIT"
    assert record["rss_sample_count"] == 0
    assert record["timeout_status"] == "RSS_UNAVAILABLE"
    assert record["result_status"] == "FAILED"
    assert record["reader_threads_stopped"] is True


def test_worker_diagnostic_uses_application_log_without_diagnostic_directory(monkeypatch, caplog) -> None:
    import circuitcare.datasheet.pdf_supervisor as supervisor

    monkeypatch.delenv("CIRCUITCARE_DIAGNOSTICS_DIR", raising=False)
    record = {"worker_pid": 4321, "exit_code": -9, "signal": "SIGKILL",
              "timeout_status": "TIMEOUT", "rss_measurement_status": "AVAILABLE"}
    supervisor._save_diagnostic(record)
    assert "worker_pid" in caplog.text
    assert "SIGKILL" in caplog.text
    assert "TIMEOUT" in caplog.text


@pytest.mark.integration
def test_timed_out_real_pdf_worker_is_reaped(tmp_path: Path, monkeypatch) -> None:
    import circuitcare.datasheet.pdf_supervisor as supervisor

    parent_pid = os.getpid()
    monkeypatch.setenv("CIRCUITCARE_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))
    temporary_paths = []

    class TrackingTemporaryDirectory(tempfile.TemporaryDirectory):
        def __enter__(self):
            path = Path(super().__enter__())
            temporary_paths.append(path)
            return str(path)

    monkeypatch.setattr(supervisor.tempfile, "TemporaryDirectory", TrackingTemporaryDirectory)
    with pytest.raises(DatasheetError, match="TIMEOUT"):
        run_pdf_worker(_pdf(), "document", timeout_seconds=0.001)
    record_path = next((tmp_path / "diagnostics").glob("pdf-worker-*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert temporary_paths and all(not path.exists() for path in temporary_paths)
    assert os.getpid() == parent_pid
    with pytest.raises(ProcessLookupError):
        os.kill(record["worker_pid"], 0)


@pytest.mark.integration
@pytest.mark.parametrize(("result_bytes", "reason"), [
    (None, "RESULT_MISSING"), (b"{", "RESULT_INVALID_JSON"),
    (b'{"status":"ERROR"}', "RESULT_INVALID_ENVELOPE"), ("TOO_LARGE", "RESULT_TOO_LARGE"),
])
def test_invalid_worker_results_are_recorded_as_failures(tmp_path: Path, monkeypatch,
                                                         result_bytes: bytes | str | None,
                                                         reason: str) -> None:
    import circuitcare.datasheet.pdf_supervisor as supervisor
    from circuitcare.datasheet.pdf_worker import MAX_RESULT_BYTES

    monkeypatch.setenv("CIRCUITCARE_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))

    class FinishedWorker:
        pid = os.getpid() + 100000
        stdout = BytesIO()
        stderr = BytesIO()

        def __init__(self, command, **_kwargs):
            self.returncode = 0
            if result_bytes is not None:
                payload = b"x" * (MAX_RESULT_BYTES + 1) if result_bytes == "TOO_LARGE" else result_bytes
                Path(command[5]).write_bytes(payload)

        def poll(self): return 0
        def wait(self, timeout=None): return 0

    monkeypatch.setattr(supervisor.subprocess, "Popen", FinishedWorker)
    with pytest.raises(DatasheetError):
        run_pdf_worker(_pdf(), "document")
    records = list((tmp_path / "diagnostics").glob("pdf-worker-*.json"))
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["failure_reason"] == reason
    assert record["result_status"] == "FAILED"
    assert record["reader_threads_stopped"] is True


def test_failed_pdf_is_not_cached_and_later_normal_document_can_load(monkeypatch) -> None:
    import circuitcare.datasheet.pdf_loader as loader
    import circuitcare.datasheet.pdf_supervisor as supervisor

    clear_pdf_cache()
    monkeypatch.setattr(supervisor, "run_pdf_worker",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(DatasheetError("worker failed")))
    data = _pdf()
    with pytest.raises(DatasheetError, match="worker failed"):
        load_pdf(data, "failed.pdf")
    assert not loader._document_cache

    digest = __import__("hashlib").sha256(data).hexdigest()
    monkeypatch.setattr(supervisor, "run_pdf_worker", lambda *_args, **_kwargs: (
        {"sha256": digest, "pages": [], "filename": "recovered.pdf"}, {"result_status": "PASSED"}))
    recovered = load_pdf(data, "recovered.pdf")
    assert recovered["filename"] == "recovered.pdf"
    assert len(loader._document_cache) == 1


def test_table_extraction_respects_page_limit_and_bounds_rows(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    import circuitcare.datasheet.pdf_loader as loader

    class FakePDF:
        def __init__(self, pages): self.pages = pages
        def __enter__(self): return self
        def __exit__(self, *_args): return False

    fake_page = SimpleNamespace(extract_tables=lambda: [[["a", "b"], ["1", "2"]]])
    monkeypatch.setitem(sys.modules, "pdfplumber", SimpleNamespace(open=lambda _stream: FakePDF([fake_page])))
    assert loader._extract_pdf_tables_local(_pdf(), max_pages=1) == [
        {"page_index": 0, "table_index": 0, "rows": [["a", "b"], ["1", "2"]]}]
    monkeypatch.setitem(sys.modules, "pdfplumber", SimpleNamespace(
        open=lambda _stream: FakePDF([fake_page, fake_page])))
    with pytest.raises(DatasheetError, match="page table processing limit"):
        loader._extract_pdf_tables_local(_pdf(), max_pages=1)


@pytest.mark.integration
def test_pdf_cache_reuses_same_hash_and_invalidates_changed_file() -> None:
    clear_pdf_cache()
    first_data, second_data = _pdf(1), _pdf(2)
    first = load_pdf(first_data, "same.pdf")
    again = load_pdf(first_data, "renamed.pdf")
    changed = load_pdf(second_data, "same.pdf")
    assert again["filename"] == "renamed.pdf"
    assert first["sha256"] == again["sha256"] != changed["sha256"]
    assert again["worker"]["worker_pid"] == first["worker"]["worker_pid"]
    assert changed["worker"]["worker_pid"] != first["worker"]["worker_pid"]


@pytest.mark.integration
def test_concurrent_pdf_worker_requests_keep_document_context_separate() -> None:
    first_data, second_data = _pdf(1), _pdf(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(load_pdf, first_data, "circuit-a.pdf")
        second_future = pool.submit(load_pdf, second_data, "circuit-b.pdf")
        first, second = first_future.result(), second_future.result()
    assert first["filename"] == "circuit-a.pdf" and first["page_count"] == 1
    assert second["filename"] == "circuit-b.pdf" and second["page_count"] == 2
    assert first["sha256"] != second["sha256"]


@pytest.mark.integration
def test_concurrent_sessions_for_same_pdf_keep_filename_context_separate() -> None:
    data = _pdf(1)
    clear_pdf_cache()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(load_pdf, data, "session-a.pdf")
        second_future = pool.submit(load_pdf, data, "session-b.pdf")
        first, second = first_future.result(), second_future.result()
    assert first["sha256"] == second["sha256"]
    assert first["filename"] == "session-a.pdf"
    assert second["filename"] == "session-b.pdf"
