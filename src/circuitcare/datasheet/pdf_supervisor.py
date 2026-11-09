"""Supervise isolated PDF parser workers with timeout and RSS termination."""
from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from .pdf_loader import MAX_PDF_BYTES, MAX_PDF_PAGES, DatasheetError
from .pdf_worker import MAX_RESULT_BYTES

ROOT = Path(__file__).resolve().parents[3]
PDF_TIMEOUT_SECONDS = 60
PDF_MAX_RSS_BYTES = 768 * 1024 * 1024
RSS_UNAVAILABLE_GRACE_SECONDS = 5.0
MAX_CAPTURED_LOG_BYTES = 64 * 1024
_logger = logging.getLogger(__name__)


def _rss_bytes(pid: int) -> int | None:
    if sys.platform == "darwin":
        class ProcTaskInfo(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "virtual_size", "resident_size", "total_user", "total_system",
                "threads_user", "threads_system"
            )] + [(name, ctypes.c_int32) for name in (
                "policy", "faults", "pageins", "cow_faults", "messages_sent",
                "messages_received", "mach_syscalls", "unix_syscalls", "csw",
                "thread_count", "running_threads", "priority"
            )]

        try:
            libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
            query = libproc.proc_pidinfo
            query.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                              ctypes.c_void_p, ctypes.c_int]
            query.restype = ctypes.c_int
            info = ProcTaskInfo()
            size = query(pid, 4, 0, ctypes.byref(info), ctypes.sizeof(info))
            return int(info.resident_size) if size == ctypes.sizeof(info) else None
        except (AttributeError, OSError):
            return None
    if sys.platform.startswith("linux"):
        try:
            resident_pages = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            return None
    try:
        import psutil
        return int(psutil.Process(pid).memory_info().rss)
    except (ImportError, OSError, ValueError):
        return None


def _drain(stream, tail: bytearray, lock: threading.Lock) -> None:
    try:
        while True:
            block = stream.read(8192)
            if not block:
                return
            with lock:
                tail.extend(block)
                if len(tail) > MAX_CAPTURED_LOG_BYTES:
                    del tail[:-MAX_CAPTURED_LOG_BYTES]
    except (OSError, ValueError):
        return


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def _save_diagnostic(record: dict) -> None:
    directory = os.environ.get("CIRCUITCARE_DIAGNOSTICS_DIR")
    if not directory:
        _logger.warning("PDF worker diagnostic: %s", json.dumps(record, sort_keys=True))
        return
    path = Path(directory)
    try:
        path.mkdir(parents=True, exist_ok=True)
        name = f"pdf-worker-{record['worker_pid']}-{time.time_ns()}.json"
        (path / name).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    except OSError:
        _logger.exception("Could not persist PDF worker diagnostic; record follows.")
        _logger.warning("PDF worker diagnostic: %s", json.dumps(record, sort_keys=True))


def _cleanup_readers(process: subprocess.Popen, readers: list[threading.Thread]) -> bool:
    for reader in readers:
        reader.join(timeout=1)
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except (OSError, ValueError):
                pass
    for reader in readers:
        reader.join(timeout=1)
    return all(not reader.is_alive() for reader in readers)


def run_pdf_worker(data: bytes, operation: str, *, filename: str = "datasheet.pdf",
                   max_bytes: int = MAX_PDF_BYTES, max_pages: int = MAX_PDF_PAGES,
                   timeout_seconds: float = PDF_TIMEOUT_SECONDS,
                   max_rss_bytes: int = PDF_MAX_RSS_BYTES) -> tuple[object, dict]:
    if operation not in {"document", "tables"}:
        raise ValueError("Unsupported PDF worker operation")
    if not isinstance(data, bytes) or not data or len(data) > max_bytes or not data.startswith(b"%PDF-"):
        raise DatasheetError("PDF worker input failed size or signature validation.")
    digest = hashlib.sha256(data).hexdigest()
    with tempfile.TemporaryDirectory(prefix="circuitcare-pdf-") as temporary_dir:
        directory = Path(temporary_dir)
        input_path, output_path = directory / "input.pdf", directory / "result.json"
        input_path.write_bytes(data)
        command = [sys.executable, "-m", "circuitcare.datasheet.pdf_worker", operation,
                   str(input_path), str(output_path), "--filename", Path(filename).name[:180],
                   "--max-bytes", str(max_bytes), "--max-pages", str(max_pages)]
        worker_env = os.environ.copy()
        source_path = str(ROOT / "src")
        worker_env["PYTHONPATH"] = os.pathsep.join(
            item for item in (source_path, worker_env.get("PYTHONPATH", "")) if item
        )
        process = subprocess.Popen(command, env=worker_env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   close_fds=False)
        stdout_tail, stderr_tail, log_lock = bytearray(), bytearray(), threading.Lock()
        readers = [threading.Thread(target=_drain, args=(process.stdout, stdout_tail, log_lock), daemon=True),
                   threading.Thread(target=_drain, args=(process.stderr, stderr_tail, log_lock), daemon=True)]
        for reader in readers:
            reader.start()
        started = time.monotonic()
        peak_rss = 0
        termination = None
        rss_sample_count = 0
        rss_failure_count = 0
        rss_unavailable_since = None
        rss_max_unavailable_seconds = 0.0
        while process.poll() is None:
            now = time.monotonic()
            current_rss = _rss_bytes(process.pid)
            if current_rss is not None:
                rss_sample_count += 1
                if rss_unavailable_since is not None:
                    rss_max_unavailable_seconds = max(
                        rss_max_unavailable_seconds, now - rss_unavailable_since)
                    rss_unavailable_since = None
                peak_rss = max(peak_rss, current_rss)
                if current_rss > max_rss_bytes:
                    termination = "MEMORY_LIMIT"
                    _terminate(process)
                    break
            else:
                rss_failure_count += 1
                if rss_unavailable_since is None:
                    rss_unavailable_since = now
                unavailable_for = now - rss_unavailable_since
                rss_max_unavailable_seconds = max(rss_max_unavailable_seconds, unavailable_for)
                if unavailable_for >= RSS_UNAVAILABLE_GRACE_SECONDS:
                    termination = "RSS_UNAVAILABLE"
                    _terminate(process)
                    break
            if now - started > timeout_seconds:
                termination = "TIMEOUT"
                _terminate(process)
                break
            time.sleep(0.05)
        exit_code = process.wait()
        readers_stopped = _cleanup_readers(process, readers)
        elapsed = round(time.monotonic() - started, 4)
        sig = -exit_code if exit_code < 0 else None
        if termination == "RSS_UNAVAILABLE":
            rss_status = "UNAVAILABLE_LIMIT"
        elif rss_sample_count == 0:
            rss_status = "UNAVAILABLE"
        elif rss_failure_count:
            rss_status = "PARTIAL"
        else:
            rss_status = "AVAILABLE"
        record = {"operation": operation, "parent_pid": os.getpid(), "worker_pid": process.pid,
                  "input_sha256": digest, "processing_time_seconds": elapsed,
                  "peak_rss_mb": round(peak_rss / 1048576, 2) if peak_rss else None,
                  "rss_measurement_status": rss_status, "rss_sample_count": rss_sample_count,
                  "rss_failure_count": rss_failure_count,
                  "rss_max_unavailable_seconds": round(rss_max_unavailable_seconds, 4),
                  "timeout_status": termination, "exit_code": exit_code,
                  "reader_threads_stopped": readers_stopped,
                  "signal": signal.Signals(sig).name if sig in signal.Signals._value2member_map_ else sig,
                  "result_status": "PENDING",
                  "stdout_tail": bytes(stdout_tail).decode("utf-8", "replace")[-2000:],
                  "stderr_tail": bytes(stderr_tail).decode("utf-8", "replace")[-4000:]}
        def fail(message: str, reason: str) -> None:
            record["result_status"] = "FAILED"
            record["failure_reason"] = reason
            _save_diagnostic(record)
            raise DatasheetError(message)

        if termination:
            fail(f"PDF worker stopped: {termination} (pid {process.pid}).", termination)
        if exit_code != 0:
            detail = ""
            if output_path.is_file() and output_path.stat().st_size <= MAX_RESULT_BYTES:
                try:
                    detail = json.loads(output_path.read_text(encoding="utf-8")).get("error", "")
                except (OSError, ValueError, AttributeError):
                    pass
            fail(f"PDF worker failed (exit {exit_code}): {detail or record['stderr_tail'][-500:]}",
                 "WORKER_EXIT")
        try:
            output_size = output_path.stat().st_size
        except OSError:
            fail("PDF worker result is missing or unreadable.", "RESULT_MISSING")
        if output_size > MAX_RESULT_BYTES:
            fail("PDF worker result exceeds the result size limit.", "RESULT_TOO_LARGE")
        try:
            envelope = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            fail(f"PDF worker returned invalid JSON: {exc}", "RESULT_INVALID_JSON")
        if not isinstance(envelope, dict) or envelope.get("status") != "OK":
            fail("PDF worker returned an invalid result envelope.", "RESULT_INVALID_ENVELOPE")
        if not readers_stopped:
            fail("PDF worker log reader threads did not stop cleanly.", "READER_CLEANUP_FAILED")
        record["result_status"] = "PASSED"
        _save_diagnostic(record)
        return envelope.get("result"), record
