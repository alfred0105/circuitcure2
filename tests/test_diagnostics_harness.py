from __future__ import annotations

import os
import sys

import pytest

from scripts.run_crash_diagnostics import _run


@pytest.mark.skipif(os.name != "posix", reason="Signal exit codes are POSIX-specific")
@pytest.mark.crash_injection
def test_diagnostic_runner_records_child_sigsegv_without_killing_parent(tmp_path):
    report = _run("sigsegv_fixture", [sys.executable, "-c", "import os,signal; os.kill(os.getpid(), signal.SIGSEGV)"],
                  tmp_path, timeout=10)
    assert report["status"] == "FAILED"
    assert report["signal"] == "SIGSEGV"
    assert report["exit_code"] < 0
    assert report["parent_pid"] == os.getpid()
    assert report["child_pid"] != report["parent_pid"]
    assert report["termination_class"] == "EXPECTED_CHILD_SIGNAL_TEST"


def test_diagnostic_runner_terminates_timed_out_child(tmp_path):
    report = _run("timeout_fixture", [sys.executable, "-c", "import time; time.sleep(5)"],
                  tmp_path, timeout=1)
    assert report["status"] == "TIMEOUT"
    assert report["exit_code"] != 0


@pytest.mark.parametrize(("sig_name", "expected"), [("SIGTERM", "CHILD_PROCESS_SIGTERM"),
                                                       ("SIGKILL", "PROCESS_KILLED")])
@pytest.mark.skipif(os.name != "posix", reason="Signal exit codes are POSIX-specific")
@pytest.mark.crash_injection
def test_diagnostic_runner_classifies_terminated_child(sig_name, expected, tmp_path):
    command = [sys.executable, "-c", f"import os,signal; os.kill(os.getpid(), signal.{sig_name})"]
    report = _run(f"{sig_name.lower()}_fixture", command, tmp_path, timeout=10)
    assert report["status"] == "FAILED"
    assert report["termination_class"] == expected
    assert report["parent_pid"] == os.getpid()
    assert report["child_pid"] != report["parent_pid"]


@pytest.mark.stress
def test_diagnostic_log_cap_does_not_limit_child_result_files(tmp_path):
    result_path = tmp_path / "result-30mb.bin"
    command = [sys.executable, "-c", (
        "import os,sys; fd=os.open(sys.argv[1],os.O_CREAT|os.O_WRONLY,0o600); "
        "os.ftruncate(fd,30*1024*1024); os.close(fd); payload=b'z'*1048576; "
        "[sys.stdout.buffer.write(payload) for _ in range(25)]; sys.stdout.flush()"
    ), str(result_path)]
    report = _run("bounded_log_fixture", command, tmp_path, timeout=10)
    assert report["status"] == "PASSED"
    assert report["log_status"] == "LOG_TRUNCATED"
    assert report["stdout_truncated"] is True
    assert (tmp_path / "bounded_log_fixture.stdout.log").stat().st_size == 20 * 1024 * 1024
    assert result_path.stat().st_size == 30 * 1024 * 1024
