#!/usr/bin/env python3
"""Run pytest-based diagnostic suites in isolated, timed child processes."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_DIAGNOSTIC_LOG_BYTES = 20 * 1024 * 1024
SUITES = {
    "fast": ["-m", "pytest", "-q", "-m", "not integration and not stress and not crash_injection"],
    "integration": ["-m", "pytest", "-q", "-m", "integration"],
    "stress": ["-m", "pytest", "-q", "-m", "stress"],
}


def _disable_core_dumps() -> None:
    if os.name == "posix":
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _copy_bounded_log(source, path: Path) -> bool:
    """Drain child output completely while retaining at most the configured log bytes."""
    written = 0
    truncated = False
    with path.open("wb") as output:
        while block := source.read(64 * 1024):
            remaining = MAX_DIAGNOSTIC_LOG_BYTES - written
            if remaining > 0:
                output.write(block[:remaining])
                written += min(len(block), remaining)
            if len(block) > remaining:
                truncated = True
    return truncated


def _probe_import(module: str, timeout: float = 20) -> dict:
    """Probe native imports out of process so an import crash cannot kill this supervisor."""
    command = [sys.executable, "-c", (
        "import importlib,json,sys; m=importlib.import_module(sys.argv[1]); "
        "print(json.dumps({'version':getattr(m,'__version__','unknown')}))"
    ), module]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False,
                                cwd=ROOT, start_new_session=(os.name == "posix"))
        try:
            details = json.loads(result.stdout.strip().splitlines()[-1]) if result.stdout.strip() else {}
        except (ValueError, IndexError):
            details = {}
        return {"available": result.returncode == 0, "version": details.get("version"),
                "exit_code": result.returncode, "signal": -result.returncode if result.returncode < 0 else None,
                "stderr_tail": result.stderr[-4000:]}
    except subprocess.TimeoutExpired as exc:
        return {"available": False, "status": "TIMEOUT", "timeout_seconds": timeout,
                "stderr_tail": (exc.stderr or b"")[-4000:].decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")[-4000:]}
    except OSError as exc:
        return {"available": False, "status": "LAUNCH_FAILED", "error": str(exc)}


def _environment() -> dict:
    packages = {}
    for module, label in (("streamlit", "streamlit"), ("ngspice", "ngspice"), ("numpy", "numpy"),
                          ("cv2", "opencv"), ("onnxruntime", "onnxruntime"), ("rapidocr", "rapidocr")):
        if module == "ngspice":
            path = shutil.which("ngspice")
            version = None
            if path:
                try:
                    version = subprocess.run([path, "--version"], capture_output=True, text=True,
                                             timeout=5).stdout.splitlines()[1].strip("* ")
                except (OSError, subprocess.SubprocessError, IndexError):
                    version = "unknown"
            packages[label] = {"available": bool(path), "version": version}
            continue
        packages[label] = _probe_import(module)
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        ram = None
    try:
        available_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        available_ram = None
    return {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "python": sys.version.split()[0],
            "os": platform.system(), "os_release": platform.release(), "architecture": platform.machine(),
            "pid": os.getpid(), "parent_pid": os.getppid(), "executable": sys.executable,
            "working_directory": str(Path.cwd()), "git_sha": _git_sha(), "available_ram_bytes": ram,
            "available_system_memory_bytes": available_ram,
            "selected_model": None, "active_workflow_step": "diagnostic_suite_start", "dependencies": packages}


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _terminate_tree(process: subprocess.Popen) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            # The group can outlive its leader; terminate remaining descendants too.
            time.sleep(0.05)
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            try:
                process.kill()
            except OSError:
                pass
    elif process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _process_tree_rss_bytes(root_pid: int) -> int | None:
    """Sample RSS for a child and descendants using ps, available on macOS/Linux."""
    try:
        result = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], capture_output=True,
                                text=True, timeout=3, check=True)
        rows = {}
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) == 3:
                rows[int(parts[0])] = (int(parts[1]), int(parts[2]) * 1024)
        selected = {root_pid}
        changed = True
        while changed:
            changed = False
            for pid, (parent, _rss) in rows.items():
                if parent in selected and pid not in selected:
                    selected.add(pid)
                    changed = True
        return sum(rows[pid][1] for pid in selected if pid in rows)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _run(name: str, command: list[str], output_dir: Path, timeout: int) -> dict:
    stdout_path = output_dir / f"{name}.stdout.log"
    stderr_path = output_dir / f"{name}.stderr.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    started = time.monotonic()
    parent_pid = os.getpid()
    peak_rss = None
    baseline_rss = None
    final_rss = None
    child_environment = os.environ.copy()
    child_environment["CIRCUITCARE_DIAGNOSTICS_DIR"] = str(output_dir.resolve())
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=child_environment, start_new_session=(os.name == "posix"),
                               preexec_fn=_disable_core_dumps if os.name == "posix" else None)
    stdout_result: list[bool] = []
    stderr_result: list[bool] = []
    stdout_thread = threading.Thread(target=lambda: stdout_result.append(
        _copy_bounded_log(process.stdout, stdout_path)), daemon=True)
    stderr_thread = threading.Thread(target=lambda: stderr_result.append(
        _copy_bounded_log(process.stderr, stderr_path)), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    timed_out = False
    try:
        # Let the interpreter exec and become visible in ps before recording its first sample.
        time.sleep(0.05)
        baseline_rss = _process_tree_rss_bytes(process.pid)
        while process.poll() is None:
            current_rss = _process_tree_rss_bytes(process.pid)
            if current_rss is not None:
                peak_rss = max(peak_rss or 0, current_rss)
                final_rss = current_rss
            if time.monotonic() - started > timeout:
                timed_out = True
                _terminate_tree(process)
                break
            time.sleep(0.1)
        exit_code = process.wait()
    finally:
        _terminate_tree(process)
        stdout_thread.join(timeout=3)
        stderr_thread.join(timeout=3)
    elapsed = time.monotonic() - started
    final_rss = _process_tree_rss_bytes(process.pid) or final_rss
    sig = -exit_code if exit_code < 0 else None
    status = "TIMEOUT" if timed_out else ("PASSED" if exit_code == 0 else "FAILED")
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")[-20000:]
    signal_name = signal.Signals(sig).name if sig and sig in signal.Signals._value2member_map_ else sig
    termination = ("TIMEOUT" if status == "TIMEOUT" else "NORMAL_EXIT" if exit_code == 0
                   else "CHILD_PROCESS_SIGSEGV" if signal_name == "SIGSEGV"
                   else "CHILD_PROCESS_SIGTERM" if signal_name == "SIGTERM"
                   else "PROCESS_KILLED" if signal_name == "SIGKILL"
                   else "PYTHON_EXCEPTION" if "Traceback (most recent call last)" in stderr_text
                   else "CHILD_PROCESS_FAILURE" if exit_code != 0 else "UNKNOWN_TERMINATION")
    if name == "sigsegv_fixture" and signal_name == "SIGSEGV":
        termination = "EXPECTED_CHILD_SIGNAL_TEST"
    return {"suite": name, "stage": name, "status": status,
            "termination_class": termination, "parent_pid": parent_pid, "child_pid": process.pid,
            "process_id": process.pid, "command": command, "started_at": started_at,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "process_name": Path(command[0]).name if command else None,
            "executable_path": command[0] if command else None, "python_version": sys.version.split()[0],
            "architecture": platform.machine(), "working_directory": str(ROOT), "git_sha": _git_sha(),
            "last_started_stage": name, "last_completed_stage": name if exit_code == 0 else None,
            "max_stdout_or_stderr_bytes": MAX_DIAGNOSTIC_LOG_BYTES,
            "stdout_truncated": stdout_result[0] if stdout_result else False,
            "stderr_truncated": stderr_result[0] if stderr_result else False,
            "log_status": "LOG_TRUNCATED" if any(stdout_result + stderr_result) else "COMPLETE",
            "wall_time_seconds": round(elapsed, 3), "baseline_rss_mb": round(baseline_rss / 1048576, 2) if baseline_rss else None,
            "peak_rss_mb": round(peak_rss / 1048576, 2) if peak_rss else None,
            "final_rss_mb": round(final_rss / 1048576, 2) if final_rss else None,
            "rss_growth_mb": round((final_rss - baseline_rss) / 1048576, 2) if final_rss and baseline_rss else None,
            "rss_measurement_source": "ps_process_tree_sampled_peak" if peak_rss else "NOT_AVAILABLE",
            "exit_code": exit_code, "signal": signal_name,
            "exception_type": None, "fatal_traceback_available": "Fatal Python error" in stderr_text,
            "child_process_status": termination,
            "stdout_log": stdout_path.name, "stderr_log": stderr_path.name}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["fast", "integration", "stress", "all"], default="stress")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/private/diagnostics")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runtime_environment.json").write_text(json.dumps(_environment(), indent=2) + "\n", encoding="utf-8")
    selected = ["fast", "integration", "stress"] if args.suite == "all" else [args.suite]
    reports = []
    for name in selected:
        command = [sys.executable, *SUITES[name]]
        report = _run(name, command, args.output_dir, args.timeout)
        (args.output_dir / f"{name}_stress.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        reports.append(report)
    summary = {"status": "PASSED" if all(row["status"] == "PASSED" for row in reports) else "FAILED",
               "suites": reports, "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()}
    (args.output_dir / "crash_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
