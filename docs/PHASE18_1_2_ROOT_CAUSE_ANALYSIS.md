# Phase 18.1.2 — Root Cause Analysis

## Finding

**`CRASH_ROOT_CAUSE_UNCONFIRMED`**. No original user crash report, matching PID/command record, macOS `.ips` report, or reproducible application crash was available in the repository or private diagnostic folder. The historical user-visible Python termination therefore cannot be attributed to a particular process or library.

The one observed SIGSEGV was deliberately injected into a separate child process. On 2026-11-04 UTC, supervisor PID 60478 launched child PID 60479 with `python -c 'import os,signal; os.kill(os.getpid(), signal.SIGSEGV)'`. It exited `-11` / `SIGSEGV`, was classified `EXPECTED_CHILD_SIGNAL_TEST`, and the supervisor completed normally. Core dumps were disabled for that child. This proves the harness can distinguish that fixture; it does **not** explain the historical termination.

The current full test run recorded no unexpected main-process or child-process crashes. Native inference and ngspice ran successfully in the integration suite. A real local Ollama CLI inference also succeeded. A separate direct API request with constrained JSON format returned HTTP 501; the application's Ollama client path was not verified by that API probe.

## Changes made

- Marked the intentional signal test `crash_injection`, excluded it from the default fast CI and diagnostic fast suite, and asserted parent/child PID separation and the explicit expected-signal classification.
- Moved Streamlit, NumPy, OpenCV, ONNX Runtime, and RapidOCR availability/version probes to short-lived child Python processes. The supervisor now records executable, working directory, parent/child PIDs, start/end timestamps, exit status and signal.
- Tightened diagnostic process-group termination, disabled POSIX core dumps, capped each supervised stdout/stderr file at 20 MiB, and distinguished timeout from elapsed duration.
- Avoided importing RapidOCR in the pytest parent during native model discovery.
- Cached at most four CSV download payloads using their content digest as the invalidation key; this preserves the download while bounding retained payloads.

## Limits and next evidence needed

The environment had no user crash timestamp, command line, affected PID, or matching macOS Crash Report to correlate. RSS sampling is process-tree sampling via `ps`, not a native allocation trace; initial samples may be affected by process startup. No historical SIGKILL/OOM event was observed. Keep the exact launch command and timestamp with any future failure, and collect the matching macOS DiagnosticReport before assigning a cause.

PDF text and table extraction remain in-process. Page and output limits do not impose a hard CPU/memory ceiling on `pdfplumber.extract_tables()`. This is an unresolved resource-isolation risk, not a reproduced crash.
