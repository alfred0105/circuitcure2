# Phase 18.1.2 — Runtime Results

## Host and process evidence

- Git baseline: `63476d247e24a7bb18127c2a709c1e5f3acad1db`; working branch `main`.
- Python executable: repository `.venv/bin/python`, Python 3.12.14; macOS 27.0.1 arm64. The unqualified `python` command is not installed in the shell.
- Physical RAM: 24 GiB. `memory_pressure` reported 52% system-wide free memory when sampled; no memory-pressure termination was observed.
- Expected intentional crash: supervisor PID 60478, child PID 60479, child exit `-11` / `SIGSEGV`, `EXPECTED_CHILD_SIGNAL_TEST`. This was the only intentional crash injection (one occurrence).
- Unexpected main-process crash: 0 observed. Unexpected test/native child crash: 0 observed.
- Historical user termination: no matching PID or crash report; `CRASH_ROOT_CAUSE_UNCONFIRMED`.

## Runtime checks

- **ngspice:** live integration tests passed for regulator, divider and RC scenarios, plus the workspace manifest integration. All expected baseline/fault simulation statuses succeeded.
- **ONNX and OCR:** both were executed in isolated child Python processes against a synthetic 512×512 blank RGB image; both completed successfully. The detector model is local and ignored by Git. This validates runtime execution only, not PCB accuracy.
- **Streamlit:** an actual server started under `-X faulthandler`; `/` and `/_stcore/health` returned HTTP 200. Existing AppTest switching soak completed 20 circuit/view transitions. Complete browser-based navigation was not exercised.
- **PDF:** existing bounded and malformed-input tests passed, including repeated PDF parse/cache tests. No adversarial complex table fixture or hard parser timeout was run; parsing remains in-process.
- **Ollama:** model listing succeeded for `gemma4:12b` (7.7 GB reported model allocation, MLX runner); `ollama run` returned `OK` with exit 0. A separate constrained JSON-format direct API request returned HTTP 501. `ollama serve` and its runner are separate from Python and are not included in Python child RSS.
- **Mixed workload:** the Ollama CLI inference ran concurrently with all five native integration tests (ONNX, OCR, and three ngspice scenarios); both processes exited 0. Repeated UI AppTest and full suites passed serially. One Streamlit session executing the entire circuit→SPICE→vision→PDF→report chain and all requested concurrency combinations were not run. This is a verified concurrent subset, not the complete `MIXED_WORKLOAD_VERIFIED` threshold.

## Memory and quality

The diagnostic supervisor sampled child process trees using `ps`. Maximum observed sampled RSS was about 488 MiB during integration. A stress-suite sample peaked around 408 MiB; figures include sampled descendants where visible. They do not measure Python allocator retention or native allocations after process exit. Ollama reports its own model allocation separately; process RSS of its runner was about 174 MiB at inspection. These measurements cannot be summed into an exact system peak.

`pytest-cov` reported 7,875 statements and 62% total coverage. Highlights: graph 89%, `spice_runner` 70%, PDF loader 62%, reporting builder 88%, report validator 78%, repair feedback persistence modules mostly 72–95%, vision component/model helpers mostly 69–90%. Some UI modules have low or zero direct line coverage. Ruff found 181 findings on the final broad scan; mypy reported 162 diagnostics; pip-audit found no known dependency vulnerabilities and skipped the local project distribution.

## Release decision

- **CODE_AUDIT_COMPLETED:** broad automated static checks covered all 142 tracked Python files and the key paths received targeted review; a manual line-by-line semantic audit is not claimed.
- **TEST_SUITE_EXECUTED:** yes; full non-crash regression and the isolated crash-injection test ran.
- **NATIVE_RUNTIME_VERIFIED:** yes; ngspice plus synthetic ONNX and OCR inference ran successfully.
- **MIXED_WORKLOAD_VERIFIED:** no; a concurrent Ollama/native subset passed, but not the complete one-session circuit-to-report workflow and all requested combinations.
- **CRASH_ROOT_CAUSE_UNCONFIRMED:** yes; there is no matching historical PID or crash report.
- **RELEASE_STABILITY_VERIFIED:** no. Before making that claim, isolate PDF parsing with a hard worker time/resource budget, validate the application's Ollama request path, exercise end-to-end browser flows and concurrent sessions, and collect PID-correlated evidence for any future unexpected Python termination.
