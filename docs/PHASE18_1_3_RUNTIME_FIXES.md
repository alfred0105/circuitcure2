# Phase 18.1.3 Runtime Fixes

## Changes

- PDF text and table extraction now runs in a child worker. The parent validates the PDF signature/size first, supervises wall time and sampled RSS, bounds captured logs and JSON output, and checks the result envelope and source SHA-256 before caching it. Cache identity remains content based; renamed inputs keep their own filename context.
- macOS RSS sampling uses `proc_pidinfo` through `libproc` rather than spawning `ps` repeatedly. The PDF worker has no subprocess children, so timeout cleanup terminates that worker directly. macOS sampling can miss a very short RSS spike; it is not a hard memory limit.
- Crash diagnostics no longer apply `RLIMIT_FSIZE` to the whole test process. The harness drains and caps captured stdout/stderr independently and records `LOG_TRUNCATED`; result artifacts remain writable under their own test budgets.
- Ollama JSON mode falls back only when the runner explicitly responds `structured output is unavailable`. It removes `format`, keeps the local model and `think=false`, validates JSON, and records `fallback_used` / `fallback_reason`. Other HTTP errors and invalid JSON still fail.
- CSV snapshots are size bounded and keyed by content digest plus analysis context. File metadata and content are checked across the read; the UI refreshes one changed snapshot and then reports failure rather than serving stale bytes.
- Report provenance reads Git metadata directly rather than invoking `git rev-parse` during Streamlit reruns. The UI CLI, ngspice, and PDF worker launches use absolute paths, omit `cwd`, and set `close_fds=False` so CPython 3.12 selects `posix_spawn` instead of forking a multithreaded macOS process. Python-created file descriptors are non-inheritable by default; this avoids the observed at-fork crash while retaining normal descriptor isolation.
- Direct navigation to the Vision view initializes its What-if state, preventing the reproduced `UnboundLocalError`.

## macOS subprocess crash evidence

A macOS DiagnosticReport generated during this validation recorded `EXC_BAD_ACCESS` / `SIGSEGV` on the child side of `fork` and explicitly reported a multithreaded process fork. The first Python fatal traceback pointed to `report_builder._git_sha()` invoking `git rev-parse` from a Streamlit rerun. A subsequent report showed the same at-fork failure in the Python CLI child while it started a nested process. CPython's `Popen` path inspection showed `close_fds=True` prevented `posix_spawn` on this platform; the affected call sites now use the spawn-safe options. After the change, a fresh-browser live simulation completed with comparison and risk outputs, the UI process stayed alive, and JSON report download succeeded. This establishes the cause of the crashes captured in this run; it does not establish that they match every previously reported Python termination.

## Regression evidence

- `tests/test_pdf_worker.py`: small and 50-page PDF, malformed PDF, timeout, RSS termination, SIGSEGV classification, cache/hash invalidation, concurrent distinct and same-content requests.
- `tests/test_local_llm_compat.py`: fallback classification and malformed structured output.
- `tests/test_diagnostics_harness.py`: 25 MiB captured output, 30 MiB sparse result, signal classification.
- `tests/test_ui_hotfix_helpers.py`: CSV cache context and changed-file race.
- `tests/test_phase18_workspace_reporting.py`: direct Git metadata read without subprocess.
- Targeted PDF/native run after macOS process monitoring changes: 12 passed; direct PDF worker suite after adding SIGSEGV record check: 8 passed.
- Full non-crash suite: 203 passed, 3 deselected. Isolated crash suite: 3 passed.
- Streamlit AppTest 100-view-switch soak passed before the last subprocess changes; full post-change regression passed afterward.

See `PHASE18_1_3_E2E_VALIDATION.md` and `PHASE18_1_3_STABILITY_RESULTS.md` for environment-specific results and remaining gaps.
