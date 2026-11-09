# Phase 18.1.2 — Repository Audit

## Scope and method

Audited the current repository at baseline `63476d247e24a7bb18127c2a709c1e5f3acad1db`, then reviewed the changed diagnostic, native integration, and download paths. Static tools traversed the 142 tracked Python files; configuration and workflow files were inspected. The test/coverage run exercised the repository but does not establish line-by-line correctness of every branch.

## Findings

| Priority | Finding | Trigger and effect | State |
|---|---|---|---|
| P1 | PDF table extraction has no hard worker timeout or memory limit | A pathological or unusually complex PDF can spend substantial CPU/memory inside `extract_tables()` before output caps apply; the Streamlit process can be affected | Open; not reproduced |
| P1 | Original Python termination has no matching crash evidence | Without the original PID, command and `.ips` report, tests cannot identify whether Codex, pytest, Streamlit, or a native child stopped | Unconfirmed |
| P2 | Direct Ollama JSON-format API request returned HTTP 501 | `gemma4:12b` CLI generation succeeded, including during the native integration suite; direct app-client behavior for this API route remains unverified | Partial; investigate constrained JSON request separately |
| P2 | Ruff reported 184 existing findings across tracked Python code | Includes import order, unused imports, executable/shebang mismatch and style issues; broad autofix was intentionally not applied | Open; baseline-wide |
| P2 | Mypy reported type errors in existing code | Errors include optional dictionary keys, nullable arithmetic, inferred collection types and benchmark result dictionaries | Open; 35+ reported diagnostics (see private log) |
| P2 | Coverage is 62% overall | `spice_runner` 70%, PDF loader 62%, graph 89%, reporting 78–90%; several Streamlit UI modules remain low or unexecuted | Open; coverage is evidence, not a quality score |

## Reviewed risk areas

- **Runtime/processes:** the diagnostic harness ran test suites in separate process groups. Intentional SIGSEGV is now opt-in and the parent stays alive. No unexpected signal was observed in the executed suites.
- **Memory:** tests use bounded fixtures and repeated UI AppTest switching. `ps` process-tree peak during integration was about 488 MiB. Ollama is a separate daemon and was measured separately (runner RSS about 174 MiB; model allocation is reported by Ollama as 7.7 GB, GPU/MLX). These figures are not directly additive physical-memory measurements.
- **Persistence/provenance:** existing atomic JSON, SQLite, hash and manifest paths were covered by regression tests, including prior integrity checks. Simultaneous multi-session write contention was not exercised in this run.
- **Inputs/native code:** malformed PDF inputs and isolated ONNX/OCR inference and live ngspice representative scenarios passed existing bounded tests. No actual PCB detection accuracy claim is made because the inference input was synthetic. CI now has a required OCR model/runtime job; the local detector ONNX artifact remains Git-ignored, so CI does not claim ONNX detector inference unless that artifact is provisioned.
- **UI/state:** 20-switch Streamlit AppTest soak passed. A real Streamlit server started and returned HTTP 200 for app and health endpoints. Browser-based interaction across all controls was not completed.
- **Security/dependencies:** `pip-audit` found no known vulnerable installable dependencies, but skipped the local `circuitcare-ai` package because it is not a PyPI distribution. Upload/path and archive checks have regression coverage; no external data was modified.

## Audit status

Broad automated static and runtime audit executed. A complete manual review of every branch, hard PDF resource isolation, Ollama generation, simultaneous sessions, and full browser workflows remain open. Do not interpret CI or this audit as release-level proof against the unconfirmed crash.
