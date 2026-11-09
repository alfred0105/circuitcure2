# Phase 18.2.7 — PDF Worker Resource Isolation Hardening

## Scope and starting point

- Repository: `alfred0105/CircuitCare-AI`, branch `main`
- Starting HEAD: `3ff1d3173e029e92fadeae0ad2bb166de0464f90`
- No dependencies, result schemas, or Datasheet approval rules changed.
- The historical Phase 18.1.2 audit remains unchanged. This report reassesses its PDF finding against the later implementation and current tests.

## Original P1 and prior mitigation

The Phase 18.1.2 P1 said that pathological `pdfplumber.extract_tables()` processing had no hard worker timeout or memory limit and could affect Streamlit. Phase 18.1.3 moved text extraction and table extraction to a separate Python child, added a 60-second supervisor timeout, sampled RSS every 50 ms, bounded logs and result JSON, and validated results before caching. The worker uses `close_fds=False` to preserve the macOS spawn-safe launch path.

Current code inspection confirms that production calls from `pdf_loader.py` dispatch to `pdf_worker.py`; `PdfReader.extract_text()` and `pdfplumber.page.extract_tables()` run in that worker. `datasheet_ui.py` catches `DatasheetError` and displays a user-facing error. The historical P1's direct parent-process parser exposure is mitigated. RSS sampling is not an OS-enforced memory ceiling, however, and the prior documentation did not make that distinction consistently.

## Findings and changes

1. RSS lookup could return `None` for the entire worker lifetime and the supervisor would continue until the wall-clock timeout. The supervisor now stops a worker when RSS remains unavailable for 5 seconds, records `RSS_UNAVAILABLE`, and reports sample count, failure count, longest gap, and measurement state. Diagnostics are written to the configured diagnostic directory or emitted through the application logger when no directory is configured or file writing fails.
2. A worker with exit code zero but a missing, oversized, malformed, or invalid result envelope could have been diagnosed as `PASSED` before the parent validated the artifact. Diagnostics are now written as `PASSED` only after the result file and envelope pass parent validation. Failure records include a reason code.
3. Reader threads were joined with a timeout but pipe handles were not explicitly closed or reported. The parent now closes both streams after process completion, joins the readers again, and rejects the result if either reader remains alive. The timeout test confirms process reaping and `TemporaryDirectory` cleanup.
4. Table extraction sliced the page list to the configured page count, silently ignoring later pages. The worker now rejects a PDF that exceeds the table page limit before extracting tables.

No global concurrency lock was added: same-content and different-content requests already run concurrently without context crossover in the two-worker test. There is no global worker-count cap, so aggregate memory use across many concurrent sessions remains a risk.

## Operating-system limits

RSS observation uses macOS `proc_pidinfo`, Linux `/proc/<pid>/statm`, and an optional `psutil` fallback on other systems. Sampling can miss peaks between observations. Sustained inability to sample now fails closed, but a brief or intermittent gap is recorded rather than treated as a hard limit.

No OS hard memory limit is claimed or applied. Python documents `resource` constants as platform-dependent; `RLIMIT_RSS` is not a portable enforcement mechanism, and `RLIMIT_AS` constrains virtual address space rather than providing a tested resident-memory budget. Linux cgroup v2 provides `memory.max` as a hard-limit interface, but creating and moving a per-worker cgroup depends on delegated writable cgroups and the host's controller configuration. Those prerequisites were not verified here, so the application does not write host cgroup settings. macOS receives no unverified `RLIMIT_AS`/`RLIMIT_RSS` limit.

- [Python `resource` documentation](https://docs.python.org/3/library/resource.html)
- [Linux cgroup v2 memory controller documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html)

## Audit status

| Area | Status | Evidence and limits |
|---|---|---|
| `PDF_PROCESS_ISOLATION` | `VERIFIED` | Production text and table paths call the worker; integration test asserts distinct parent/worker PIDs. |
| `PDF_TIMEOUT_ENFORCEMENT` | `VERIFIED` | macOS local and Linux CI tests confirm worker timeout supervision, reaping, temporary directory cleanup, and parent survival. |
| `PDF_MEMORY_MONITORING` | `PARTIAL` | macOS and Linux worker tests observe RSS and trigger the sampled limit; sustained missing readings terminate. Sampling still cannot guarantee detection of a short peak. |
| `PDF_HARD_MEMORY_LIMIT` | `NOT_VERIFIED` | No per-worker OS hard limit is installed. Cgroup delegation and safe enforcement are not verified. |
| `PDF_FAILURE_CLEANUP` | `VERIFIED` | macOS local and Linux CI coverage confirms timeout child reaping, temporary directory removal, reader cleanup, and rejection of invalid/missing/oversized results. |
| `PDF_CONCURRENT_WORKER_SAFETY` | `PARTIAL` | Two simultaneous requests are tested for same and different PDFs. Larger multi-session load and aggregate memory are not bounded or measured. |
| `PDF_PARENT_SURVIVAL` | `VERIFIED` | macOS local and Linux CI timeout/failure tests complete in the test parent; malformed input and worker errors return controlled `DatasheetError`s. Live browser upload E2E was not run. |

## Regression coverage

`tests/test_pdf_worker.py` now covers one-page and 50-page PDFs, real text extraction, real table extraction, same-hash cache reuse, content-hash invalidation, concurrent same/different documents, missing RSS, RSS limit termination, timeout, SIGSEGV classification, process reaping, temporary-file cleanup, reader-thread cleanup, missing/oversized/invalid JSON results, no cache insertion on failure, recovery with a later valid result, and table page-limit enforcement.

Other PDF tests in `tests/test_datasheet_ai.py` cover malformed, encrypted, oversized, and over-page-limit inputs. No actual OOM is induced; the RSS cap test uses a low threshold to trigger controlled supervisor termination.

Normal extracted text and table values are checked against small synthetic PDFs. A benchmark against an older version was not run; the parser and extraction limits remain unchanged, except that table requests over the existing page limit now return an explicit error instead of silently skipping extra pages. Performance measurements were not collected.

## Validation record

- `compileall -q src scripts tests`: `PASS`
- `tests/test_pdf_worker.py -rs`: `18 passed`
- Standard suite (`not integration and not stress and not crash_injection`): `203 passed, 35 deselected`
- Integration suite: `21 passed, 217 deselected`
- Stress suite: `11 passed, 227 deselected`
- Ruff for changed implementation modules and `tests/test_pdf_worker.py`: `PASS`
- Full requested Ruff scope (`src/circuitcare/datasheet tests/test_pdf_worker.py`): `FAIL` with 29 findings in unchanged Datasheet modules; no findings in changed PDF modules or test file.
- Changed-source mypy: `FAIL` with 7 diagnostics, same 7 diagnostics in the unchanged baseline versions of these three files; no new type diagnostic. An unrelated imported module also reports an existing nullable-value diagnostic in the baseline run.
- Local tests used Python 3.12 on macOS Apple Silicon. Test output includes existing duplicate-ZIP fixture and Streamlit/Pandas deprecation warnings.
- GitHub Actions: `PASS` — [run 41854035541](https://github.com/alfred0105/CircuitCare-AI/actions/runs/41854035541), all six jobs passed: Python 3.11, Python 3.12, Linux ngspice Integration, native OCR Integration, bounded Stress, isolated Crash Injection.
- Linux CI and Python 3.11/3.12: `PASS` for CI-covered flows; local manual validation used macOS/Python 3.12 only.
- Local platform: macOS Apple Silicon, Python 3.12; all 18 targeted PDF tests passed.
- Browser: app opened on isolated port 8503 before code changes; no live PDF upload/timeout browser flow was performed.

## P1 disposition

**PDF-related P1: PARTIAL, not CLOSED.** The worker isolation and parent survival path address the original direct-parent parsing hazard, and wall-clock termination is enforced. The remaining limits are lack of OS hard memory enforcement, sampled RSS gaps/peaks, no global concurrency cap, and no large multi-session memory test. The record does not claim `HARD_MEMORY_LIMIT_VERIFIED` or release-level PDF safety.

Phase 18.2.7 scoped changes and CI verification are complete. The project-wide release stability status remains subject to the historical non-PDF crash attribution and other unresolved audit items.
