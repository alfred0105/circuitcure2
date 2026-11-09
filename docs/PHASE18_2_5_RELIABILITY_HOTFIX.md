# Phase 18.2.5 — Reliability & Regression Hotfix

## Scope and baseline

- Repository: `alfred0105/CircuitCare-AI`, branch `main`
- Starting HEAD: `3ca5b82a0d86f98269bca7ed0d0b46bbd3cd914a`
- No dependency or report schema changes.
- The existing workspace, manifest, fingerprint, datasheet approval, and subprocess contracts remain in use.

## Findings and fixes

### P1-01 — Datasheet application cache identity

**Before:** `ui.py` keyed the active applied-rating cache with the complete dashboard context, while the Datasheet view received a shortened context token and removed a different key on apply/revoke. A same-page rerun could therefore reuse stale risk reassessment state.

**Change:** `rating_review.applied_rating_session_key()` is the single key rule. The UI computes a short dashboard token once, pairs it with the active stress-file SHA-256, and uses the helper for cache reads. Datasheet apply and revoke use that same helper and inputs before invalidating the cache. The persistent Registry remains scoped by circuit path, scenario ID, and stress signature; approval and MPN checks are unchanged.

**Verification:** Regression tests cover stable identity, context and stress separation, apply → revoke → reapply invalidation through the Datasheet application callback, blocked re-evaluation when trusted stress/risk data are absent, and the exact token/signature forwarded at the UI boundary. Existing Registry and approval tests pass. An actual PDF-backed approval/revoke flow is **NOT_VERIFIED** because no real datasheet PDF and matched MPN were supplied.

### P1-02 — Live and cached result provenance

**Before:** a matching live session was classified `SUCCESS` before checking the result manifest; manifest failure did not always suppress its comparison, risk, and stress objects. The downloads view could still expose `comparison.csv` directly from disk.

**Change:** `load_analysis_snapshot()` now requires `VERIFIED` provenance for both live and cached successful results, including SHA-256 entries for all three required JSON outputs (`comparison.json`, `risk_assessment.json`, `component_stress.json`). Missing manifest/fingerprint data or a missing required digest yields `PROVENANCE_UNVERIFIED`; changed fingerprints or result hashes yield `STALE`. Both outcomes clear comparison, risk, and stress and are reflected in the displayed status. The existing Vision overlay integrity gate and report snapshot receive that state. `render_downloads()` only exposes the raw simulation CSV when the current result is a verified `SUCCESS` or `CACHED_SIMULATION`; otherwise it shows a block notice. Datasheet reassessment receives no usable risk/stress when provenance is invalid. No automatic simulation or inference was added.

**Verification:** Tests cover verified live success, verified cached success, live JSON tampering, missing live manifest, missing required result digest, cached fingerprint mismatch, failed simulation status, CSV download blocking, and Datasheet reassessment blocking. Existing Live ngspice integration passed and the browser showed `SUCCESS · Live Simulation` with waveform data after a new run. The checked-in legacy cached demo has no verifiable manifest and correctly displays `PROVENANCE_UNVERIFIED`; reuse of that legacy cache as a verified result is **NOT_VERIFIED** and remains blocked.

### P2-01 — Circuit view fault selection

**Before:** the extracted view displayed `case["fault_type"]` and `case["value"]`, which can be defaults and stale after user selection changes.

**Change:** `ui.py` passes the currently selected fault type and only the active fault parameter to `render_circuit_view()`. Parameterless faults render `해당 없음`. Graph construction and connectivity analysis are unchanged.

**Verification:** A renderer regression test checks live parameter and parameterless fault display. Browser E2E changed the selected value to `13k` and showed `13k` in the circuit view, then restored the scenario input.

### P2-02 — Report inspection ranking context

**Before:** report construction read `inspection_ranking` from the active diagnosis state without confirming the current circuit/fault context and measurement-set hash. The fault-hypothesis ranking branch was not gated by the same validation used for its hypotheses.

**Change:** Bayesian diagnosis results are stamped at the point they are produced with a dashboard context ID and the canonical measurement-set SHA-256. Report construction accepts the diagnosis and ranking only when both match the current report context. The fault-hypothesis branch continues to require its matching context and measurement hash. A missing or stale ranking is represented as `NOT_EVALUATED`. What-if remains in the separate `PREVIEW_ONLY` fields. Measurement source labels and explicit `0.0 V` values are preserved. Report schema and report generation side effects are unchanged.

**Verification:** Tests cover matched ranking inclusion, context mismatch, measurement hash mismatch, preview/active separation, and preservation of a user-reported `0.0 V`. Report construction does not invoke SPICE or LLM execution.

## Changed files

- `src/circuitcare/ui.py` — shared Datasheet cache identity, live integrity-gated downloads, selected fault values, and report context handoff.
- `src/circuitcare/dashboard/analysis_loader.py` — fail-closed result integrity/status handling for Live and Cached data.
- `src/circuitcare/dashboard/views/downloads.py` — raw CSV gate.
- `src/circuitcare/dashboard/views/circuit.py` — render the selected fault configuration.
- `src/circuitcare/dashboard/views/bayesian.py` — stamp diagnosis context and measurement identity when generated.
- `src/circuitcare/datasheet/rating_review.py` — shared scoped session-cache key helper.
- `src/circuitcare/datasheet/datasheet_ui.py` and `ui_application.py` — pass and invalidate that shared key.
- `src/circuitcare/services/report_service.py` — validate ranking and diagnosis context before including them.
- `tests/test_dashboard_architecture.py` and `tests/test_phase18_workspace_reporting.py` — regression coverage for the four hotfixes.
- `docs/PHASE18_2_5_RELIABILITY_HOTFIX.md` — this record.

No dependencies were added. No report JSON/HTML/Markdown/ZIP schema was changed.

## Verification record

Environment available locally: macOS Apple Silicon, Python 3.12.14. Python 3.11 was not installed locally; the pushed CI matrix passed on both Python 3.11 and 3.12.

| Check | Result |
|---|---|
| `compileall -q src scripts tests` | PASS |
| Focused changed-module Ruff | PASS |
| Full Ruff (`src scripts tests`) | FAIL on existing repository lint debt: 160 findings after this change versus 166 at starting HEAD; changed modules pass focused Ruff. |
| Full mypy (`src/circuitcare`) | FAIL on existing type debt: 151 errors, unchanged from starting HEAD; no new diagnostic after line-normalized baseline comparison. |
| Standard pytest (`not integration and not stress and not crash_injection`) | PASS — 196 tests (after final test additions). |
| Integration pytest | PASS — 14 tests. |
| Stress pytest | PASS — 11 tests. |
| Combined non-crash suite | PASS — 221 tests (after final test additions). |
| Isolated crash-injection marker | PASS — 3 tests; signal cases terminate child processes through the diagnostics harness. |
| Streamlit browser E2E | PASS for app start on isolated port 8503, circuit view parameter `13k`, Live ngspice success, and waveform rendering. The user's existing port 8501 was left untouched. |

GitHub Actions for commit `4b8258e6c70569677600078684f1191d9866f15a` completed successfully across the Python 3.11/3.12, ngspice integration, native OCR integration, bounded stress, and isolated crash-injection jobs: [run 41835573690](https://github.com/alfred0105/CircuitCare-AI/actions/runs/41835573690).

The legacy demo cache, real datasheet application, and real PCB image/position-file overlay remain **NOT_VERIFIED** for the reasons above or lack of supplied source files. Tests verify their provenance gates without substituting synthetic evidence.

## Compatibility and remaining issues

- Existing manifest/fingerprint JSON shape, cached file names, applied-rating Registry, approval records, subprocess `close_fds=False`, report schema, and dependency set are unchanged.
- No new P0/P1 issue was found in the four reviewed paths.
- Full Ruff and mypy remain repository-wide cleanup items. Their existing diagnostics did not increase; targeted changed-file Ruff passes and the mypy baseline comparison reports no new diagnostics.
- A valid, previously cached demo result is not available in the checked-in legacy case, so its real UI reuse remains **NOT_VERIFIED**; current behavior fails closed.

## Completion assessment

The four Phase 18.2.5 regression paths are fixed and regression-covered, and the pushed CI matrix passed. Phase 18.2.5 is complete. The next circuit-diagram-centered UI/UX phase can proceed; the unverified source-dependent datasheet, PCB, and legacy cached-demo paths should remain explicitly gated until their real input artifacts are available.
