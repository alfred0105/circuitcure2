# Phase 18.2.1–18.2.4 — Regression Results

## Environment and baseline

- Branch: `main`
- Requested review baseline: `ac49c5b96caa624caddc5fd0c0855bac224bbe89`
- Runtime: macOS, Python 3.12.14, repository `.venv`
- The user's Streamlit app at port 8501 was left untouched. Browser checks used this isolated app on port 8503.
- No fixture PCB image, KiCad position CSV, or datasheet PDF was present in `examples/` or `tests/fixtures/`.

## Regression tests

| Test group | Before | After | Result |
| --- | ---: | ---: | --- |
| `not integration and not stress and not crash_injection` | 179 passed | 188 passed | Pass; includes new architecture coverage |
| `integration` | 14 passed | 14 passed | Pass |
| `stress` | 11 passed | 11 passed | Pass |
| `not crash_injection` | 204 passed, 3 deselected | 213 passed, 3 deselected | Pass |
| `crash_injection` | 3 passed | 3 passed | Pass |
| Focused dashboard/datasheet/Vision/reporting regressions | — | 81 passed | Pass |

Recurring warnings are the existing pandas `reset_index(inplace=...)` deprecation and duplicate ZIP member in a verifier test.

## Browser regression on port 8503

- Scenario 1: default demo overview rendered; connectivity graph and report panel rendered; JSON report downloaded to the local Downloads folder.
- Scenario 2: divider circuit selected; R1 fault parameter set to `13k`; ngspice reached `SUCCESS · Live Simulation`; the extracted SPICE waveform view rendered 508 samples, `2.5 V` baseline, and `0.3571 V` fault output; JSON report downloaded.
- Scenario 3: after the final view extraction, circuit A → circuit B → circuit A completed. Each change showed `PROVENANCE_UNVERIFIED`, the parameter returned to the circuit default (`10k` on circuit A), and the prior live result was not reused across circuit contexts.
- Scenario 4: Vision upload screen loaded and showed local image-input guidance. `NOT_VERIFIED` after the upload step because no PCB image or KiCad Position CSV fixture was available.
- Scenario 5: Datasheet review screen loaded, including the PDF upload step. `NOT_VERIFIED` after PDF input because no source PDF and matching real MPN were available. No synthetic rating was approved.
- Existing user port 8501 was not accessed or stopped. Browser console errors were absent during the checks.

## Architecture and static checks

- `ui.py`: 942 → 426 lines; main function: 253 lines; overview renderer: 69 lines. Case definitions, state invalidation, snapshot loading, simulation orchestration, workspace records, diagnostics, report assembly, and view sections now have explicit module boundaries.
- `vision_ui.py`: 330 → 369 lines after extraction and context integration; the confirmed overlay now receives a `VisionViewContext`, a `ConfirmedOverlayContext`, and two export callbacks instead of dozens of independent arguments. Risk attachment requires verified provenance for both live and cached results.
- `datasheet_ui.py`: 295 → 153 lines; document/source rendering, candidate approval controls, and rating application/revocation are separate view modules. The existing MPN, evidence, user-approval, and conditions gates remain in place.
- Focused Ruff check for all changed application/view/controller/service modules and the new architecture tests: passed.
- Full-repository mypy: 151 errors across 37 files (139 files checked). Existing findings remain across risk, repair feedback, datasheet evaluation, graph, benchmark, workspace, and UI typing; no repository-wide type cleanup was included.
- `compileall`: passed.

## Remaining verification

- Real PCB registration/overlay and datasheet approval/application need representative image/CSV/PDF inputs for browser E2E. These workflows were not represented as passed.
- Full app static typing remains noisy; the local mypy run is informational and the workflow CI remains the authoritative integration signal.
