# Phase 18.2 — Architecture Refactor

## Goal

Clarify application, state, service, and Streamlit view responsibilities while keeping Python, Streamlit, ngspice, Ollama, ONNX, existing algorithms, persisted contracts, and report formats.

## Refactored boundaries

- `dashboard/case_loader.py` owns static demo cases and fault options; `dashboard/session_state.py` owns tested invalidation policy with a thin Streamlit state adapter.
- `dashboard/analysis_loader.py` loads the current analysis snapshot and rejects stale or unverified cached evidence. `dashboard/simulation_controller.py` builds requests and normalizes outcomes while delegating execution and validation to `services/simulation_service.py`.
- `dashboard/workspace_adapter.py` builds the existing workspace record contract. `dashboard/views/` contains the workflow status, What-if preview, fault map, report panel, and download renderers.
- `services/report_service.py` builds a report snapshot from explicit immutable context and read-only state input. It does not call SPICE or Ollama and does not access Streamlit state.
- Vision context is represented by `VisionViewContext` and `ConfirmedOverlayContext`; risk overlay requires a verified manifest and an approved same-board/reference context. `CACHED_SIMULATION` is supported; `STALE`, `PROVENANCE_UNVERIFIED`, `FAILED`, `NOT_RUN`, and legacy `CACHED` are blocked.
- Datasheet document presentation, evidence review controls, and approved rating application are separated into `ui_document.py`, `ui_candidate_review.py`, and `ui_application.py`. Existing approval, MPN, evidence, and conditions policies are preserved.

## Structure after the refactor

- `ui.py`: 426 lines; `main()`: 253 lines; `_render_analysis_overview()`: 69 lines. Circuit, waveform, risk, Bayesian, and repair-advisor rendering now dispatches to dedicated view modules.
- `vision_ui.py`: 369 lines; confirmed overlay rendering: 174 lines with context-based inputs and two callbacks.
- `datasheet_ui.py`: 153 lines.

The top-level dashboard remains the entry point and retains compatibility helper imports where existing callers rely on them. The extraction focused on separately testable policies, file/result loading, report construction, process boundaries, and independent UI steps rather than creating modules for trivial helpers.

## Preserved behavior and contracts

- Existing Session State key names and invalidation rules.
- Existing ngspice worker/process behavior, absolute arguments, timeout, output validation, and spawn-safe `close_fds=False`.
- Workspace hashes, simulation manifest checks, cache fingerprint checks, evidence hashes, report JSON/HTML/Markdown/ZIP formats, and download controls.
- What-if remains preview-only and simulated observations do not overwrite recorded measurements.
- Vision same-board confirmation, netlist reference matching, and individual AI Reference confirmation remain required before risk is attached.
- Datasheet proposals remain untrusted until explicit review. Reassessment uses the existing simulation stress and does not rerun SPICE.
- No new framework, production dependency, AI algorithm, or persisted schema was introduced.

## Verification

See [PHASE18_2_REGRESSION_RESULTS.md](PHASE18_2_REGRESSION_RESULTS.md) for the before/after test groups, browser outcomes, static checks, and unverified input-dependent workflows.
