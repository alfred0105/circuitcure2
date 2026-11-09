# Phase 21.0.1 — Roadmap Alignment and Verification Record

## Status

Roadmap and product-scope alignment are complete. Responsive browser acceptance is **partial**: the live local application was opened in the in-app browser, but this session exposed a narrow viewport and did not provide a way to set the requested 1920×1080, 1440×900, and 1280×720 viewport sizes or save fresh screenshots into the repository. Phase 21.0.1 is therefore not marked fully complete.

## Baseline

- Repository: `alfred0105/CircuitCare-AI`
- Baseline commit supplied for this phase: `a69da7aeaa21af0edd31348c045b18ae0439d7cb`
- Working branch: `main`
- Phase 21.0 audit remains the source of the prior technical inventory and evidence limits.

## Roadmap and scope changes

The roadmap keeps the detailed technical/evaluation plan for 21.1–21.7 and extends the former integrated demo/release work into 21.10:

- **21.0:** technical audit, scope and evidence inventory.
- **21.0.1:** roadmap alignment, UI verification and final documentation.
- **21.1–21.5:** PCB data/registration, annotation, image-only reconstruction, design-assisted ablation and electrical connectivity reconstruction.
- **21.6–21.7:** physics-informed fault diagnosis and active measurement evaluation.
- **21.8:** natural-language symptom interpretation, fault candidates, secondary-impact candidates, bounded compound-fault hypotheses and measurement recommendations.
- **21.9:** user-centered repair workbench from diagnosis through PCB viewing, measurement updates and reporting.
- **21.10:** independent competition evaluation, integrated demo and release.

PCB reconstruction and diagnosis research can proceed on separate data tracks. Phase 21.8 must rely on verified design connectivity or expose uncertainty and request user input. The three research contributions remain independently evaluated. Product scope, evidence gates and evaluation distinctions are aligned in the project-scope and evaluation-strategy documents.

## README and documentation

README now distinguishes the supported-fault local prototype from the target product, which begins with multiview PCB photos and natural-language symptoms. It records that Phase 21.0.1 has aligned the roadmap while responsive browser acceptance remains open. Future phases are presented as planned, not implemented.

Updated documents:

- `README.md`
- `docs/PHASE21_ROADMAP.md`
- `docs/PHASE21_0_PROJECT_SCOPE.md`
- `docs/PHASE21_EVALUATION_STRATEGY.md`

The prior Phase 21.0 technical audit was not changed. Historical performance numbers and evidence claims were not altered.

## Browser verification evidence

The Streamlit application was served locally at `http://127.0.0.1:8503/` and opened in the in-app browser. The browser surface reported an approximately 525-pixel-wide viewport, so this is a narrow-screen spot-check and cannot stand in for the three required desktop/presentation viewports.

Observed in the actual browser:

- The CircuitCare AI workbench title, circuit/fault status, provenance warning, and main navigation rendered.
- Component search and selection for R1 updated the selected-component inspector.
- Viewer Fit and zoom controls changed or reset the displayed zoom state.
- Dragging the schematic panned the drawing.

Not verified to acceptance standard:

- 1920×1080, 1440×900 or 1280×720 layout behavior.
- Full component click-to-inspector synchronization from the schematic. A viewer click attempt did not produce conclusive inspector evidence.
- Risk and measurement overlay status/legend behavior on the restarted app.
- JavaScript console error state.
- Fresh screenshots at the required sizes or interaction states. No new screenshots are included; older Phase 19/20 captures were not reused.

No UI code change was made in this phase because the required desktop viewport evidence was unavailable and no new defect was confirmed strongly enough to justify a code patch. The previous Phase 21.0 UI work remains in place.

## Tests and CI

Automated test suites and GitHub Actions have not yet been run for this change set. Browser spot-checks are manual observations only and are not an automated browser E2E result. These items remain open completion gates.

## Remaining work

1. Run the fast Python suite plus the repository's integration and stress suites in the intended local environment.
2. Repeat browser checks at 1920×1080, 1440×900 and 1280×720, including component click synchronization, both overlays, and console errors.
3. Save fresh evidence to `docs/assets/phase21_0_1/`.
4. Push the verified change and inspect the resulting GitHub Actions run.

## Phase 21.1 readiness

The roadmap and research gates are aligned, so Phase 21.1 planning can proceed. Data execution should begin only after identifying a physical board/revision, obtaining front/back photos and matching KiCad PCB/Position CSV/netlist sources, and establishing independent calibration and validation points. Phase 21.0.1 itself remains partially open until the browser and CI gates above are recorded.
