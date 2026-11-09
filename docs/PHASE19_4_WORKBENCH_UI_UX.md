# Phase 19.4 — Workbench UI/UX and Validation

Date: 2026-11-05  
Baseline: `97b59ce93e922f8f54f2a4cc25c505c66d796ea7`  
Branch: `main`

## 1. Before UI structure

The prior landing page opened on **분석 개요**. The schematic and fault workspace was a separate choice in an 11-entry analysis selector. Circuit and fault setup lived in the sidebar; waveform, risk, Bayesian, and repair/report work were reached through separate analysis pages. The actual baseline browser was inspected, but no before screenshots were saved and the task-to-result click counts were not measured.

## 2. UX issues verified

- The real schematic was not the default landing view.
- The user had to locate the schematic/fault entry among eleven peer options.
- Fault source and currently inspected component needed a persistent distinction.
- Long safety/provenance caveats occupied the old landing header instead of being shown alongside relevant context.
- The schematic uploader and fault configuration competed with the canvas for vertical space.

These are findings from the prior live page and current source, not inferred performance claims.

## 3. Information architecture

The default is now **회로도 Workbench**. The existing analysis choices remain available in one grouped selector: MAIN WORKBENCH, ANALYSIS DETAIL, and ADVANCED TOOLS. Navigation keeps the existing view identifiers and handles one-time migration of the old default. Circuit and fault configuration remain in the sidebar so the prior analysis renderers can continue to be reused.

## 4. Workbench layout

The page places a compact project header and diagnostic context strip above the circuit area. The central workspace retains the KiCad SVG viewer and its existing navigation/search controls. The left in-canvas column provides component selection and sheet controls; the right Inspector summarizes the inspected component, connected nets, mapping, connection candidates, selected overlay evidence, simulation provenance, and direct waveform/report actions. KiCad schematic upload is in the sidebar and capped at 20 MB, matching the existing storage limit.

## 5. Design system

A neutral light slate palette, single blue Streamlit accent, restrained borders, compact status labels, and scoped app-owned header styles were added. Dynamic header values are HTML-escaped. Existing SVG risk/selection colors and meanings were not altered. The global app theme is in `.streamlit/config.toml`; custom CSS targets only app-owned classes and avoids Streamlit internal DOM class selectors.

## 6. UI states and navigation

The four existing schematic modes remain: Basic, Connection Candidates, Electrical Risk, and Measurement/Inspection. Workbench-to-analysis navigation uses a one-shot session target; circuit and fault controls remain in session state. A separate `FAULT SOURCE` and `INSPECTED` line is shown in the Inspector. Inspection selection does not write the fault selector. Analyses continue to use existing provenance and context validation.

## 7. Existing 11 analysis routes

All original destinations remain selectable: Analysis Overview, SPICE Waveforms, Secondary Damage Risk, Bayesian Diagnosis, Local LLM Repair Support, PCB Photo Recognition, Datasheet AI, AI Fault Hypothesis Analysis, Repair History and AI Learning, AI Performance Evaluation, plus the new default Schematic Workbench. The former “회로 및 고장” entry is represented by the default Workbench. Waveform and report shortcuts route to the existing Waveform and Analysis Overview pages.

## 8. Inspector integration

The Inspector presents summary, connection graph evidence, overlay-specific risk/measurement evidence when available, simulation state, and quick links. It explicitly labels Fault Source separately from inspected reference, and preserves wording that graph candidates are not damage evidence and risk is not failure probability. Missing or unverified evidence remains labelled as such.

## 9. Before/after click comparison

No baseline click counts were measured, so a numeric before/after reduction is **NOT VERIFIED**. In the live app, the initial landing view now shows the schematic without a navigation click (0 clicks after launch). With Risk mode active, selecting a schematic symbol is one click; in the observed R2 check, the Inspector retained U1 as Fault Source. Measurement/Inspection mode is one mode change and shows recommendation information in the Inspector. Waveform and report quick actions each take one click from the Inspector. Circuit selector interaction was two pointer actions (open and choose). These are observed interaction counts, not an overall usability study.

## 10. Responsive validation

The browser was inspected at 1920×1080, 1440×900, and 1280×720. The header/context strip wraps cleanly and the three main columns remain visible. At 1280×720 with the sidebar expanded, the Inspector is narrow and text wraps substantially; there is no automatic Inspector collapse or bottom-panel fallback yet. This is a known limitation and the narrow layout is usable but less comfortable. Temporary viewport overrides were reset after validation.

## 11. Browser E2E

Verified in the running local Streamlit app:

- Initial Workbench shows the actual `regulator.svg` schematic and Inspector.
- Regulator and divider schematic selection paths were exercised; the Inspector reflected the inspected part.
- In Risk mode, R2 was inspected while U1 stayed the Fault Source. The observed R2 row was `WITHIN_LIMIT`, priority 1, with a demo-assumption rating source; J1 remained `UNKNOWN`. No confirmed damage was claimed.
- Measurement/Inspection showed a prior-only VOUT recommendation and logical pin candidates.
- A `simulated_demo` measurement and a self-reported `user_measurement` input were applied in the browser; the Bayesian posterior/source label updated. A mixed-source/model-mismatch state blocked overlay markers as expected. The measurement set was reset to prior-only afterward.
- Waveform and report navigation were exercised by AppTest.
- All 11 selector routes and context return were exercised in AppTest; selected pages were not all manually browser-opened.
- Zoom, pan, and Fit interaction coverage includes previous Phase 19.3 browser validation; this pass used Fit and inspected responsive layouts.

Not browser-verified here: XML-only upload flow, every advanced page rendered in a real browser, console logs, marker alignment at each viewport, and full keyboard-only operation. No before/after screenshots were saved as repository artifacts.

## 12. Phase 19.3 residual validation

- STALE manifest: browser injection was not performed to avoid changing operational artifacts. Existing integrity/risk-gate tests cover stale data handling; browser-level stale state is **NOT VERIFIED**.
- Bayesian source transition and posterior update: **VERIFIED** in the browser for demo and self-reported source labels. A mixed-source scenario showed model-mismatch warning and fail-closed marker behavior.
- Recommendation marker update after Bayesian source change: **NOT VERIFIED**; the observed mixed-source condition intentionally blocked markers.
- Regulator LOAD marker: **NOT VERIFIED** in the browser.
- Browser console inspection: **NOT VERIFIED** through the available browser surface.
- Existing source policy labels reported values as self-reported/demo inputs, not independently authenticated measurements.

## 13. Regression checks

Final default regression group: 270 passed, 48 deselected. Integration: 34 passed, 284 deselected. Stress: 11 passed, 307 deselected. Crash injection: 3 passed, 315 deselected. The focused Workbench AppTest, direct vision-view switching, and report/navigation tests are included in the default group. The legacy navigation assertion was updated to check the grouped display labels while confirming its destinations remain present. Compileall passed. Ruff passed on all changed Python modules and the adjusted regression test.

## 14. Performance validation

No benchmark or latency claim was made. Selection and overlay changes were exercised without explicitly starting ngspice; this was a UI behavior check, not an instrumented process-count test. Existing simulation remains behind its explicit button. Local LLM availability is checked only on its related pages. Pan/zoom rerun counts and SVG cache hit rates were not instrumented in this phase. JavaScript syntax check passed with `node --check`. Targeted mypy reports 40 errors in the existing `ui.py` monolith (primarily case-loader/session-state typing); the three new/changed helper modules and schematic renderer emitted no mypy errors. These warnings are recorded as existing-scope typing debt; no broad typing cleanup was attempted.

## 15. Known limitations

- At 1280×720, the Inspector wraps content and does not collapse automatically.
- Existing selector groups are conveyed through formatted option labels, rather than separate navigation panels.
- Workbench quick Report opens the existing Analysis Overview report area; there is no separate one-click generated report document.
- Full browser E2E for stale injection, XML-only fallback, all advanced routes, LOAD markers, and console errors remains outstanding.
- No screenshot artifacts were retained.

## 16. Phase 19 status

- Phase 19.1 and 19.2: consult their existing phase reports for the exact prior acceptance scope; this UI task did not re-run their complete acceptance suites.
- Phase 19.3: partial; earlier viewer/evidence work remains, but listed browser residuals above are not all closed.
- Phase 19.4.1–19.4.3: implemented for the default Workbench, grouped navigation, themed header/context, Inspector evidence, and direct actions; browser checks cover the primary circuit path.
- Phase 19.4.4: partial. Required regression groups, final CI, full advanced-browser coverage, and some Phase 19.3 residuals must be reported from actual final results. No new P0/P1 issue was observed in the covered flows; this is not a full release certification.

### Final local command results

- `python -m compileall -q src scripts tests`: **PASS**
- `python -m pytest -q -m "not integration and not stress and not crash_injection"`: **270 passed, 48 deselected**
- `python -m pytest -q -m integration -rs`: **34 passed, 284 deselected**
- `python -m pytest -q -m stress -rs`: **11 passed, 307 deselected**
- `python -m pytest -q -m crash_injection -rs`: **3 passed, 315 deselected**
- Ruff changed Python files: **PASS**
- `node --check src/circuitcare/schematic/assets/viewer.js`: **PASS**
- Targeted mypy: **40 existing-scope errors in `ui.py`; no errors in added navigation/theme modules or schematic renderer**
- GitHub Actions for `d37ae68` and its tested source tree: **PASS**, all six jobs including Python 3.11/3.12, ngspice integration, OCR inference, stress, and crash injection. [Run 41988485418](https://github.com/alfred0105/CircuitCare-AI/actions/runs/41988485418)
- An earlier CI attempt caught an AppTest assertion tied to version-specific widget exposure; the assertion was narrowed to stable Workbench/Inspector content, and the subsequent cross-version CI run passed.
