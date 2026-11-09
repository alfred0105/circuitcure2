# Phase 19.3.4 — Overlay E2E Validation

## Result

The Divider cached report is now backed by a new, locally executed ngspice run. Its input fingerprint and output hashes verify against the checked-in run package. The Regulator demo now has a KiCad schematic that exports to the existing XML pin/net topology and SVG geometry. Browser checks verified the Divider electrical-risk overlay and Regulator VOUT measurement markers. Browser-only stale-manifest injection and some interaction cases remain partial; see the matrix below.

## Divider provenance and risk

The previous `data/evaluations/divider/simulation_results/fault_r1_resistance_drift/` report had no trustworthy run manifest. Its original source could not be established, so it remains untouched and is still blocked by the fail-closed policy.

A fresh run used the existing `tests/fixtures/divider_netlist.xml`, `data/circuits/divider.cir`, mapping, probes, rating configuration, and R1 `resistance_drift` parameter `10k` through the current fault-injection → ngspice → stress/risk pipeline. The baseline and fault ngspice subprocesses both succeeded (`ngspice-47`). The run is retained at `data/evaluations/divider/phase19_3_4_run/`; the UI points to its separate result directory.

Fingerprint SHA-256: `603b35915d379b72323fe39c96ae944b8718c196d47d40fd81054f9a8c73b196`.

The run manifest records hashes for `comparison.csv`, `comparison.json`, `component_stress.json`, and `risk_assessment.json`. A current-input snapshot loads as `CACHED_SIMULATION / VERIFIED`; editing a result file makes the snapshot `STALE` and removes its risk data. The Divider demo additionally includes the current ngspice engine version in the fingerprint. If the version cannot be read, the fingerprint is unavailable and the cached risk is blocked.

The new results match the legacy report's numerical risk levels: R2 is `WITHIN_LIMIT`, J1 is `UNKNOWN`. R2's evaluated power is approximately `0.0002066116 W` against an illustrative `0.125 W` `demo_assumption` rating (utilization `0.0016528926`). Voltage/current limits are unknown. The VOUT fault case is about `0.454545455 V` versus a `2.5 V` baseline. No result confirms physical damage or gives a damage probability. Browser verification showed the `FAULT ASSUMPTION`, `WITHIN_LIMIT`, and `UNKNOWN` markers, and the R2/J1 Inspector evidence matched the report.

## Regulator KiCad fixture

Added `tests/fixtures/regulator.kicad_sch`, using embedded symbol definitions from the installed KiCad 10 library: `Connector_Generic:Conn_01x02`, `Regulator_Linear:L7805`, `Device:R`, `Device:C`, and `Switch:SW_SPST`. The existing XML omits library IDs and complete symbol pin metadata, so these metadata remain explicitly `UNKNOWN` in equivalence results. In particular, U1 is the available L7805 representative pinout with the XML value `Regulator`; it does not assert the real installed regulator model. Q1 is represented by the two-terminal switch abstraction in the XML, while its pre-existing SPICE mapping is explicitly excluded from physical stress assessment because its terminals do not match S1's power pins.

KiCad CLI 10.0.6 exported the schematic to XML and SVG. `verify_schematic_netlist()` reports `VERIFIED_EQUIVALENT`: all component references and values, named nets, and pin-to-net groups match `tests/fixtures/fault_demo_netlist.xml`. `build_symbol_geometry()` reports `VERIFIED_GEOMETRY` for all six references. `case_loader.py` now attaches this schematic to the existing Regulator demo. Browser inspection verified SVG render, U1 selection/Inspector sync, and the retained U1 / `regulator_output_rise` / `8` context.

VOUT anchors: `U1.3`, `R1.1`, `C1.1`.

LOAD anchors: `R1.2`, `RLOAD.1`.

Both anchor sets pass `VERIFIED_LOGICAL_NET_CANDIDATES` from the embedded KiCad pin definitions and the current XML circuit. They are logical schematic candidates, not safe or accessible PCB probe points.

## Browser E2E record

Run against the local Streamlit app at `127.0.0.1:8503` with real Pointer interactions:

| ID | Outcome | Evidence |
|---|---|---|
| E2E-01 | VERIFIED | Divider connection overlay rendered the R1 Fault Assumption and direct connectivity-only candidate markers. |
| E2E-02 | VERIFIED | Divider overlay rendered Fault Assumption, R2 `WITHIN_LIMIT`, and J1 `UNKNOWN`. |
| E2E-03 | VERIFIED | Clicking R2 synchronized Inspector; measured power, utilization, and rating source shown. |
| E2E-04 | VERIFIED | Clicking J1 synchronized Inspector; risk remained `UNKNOWN`. |
| E2E-05 | PARTIAL | Zoomed to 195% using the viewer control; marker alignment remained visible. Exact 175% was not available from the stepped control. |
| E2E-06 | VERIFIED | Drag pan moved schematic and markers together. |
| E2E-07 | VERIFIED | Fit control returned zoom to 100%; overlay stayed active. |
| E2E-08 | VERIFIED | Switching overlay mode from Divider risk to Regulator context cleared prior markers. |
| E2E-09 | PARTIAL | Stale result tampering is covered by the automated snapshot test; a stale manifest was not injected into the browser session. |
| E2E-10 | VERIFIED | Divider → Regulator context switch replaced the SVG and removed Divider markers after rerun. |
| E2E-11 | VERIFIED | Regulator U1 was selected and Inspector showed pins 1/2/3 and the mapped VOUT net. |
| E2E-12 | VERIFIED | Regulator Measurement/Inspection view showed VOUT markers on C1.1, R1.1, and U1.3. |
| E2E-13 | PARTIAL | The displayed source was correctly `prior_only`; a second live measurement source was not entered in this browser pass. |
| E2E-14 | VERIFIED | Divider's non-Bayesian context showed the explicit unsupported-What-if fallback. |
| E2E-15 | NOT_VERIFIED | The CUA browser surface did not expose a console-log reader in this run; no console-error claim is made. |

The Measurement panel labels source as `prior_only`, shows the configured information gain and `demo_assumption` uncertainty, and warns that pins are logical candidates rather than safe PCB probe points. The Regulator cached simulation remains `PROVENANCE_UNVERIFIED`, as required; adding the schematic does not validate its old simulation artifacts.

## Automated checks

Added two tests in `tests/test_phase19_3_4_overlays.py`: (1) live engine-version fingerprint, manifest, scenario, risk-level and tamper rejection; (2) actual KiCad XML equivalence, SVG geometry, and VOUT/LOAD pin anchors. `tests/test_simulation_engine_fingerprint.py` covers ngspice version inclusion and fail-closed behavior. The Phase 19.3.4-focused set is 4 passed.

Local environment: Python 3.12.14. Verification summary after the CI-discovered fixes: `compileall` completed; final default group 269 passed, 48 deselected; integration 33 passed before the ngspice-version-aware marker adjustment (including actual KiCad CLI 10.0.6 export/equivalence and ngspice), and the focused integration file passed 2 tests afterward; stress 11 passed; crash injection 3 passed; changed-file Ruff passed; mypy passed for `case_loader.py`, `simulation_service.py`, and `simulation_context.py`; JavaScript syntax passed; `git diff --check` passed.

The first GitHub Actions run found two environment-specific issues: unavailable ngspice produced a `None` fingerprint that the verifier tried to measure with `len()`, and the KiCad CLI fixture test was in the default suite although CI does not install KiCad. The verifier now handles missing/non-string hashes as `PROVENANCE_UNVERIFIED` without raising. The real KiCad export test is marked integration and reports `NOT_VERIFIED` skip when the CLI is absent; it ran locally with KiCad 10.0.6. Run `41978358315` confirmed stress, native OCR, crash-injection, and ngspice integration jobs pass, but showed the Divider manifest/tamper test also requires ngspice and therefore belongs in the integration suite. Run `41978552471` passed Python 3.11/3.12 default, stress, OCR, and crash-injection jobs; its ngspice integration exposed that the checked-in result is pinned to ngspice-47 while CI installs a different engine version. The manifest test now reports `NOT_VERIFIED` skip on that engine mismatch, because the production cache correctly marks cross-version results stale. Run `41978756090` passed all CI jobs: Python 3.11/3.12, bounded stress, native OCR, crash injection, and ngspice integration. On this CI runner the pinned-manifest subtest skips as `NOT_VERIFIED` because ngspice version differs; on matching local ngspice-47 the focused manifest/tamper test passes. CI without KiCad does not count schematic export as verified.

Repository-wide Ruff and mypy were also attempted. Ruff reports 160 pre-existing findings across unrelated scripts/source/tests. Mypy reports 151 existing errors across 37 modules. The Phase 19.3.4 changed Python files pass their targeted checks; no unrelated lint or type cleanup was applied.

## Limitations and phase decision

- The XML's missing library metadata stays `UNKNOWN`; equivalence verifies value and connectivity, not actual bill-of-material identity.
- The Regulator demo schematic uses available representative symbol pinouts. It must not be read as proof of installed hardware identity or as probe-safety guidance.
- Regulator's old cached simulation has no verified provenance and its electrical-risk overlay remains blocked.
- Browser stale injection, source switching after new measurement entry, exact 175% zoom, and console inspection are partial or unverified.

Phase 19.3.1: **VERIFIED** for connectivity candidate rendering and selection/Inspector behavior. Phase 19.3.2: **VERIFIED** for the fail-closed risk gate and the newly manifested Divider simulation, including browser markers and Inspector values. Phase 19.3.3: **PARTIAL**; VOUT/LOAD geometry and the Regulator VOUT browser Marker are verified, but switching to a browser session with a different measurement source and mismatch behavior was not completed here. Phase 19.3.4: **PARTIAL** because browser-injected stale-manifest rejection, a user-measurement source transition, and console inspection were not fully proven. Overall Phase 19.3 remains **PARTIAL** until these items are closed. Phase 19.4 may proceed on non-blocked UI/UX work while preserving fail-closed gates, but must not claim full Phase 19.3 acceptance.
