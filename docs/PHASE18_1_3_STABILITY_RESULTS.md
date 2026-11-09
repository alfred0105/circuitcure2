# Phase 18.1.3 Stability Results

## Environment

- Baseline commit: `bb26ff26194a57286a492f4b73639d2c52eab172`
- Runtime: `.venv/bin/python`, CPython 3.12.14, macOS 27.0.1, arm64
- Dependency check: `pip check` passed.
- Compile and whitespace check: `compileall` and `git diff --check` passed at the time recorded; rerun after final edits.

## Test matrix

| Check | Result |
|---|---:|
| Fast suite (before CSV race-test correction) | 177 passed, 1 failed; the test injected a mutation through `Path.read_bytes`, while production now reads through `Path.open`. Test was corrected to mutate at file-open boundary. |
| Integration marker | 14 passed in final full run |
| Stress marker | 11 passed in earlier isolated run; all included in final full run |
| PDF worker + native integration after process-monitor changes | 12 passed; final PDF worker suite 8 passed |
| Full suite, excluding isolated crash injection | 204 passed, 3 deselected |
| Isolated crash injection | 3 passed, 204 deselected |
| Crash diagnostics harness `--suite all` | fast, integration and stress subprocess stages passed; sampled peaks 249.69 / 437.55 / 404.00 MiB; final RSS 249.69 / 110.44 / 393.73 MiB. These are harness-child measurements, not leak conclusions. |
| Coverage | 62% (8,122 statements, 3,048 missed) |
| Ruff | 178 repository-wide findings; mostly pre-existing style/import/shebang findings. No broad autofix applied. |
| Mypy | 166 repository-wide errors across 33 files; no broad type rewrite applied. |
| Python 3.11 / 3.12 CI | PASS on both versions; all 6 jobs passed (Python 3.11, Python 3.12, ngspice integration, native OCR inference, bounded stress, isolated crash injection). [GitHub Actions run #41794469574](https://github.com/alfred0105/CircuitCare-AI/actions/runs/41794469574). |

The full suite ran on Python 3.12 only. Earlier local measurements from this run included 100 AppTest view switches, 20 ngspice repeats, 20 PDF cache loads, and 30 synthetic board detections. RSS observations are samples; they do not prove a memory leak or its absence.

## Phase status

| Requirement | Status |
|---|---|
| PDF_ISOLATED | PASS |
| HARNESS_VERIFIED | PASS |
| OLLAMA_CHAT_VERIFIED | PASS (real local client fallback path) |
| SPICE_VERIFIED | PASS for direct CLI/native repeats; browser child launch needs clean rerun |
| NATIVE_VISION_VERIFIED | PASS for isolated ONNX/OCR inference and browser upload/rectification/inference panel; no accuracy claim and no complete Reference/overlay flow |
| BROWSER_E2E_VERIFIED | PARTIAL (known fault, measured hypothesis candidate, report download, context switch pass; Vision Reference/overlay and datasheet approval/reassessment incomplete) |
| CONCURRENT_SESSION_VERIFIED | PARTIAL (separate circuit selections in 2 concurrent browser contexts; heavy tasks not concurrent) |
| MEMORY_STABILITY_VERIFIED | PARTIAL (bounded repeats and samples; mixed workload passed without ONNX; not a long-duration soak) |
| CRASH_ROOT_CAUSE_IDENTIFIED | PASS for captured at-fork SIGSEGV and affected call sites; relationship to any earlier report remains unconfirmed |
| RELEASE_STABILITY_VERIFIED | NOT VERIFIED |

No full release-stability claim is made while browser-triggered simulation and mixed workload remain unresolved or unverified. Diagnostic evidence identifies a real macOS at-fork crash path in this validation, but not every earlier termination report.
