# Phase 18.1.1 — Crash Stability and Reliability

## Scope and environment

This work followed `main` at `482105e01cca46dd9f6b7aedab91f9f51bc27c4b`. The repository inventory contains 106 application Python files, 13 Python scripts, and 16 Python test files (135 total before this change). The current checkout is macOS 15/Apple Silicon arm64, Python 3.12.14, Streamlit 1.66.1, NumPy 2.5.4, OpenCV 4.14.0, ONNX Runtime 1.31.1, and ngspice 47. No user crash report was supplied; therefore no single runtime crash cause is claimed as confirmed. No evaluation ground truth or repair case database was edited.

## Changes

- Graph path search now limits expanded states, peak queued states, total path results, depth and optional elapsed time. `PathSearchResult` reports `COMPLETE`, `PARTIAL_RESOURCE_LIMIT`, or `PARTIAL_DEPTH_LIMIT` with counts and the limit reason. The existing `find_paths()` list API remains compatible and is bounded. CLI, UI, fault map and risk output surface the search state; UI labels partial output as incomplete.
- The Streamlit main view now selects one of its 11 areas at a time. Expensive area renderers and Ollama model discovery run only when that area is active. PDF parsing has a four-document LRU keyed by content SHA, parser version and settings; it also bounds pages, extracted characters, table count, and cell count. Replacing a PDF clears the previous large document from session state.
- PCB image decode results and board contour candidates use small content/settings-keyed caches. Changing image removes prior image-scoped large temporary results. Perspective transforms reject invalid output sizes and ill-conditioned matrices. ONNX sessions remain bounded, now set CPU thread counts explicitly, and model digest cache invalidation includes file change metadata. OCR handles NumPy `scores` without ambiguous truth testing and validates text sequence shape.
- ngspice output is streamed row-by-row into exact full-resolution numeric columns and rejected above 256 MiB, 1,000,000 samples, or 128 vectors; no numerical downsampling is used. Child stdout/stderr go to files instead of unbounded in-memory capture. A POSIX `-11` child exit gets at most one retry; each attempt's logs are kept, and a repeated crash is reported as `SIGSEGV`. Timeout remains a failure and never yields a success result.
- Evidence export keeps the source evidence digest and stores the bundle file digest separately. A changed file is marked `ARTIFACT_HASH_MISMATCH`; missing source digest is `SOURCE_HASH_UNAVAILABLE`. Report validation now rejects mixed evidence scenario IDs. Datasheet review/application JSON writes use flush, fsync and same-directory atomic replacement.
- Added an isolated diagnostics runner that records environment, wall time, sampled process-tree RSS where `ps` is available, exit code and signal. Timeout cleanup kills the process group on POSIX. Generated run cleanup is a dry-run by default, only targets old `data/ui_runs` folders, and skips folders with `.circuitcare-active` markers.

## Verification results

- Graph stress inputs of 100, 300 and 1,000 components ran with explicit queue/state/path budgets; they returned resource-limited partial results within budget. The small graph remains complete and a depth cutoff is explicitly partial.
- PDF cache test: one parse followed by reuse; a 50-page synthetic PDF reused across 20 calls; malformed PDF returned a handled error. Image cache tests cover repeated content, 20 sequential images, and 2,000 × 2,000 image decoding.
- ONNX Runtime inference: **PASSED** in an isolated child using the locally present `models/pcb_component_detector.onnx` and a synthetic 512 × 512 blank image. OCR model inference: **PASSED** in an isolated child using locally installed RapidOCR ONNX models on the same synthetic image. OCR NumPy-array score handling also passed. This verifies local runtime execution only, not detection or OCR accuracy. The detector model is Git-ignored and is not guaranteed to exist in CI.
- ngspice 47 ran three isolated end-to-end scenarios: regulator output rise, divider resistance drift, and RC resistance drift. All three produced successful baseline/fault results. Mocked subprocess tests cover one-retry recovery and repeated SIGSEGV failure while preserving both attempt logs. A deliberate convergence-failure fixture and actual ngspice timeout were **NOT RUN**.
- Streamlit AppTest verified all 11 areas remain selectable and switching to the evaluation area completes without an app exception. This checks lazy view dispatch, not a long-running browser soak.
- Atomic-write tests verify invalid JSON data leaves an existing registry untouched. Bundle tests verify a mismatched original evidence hash is not replaced by the exported file hash.
- Isolated child tests confirm a synthetic child SIGSEGV is reported as `CHILD_PROCESS_SIGSEGV` without terminating pytest, and a timed-out child is terminated. RSS measurements are sampled OS process-tree values, not `tracemalloc` and not continuous native allocator accounting.
- Local macOS tier runs: fast **170 passed / 16 deselected, 2.55 s**; integration **6 passed / 180 deselected, 1.36 s**; stress **10 passed / 176 deselected, 2.16 s**. The combined local suite: **186 passed, 11 warnings, 5.57 s**. Warnings are the intentional duplicate-ZIP-path rejection fixture and Streamlit/Pandas deprecation notices.
- A standalone 20-circuit Streamlit AppTest switch soak sampled RSS at 171.08 MiB after initial load, 178.03 MiB at 5 switches, 179.55 MiB at 10, and 180.73 MiB at 20 (1.72 s). In the combined stress suite, after earlier image/cache tests, the same switch soak sampled 399.66, 406.12, 406.62, and 409.25 MiB at those points. It completed without app errors. These AppTest runs are not a real browser/server long-duration soak.
- Latest isolated suite sampled peak RSS: fast 250.20 MiB, integration 423.12 MiB, stress 411.28 MiB. Integration's peak included actual ONNX/OCR child inference; its final pytest RSS was 118.44 MiB after child exit. Stress included 20 synthetic 2,000² images and repeated UI reruns. These bounded tests completed without process failure; allocator retention and high-water RSS remain possible.
- Latest diagnostics save per-suite timing/RSS, child signal classification and `streamlit_switch_soak.json`. `ps` samples process-tree RSS and can miss short peaks. Baseline-to-peak differences include Python and optional dependency startup and should not be called a leak measurement.
- GitHub Actions run [41760254583](https://github.com/alfred0105/CircuitCare-AI/actions/runs/41760254583) passed for code commit `769c79a`: Python 3.11 fast CI, Python 3.12 fast CI, and Linux ngspice integration all succeeded. GitHub emitted non-blocking Node.js action deprecation and upcoming Ubuntu image migration notices.

## Limits and remaining risks

- The originally reported Python process termination has not been reproduced because no crash timestamp, traceback, `.crash` report, or diagnostic log was supplied. This change adds evidence collection for a future occurrence; it does not prove that every possible native crash source has been eliminated.
- macOS native crash-report inspection, long-duration browser/server soak, large near-limit PDF, and SPICE convergence failure remain unverified. A 20-switch AppTest was run; it is not a substitute for a long live-browser session.
- Streamlit and some Python libraries may retain allocator memory after cached objects are evicted. RSS samples can miss short peaks; tracemalloc does not include native allocations.
- Physical repair accuracy remains `NOT_EVALUATED`. User-entered readings are self-reported and not independently verified.

## Release and data integrity

`v0.1.0-mvp` is untouched. Evaluation ground truth, original benchmark results, and repair case DB are unchanged. The generated diagnostics directory is ignored by Git under `data/private/diagnostics/`.
