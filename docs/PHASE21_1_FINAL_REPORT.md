# Phase 21.1 — Implementation and Readiness Report

## Result

**Implementation: PARTIAL. Accuracy validation: BLOCKED.** The dataset manifest, coordinate-frame and correspondence models, RANSAC fitting, independent validation metrics, guarded reference projection, and CLI records are implemented. The available pinned PRISM data does not provide reviewed image-to-KiCad correspondences or independent held-out points, so no view is called registered and no pixel/mm error is claimed.

## PRISM 12-board attempt

Command: `python scripts/calibrate_prism.py --all --mode AUTO_REGISTRATION` using PRISM commit `9189c87bbec6995b745b809db99cef6d3269401f`.

For every row, TOP and BOTTOM statuses are both `REGISTRATION_UNVERIFIED`; calibration and validation points are 0; median and P95 errors are `NOT_VERIFIED`; no transform was emitted. The image-only existing board/pad candidate detectors were run as diagnostics, but their detections were not promoted to point correspondences.

| Board | Reference pads | vias | segments | zones | outline points | Image-only TOP candidates (board/pad-via) | BOTTOM candidates (board/pad-via) | TOP / BOTTOM status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Cascade | 46 | 0 | 70 | 0 | 0 | 1 / 14 | 1 / 8 | UNVERIFIED / UNVERIFIED |
| Connector | 13 | 0 | 23 | 1 | 0 | 0 / 2 | 0 / 2 | UNVERIFIED / UNVERIFIED |
| Crossover | 12 | 0 | 23 | 0 | 0 | 0 / 0 | 0 / 1 | UNVERIFIED / UNVERIFIED |
| Divider | 8 | 0 | 10 | 0 | 0 | 1 / 0 | 1 / 0 | UNVERIFIED / UNVERIFIED |
| Elements | 40 | 2 | 112 | 0 | 0 | 0 / 38 | 0 / 31 | UNVERIFIED / UNVERIFIED |
| Filter | 18 | 0 | 40 | 0 | 0 | 1 / 6 | 1 / 4 | UNVERIFIED / UNVERIFIED |
| Floating | 8 | 2 | 7 | 1 | 0 | 0 / 4 | 0 / 1 | UNVERIFIED / UNVERIFIED |
| Groundplane | 8 | 2 | 7 | 2 | 0 | 0 / 1 | 0 / 8 | UNVERIFIED / UNVERIFIED |
| Meander | 6 | 0 | 155 | 0 | 0 | 1 / 1 | 1 / 1 | UNVERIFIED / UNVERIFIED |
| Round | 8 | 0 | 6 | 0 | 0 | 1 / 5 | 1 / 6 | UNVERIFIED / UNVERIFIED |
| Simple | 2 | 2 | 4 | 1 | 0 | 1 / 4 | 0 / 2 | UNVERIFIED / UNVERIFIED |
| Transistors | 14 | 0 | 32 | 0 | 0 | 1 / 9 | 1 / 11 | UNVERIFIED / UNVERIFIED |

The reference counts come from PRISM's KiCad-derived evaluation JSON. The candidate counts are detector outputs, **not** matched accuracy or calibration anchors. The 24-view attempt summary is generated under ignored `data/benchmarks/phase21_1/registration/summary.json`. Bundle manifests are generated under ignored `data/benchmarks/phase21_1/bundles/`. As no view was verified, `build_pcb_reference.py` preserved all 24 views as not projected; it did not create image-space masks.

## Geometry availability

| Geometry | PRISM reference availability | Verification meaning |
|---|---|---|
| Board outline | Empty (`point_count=0`) on all 12 inspected cases | No outline reference to project |
| Component position | Present in component references; no image-space validation | Design-derived only |
| Pad geometry | Present in expected geometry, counts in table | No image-space/visibility score |
| Via geometry | Present on Elements, Floating, Groundplane, Simple only | Layer text is reference data; physical via continuity not independently measured |
| Track segments | Present in expected geometry | No trace segmentation score without transform and visibility labels |
| Copper zones | Present only on selected cases | Not equivalent to photo-visible copper |
| Net identity | Present in PRISM expected netlist/geometry JSON | Design-reference truth for rendered examples, not physical continuity validation |

This phase reuses PRISM reference JSON and does not add a native `.kicad_pcb` geometry parser. The local environment lacks `kicad-cli`. Arbitrary design parsing remains out of scope for this partial implementation.

## Evaluation tracks

- `IMAGE_ONLY`: existing image processing receives pixels only; its candidate diagnostics do not use design references.
- `DESIGN_ASSISTED`: a separately supplied image/design correspondence file can fit a transform. Such fitting is explicitly identified as design-assisted.
- Held-out IDs must be disjoint from calibration IDs; acceptance requires an explicit threshold. Without held-out validation the result cannot be `VERIFIED`.
- Projected geometry is kept separate from observations and labeled as not visibility-grounded. It is rejected unless registration is independently verified.

## Verification run

- Pinned PRISM fetch/inventory: **12/12 ready**.
- Phase 21.1 registration and existing PRISM adapter tests: **14 passed**.
- Fast Python suite: **296 passed**, 48 deselected, 3 existing warnings.
- Integration suite: **34 passed**.
- Stress suite: **11 passed**.
- Existing PRISM baseline pipeline: ran all 12 samples / 24 images; pipeline completed for all; its JSON continues to label geometry and electrical metrics `NOT_VERIFIED` because alignment is absent.
- Registration record integrity validation: **24 views checked, 0 verified, no record-integrity issues**.
- New transforms tests use synthetic correspondences only and demonstrate software behavior, not PCB registration accuracy.
- GitHub Actions: [run 42051982612](https://github.com/alfred0105/CircuitCare-AI/actions/runs/42051982612) completed successfully for result commit `eaf2b79d4351052e9bba70cf21d96bec59da5019` (Python 3.11/3.12, integration, stress, OCR and isolated crash-injection jobs).

## Phase 21.2 readiness

The repository is ready to start annotation protocol and bundle intake work. It is **not ready** for quantitative image-only training/evaluation or physical-board generalization claims. PRISM supports controlled pipeline regression and design-reference structure checks, but currently lacks image-to-design anchor truth, held-out validation, visibility/occlusion labels, and independently reviewed image-space pad/via/trace labels.

For a physical PCB, acquire matching board/revision, original TOP/BOTTOM photos with camera orientation, matching KiCad PCB and netlist/Position CSV, reviewed distributed calibration points, separate held-out points, layer-aware pad/via/track references, visibility masks for occluded copper, and independent continuity-backed pad/pin-to-net truth. Do not fit and score on the same labeled pad/net correspondences.

## Phase 21.1.1 follow-up — 2026-11-06

This report's 0-transform summary records the original Phase 21.1 state. The first Phase 21.1.1 experiment reported 21 transforms emitted and three views failed; that is a historical experiment-stage result, not the final pinned regression. The reproducible Phase 21.1.2 run yields 23 views with automatic candidates, 16 transforms emitted, eight rejected, and zero of 24 independently verified. Emission rate and verified accuracy are separate. See [Phase 21.1.1 automatic registration](PHASE21_1_1_AUTOMATIC_REGISTRATION.md), [its historical 24-view benchmark](PHASE21_1_1_BENCHMARK_RESULTS.md), and [the Phase 21.1.2 service and validation report](PHASE21_1_2_REGISTRATION_BACKEND.md).
