# Phase 21.1.1 — PRISM Automatic Registration Benchmark

**Run date:** 2026-11-06

**CircuitCare baseline:** `1877b58880f0f725157bb0fa16c782b35fe077d7`

**PRISM commit:** `9189c87bbec6995b745b809db99cef6d3269401f`
**Command:** `python scripts/calibrate_prism.py --all --mode AUTO_REGISTRATION --artifact-dir data/benchmarks/phase21_1_1`

## Result summary

| Measure | Result |
|---|---:|
| TOP/BOTTOM image views processed | 24 |
| Views with candidate correspondences | 23 / 24 |
| Candidate correspondence pairs | 230 |
| Geometric inliers retained by the fitted model | 191 |
| Transforms emitted | 16 / 24 |
| `AUTO_ESTIMATED_UNVERIFIED` | 16 |
| `REGISTRATION_FAILED` | 8 |
| Independently verified (`VERIFIED`) | 0 / 24 |
| Views needing human review | 24 |
| Ground-truth projections emitted | 0 / 24 |

`Candidate correspondences`, geometric inliers, emitted transforms and independently verified registrations are distinct counts. Residual values below are **in-sample errors on automatically matched points used to estimate/refine the transform**. They are not held-out error, accuracy, or independent validation. A few exact-looking tiny residuals occur because three or four points can determine an affine transform nearly exactly; the transform may still be ambiguous. Low-support, low-coverage and ambiguous hypotheses are now rejected as `REGISTRATION_FAILED` and retain diagnostic candidate/inlier counts.
## Per-view results

The “Error” column is training-fit residual only. Per-view `summary.json` also records mean, median, RMSE, P95 and maximum in-sample residuals where matches were available. `NOT_AVAILABLE` means no residual was available. `LOW_MATCH_SUPPORT`, `LOW_SPATIAL_COVERAGE`, and `AMBIGUOUS_TRANSFORM_HYPOTHESES` are flags; any of these blocking flags causes the candidate transform to be rejected.

| PCB | Side | Auto Matches | Inliers | Error | Registration Status |
|---|---|---:|---:|---|---|
| Simple | TOP | 0 | 0 | NOT_AVAILABLE | REGISTRATION_FAILED |
| Simple | BOTTOM | 3 | 0 | NOT_AVAILABLE | REGISTRATION_FAILED |
| Divider | TOP | 3 | 3 | rejected candidate; fit median 0.00px / P95 0.00px; LOW_MATCH_SUPPORT, AMBIGUOUS_TRANSFORM_HYPOTHESES | REGISTRATION_FAILED |
| Divider | BOTTOM | 5 | 3 | rejected candidate; fit median 0.00px / P95 0.00px; LOW_MATCH_SUPPORT | REGISTRATION_FAILED |
| Crossover | TOP | 9 | 6 | fit median 0.55px / P95 0.69px | AUTO_ESTIMATED_UNVERIFIED |
| Crossover | BOTTOM | 10 | 8 | fit median 0.76px / P95 1.50px | AUTO_ESTIMATED_UNVERIFIED |
| Groundplane | TOP | 7 | 7 | fit median 0.29px / P95 1.17px | AUTO_ESTIMATED_UNVERIFIED |
| Groundplane | BOTTOM | 7 | 7 | fit median 0.67px / P95 1.09px | AUTO_ESTIMATED_UNVERIFIED |
| Meander | TOP | 4 | 4 | rejected candidate; fit median 0.21px / P95 0.21px; AMBIGUOUS_TRANSFORM_HYPOTHESES | REGISTRATION_FAILED |
| Meander | BOTTOM | 5 | 5 | fit median 0.31px / P95 0.64px | AUTO_ESTIMATED_UNVERIFIED |
| Cascade | TOP | 27 | 26 | fit median 0.85px / P95 1.40px | AUTO_ESTIMATED_UNVERIFIED |
| Cascade | BOTTOM | 27 | 20 | fit median 1.02px / P95 1.61px | AUTO_ESTIMATED_UNVERIFIED |
| Connector | TOP | 7 | 6 | fit median 0.66px / P95 0.92px | AUTO_ESTIMATED_UNVERIFIED |
| Connector | BOTTOM | 7 | 6 | fit median 0.28px / P95 1.19px | AUTO_ESTIMATED_UNVERIFIED |
| Elements | TOP | 28 | 22 | fit median 0.69px / P95 1.66px | AUTO_ESTIMATED_UNVERIFIED |
| Elements | BOTTOM | 16 | 16 | fit median 0.68px / P95 1.29px | AUTO_ESTIMATED_UNVERIFIED |
| Filter | TOP | 16 | 13 | fit median 0.32px / P95 0.52px | AUTO_ESTIMATED_UNVERIFIED |
| Filter | BOTTOM | 16 | 14 | fit median 0.47px / P95 1.37px | AUTO_ESTIMATED_UNVERIFIED |
| Floating | TOP | 8 | 7 | fit median 0.67px / P95 1.01px | AUTO_ESTIMATED_UNVERIFIED |
| Floating | BOTTOM | 9 | 7 | fit median 0.49px / P95 1.34px | AUTO_ESTIMATED_UNVERIFIED |
| Round | TOP | 3 | 0 | NOT_AVAILABLE | REGISTRATION_FAILED |
| Round | BOTTOM | 5 | 4 | fit median 0.44px / P95 0.53px | AUTO_ESTIMATED_UNVERIFIED |
| Transistors | TOP | 4 | 3 | rejected candidate; fit median 0.00px / P95 0.00px; LOW_MATCH_SUPPORT | REGISTRATION_FAILED |
| Transistors | BOTTOM | 4 | 4 | rejected candidate; fit median 0.34px / P95 0.40px; LOW_SPATIAL_COVERAGE, AMBIGUOUS_TRANSFORM_HYPOTHESES | REGISTRATION_FAILED |

## Failure and uncertainty analysis

- Simple TOP had no geometrically consistent consensus; Simple BOTTOM yielded three candidates but the fit was degenerate.
- Round TOP produced three candidate pairs but the fit was degenerate. Round BOTTOM produced an unverified estimate.
- Divider TOP/BOTTOM, Meander TOP and both Transistors sides were rejected for low support, low coverage or transform ambiguity even when an affine candidate could be fitted. Their in-sample residuals must not be interpreted as good registration.
- TOP and BOTTOM are fit independently. Similarity of selected shared anchor IDs does not verify a common board coordinate system; the aggregate TOP/BOTTOM state remains `NOT_VERIFIED`.
- Mirror is searched as one possible negative-determinant hypothesis. No side is blindly assigned a mirror rule.
- Image feature diagnostics from the pre-existing board/copper detectors remain candidate-only and are not treated as calibration matches.

## Visual review artifacts

Generated locally from the pinned PRISM checkout (not committed: images are external dataset-derived outputs and their redistribution status has not been cleared):

- `data/benchmarks/phase21_1_1/overlays/Crossover_TOP_registration.png` — overlay looked visually aligned on a selected stable example; it remains unverified because anchors were selected using design geometry.
- `data/benchmarks/phase21_1_1/overlays/Divider_TOP_registration.png` — shows the image candidates only because low-support/ambiguous fit was rejected; the candidate fit and rejection diagnostics remain in the match JSON and summary.
- Per-view `*_registration.png`, per-view `_top_bottom_unregistered.png`, raw anchor JSON and match JSON are in the same ignored artifact directory.

Overlay key: green circles are image-only round-hole candidates; magenta crosses are design anchors transformed into image space; orange lines are matched residual vectors. Every registration overlay is watermarked `DESIGN-ASSISTED ... AUTO_ESTIMATED_UNVERIFIED`.

## Existing PRISM baseline regression

The existing `benchmark_prism.py --all --baseline-only` pipeline was run against all 12 samples (24 views). It completed its pipeline for all views and retained its original caveats: candidate copper/contact outputs are not validated electrical connectivity, and all alignment-dependent scores remain `NOT_VERIFIED`. The raw baseline report and images remain in the ignored local `data/benchmarks/phase21_1_1/prism_baseline/` directory. No prior Phase 21.0/21.1 regression pipeline was changed.

## Registration integrity and projection gate

`scripts/validate_registration.py` reports 24 views checked, 0 verified, and a valid record structure. `scripts/build_pcb_reference.py --all` emits 24 `NOT_PROJECTED_REGISTRATION_NOT_VERIFIED` records and no masks. This is intentional: even a visually plausible design-assisted estimate is insufficient for projection into evaluation labels until independent reviewed validation points satisfy the predeclared criterion.

The automatic transform status is progress from the earlier Phase 21.1 record (0 emitted transforms); the stability gate rejected five additional candidate transforms, leaving 16 emitted estimates. These counts are not a verified-registration success rate. Historical Phase 21.1 metrics remain in [the Phase 21.1 report](PHASE21_1_FINAL_REPORT.md); this experiment does not overwrite their provenance.

## Phase 21.2 readiness

1. **Can PRISM now support quantitative trace-detection evaluation?** No. Registration is not independently verified, and visible-copper/occlusion masks are absent.
2. **How controlled are image/design matching errors?** The pipeline rejects degenerate fits and exposes support, residual, coverage and competing-transform ambiguity, but repeated patterns and sparse holes still produce plausible false hypotheses. Independent adjudication is missing.
3. **Can training labels be made now?** Not as verified image-space labels. Design geometry may be stored as a separate design-assisted reference, but not used as visible-copper ground truth.
4. **What independent data is missing?** Matching physical board revision and photos, image-to-design correspondences reviewed independently, separate held-out landmarks, and visibility/occlusion plus pin/net continuity annotations.
5. **What must come first?** Obtain at least one physical or independently adjudicated board pair with distributed calibration and held-out landmarks; agree a pixel threshold before fit; model pad/track/via geometry and visibility before evaluating segmentation.

PRISM remains useful for synthetic pipeline regression and matching-algorithm development, not for physical-board performance claims or a quantitative image-only accuracy score.
