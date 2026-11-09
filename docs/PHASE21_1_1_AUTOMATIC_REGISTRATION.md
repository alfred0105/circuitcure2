# Phase 21.1.1 — Automatic Anchor Matching and Registration

## Scope and provenance

This follow-up adds actual automatic image/design anchor matching to the Phase 21.1 coordinate-frame and RANSAC foundation. It processes all 12 pinned PRISM boards and both sides. The PRISM checkout is pinned to `9189c87bbec6995b745b809db99cef6d3269401f`; no PRISM source files are modified.

The PRISM `Dataset/METADATA.md` specifies image origin at the top-left, KiCad design coordinates in millimeters, and no supplied pixel/mm scale, DPI, or calibration mapping. Its image dimensions vary by board and sometimes by side. Inspection of the pinned checkout found no image-generation transform metadata or source capable of reconstructing one exactly. PRISM's source has a separate TOP/BOTTOM image-alignment method based on drill-hole candidates and tests mirror/rotation hypotheses; that image-to-image procedure does not define an image-to-KiCad transform. Therefore no fixed render transform is assumed. BOTTOM mirroring is searched as a candidate rather than forced.

Reference: [PRISM repository](https://github.com/KamMalinski/PRISM), pinned revision above; local inspection note: `data/external/prism/Dataset/METADATA.md`.

## Implemented matching path

### Image-only candidates

`anchor_detection.py` defines `ImageAnchor` separately from design geometry. The detector thresholds grayscale pixels below 28, filters connected components by area, aspect ratio, maximum size and circularity (at least 0.35), merges centers within 5 px, and returns only `DETECTED_CANDIDATE` records. It assigns no component, pin, net, or electrical identity. The method targets dark drill-hole centers in flat PRISM renders; it does not detect or validate rectangular SMD pads and is not claimed to generalize to physical PCB photos.

### Design-assisted anchors

`anchor_matching.py` extracts through-hole pads and vias from PRISM `expected_geometry.json`. Each `DesignAnchor` retains ID, KiCad mm coordinates, component/pin, layer, pad type/shape/dimensions, and source evidence. The image and design anchor collections remain separate. This is explicitly `DESIGN_ASSISTED_REGISTRATION`; the design reference participates in correspondence search.

### Correspondence and transform search

The matcher compares pairwise design/image distances to find plausible pixel/mm scales in [1, 80], scores distance-consistent pairs over the leading scale histogram peaks, and searches both pair orientations and both transform handednesses. It assigns one-to-one candidate correspondences by geometric distance, refines candidates with affine RANSAC, then passes candidate pairs through the existing `fit_transform(..., TransformType.AFFINE)` RANSAC implementation. The existing coordinate-frame model is reused. A negative-determinant result is labeled `MIRROR`; positive determinant is `AFFINE`/`ROTATION_SCALE`. The bottom side is not predetermined to be mirrored.

The current defaults are: 720 log-scale histogram bins, pair-distance tolerance 1.5%, match tolerance `max(3.5 px, 0.006 × min(image width, height))`, and RANSAC threshold 2 px. These are search parameters, **not validated acceptance thresholds**. The 2 px fit threshold is not an independent quality criterion.

The result reports candidate match pairs, geometric inliers, inlier ratio, training residual median/P95, spatial coverage, transform stability across leading hypotheses, orientation, and review flags. Low support, low coverage, or ambiguity remains visible in the report. Degenerate fits are rejected. Tiny in-sample residuals from three/four points are not evidence of accuracy.

## Verification and leakage controls

Automatic correspondences are calibration data produced by a design-assisted search. They are not independent validation anchors. Every automatic transform remains `AUTO_ESTIMATED_UNVERIFIED`; the algorithm does not mark its own matches `REVIEWED`. `fit_transform()` can return `VERIFIED` only when separate validation points have unique IDs, are labeled `REVIEWED`, and satisfy a predeclared max held-out error threshold. The registration report validator independently rejects a `VERIFIED` record unless the validation-review metric is true and the threshold is satisfied.

Validation review status is a provenance declaration; it cannot prove that a person reviewed a point correctly. Calibration/validation ID overlap is rejected, but future real data intake must also detect duplicate physical locations under different IDs. Randomly splitting auto-matched points would not create independent ground truth.

`IMAGE_ONLY` remains separate: dark round candidates use pixels only. `DESIGN_ASSISTED_REGISTRATION` uses design anchors to match and fit. Registration correspondences are not reused as an independent accuracy score or training label. The PRISM boards remain development data, not a new holdout.

## Review outputs and projection gate

Each view gets anchor JSON, match JSON and a watermarked overlay under ignored `data/benchmarks/phase21_1_1/anchors/`, `matches/`, and `overlays/`. Registration overlays mark image candidates in green, transformed design anchors in magenta and match residual vectors in orange; the watermark says `DESIGN-ASSISTED ... AUTO_ESTIMATED_UNVERIFIED`. TOP/BOTTOM side-by-side files explicitly say they are not co-registered.

No view has independent validation, so `build_pcb_reference.py` correctly produces 24 `NOT_PROJECTED_REGISTRATION_NOT_VERIFIED` entries and no image-space reference masks. Even a future verified coordinate transform would only project design geometry: current rasterization does not model exact rotated pad shapes, true copper visibility, component occlusion, or all track/zone width semantics. `PROJECTED_NOT_VISIBILITY_GROUNDED` must not be interpreted as observed visible copper. No quantitative copper segmentation ground truth is available yet.

See [benchmark results](PHASE21_1_1_BENCHMARK_RESULTS.md) for all 24 measured outcomes and Phase 21.2 readiness. The earlier [Phase 21.1 registration](PHASE21_1_COORDINATE_REGISTRATION.md), [ground-truth validation](PHASE21_1_GROUND_TRUTH_VALIDATION.md), and [final report](PHASE21_1_FINAL_REPORT.md) remain the historical foundation; the benchmark document supersedes their earlier statement that the automatic stage emitted no transforms.

## Commands

```bash
python scripts/calibrate_prism.py --sample Simple --mode AUTO_REGISTRATION
python scripts/calibrate_prism.py --all --mode AUTO_REGISTRATION
python scripts/validate_registration.py --registration data/benchmarks/phase21_1_1/summary.json
python scripts/build_pcb_reference.py --all --registration data/benchmarks/phase21_1_1/summary.json --output-dir data/benchmarks/phase21_1_1/projected_references
```

The generated benchmark run stored 16 unverified transforms, eight failures (including five rejected unstable candidates), and zero independently verified transforms. The local image/data-derived artifacts are ignored because the pinned external dataset's output redistribution status is not established; the source code and compact benchmark report are committed.
