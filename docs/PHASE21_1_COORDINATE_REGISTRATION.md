# Phase 21.1 — Coordinate Frames and Registration

## Frames and direction

The registration model names `IMAGE_RAW`, `IMAGE_RECTIFIED`, `BOARD_CANONICAL`, `KICAD_DESIGN`, and `LAYER_LOCAL`. Every transform declares source frame, target frame, transform type, board side, evidence, calibration-point IDs, fit error, validation errors and verification state. `apply()` rejects frame-direction and TOP/BOTTOM mismatches. Inversion is explicit; bottom-side mirroring is never applied implicitly.

The current fitter supports RANSAC affine and homography transforms. Homography requires at least four finite, unique-ID, non-collinear calibration correspondences, with source and target spread checks. The RANSAC threshold is recorded in target coordinate units. A caller must pass a separate held-out set and a predeclared acceptance threshold before status can become `VERIFIED`. Without held-out points, status stays `CALIBRATED_NOT_INDEPENDENTLY_VERIFIED`; failed fit/threshold is retained as `REGISTRATION_FAILED`.

Use `scripts/calibrate_prism.py`:

```bash
python scripts/calibrate_prism.py --all --mode AUTO_REGISTRATION
python scripts/calibrate_prism.py --sample Simple --mode MANUAL_CALIBRATION --points-json reviewed_points.json --ransac-threshold-px 2 --acceptance-threshold-px 3
python scripts/validate_registration.py --registration data/benchmarks/phase21_1/registration/summary.json
```

The optional point file is keyed by sample and side, with separate `calibration` and `validation` arrays. Each point needs `point_id`, `x_mm`, `y_mm`, `u_px`, `v_px`, and a non-empty `evidence` string. A validation point must also set `review_status: "REVIEWED"` before the result can be marked verified; this records declared review provenance, not an audit of human correctness. IMAGE_ONLY and DESIGN_ASSISTED are separately labeled; providing design-derived correspondences makes registration DESIGN_ASSISTED.

## TOP / BOTTOM convention

Each image is independently registered to `KICAD_DESIGN` and labeled TOP or BOTTOM. This implementation does not automatically mirror either side. Explicit point pairs establish orientation and transform direction. Mixed-side points are rejected. A through-hole via appearing in both images does not by itself prove a cross-layer connection.

## PRISM registration attempt

The pinned data has rendered TOP/BOTTOM views and matching KiCad PCB designs, but the current available PRISM reference schema reports an empty outline polygon for the inspected samples and does not provide reviewed image-to-design anchor pairs. Image-only board/pad candidates are collected as diagnostics and are not promoted to correspondences. Therefore the 12 × 2 initial attempts remain `REGISTRATION_UNVERIFIED` until reviewed correspondences and independent validation are supplied. No calibration or held-out error is reported for these views. The 2 px RANSAC default is an algorithm setting, **not** a validated acceptance criterion; acceptance remains unset unless declared before a run.

## Errors to retain

The report preserves per-image feature diagnostic failures and distinguishes missing correspondences, insufficient/degenerate points, transform fit failure, held-out threshold failure and absent independent validation. Wrong-side, mirror-orientation, scale and drift require adjudicated evidence; they are not guessed when the cause is unknown.

## Limits

PRISM rendered examples do not establish real-photo registration accuracy. A calibration point derived from KiCad is design-assisted. If the same pad/net labels are used to fit and score a transform, that metric is not independent; held-out point IDs must be disjoint. Independent validation should come from points reserved before fitting and, for physical claims, independently measured or reviewed physical features.

## Phase 21.1.1 follow-up

The statements above describe the original Phase 21.1 attempt and are historical. Phase 21.1.1 subsequently implemented design-assisted automatic anchor matching and produced 21 `AUTO_ESTIMATED_UNVERIFIED` transforms from 24 PRISM views; three fits failed and zero are independently verified. See [automatic registration method](PHASE21_1_1_AUTOMATIC_REGISTRATION.md) and [all-view benchmark](PHASE21_1_1_BENCHMARK_RESULTS.md). No fixed pixel/mm mapping was inferred from PRISM metadata.
