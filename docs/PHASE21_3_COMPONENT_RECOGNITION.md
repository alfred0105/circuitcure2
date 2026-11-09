# Phase 21.3 — Component Recognition & Evaluation

## Outcome

Phase 21.3 adds a reproducible evaluation path for the existing local ONNX PCB detector, board-grouped annotation manifest/schema checks, native-resolution overlapping-tile inference, and safer observation identity/fusion in the multiview component map. No detector retraining was performed.

The source-provided validation split can be measured, but the project does **not** have reviewed ground truth or board IDs for that data. Thus Phase 21.3 does not establish board-independent detection accuracy, sealed-test performance, identity precision, or real-board multi-image recall. Those gates remain blocked until reviewed, board-grouped labels are available.

## Changes

- Missing or repeated detector IDs now receive deterministic photo-scoped IDs; observations are sorted by class/geometry before fusion to reduce order dependence.
- Observation groups use complete-link matching: a new item must agree with every group member. Same-photo high-overlap candidates remain ambiguous instead of being silently fused; conflicting classes and pairwise disagreement are flagged for review.
- The opt-in tiled detector uses overlapping native-resolution tiles, restores full-frame coordinates, and runs global class-aware NMS. Component Map tiled inference now decodes the preserved original image and maps detected polygons back into the existing processed/canonical frame.
- The service default remains single-image inference. Tiling is an explicit recall/latency tradeoff and is not the default because the source validation comparison reduced aggregate scores.
- Detection evaluation reports one-to-one class-aware precision, recall, F1, AP@0.5, AP@0.5:0.95, per-class support/status, small-box recall, threshold curves, and per-image results.
- Annotation manifests have a versioned JSON Schema and a validator for required provenance, bounding boxes, label review status, duplicate image IDs/hashes, and board split leakage. Any unreviewed annotations block scoreable ground truth.

## Current decision

Keep single-image inference as the default. The tiled mode increased small-box recall on the source split (2/8 to 6/8) but reduced precision, recall, and mAP, while taking about 8.3 times longer per image. The small-box sample is only eight capacitors and is a pixel-area proxy, so this result is exploratory rather than proof of a general small-component gain.

See [dataset and evaluation protocol](PHASE21_3_DATASET_AND_EVALUATION.md), [benchmark results](PHASE21_3_BENCHMARK_RESULTS.md), and [backend contract](PHASE21_3_BACKEND_CONTRACT.md).
