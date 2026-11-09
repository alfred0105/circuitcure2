# Phase 21.3.1 Benchmark Results

## Evaluation status

The available 13-image / 196-box public validation split is a **source-label comparison**, not reviewed validation or a sealed board-level test. Its board IDs are unresolved and its annotations have not been independently adjudicated. This Phase does not promote those labels or model outputs to ground truth.

| Track | Status | Images | Boxes | Board independence |
|---|---|---:|---:|---|
| Source-provided labels | `SOURCE_LABEL_COMPARISON` | 13 | 196 | `BLOCKED_BOARD_ID_UNRESOLVED` |
| Reviewed validation | `BLOCKED_REVIEWED_GROUND_TRUTH_UNAVAILABLE` | — | — | unavailable |
| Sealed board-level test | `BLOCKED_SEALED_TEST_UNAVAILABLE` | — | — | unavailable |

## Current comparison

The manifest-backed clean-commit evaluation reproduced the Phase 21.3 single-image result:

| Metric | Result |
|---|---:|
| Precision at confidence 0.25 | 0.497925 |
| Recall at confidence 0.25 | 0.612245 |
| F1 at confidence 0.25 | 0.549199 |
| mAP@0.5 | 0.584774 |
| mAP@0.5:0.95 | 0.333614 |
| TP / FP / FN at confidence 0.25 | 120 / 121 / 76 |

The measured class support and results are available in the machine-readable summary. Classes with fewer than 10 source boxes are flagged `INSUFFICIENT_SUPPORT`; they should not be treated as reliable class-level estimates.

## Reproduction record

- CircuitCare code: `4f771da54322f64528ff586b94fc6573adcd80e8`, working tree clean at evaluation
- Dataset source commit: `0d7ae7310fa607389779b6132285336ad73e96e1`
- Dataset manifest SHA-256: `7d09d4b88672d1db26e024bc67a1065353c8dd974d2079df35c1a3386338eaa7`
- Model: `pcb-yolov8s-2026-07-11`, SHA-256 `45a1dc54bb2eaff067edc631be30fee650deb1236e97a038610faa11aef28424` (44,943,271 bytes; local ignored artifact)
- Runtime: Python 3.12.14, macOS 27.0.1 arm64, OpenCV 4.14.0, NumPy 2.5.4, ONNX Runtime 1.31.1, Pillow 13.0.0
- Inference: CPU, single-image mode, input 960 px, raw prediction floor 0.001, reporting confidence 0.25, NMS IoU 0.45, metric IoU 0.50; AP sweep 0.50–0.95
- Runtime/resource observation over 13 images: mean 250.4 ms, median 241.4 ms, P95 288.8 ms; process peak RSS 1,172,848,640 bytes, including Python and imported libraries
- First model-load plus inference: recorded in the JSON summary; excluded from steady per-image latency
- Per-image image and label hashes: recorded in the JSON summary

The Phase 21.3.1 implementation-stage run and clean run produced identical aggregate metrics, but their latency distributions varied substantially; runtime is therefore descriptive and is not evidence of a speed improvement. Full machine-readable provenance and per-class metrics are in [`clean_baseline_summary.json`](../data/benchmarks/phase21_3_1/clean_baseline_summary.json).

## Interpretation and next data gate

These values compare predictions to upstream labels that may be incomplete or incorrect. They do not establish real-PCB detection accuracy, board-independent generalization, or a damage probability. Per-class support below 10 is flagged as insufficient for reporting. Small-object misses and source-label false positives remain candidates for human review.

To unblock independent validation, reviewers must identify physical board/revision and side for each image, adjudicate boxes and omissions, record reviewer/provenance, and reserve board groups for validation and sealed test before tuning against those splits. Until then, retain `BLOCKED_REVIEWED_GROUND_TRUTH_UNAVAILABLE`, `BLOCKED_BOARD_ID_UNRESOLVED`, and `BLOCKED_SEALED_TEST_UNAVAILABLE`.

See [dataset audit](PHASE21_3_1_DATASET_AUDIT.md), [evaluation validation](PHASE21_3_1_EVALUATION_VALIDATION.md), and [recognition reliability report](PHASE21_3_1_RECOGNITION_RELIABILITY.md) for source inventory and protocol details.
