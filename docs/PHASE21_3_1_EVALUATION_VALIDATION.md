# Phase 21.3.1 — Evaluation Validation

## Metric contract

`evaluate_detections` validates taxonomy IDs, finite positive-area boxes, confidence values, class names, and threshold ranges before scoring. It uses class-aware greedy one-to-one IoU matching, confidence-ranked AP with the interpolated precision envelope, and averages classes that have ground truth. Stable tie ordering makes repeated runs deterministic.

An image with no GT is represented by an explicit `image_id: []` entry; its predictions are false positives. Prediction-only image keys are also counted as evaluated and contribute false positives. A blank source label file can be included in the diagnostic source-label comparison, but remains unreviewed. To count an image as a reviewed negative, set `label_state: REVIEWED_EMPTY` with a reviewer and `review_provenance`.

Fixtures cover perfect matching, all missed, all false positive, duplicate one-to-one matching, wrong-class FP+FN, IoU/AP, confidence threshold curves, deterministic confidence ties, unsupported classes/invalid boxes, image-hash and board-split leakage, reviewed-empty versus unannotated, and mixed reviewed/draft label gates. Analytic fixture values were used; a separate full COCO/Ultralytics evaluator was not available in this environment.

## Run with an explicit dataset manifest

```bash
python scripts/evaluate_pcb_detector.py \
  --dataset-manifest data/datasets/public_pcb_component_validation_manifest.json \
  --split validation --model models/pcb_component_detector.onnx \
  --mode single --output reports/pcb-validation.json
```

The report captures Dataset ID/name/URL/version/license, annotation source/review status, split method, board grouping policy, manifest hash, every image/label hash, model hash, Code SHA and clean/dirty state, Python/OpenCV/NumPy/ONNX Runtime/Pillow versions, platform, input/NMS/metric IoU/confidence settings, runtime and peak RSS. The path-only compatibility mode records metadata as null and status `PROVENANCE_INCOMPLETE`.

The manifest validator detects duplicate image IDs, exact duplicate image hashes, board IDs crossing splits, invalid hashes/boxes/classes, missing metadata, duplicate per-image component IDs, review provenance gaps, and distinguishes `UNANNOTATED` from `REVIEWED_EMPTY`. The evaluator additionally checks that referenced image/label files exist and match their hashes. A `BOARD_ID_UNRESOLVED` sentinel is not counted as a physical board and blocks board-level scoreability.

## Failure analysis

Use an evaluation JSON containing the full `ground_truth` and `predictions` arrays:

```bash
python scripts/report_detection_failures.py \
  --evaluation reports/pcb-validation.json \
  --manifest data/datasets/public_pcb_component_validation_manifest.json \
  --output-dir reports/pcb-failure-review
```

The report keeps per-image FN/FP rows with class, box, detection confidence, annotation/detection identifier, and possible class confusion; marked-up image previews use green GT and red predictions. These are comparisons to supplied annotations, not adjudicated failures when annotation status is unreviewed.

## CI

No workflow restructuring was required. Existing Actions already cache pip, filter documentation-only changes, test Python 3.11/3.12, and keep native OCR, ngspice, stress, crash-injection, and data-heavy PRISM work in the existing separate jobs. The new fast unit tests join the existing Python matrix; the public dataset/model are not downloaded in ordinary CI.
