# Phase 21.3.1 — Recognition Reliability

## Outcome

The Component Map now validates detector response shape and each detection before sorting, coordinate mapping, identity assignment, or fusion. A bad record is saved as a structured `detection_rejections` entry and skipped; valid records in the same response still reach the map. Duplicate detector IDs are disambiguated while a separate `DUPLICATE_DETECTION_ID` diagnostic records their origin.

Missing confidence is retained as `LOW_CONFIDENCE` with a null score. Non-numeric, non-finite, or out-of-range confidence is rejected. Unsupported/missing classes, malformed boxes/polygons, and incomplete detector responses receive explicit error codes. Rejections never enter identity fusion. Existing detector-level status values remain intact.

Observation IDs identify a detection observation; component IDs represent a cross-view physical-identity hypothesis. Reprocessing geometry without a trusted prior correspondence does not prove physical identity. Existing one-to-one prior/user-confirmed matching and measurement-reference preservation remain intact; ambiguous or conflicting mappings are kept separate.

## Tools added

- `scripts/create_yolo_review_manifest.py` converts YOLO labels to a provenance manifest with unresolved board/side fields and draft labels. It requires explicit dataset/source/license/split metadata and refuses to treat missing label files as negative images.
- `scripts/audit_pcb_dataset.py` inventories source split files/labels, flags filename-derived augmentation families, and reports exact/pHash candidates for local images. Hash similarity is a review lead only; it never creates a Board ID.
- `scripts/export_annotation_review.py` writes an image preview packet and editable manifest while explicitly leaving every label unpromoted.
- `scripts/report_detection_failures.py` produces per-image FN/FP/class-confusion suggestions and overlay previews against the supplied labels. These error categories do not certify the labels as ground truth.
- `scripts/evaluate_pcb_detector.py` accepts the versioned annotation manifest or the legacy `--images`/`--labels` pair. Path-only runs are marked `PROVENANCE_INCOMPLETE`; they no longer assert a fixed dataset name, license, or split.

## Status

The source dataset manifest contains 13 source validation images and 196 draft annotations. All 13 board IDs are unresolved and board side is unknown. The one empty source label file is explicitly `UNANNOTATED`, not a reviewed negative. Therefore the current comparison remains `SOURCE_LABEL_COMPARISON`; independent reviewed accuracy, Board-level leakage, and sealed-test claims are blocked.

See [dataset audit](PHASE21_3_1_DATASET_AUDIT.md), [evaluation validation](PHASE21_3_1_EVALUATION_VALIDATION.md), and [baseline results](PHASE21_3_1_BENCHMARK_RESULTS.md).
