# Phase 21.3 — Component Map Backend Contract

## Detector entry points

`detect_components(image, model_path, confidence_threshold=0.25, iou_threshold=0.45)` remains the default, bounded single-image model inference path.

`detect_components_tiled(image, model_path, confidence_threshold=0.25, iou_threshold=0.45, tile_size_px=1536, tile_overlap_px=320, timeout_seconds=None)` is opt-in. Resource guards cap images at 40 megapixels, tile size at 256–4096 px, overlap at half a tile, total tiles at 256 and accumulated candidates at 50,000. It returns full input-frame `bbox_xyxy`, deterministic tile detection IDs, `inference_mode: overlapping_tiles`, tile provenance and aggregate latency. It raises `ValueError` for invalid geometry/resource budgets and `TimeoutError` when the configured budget expires. A stopped run is not an inference success.

The component-map service accepts `inference_mode="single"|"tiled"`; it defaults to `single`. Tiled mode runs on preserved native-resolution image bytes, constrained by intake pixel/byte limits, then maps each detection polygon into the existing processed image frame before applying the same photo-to-mosaic transform. TOP and BOTTOM remain separate. Unregistered photos do not enter the canonical component map.

## Observation and identity semantics

- Every model result is a candidate observation, not an electrical connection or confirmed part identity.
- Missing IDs are generated from photo, class, rounded geometry, model and OCR signature; repeated explicit IDs are disambiguated deterministically within the photo.
- `source_detection_id` retains the supplied identifier when one exists.
- Same-photo overlapping candidates are marked for identity review; pairwise-incompatible candidates cannot be fused through transitive similarity.
- Previous user-confirmed identity and measurement references are preserved only when a one-to-one continuation is supported. Ambiguous matches are not silently assigned.
- Confidence is detector confidence, never fault probability.

## Frontend fields

The Phase 21.2.1 map contract remains compatible. Each component observation provides or inherits:

| Field | Meaning |
|---|---|
| `board_id`, `board_side` | Declared board session and TOP/BOTTOM partition |
| `canonical_image_reference` / side `canonical_image.image_id` | Image canvas for overlay coordinates |
| `component_id`, `class_name`, `bbox_xyxy`, `polygon_xy` | Candidate identity/type and canonical map geometry |
| `confidence`, `identification_status`, `identity_review_status` | Separate detector/type and identity evidence states |
| `source_image_ids`, observation `photo_id` | View provenance |
| `ocr_marking`, `possible_part_number` | OCR evidence / unverified candidate only |
| `registration_status`, residuals and coordinate frame | Photo alignment provenance and uncertainty |
| `evidence`, `inference_mode`, `source_tile` | Detection evidence and inference provenance |
| `suggested_functional_role` | Not evaluated in Phase 21.3; absent until later inference |

Fault overlay remains scenario/evidence-gated and receives stable component IDs only. A detector candidate is not automatically used to assert a fault, physical damage, connectivity, netlist, or SPICE wiring.

## Evaluation contract

The evaluator scores supplied YOLO labels but carries provenance status `SOURCE_PROVIDED_UNADJUDICATED`. Board-level acceptance requires reviewed annotations, a board-grouped split, and no validation/test leakage. The manifest checker returns `BLOCKED_UNREVIEWED_ANNOTATIONS` or `BLOCKED_NO_REVIEWED_GROUND_TRUTH` when that gate is not satisfied.
