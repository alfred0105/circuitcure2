# Phase 21.2.1 Backend Contract

## Entry points

```python
session = MultiImageComponentMapService().ingest(board_id, photo_inputs)
component_map = service.build_map(
    session,
    model_path="models/pcb_component_detector.onnx",  # optional local ONNX model
    previous_component_map="component-identities.json",  # optional prior map/manifest
    confirmed_identity_by_observation_id={},  # optional explicit user-confirmed matches
)
save_identity_manifest("component-identities.json", component_map)
```

The service accepts an injected detector for tests/adapters and otherwise calls the existing ONNX detector when `model_path` is provided. Image sides remain isolated. `fault_overlay()` maps component IDs only within the selected side and preserves the existing measurement-confirmation guard for `CONFIRMED_FAULT`.

## Map response fields

The current response schema remains `circuitcare.pcb-component-map.v1`. Phase 21.2.1 adds fields without changing the existing schema identifier:

| Field | Meaning |
|---|---|
| `photos[].photo_id`, `sha256`, `legacy_photo_id` | Stable content-derived upload identity, digest, and migration reference to the previous position-dependent ID |
| `photos[].coordinate_transforms` | Matrix chain among `IMAGE_FILE_PIXEL`, `EXIF_CORRECTED_IMAGE_PIXEL`, `DOWNSCALED_IMAGE_PIXEL`, optional `RECTIFIED_IMAGE_PIXEL`, and `MOSAIC_PIXEL` |
| `registrations[].matrix_direction` | `SOURCE_TO_CANONICAL`; matrix coordinates and units are pixels |
| `registrations[].source_coordinate_frame`, `target_coordinate_frame` | Explicit transform endpoints |
| `canonical_image.coverage_metrics` | Valid mask union, canvas footprint, uncovered area, overlap, per-source contribution, and photometric residual diagnostics |
| `components[].component_observation_ids` | Stable observation references for the current photos and detector IDs |
| `components[].identity_match_status` | `NEW_COMPONENT`, `IDENTITY_MATCHED`, `USER_CONFIRMED`, `IDENTITY_AMBIGUOUS`, or `IDENTITY_CONFLICT` |
| `components[].identity_confidence`, `identity_evidence` | Heuristic score and traceable match gates; not calibrated probability or accuracy |
| `components[].measurement_history_refs` | Existing measurement references carried forward only when identity matching succeeds |
| `historical_component_id_mappings[]` | `PRESERVED` or `NOT_OBSERVED_NO_ID_REASSIGNMENT` with identity-review marker |
| `detection_status` | `DETECTOR_NOT_CONFIGURED`, `BLOCKED_MODEL_UNAVAILABLE`, `INFERENCE_COMPLETE`, or a failure/partial state |

## Coordinate convention

Matrices are 3×3 homogeneous transforms in pixel coordinates, applied to column vectors. The image-to-canonical direction is explicit. Detection `bbox_xyxy` and `polygon_xy` are in `MOSAIC_PIXEL`; `processed_bbox_xyxy` is in the rectified-or-decoded detector input; `original_bbox_xyxy` remains in the EXIF-corrected source frame for compatibility; `encoded_bbox_xyxy` is in the original file pixel frame. The `encoded_to_canonical` matrix is composed from the EXIF orientation, decode/downscale, optional rectification, and registration/canvas shift. The inverse chain is used to map detection boxes back to both source frames.

No physical millimeter transform is emitted. Image registration does not produce KiCad registration, and neither this map nor unverified component identity may be treated as electrical connectivity or SPICE input.

## Identity persistence and safety

Pass the prior map or the output of `save_identity_manifest()` to reuse IDs. Matching requires compatible board ID and side, class agreement, bounded position/scale, minimum score, and a runner-up gap. Each previous ID can be assigned once. Ambiguous, missing, or contradictory identity is not silently transferred. Explicit confirmation input is keyed by current `observation_id` and names an existing previous `component_id`; unknown, duplicate, or class-conflicting assignments become `IDENTITY_CONFLICT`.

Fusion additionally requires different source photos, same detector class, no contradictory non-empty OCR markings, valid candidate registration, acceptable registration residual/inlier ratio, high box IoU, bounded center distance, and compatible box scale. Every fused pair retains numeric gate evidence. These are conservative heuristics, not independent identity truth.

## Fail-closed states

| State | Caller behavior |
|---|---|
| `RECTIFICATION_UNVERIFIED` / `RECTIFICATION_FAILED_KEEP_ORIGINAL` | Use original decoded image geometry and expose reason; do not interpret as a verified board plane |
| `UNREGISTERED` | Do not fuse detections from that photo into the canonical map |
| `IDENTITY_AMBIGUOUS` / `IDENTITY_CONFLICT` | Require human mapping before attaching historical measurement data |
| `BLOCKED_MODEL_UNAVAILABLE` | Display no component predictions from an absent/unloadable model |
| `NOT_OBSERVED_NO_ID_REASSIGNMENT` | Preserve historical record, but do not attach it to a current component |
| unmapped overlay ID | Return `UNMAPPED_COMPONENT_ID`; do not guess a nearest component |

## Compatibility

The Phase 21.2 service and fault-overlay entry points remain available. `legacy_photo_id` supports consumer migration from the former input-index-based photo key, but new identity reuse should persist `photo_id` and `component_id` in the identity manifest. The map schema label stays `v1` to preserve readers; consumers should ignore unknown additive fields. A breaking matrix direction or component ID change requires a future schema version.
