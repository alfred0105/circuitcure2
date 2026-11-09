# Phase 21.2.1 — Component Map Stabilization

## Result summary

This phase stabilizes the existing Phase 21.2 pipeline in `MultiImageComponentMapService`. It does not implement PCB connectivity, physical damage prediction, or a UI. The backend remains a local Python service; it creates image-space candidate geometry only.

| Area | Result | Evidence state |
|---|---|---|
| Stable identity | Prior map/JSON identity manifest can be supplied; one-to-one class/geometry matching reuses IDs and preserves measurement references and user confirmation | Synthetic repeat/add-view test passed; real identity precision `NOT_EVALUATED` |
| False-merge protection | Same-photo duplicate boxes, class mismatch, conflicting OCR, uncertain/weak registration, and loose geometry are excluded from fusion; pairwise evidence is serialized | Unit tests passed; real false-merge rate `NOT_EVALUATED` |
| Automatic rectification | Existing boundary candidate detector and perspective rectifier are reused; clear interior candidates produce an unverified rectification candidate; ambiguous, frame-touching, or absent boundaries preserve the original | Synthetic positive and fail-closed checks passed; no physical-board validation |
| Coverage and provenance | Valid warped pixel masks, overlap, uncovered pixels, pixel source map, per-source contribution, canvas footprint, and overlap photometric residual are reported independently | Synthetic geometry smoke only |
| Coordinate frames | Encoded image → EXIF corrected → decoded → rectified → mosaic transforms include direction/frame labels; detection boxes map back through inverse transforms | EXIF rotation regression passed; no mm coordinates generated |
| Detector | Existing ONNX CPU detector loaded locally and was called through the service for all 24 pinned PRISM images | Pipeline inference smoke passed; recognition accuracy `NOT_EVALUATED` |
| Physical multiview | No matching real PCB capture set with multiple same-side photographs was found in the checkout | `BLOCKED_REAL_IMAGE_DATA_UNAVAILABLE` |

## Component identity policy

New IDs derive from a stable primary observation ID, not rounded canonical coordinates. Photo IDs derive from board, side, role, basename, and content digest; they no longer depend on input-list position. A prior full map or identity manifest enables identity continuity. Candidate matching is one-to-one, same-class, spatially bounded, and requires a score and runner-up margin. Ambiguous matches receive `IDENTITY_AMBIGUOUS`; IDs are not transferred. Explicit user mappings require a known previous component and class compatibility. `measurement_history_refs`, user confirmation, identity evidence, and historical mapping state travel with matched IDs.

An identity manifest can be written without embedding the mosaic images:

```python
from circuitcare.vision.reconstruction.multi_image_component_map import save_identity_manifest

save_identity_manifest("component-identities.json", component_map)
next_map = service.build_map(session, previous_component_map="component-identities.json")
```

If no previous map is supplied, the service only promises a deterministic ID for the same observation input. It does not claim to recover a historical physical identity.

## Automatic rectification policy

The existing `detect_board_candidates()` output is accepted only when one candidate is clearly larger, covers a sufficient fraction of the image, and its four corners remain inside an image-edge margin. A candidate touching the image frame is rejected because a detector can mistake the image boundary or a cropped texture for the PCB boundary. Rectification output is capped at the input's longest edge to avoid arbitrary upscaling. Accepted transforms are still `RECTIFIED_CANDIDATE_NOT_PHYSICALLY_VALIDATED`; planar correction does not remove parallax from tall components. Manual corners remain supported.

## Mosaic coverage semantics

- `valid_pixel_coverage_ratio`: union of warped source masks divided by canvas pixels.
- `canvas_coverage_ratio`: convex footprint area divided by the output canvas; it is not a count of observed pixels.
- `overlap_union_ratio_to_canvas`: pixels covered by at least two sources divided by canvas pixels.
- `unobserved_area_ratio`: pixels covered by no source divided by canvas pixels.
- `source_contribution_ratio`: selected source pixels after the current sharpness heuristic divided by valid pixels. A source may contribute zero if another source wins every overlap pixel.
- `overlap_photometric_residual_mean_abs_rgb`: simple pre-correction overlap difference diagnostic; it is not a ghosting score.
- `ghosting_diagnostics`: remains `NOT_EVALUATED` without parallax/artifact truth.

## Detector and public model provenance

The optional detector is the existing `pcb-yolov8s-2026-07-11` ONNX model. The local ignored file used in this run had SHA-256 `45a1dc54bb2eaff067edc631be30fee650deb1236e97a038610faa11aef28424` and 44,943,271 bytes. Its pinned checkpoint source, dataset/model license notes, exporter caveat, and setup instructions are documented in [PCB AI detector setup](PCB_AI_DETECTION.md). The ONNX model is not committed. One sample prediction and all 24 PRISM inference calls establish only that the runtime path executes; no independent component labels were scored in this phase.

## Test and benchmark records

Run the targeted tests with:

```bash
.venv/bin/python -m pytest -q tests/test_phase21_2_multi_image_component_map.py
.venv/bin/python scripts/benchmark_multi_image_component_map.py --output-dir data/benchmarks/phase21_2_1
```

See [benchmark results](PHASE21_2_1_BENCHMARK_RESULTS.md) for exact input hashes, synthetic geometry metrics, public dataset smoke counts, and resource caveats. The PRISM TOP and BOTTOM images are separate single views, not same-side multiview physical captures.

## Remaining gates

Independent physical same-side photographs, reviewed landmarks, component identity labels, and detector annotations are still required for accuracy claims. The synthetic registration smoke passes with approximately 1 px known-translation error, but those same generated pixels are not an independent dataset. There is no validated physical scale, electrical net reconstruction, or automatic fault confirmation in this service.
