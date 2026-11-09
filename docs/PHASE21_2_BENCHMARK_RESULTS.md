# Phase 21.2 Benchmark Results

## Scope and limits

There are no independently annotated real multi-view PCB images or available ONNX detector weights in the repository. Real component precision, recall, mAP, type recall, SMD recall, unknown rate, false positives/negatives, and identity-fusion precision are therefore **`NOT_EVALUATED`**. Image registration inliers are not independent truth. The Phase 21.1.2 KiCad alignment status remains unchanged and this report makes no electrical connectivity claim.

## Synthetic geometry fixture

`scripts/benchmark_multi_image_component_map.py` creates three deterministic 420×420 textured overlapping crops with known horizontal offsets (0, 120, 240 px), then runs the full local session/map path. The saved output is [synthetic_summary.json](../data/benchmarks/phase21_2/synthetic_summary.json).

| Metric | Observed result |
|---|---:|
| Input views | 3 |
| Registration | 1 reference, 2 `REGISTERED_CANDIDATE` |
| Inlier ratio, moving views | 1.000 / 0.998 |
| Inlier RMSE, moving views | 0.512 / 0.315 px |
| Inlier P95 residual, moving views | 0.941 / 0.431 px |
| Known-offset transform error | 1.000 / 0.995 / 0.973 px for reference and views |
| Mosaic dimensions | 660×422 px (expected scene extent 660×420; 2 px padding from estimated warp extent) |
| One local full-flow run | 0.125 s |

These are synthetic geometry smoke measurements; they do not estimate real-photo registration accuracy. Synthetic injected detections verify component-map coordinates, two-observation fusion, provenance, and overlay behavior in unit tests, but do not evaluate the detector.

## Real PRISM image intake smoke

Three pinned PRISM boards (Cascade, Elements, Simple) were passed through intake and side-isolated map generation, for six real image files. Dataset commit: `9189c87bbec6995b745b809db99cef6d3269401f`. Input SHA-256, decoded image sizes, per-board runtime, and detector state are saved in [prism_image_intake.json](../data/benchmarks/phase21_2/prism_image_intake.json).

- TOP and BOTTOM produced separate side maps for all three boards. No cross-side registration was attempted.
- Each side had one input view, so this run tested intake, orientation-aware decode, bounded downscaling and map generation; it did not exercise same-side multi-image stitching.
- All six detections were `DETECTOR_NOT_CONFIGURED`; no trained model file was present. Thus component count was zero, not evidence that those boards contain no components.
- Total observed wall time for the two-side board runs ranged from about 0.087 to 0.118 seconds on the current local host. These single-run figures are diagnostic only.

## Fail-closed cases exercised

Unit tests cover corrupted input and image-count limits, duplicate SHA detection, EXIF rotation, TOP/BOTTOM separation, homography repeatability, overlap stitching, close-up detection projection, multi-view identity fusion, unregistered non-overlap rejection, component-ID mapping, confirmed-fault gating, and JSON serialization. Canvas limits and wrong-side inputs cannot produce a cross-side transform. Existing Phase 21.1/21.1.2 coordinate-registration tests remain in the focused regression.

No maximum RSS claim is made: pixel, file, decoded-edge and mosaic-canvas limits bound inputs/arrays, but Python/OpenCV process RSS is allocator- and platform-dependent. The current run records elapsed time but does not sample RSS.
