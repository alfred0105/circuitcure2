# Phase 21.3 — Dataset & Evaluation Protocol

## Data sources and eligibility

The detector benchmark uses the `val` images and YOLO labels from [wangshnegqi/pcb-component-detection-dataset-yolo](https://github.com/wangshnegqi/pcb-component-detection-dataset-yolo), pinned at commit `0d7ae7310fa607389779b6132285336ad73e96e1`.

| Item | Source evidence / limitation |
|---|---|
| Validation images and labels | 13 images / 196 boxes in the pinned `val` split |
| Image type | Repository describes real PCB/electronics-board photographs |
| Board count / board IDs | Not supplied in the dataset manifest; cannot establish unique board count |
| Annotation format | YOLO normalized class + center + width/height boxes |
| Source classes | Resistor, Capacitor, Diode, LED, IC, Connector, Switch/Button, Header, Crystal, Debug/TP |
| Dataset license | CC BY 4.0, declared by source repository (`LICENSE-DATASET`) |
| Model/repository license | Apache-2.0, declared by source repository (`LICENSE`) |
| Reviewed status | Source-provided labels have not been independently re-reviewed by CircuitCare |
| Board-group split | Unknown; source split is by image and same-board leakage cannot be ruled out |

The source validation split is used as an **unadjudicated source-label comparison set only**. It is not a sealed test set, and its scores are not presented as reviewed ground-truth or board-level estimates. PRISM is not used for component detection accuracy because it has no independent per-component photo boxes/classes. No dataset photographs or source label files are committed to this repository; the optional local checkout is ignored.

## Annotation and split contract

Use [`schemas/component_annotation.schema.json`](../schemas/component_annotation.schema.json) and [`data/datasets/component_annotation_manifest.template.json`](../data/datasets/component_annotation_manifest.template.json). The manifest records board/revision, side, image ID/hash/size, split, source and per-component class, box/polygon, occlusion, small-component flag, review state, reviewer and provenance. Allowed review states are `REVIEWED_GROUND_TRUTH`, `DRAFT_ANNOTATION`, `MODEL_PSEUDO_LABEL`, and `UNRESOLVED`.

Assign all images, sides, closeups, crops, and augmentations from the same physical board revision to one split **before** generating crops or augmentation. The validator rejects board IDs or identical image hashes in multiple splits. Source data without board IDs cannot pass as board-independent validation. Any draft, pseudo-label, or unresolved annotation blocks a manifest from being scoreable as reviewed ground truth.

Validate a manifest with:

```bash
python scripts/validate_component_dataset.py manifest.json
```

## Metrics

- P/R/F1 use class-aware one-to-one matching at IoU 0.50 and the stated confidence cutoff.
- AP uses a confidence-ranked precision-recall curve with raw detector predictions retained down to the run's recorded confidence floor (0.001 in this experiment); mAP@0.5:0.95 averages IoU thresholds 0.50–0.95 in 0.05 increments.
- Small-object recall is an explicitly labeled pixel-area proxy (box area ≤ 32×32 px in the source image). It is not the missing human-reviewed `small_component` label.
- Per-class support below 10 is marked `INSUFFICIENT_SUPPORT`; metrics remain visible but should not guide a class-level policy.
- Latency reports warmup separately from steady per-image inference. Peak RSS is process high-water memory, not isolated model memory.
- Board-level precision/recall, false merge/split and identity preservation require reviewed board/component IDs and are therefore not estimated on this source split.

## Reproduction

```bash
python scripts/evaluate_pcb_detector.py \
  --images data/external/pcb-component-detection/dataset/images/val \
  --labels data/external/pcb-component-detection/dataset/labels/val \
  --model models/pcb_component_detector.onnx --mode single \
  --output /tmp/pcb-single.json

python scripts/evaluate_pcb_detector.py \
  --images data/external/pcb-component-detection/dataset/images/val \
  --labels data/external/pcb-component-detection/dataset/labels/val \
  --model models/pcb_component_detector.onnx --mode tiled \
  --tile-size 1536 --tile-overlap 320 --output /tmp/pcb-tiled.json
```

The output records model and image/label hashes, evaluation settings, per-image scores, latency and memory. Full predictions are meant for local `/tmp` outputs; compact reproducible summaries are checked in under `data/benchmarks/phase21_3/`.
