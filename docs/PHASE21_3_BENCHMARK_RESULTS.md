# Phase 21.3 — Benchmark Results

## Experiment record

| Item | Value |
|---|---|
| Repository base SHA | `5ad38e76529808902977b9e93a1f29726fd9e56f` |
| Working tree during benchmark | Dirty; experiment code was not yet committed |
| Dataset source SHA | `0d7ae7310fa607389779b6132285336ad73e96e1` |
| Dataset split | Source-provided `val`, 13 images / 196 boxes; no board IDs |
| Model | `pcb-yolov8s-2026-07-11`, ONNX SHA-256 `45a1dc54bb2eaff067edc631be30fee650deb1236e97a038610faa11aef28424`, 44,943,271 bytes |
| Machine/runtime | macOS Apple Silicon M5, Python 3.12.14, ONNX Runtime CPU |
| Reported threshold / IoU | confidence 0.25 / IoU 0.50 |
| AP prediction floor | confidence 0.001; threshold sweep predictions retained down to this value |

The compact run outputs preserve model hash, every input image/label hash, per-image metrics, configuration, runtime and process peak RSS: [`single_baseline_summary.json`](../data/benchmarks/phase21_3/single_baseline_summary.json), [`tiled_comparison_summary.json`](../data/benchmarks/phase21_3/tiled_comparison_summary.json). No images, labels, or full prediction dumps are committed.

## Main comparison

| Metric | Single image | Native overlapping tiles (1536 px / 320 px overlap) |
|---|---:|---:|
| Precision @ 0.25 | 0.498 | 0.212 |
| Recall @ 0.25 | 0.612 | 0.347 |
| F1 @ 0.25 | 0.549 | 0.263 |
| mAP@0.5 | 0.585 | 0.309 |
| mAP@0.5:0.95 | 0.334 | 0.158 |
| TP / FP / FN | 120 / 121 / 76 | 68 / 253 / 128 |
| Small-box recall (8 GT boxes) | 2/8 = 0.250 | 6/8 = 0.750 |
| Mean latency / image | 124 ms | 1,027 ms |
| Median / P95 latency | 120 / 147 ms | 1,017 / 1,112 ms |
| Process peak RSS | 774,832,128 bytes | 770,654,208 bytes |

Tiling is approximately 8.3× slower in this run. It improves the small-box proxy on a tiny sample while substantially reducing total detection quality. Keep single-image inference as the default; tiled inference remains an explicit exploration option. The small-box set contains only eight capacitors and was defined by pixel area, not human annotation, so the observed difference is not a generalizable accuracy claim.

## Per-class source-split results at confidence 0.25

| Class | Support | Single recall / AP50 | Tiled recall / AP50 | Support status |
|---|---:|---:|---:|---|
| Resistor | 52 | 0.423 / 0.324 | 0.308 / 0.138 | Measured on source split |
| Capacitor | 59 | 0.678 / 0.618 | 0.322 / 0.116 | Measured on source split |
| Diode | 2 | 0.000 / 0.000 | 0.000 / 0.700 | Insufficient support |
| LED | 15 | 0.667 / 0.667 | 0.200 / 0.153 | Measured on source split |
| Integrated circuit | 32 | 0.750 / 0.628 | 0.688 / 0.375 | Measured on source split |
| Connector | 9 | 0.444 / 0.726 | 0.222 / 0.222 | Insufficient support |
| Switch/button | 6 | 0.833 / 0.976 | 0.500 / 0.383 | Insufficient support |
| Header | 8 | 0.625 / 0.658 | 0.125 / 0.263 | Insufficient support |
| Crystal | 4 | 0.250 / 0.250 | 0.000 / 0.063 | Insufficient support |
| Debug/test point | 9 | 1.000 / 1.000 | 0.222 / 0.678 | Insufficient support |

AP is confidence-ranked on outputs retained to a 0.001 floor. At confidence 0.40, the single-view F1 on this source split is 0.591 (P=0.612, R=0.571); this is a validation-split observation only and is not adopted as a production threshold without board-held-out reviewed labels.

## Limits and Phase gate

- Labels are source-supplied and not independently adjudicated; board IDs and split grouping are absent. Same-board leakage and board-level generalization cannot be determined.
- The dataset has boxes/classes only. Occlusion, manually reviewed small-component labels, physical component identity, multi-view identity, false merge/split, and reference-matching labels are unavailable.
- Tiled prediction duplicates and aggregate false positives are measurable, but identity fusion performance is not measurable without component identities.
- The source split is a development comparison, not the required validation/sealed-test structure. No accuracy claim is gated as complete.
- **Independent real-board recognition validation: BLOCKED** pending a reviewed, board-grouped dataset and a protected test split.

The previous Phase 12 record reported mAP@0.5 0.509 because its evaluation AP path applied the 0.25 operating threshold before constructing the PR curve. This Phase 21.3 evaluator retains low-confidence outputs to 0.001 for AP, so the recomputed mAP@0.5 is 0.585 while P/R at 0.25 remain 0.498/0.612. This is an evaluation-protocol correction, not a model improvement; the historical Phase 12 result is preserved in [PCB_AI_EVALUATION.md](PCB_AI_EVALUATION.md).
