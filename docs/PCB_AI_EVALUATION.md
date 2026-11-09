# PCB AI evaluation record — Phase 12 (historical)

> **Historical result:** the Phase 12 mAP@0.5 value below used predictions truncated at the 0.25 operating threshold when constructing the PR curve. Phase 21.3 re-evaluated the same checkpoint with predictions retained down to confidence 0.001 for AP: mAP@0.5 = 0.585, mAP@0.5:0.95 = 0.334, while precision/recall at 0.25 remain 0.498/0.612. This is a metric-protocol correction, not a model change. See [Phase 21.3 benchmark results](PHASE21_3_BENCHMARK_RESULTS.md).

## Detector

| Item | Recorded result |
|---|---|
| Model | Third-party YOLOv8s PCB detector, 960 px input |
| Source commit | `0d7ae7310fa607389779b6132285336ad73e96e1` |
| Checkpoint SHA-256 | `ad21e33f0962e3a23119223e0f82ef7d189ef118a8d4b79d51ab62187612e93c` |
| Evaluated ONNX SHA-256 | `45a1dc54bb2eaff067edc631be30fee650deb1236e97a038610faa11aef28424` |
| Runtime | ONNX Runtime CPU, macOS / Apple Silicon M5, Python 3.12.14 |
| Dataset | Source repository's `val` split, 13 images, 196 labeled objects |
| Matching rule | Class-aware IoU matching at 0.50; confidence threshold 0.25 for precision/recall |
| Precision / recall | 0.498 / 0.612 at confidence threshold 0.25 |
| mAP@0.5 | 0.509 |
| First call | 2,687 ms including session/model load and first inference |
| Steady per-image CPU inference | Mean 1,285 ms; median 1,305 ms (13 image loop after warmup) |

Per-class AP@0.5:

| Class | Labeled objects | AP@0.5 |
|---|---:|---:|
| Resistor | 52 | 0.280 |
| Capacitor | 59 | 0.609 |
| Diode | 2 | 0.000 |
| LED | 15 | 0.667 |
| Integrated circuit | 32 | 0.600 |
| Connector | 9 | 0.333 |
| Switch/button | 6 | 0.833 |
| Header | 8 | 0.522 |
| Crystal | 4 | 0.250 |
| Debug/test point | 9 | 1.000 |

These metrics are from the source-supplied validation split. Its board-level split provenance was not established, and augmented or related board views may occur across subsets. Therefore they are **not** an independent board-level generalization estimate and do not measure CircuitCare Reference matching, physical registration, electrical risk ranking, or damage detection. The small diode count also makes its class AP especially unstable. Runtime is a single local machine observation, not a performance guarantee.

The evaluation can be repeated with the optional ONNX model and the dataset's YOLO images/labels:

```bash
python -m pip install -e ".[vision,vision-ai]"
python scripts/evaluate_pcb_detector.py \
  --model models/pcb_component_detector.onnx \
  --images /path/to/dataset/images/val \
  --labels /path/to/dataset/labels/val \
  --output /tmp/circuitcare-pcb-evaluation.json
```

The script's AP summary is a compact implementation for this fixed evaluation, not a replacement for a benchmark suite. It computes class-wise average precision at IoU 0.5, plus aggregate precision/recall at the configured confidence cutoff.

## OCR and example inference

RapidOCR / PP-OCR ONNX was run locally on one 2304×4096 image from the same public PCB dataset. The first process call returned six text lines in about 5.3 seconds including lazy runtime/model initialization; one literal `U26` Reference-shaped candidate was extracted. There are no OCR ground-truth transcripts or Reference annotations in this sample, so character accuracy, line recall, Reference precision/recall, and steady-state timing are **not measured**. This is an inference smoke check only.

The detector also ran on that actual public PCB image and returned 16 component candidates. This verifies that the selected model artifact can execute and produce boxes; it is not an accuracy result for that image because an independent per-image review was not recorded. No prediction asserts electrical connectivity, physical damage, or a damage probability.

## CircuitCare design matching

No CircuitCare real-board bundle containing a matching PCB photo, `.kicad_pcb` / Position CSV, and XML netlist was available. As a result, the following are **NOT VERIFIED**:

- Detector-to-KiCad Reference matching quality on a CircuitCare board.
- Real-board homography accuracy and independent physical validation points.
- Whether an image/CSV/netlist identity assertion accurately describes a physical board.
- Risk overlay placement accuracy on a real board.

Unit tests verify score ordering, ambiguity handling, explicit per-detection user confirmation, duplicate Reference prevention, and scenario-gated overlay mechanics. Those tests do not substitute for the missing real-board data. Phase 13 can continue; real design-matching validation remains a separate data-dependent validation task.
