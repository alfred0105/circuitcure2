# PCB AI detector and OCR setup

This optional feature proposes PCB component boxes and silkscreen text locally. It does not infer electrical connections, confirm component identity, or diagnose damage. The electrical CLI and dashboard work without these extras.

## Install and prepare

```bash
python -m pip install -e ".[ui,vision,vision-ai,vision-ocr]"
python scripts/prepare_pcb_detector.py
streamlit run src/circuitcare/ui.py
```

The preparation script explicitly downloads a pinned checkpoint into the user model cache and invokes an external Ultralytics `yolo` exporter to create an ONNX model at `models/pcb_component_detector.onnx` by default. The ONNX path is ignored by Git. It does not install Ultralytics. To prepare the exporter in a separate environment, install a compatible Ultralytics release there and pass its executable:

```bash
python scripts/prepare_pcb_detector.py --exporter /path/to/yolo
```

Use `--weights` for a checkpoint already obtained and verified from the pinned source. The default output is in the user's application cache. `--output models/pcb_component_detector.onnx` places a local file under an ignored path; neither checkpoint nor exported model is committed. Runtime inference uses ONNX Runtime CPU. RapidOCR downloads/caches its ONNX assets through its own first-use behavior, then runs local inference. Both model workflows require connectivity for initial artifact acquisition unless the artifacts have already been cached.

The checkpoint is an 89,552,854-byte file with SHA-256 `ad21e33f0962e3a23119223e0f82ef7d189ef118a8d4b79d51ab62187612e93c`, pinned at source commit `0d7ae7310fa607389779b6132285336ad73e96e1`. Exported ONNX used for Phase 12 evaluation has SHA-256 `45a1dc54bb2eaff067edc631be30fee650deb1236e97a038610faa11aef28424`. The detector is YOLOv8s at a 960 px square input with these classes: resistor, capacitor, diode, LED, integrated circuit, connector, switch/button, header, crystal, and debug/test point.

## User review

- Select a rectified board image and run detection and OCR separately with the buttons in the PCB tab.
- A missing or invalid detector produces `MODEL_UNAVAILABLE`; no generated or contour-based result is substituted for a trained-model prediction.
- Boxes, class labels, model confidence, and OCR strings are unverified candidates. Low detector confidence is displayed as `unknown_component` while preserving the raw predicted class.
- The user can reject a box, correct its class, or add a manual box. The output preserves source, model class, correction, and confidence separately.
- OCR keeps raw text and only extracts literal Reference-shaped strings. It does not silently change `O/0`, `I/1`, or other characters.
- Position/OCR/type matching returns a heuristic ordering index, not a probability. Every component Reference must be manually confirmed. Risk color is available only when the existing scenario-matched report is current and both the board identity and individual Reference were confirmed.
- The original image is not saved by the workflow. Explicit result saving writes derived JSON and overlay artifacts, excluding the original image.

## Artifact and license provenance

Sources consulted for this implementation:

- [PCB component detection dataset/model repository](https://github.com/wangshnegqi/pcb-component-detection-dataset-yolo), pinned to commit `0d7ae7310fa607389779b6132285336ad73e96e1`.
- The repository declares its checkpoint/repository license as [Apache-2.0](https://github.com/wangshnegqi/pcb-component-detection-dataset-yolo/blob/main/LICENSE). Its image and annotation dataset has a separate [CC BY 4.0 license](https://github.com/wangshnegqi/pcb-component-detection-dataset-yolo/blob/main/LICENSE-DATASET). The model file is not redistributed here.
- The checkpoint was produced using Ultralytics tooling. Ultralytics publishes its framework under [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE); framework licensing and a publisher's separate weights declaration are distinct. CircuitCare does not bundle Ultralytics, invoke it at runtime, or include its code in the application package. The exporter is user-installed separately. Anyone redistributing the checkpoint, exporter, or a derivative should review the applicable terms and their own obligations.
- [RapidOCR source](https://github.com/RapidAI/RapidOCR) declares Apache-2.0. OCR model artifacts are fetched/managed by RapidOCR; review the current upstream model-artifact notices before redistributing those artifacts.

This attribution is provenance documentation, not a legal opinion. Third-party repositories and license declarations can change; this implementation pins the detector source commit and verifies the downloaded checkpoint size and digest.

## Limits

The detector is trained on a public PCB image dataset and may miss unfamiliar packages, tiny parts, occlusions, or unusual layouts. The score threshold is an engineering display policy, not a calibrated confidence boundary. OCR Reference candidates can be wrong, especially on small or silkscreen-poor images. A human must verify every design mapping. See [the evaluation record](PCB_AI_EVALUATION.md); its public validation split is not evidence that a CircuitCare board was independently validated.
