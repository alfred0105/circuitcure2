# CircuitCare AI PCB Component Recognition

This local web module provides PCB component location detection, component type classification, and review-only pin/pad candidates. It is packaged under CircuitCare AI as the selected browser UI (`apps/pcb-component-recognition/index.html`). It does not infer wire paths or electrical connections.

## Run locally

Python 3.11+ is recommended. Install PyTorch for the user's CUDA version first, then install the remaining packages:

```powershell
python -m pip install -r requirements.txt
$env:CIRCUITCARE_PCB_WEIGHTS = 'C:\path\to\pcb-model-weights'
python app_server.py --port 8768
```

Open `http://127.0.0.1:8768`. The weight root can also be the module's `weights` directory. Model files and sample PCB photos are intentionally excluded from Git; image upload remains available. Place optional sample photos under `data/samples/`.

## Required local checkpoints

For the main automated pipeline:

- `any_best.pt` — Parsa YOLO26s component locator.
- `external/pcb-ai-doctor/best_v12n.pt` — AI Doctor location supplement.
- `external/pcb-yolov8s-detection/best.pt` — Arshia YOLOv8s component classifier.
- `fine_best.pt` — Parsa fine classifier checkpoint loaded by the current server startup path.

Additional model-comparison features use `external/pcb-ai-doctor/best_v8n.pt`, `external/PCBInspect-AI/YoloV12-Medium-160-FineTuned.pt`, `external/nuts_best.pt`, and `external/paine2026/unet_trained.pth`.

Check each model source's license and usage terms before downloading or redistributing weights. Model checkpoints are not committed.

## What the output means

- Component boxes combine the YOLO26s locator and AI Doctor v12n proposals; Arshia classifies matching boxes. Pin/pad detections from the classifier are excluded from component types.
- After detection, use **박스 검수** to draw an added box, select and move a box on the image, disable or delete incorrect boxes, edit its class or pixel coordinates, and ask the type model to classify the selected crop. Apply the model suggestion only when it looks right; export the reviewed boxes to CSV when finished. Review edits stay in the browser session until exported.
- Pin and pad markers are candidates for visual review. Yellow is a low-confidence geometric prior; green is an image-derived pad candidate; magenta indicates a supported or observed prior. They are not verified electrical terminals.
- The page includes single-image and YOLO ZIP evaluation tools. Only annotated regions are scored; unlabeled image areas are not treated as negatives.

The model's benchmark results are tied to the displayed datasets and annotation sources. They are not a guarantee of performance on a new board.
