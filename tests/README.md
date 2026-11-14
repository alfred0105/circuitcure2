# Test suites

The test jobs are split by what they prove. Test files and fixtures are kept in
the repository; marking a suite does not remove its coverage.

## Current PCB workbench

The default Python job checks the Streamlit photo intake/workbench, component
detection and review, local OCR/Qwen request handling, overlays, measurements,
diagnosis state, and supporting code. `test_ui_soak.py` remains in the separate
`stress` job because it runs repeated full workbench transitions.

## Optional circuit regressions

`test_ui_hotfix_helpers.py`, `test_fault_ai.py`, and `test_netlist.py` are marked
`optional_circuit`. They cover the older simulation-oriented helper UI and
optional SPICE/KiCad circuit-analysis paths. The `optional-circuit-regression`
job runs them separately so their failures are reported apart from the current
PCB photo workbench; these tests are still required and are not deleted.

Run locally with:

```bash
python -m pytest -q -m optional_circuit
```

## Native integration and fault isolation

Tests requiring ngspice, ONNX/OCR runtime support, stress loops, or isolated
crash injection keep their existing `integration`, `stress`, and
`crash_injection` markers and CI jobs.

## Model performance

`test_pcb_detection_parity.py` is marked `model_performance`. It requires a
Windows reference checkout, the local YOLO weights, and a labeled comparison
photo, so it is not run in hosted CI where those private assets are absent.
Run it locally with:

```bash
CIRCUITCARE_WINDOWS_REPO=/path/to/CircuitCare-AI \
CIRCUITCARE_PCB_WEIGHTS=/path/to/pcb_component_pipeline \
CIRCUITCARE_PARITY_IMAGE=/path/to/pcb-photo.jpg \
python -m pytest -q -m model_performance
```

The test does not download or upload model weights.
