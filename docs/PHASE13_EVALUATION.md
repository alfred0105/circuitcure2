# Phase 13 verification record

## Phase 12 hotfix

- The detector UI default resolves to repository-root `models/pcb_component_detector.onnx`, matching `scripts/prepare_pcb_detector.py`. The UI reports `READY` when the local ONNX file and runtime are usable; absent model artifacts remain `MODEL_UNAVAILABLE`. User-entered model paths remain supported.
- Corrected the editable-install command in PCB AI setup documentation to include `.`.
- The local ignored ONNX file was available for a real inference smoke check: the repository-root default returned `READY`; a public PCB image produced 4 detector candidates and OCR completed with one text line. This is a smoke check, not an accuracy estimate.
- The model-free CI path tests the default path without requiring or downloading model weights. An initial hotfix CI run exposed an overly strict test assumption; the test now allows both installed and absent model states while verifying the same path.
- The corrected hotfix commit passed GitHub Actions run [41578345586](https://github.com/alfred0105/CircuitCare-AI/actions/runs/41578345586).

## Fault map and What-if

- Component–Net graph and existing bounded path search feed a separate presentation payload. Scenario-matched risk report rows supply electrical risk; U1 demo inspection ranking supplies the posterior-weighted inspection view. These sources remain distinct.
- Selecting a component highlights the connectivity route and shows the path, shared Net names, hops, risk metrics, rating utilization, inspection score, and unknowns.
- The slider uses only `simulated_demo` input through existing diagnosis/Bayesian engines. It does not update actual measurement state or rerun SPICE. Model mismatch, next-measurement recommendation, posterior, and posterior-weighted inspection ranking update from the preview.
- At U1 VOUT=7.9 V, σ=0.2 V: `U1_OUTPUT_8V` posterior is effectively 1.0. C1 score changes 67.3267 → 98.93 while its rank stays 1; ranks do not change in the available candidate set.
- At U1 VOUT=2.0 V, σ=0.2 V: `U1_OUTPUT_3V` has the highest conditional weight, but all configured predictions are beyond the 4σ mismatch threshold; the UI shows a prominent mismatch warning rather than treating the top row as a diagnosis. C1's weighted score becomes 33.35; rank order remains unchanged.
- With VOUT consumed, this configured model returns no meaningful next measurement recommendation. The UI reports that result without inferring inspection is unnecessary.
- The existing C1 electrical `EXCEEDED` color remains red in the PCB overlay while `inspect #rank` is appended as a separate Bayesian priority annotation. Only already-confirmed Reference links can receive either overlay annotation.

## Local verification

- `.venv/bin/python -m pytest -q`: **44 passed** locally (Phase 12 baseline 39 plus 1 detector path regression and 4 Phase 13 visualization/What-if tests). Tests cover stress/connection separation, mismatch, posterior-score changes, saved-SPICE immutability, selected indirect path highlighting, and PCB electrical-color/inspection-priority separation.
- GitHub Actions run [41578966702](https://github.com/alfred0105/CircuitCare-AI/actions/runs/41578966702) passed on Python 3.11 after a CI-discovered f-string compatibility fix.
- Streamlit `AppTest`: overview tab, all existing tabs, VOUT slider 7.9→2.0 V, posterior change 8 V→3 V, mismatch warning state, reset, and preservation of the actual measurement list.
- The local What-if diagnosis/Bayesian calculation measured about **0.0049 seconds** in this AppTest run, excluding Streamlit rerender and visualization.

## Validation limits

- The map is a Component–Net connectivity visualization, not a schematic-layout reconstruction or directional propagation model.
- Bayesian scores are conditional inspection indices, not damage probabilities. A model mismatch invalidates the configured hypotheses as a good explanation; it does not supply a new fault hypothesis.
- Electrical stress remains the existing scenario-matched SPICE/rating evidence. What-if inputs never alter it.
- Real PCB design/photo Reference matching remains unverified because no matching physical board and KiCad source bundle is available.
