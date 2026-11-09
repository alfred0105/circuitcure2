# Frontend F5.1 E2E Validation

## Environment

- Browser: ChatGPT in-app browser at `http://127.0.0.1:8519/` (browser engine/version not exposed by the UI automation surface)
- OS: macOS 27.0.1, Apple Silicon development machine
- Python: 3.12.14 (`.venv`)
- Streamlit: 1.67.0
- Test target: F5.1 working tree based on `87759e7b7ad296dce0cbf01e093873f273856d5b`
- Fixture command: `PYTHONPATH=src .venv/bin/streamlit run tests/e2e/f5_measurement_workbench_app.py --server.headless true --server.address 0.0.0.0 --server.port 8519`

The fixture banner identifies its synthetic PCB and components as test-only and not real analysis. Browser checks below used direct pointer and form interactions in the visible app. The automation surface does not expose a pointer-only move, browser console logs, or viewport resizing, so those checks are explicitly marked unverified.

## Browser scenarios

| Scenario | Result | Evidence observed |
|---|---|---|
| Component click | PASS | Clicking the synthetic capacitor and resistor changed the right panel to the matching Component ID and type. |
| Hover tooltip | PARTIAL | Pointer entry during component clicks displayed the Korean tooltip. A hover-only action with no click or drag was not available through the browser controls, so selection independence was not proven in the browser. |
| Pan | PASS | Dragged the board from an empty area; the selected capacitor remained selected after the drag, with no clear event visible. |
| Zoom and hit test | PASS | Viewer displayed 100%, 156%, and 244% after toolbar controls. At 156% and 244%, clicking inside the visible overlay selected the capacitor correctly. |
| Resize | PARTIAL | The in-app browser viewport was 1280 × 720. No viewport-resize control was available, so 1920 × 1080 and 1440 × 900 remain unverified. |
| 0 V and negative voltage | PASS | Saved 0 V and -0.25 V; both appeared in the selected capacitor's history. |
| Unit conversion/display | PASS | Saved 3.3 mV on the resistor and the UI retained the entered value and unit in history. |
| Per-component history | PASS | Switched from resistor back to capacitor; the capacitor's two records remained and the resistor's record did not appear in its list. |
| Edit kind/unit and uncertainty | PASS | Changed a capacitor record from voltage to 3.3 kΩ resistance. The editor selected Ω-family units and cleared the old voltage σ. The resulting Measurement Set changed. |
| Uncertainty unit conversion | PASS | Automated regression confirms UI σ is entered in the measurement type's base unit, then converted for the selected input unit (e.g. 0.1 V ↔ 100 mV, 5 Ω ↔ 0.005 kΩ). |
| Delete one record | PASS | Deleted the edited resistance record; the separate 0 V record remained. |
| New project isolation | PASS | Saved a measurement in fixture project A, activated the test-only new-project control, selected a component in project B, and observed an empty measurement history. The diagnosis panel showed no stored diagnosis. |
| Diagnosis guard | PASS | With saved measurements but no electrical mapping/model/provider, the panel showed the blocking reasons and kept “진단 갱신” disabled. |
| Fault-state tooltip | PARTIAL | The no-diagnosis tooltip displayed “고장 진단 미실행.” Suspected/mismatch states are guarded in code and unit tests; no fake fault result was injected into the browser fixture. |
| Browser console | NOT TESTED | Console-log inspection was not exposed by the connected in-app browser controls. |

The app was visually observed at 1280 × 720. Screenshots appeared in the interactive browser session during verification but were not saved as repository files; therefore no screenshot asset paths are claimed here.

## Automated checks

```text
.venv/bin/python -m compileall -q src tests
.venv/bin/python -m pytest -q
```

Result: **432 passed, 23 skipped**. One existing test emits a `zipfile` duplicate-name warning in `test_phase18_workspace_reporting.py`.

Focused F5.1 regression run before the final full suite: **60 passed** across frontend state, measurement, diagnosis, PCB viewer, entry, and overlay adapter tests.

No separate JavaScript test runner or Streamlit AppTest count was run for this phase. The full Python suite includes the repository's existing app tests. Backend integration was not exercised because no board-specific diagnosis Provider/model/Node Mapping contract is connected in this frontend environment.

## Remaining browser verification

- Hover without selecting or dragging, including proving SelectionState does not change on hover alone.
- 1920 × 1080 and 1440 × 900 layout, overflow, overlay, and hit-test checks.
- Browser console error inspection.
- Fault-candidate and model-mismatch tooltip rendering with a verified test provider/result envelope.
