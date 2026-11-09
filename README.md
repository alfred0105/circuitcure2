# CircuitCare AI

**Local, lightweight AI-assisted circuit fault propagation analysis and repair prioritization.**

CircuitCare AI aims to answer a practical repair question: **if one component has failed, which *other* components might have been affected, why, and what should be checked next?**

> **Status: integrated local prototype; Phase 21.0 audit is recorded and Phase 21.0.1 roadmap alignment is complete, with responsive browser verification still open.** Existing capabilities include KiCad/netlist analysis, supported SPICE scenarios, stress/risk review, conditional diagnosis and measurement suggestions, schematic and PCB candidate views, and evidence-grounded reports. The three competition research claims are electrical connectivity reconstruction, physics-informed fault reasoning, and active-measurement efficiency. No physical repair accuracy or real-board electrical connectivity performance has been established. Risk levels, detector scores, heuristic matches, and Bayesian weights do **not** establish damage or real-world failure probability.

### Validation at a glance

- Synthetic fault benchmark: 10 cases / 5 topology groups; repeated runs are not independent cases. Rules exact precision .0495 and recall 1.0; LLM exact precision/recall 0; combined methods showed no exact-diagnosis gain over Rules. See [Phase 17 results](docs/PHASE17_EVALUATION.md) and [Phase 17.1 correction](docs/PHASE17_1_EVALUATION_HOTFIX.md).
- PCB detector: source-provided validation split, 13 images / 196 boxes; precision .498, recall .612, recomputed mAP@.5 .585 (AP uses a 0.001 prediction floor). Board split independence and reviewed labels are not verified; this is not a board-level accuracy claim. See [Phase 21.3 benchmark](docs/PHASE21_3_BENCHMARK_RESULTS.md).
- PRISM: 12 boards / 24 sides passed through image processing; no image-to-design calibration, valid pixel accuracy metrics, or electrical connectivity validation. See [PRISM benchmark](docs/PHASE20_1_1_PRISM_BENCHMARK.md).
- Schematic pin-link prediction: 4,948 graphs with project-grouped splits; on the fixed test set the LightGBM baseline reached PR-AUC .2785 and end-to-end Recall@5 .1131 under the candidate graph's truth-pair-weighted recall of .648. Ranking and training-sampling experiments are documented in [Phase 22 connectivity results](docs/PHASE22_CONNECTIVITY_DATASET.md#ranking-hard-negative-net-aware-and-ensemble-experiments).
- Active-measurement policy benefit: `NOT_EVALUATED`; expected-information-gain selection exists, but no matched Random / Rule / EIG efficiency experiment exists.

### Read order

1. [Project overview and setup](README.md)
2. [System architecture](docs/ARCHITECTURE.md)
3. [Evaluation methodology](docs/BENCHMARK_METHODOLOGY.md) and [Phase 21 evaluation strategy](docs/PHASE21_EVALUATION_STRATEGY.md)
4. [Current validation results](docs/PHASE21_0_TECHNICAL_AUDIT.md)
5. [Phase 21 roadmap](docs/PHASE21_ROADMAP.md)

Phase 21 records: [technical audit](docs/PHASE21_0_TECHNICAL_AUDIT.md), [project scope](docs/PHASE21_0_PROJECT_SCOPE.md), [evaluation strategy](docs/PHASE21_EVALUATION_STRATEGY.md), [roadmap](docs/PHASE21_ROADMAP.md).

### Phase 18.1.1 stability work

Graph traversal now enforces global queue, expanded-state, result, depth, and optional time budgets; partial results are labelled. The dashboard selects one analysis area per rerun. PDF and PCB image work uses bounded content-based caches. ngspice result files are parsed as streams under file, sample, and vector ceilings. Crash diagnostics and test tiers are described in [Crash Diagnostics](docs/CRASH_DIAGNOSTICS_GUIDE.md), [Stress Tests](docs/STRESS_TEST_GUIDE.md), [Full Code Stability Audit](docs/FULL_CODE_STABILITY_AUDIT.md), and [Phase 18.1.1 results](docs/PHASE18_1_1_CRASH_STABILITY.md).

Run the fast suite with `python -m pytest -q -m "not integration and not stress"`, native integrations with `python -m pytest -q -m integration`, and synthetic stress coverage with `python -m pytest -q -m stress`. The previously reported Python termination remains unattributed because no crash log was supplied.

## Current prototype and target product

**Current prototype:** the user selects a circuit and a supported fault component/mode. CircuitCare evaluates candidate connectivity, runs supported normal-vs-fault SPICE analyses, compares stress with configured ratings, and ranks conditional Bayesian hypotheses and measurement candidates. PCB Vision and multiview reconstruction are partial; their electrical connectivity accuracy has not been verified on a CircuitCare physical board.

**Target product:** start with multiview PCB photos and a natural-language symptom, even when the user does not know the failed component. The system will structure symptoms into module and fault candidates, analyze root cause separately from possible secondary stress, recommend measurements, accept measured values, and guide repair. When a net or cause is uncertain, it must ask for design data, user confirmation, or additional measurements, or show explicitly limited candidates. It must not present an LLM hypothesis, image-derived contact, simulated rating stress, or conditional Bayesian score as a confirmed failure.

## Roadmap

| Phase | Deliverable | State |
|---|---|---|
| 01 | Python environment, conventions, sample data | Complete |
| 02 | KiCad netlist parser | Complete; verified with KiCad 10.0.6 export |
| 03 | Component-net graph and affected-neighbor candidates | Complete |
| 04 | Failure modes and fault injection models | Complete; simulation not run |
| 05 | ngspice baseline/fault electrical comparisons | Complete; supported demo circuit verified |
| 06 | Secondary-damage stress assessment and evidence | Complete; overvoltage demo flags C1 above its declared demo rating |
| 07 | Local lightweight AI and active measurement ranking | Complete; simulated measurement updates hypothesis posterior |
| 08 | Evidence-grounded repair recommendations and audit trail | Complete; local Ollama explanation plus deterministic fallback |
| 09 | UI, benchmark, offline evaluation | Complete; divider and RC live runs verified in the dashboard, with fault-scoped session state and accumulated Bayesian measurements |
| 10 | Stabilization, poster, reproducible demo | Complete; evidence and limits recorded |
| 11 | PCB photo input, manual registration, evidence-based risk overlay | Complete; synthetic fixture and KiCad CSV format verified, real board not verified |
| 12 | Local PCB component detection, silkscreen OCR, design-aware matching | Complete; public validation split evaluated, matching CircuitCare board unavailable |
| 13 | Interactive influence map and Bayesian What-if dashboard | Implemented; U1 virtual interaction and stress-data separation verified locally |
| 14 | Local datasheet extraction and evidence-gated rating review | Implemented; one YAGEO PDF and local Ollama smoke-checked; broad accuracy and physical Reference match not evaluated |
| 15 | Local AI fault hypotheses, physics verification, separate evaluation tracks | Implemented; local Gemma and ngspice exercised on three synthetic development cases; datasheet results are a development pilot after parser feedback |
| 16 | Verified repair history, controlled prior adaptation, longitudinal evaluation | Implemented; synthetic workflow verified; no real repair records, active learned prior, or independent improvement evaluation |
| 17 | Independent benchmark, fair baselines, statistical validation, efficiency | Synthetic benchmark complete; see Phase 17 evaluation and Phase 17.1 metric correction |
| 17.1 | Metric integrity, off-grid evaluation, SPICE identifiability | Complete; corrected reused-holdout scoring and off-grid parameter evaluation on reused topology templates |
| 18 | Unified workspace, guided workflow, evidence-grounded reports | Implemented; cached synthetic report flow verified; physical repair validation not evaluated |
| 18.1 | Diagnostic context and evidence integrity | Implemented; see [verification record](docs/PHASE18_1_RELIABILITY_HOTFIX.md) |
| 19–20 | Schematic workbench, multi-view PCB pipeline, PRISM adapter and segmentation candidates | Partially implemented; real-board registration and electrical reconstruction remain unverified |
| 21.0 | Scope, technical/evaluation audit and evidence inventory | Audit and docs pushed; no AI accuracy improvement claimed |
| 21.0.1 | Roadmap alignment, actual browser UI verification, final documentation | Roadmap/docs aligned; narrow IAB interaction spot-check only. Target viewport screenshots and full browser E2E remain open; see [final alignment record](docs/PHASE21_0_1_FINAL_ALIGNMENT.md) |
| 21.1 | PCB data and photo↔KiCad registration | Backend service and 24-view PRISM regression complete; 16 estimates remain unverified, independent validation is blocked; see [21.1.2 report](docs/PHASE21_1_2_REGISTRATION_BACKEND.md) |
| 21.2 | Multi-image PCB component mapping | Same-side candidate registration, mosaic, partial component map, and fault overlay contract implemented — see [Phase 21.2 report](docs/PHASE21_2_MULTI_IMAGE_COMPONENT_MAPPING.md) |
| 21.2.1 | Component map stabilization | Stable identity manifests, conservative auto-rectification, valid-pixel coverage, EXIF-aware coordinate provenance, and local ONNX pipeline smoke verified; physical multiview validation and accuracy remain blocked — see [stabilization report](docs/PHASE21_2_1_COMPONENT_MAP_STABILIZATION.md) |
| 21.3 | Component recognition and evaluation | Evaluator, annotation contract and native-resolution tiled comparison implemented; tiled mode did not improve aggregate metrics. Reviewed, board-grouped accuracy validation remains blocked — see [Phase 21.3 report](docs/PHASE21_3_COMPONENT_RECOGNITION.md) |
| 21.3.1 | Recognition reliability and board-level dataset validation | Invalid detections fail closed; evaluation provenance is manifest-backed; source dataset audited. Reviewed GT, board IDs and sealed test remain blocked — see [Phase 21.3.1 report](docs/PHASE21_3_1_RECOGNITION_RELIABILITY.md) |
| 21.4 | Functional-block inference | Planned; reviewed block labels and component evidence required |
| 21.5 | Symptom-driven fault localization | Planned; symptom catalog and evidence-ranked component/block hypotheses |
| 21.6 | Physics-informed fault diagnosis | Planned; independent circuit families and supported SPICE cases |
| 21.7 | Active measurement policy evaluation | Planned; paired Random/Rule/EIG cases and fixed stopping criteria |
| 21.8 | Measurement feedback and secondary-fault reasoning | Planned; update candidate ranking from measurements while preserving evidence boundaries |
| 21.9 | Frontend/backend integration | Planned; connect photo map, symptom hypotheses, measurements and fault overlays |
| 21.10 | Real-fault evaluation and competition release | Planned; sealed board-level component evaluation and independently reviewed fault cases |
| 22 | Schematic connectivity dataset | 5,046 unique KiCad schematics acquired; 4,957 wire-free/original pairs generated with 19-class labels. Full-dataset scaling, mapping review, and image-rendered input validation remain open — see [Phase 22 dataset record](docs/PHASE22_CONNECTIVITY_DATASET.md) |

Historical phase acceptance criteria: [docs/ROADMAP.md](docs/ROADMAP.md). Current direction and gates: [Phase 21 roadmap](docs/PHASE21_ROADMAP.md). Phase 21.1 artifacts and limits: [dataset foundation](docs/PHASE21_1_DATASET_FOUNDATION.md), [coordinate registration](docs/PHASE21_1_COORDINATE_REGISTRATION.md), [ground-truth validation](docs/PHASE21_1_GROUND_TRUTH_VALIDATION.md), [results/readiness](docs/PHASE21_1_FINAL_REPORT.md), [21.1.1 automatic match benchmark](docs/PHASE21_1_1_BENCHMARK_RESULTS.md), [21.1.2 registration backend](docs/PHASE21_1_2_REGISTRATION_BACKEND.md), and [21.2 multi-image component mapping](docs/PHASE21_2_MULTI_IMAGE_COMPONENT_MAPPING.md), [backend interface](docs/PHASE21_2_BACKEND_INTERFACE.md), [benchmark results](docs/PHASE21_2_BENCHMARK_RESULTS.md), [21.2.1 stabilization](docs/PHASE21_2_1_COMPONENT_MAP_STABILIZATION.md), [backend contract](docs/PHASE21_2_1_BACKEND_CONTRACT.md), and [benchmark results](docs/PHASE21_2_1_BENCHMARK_RESULTS.md). Phase 21.3 artifacts: [recognition report](docs/PHASE21_3_COMPONENT_RECOGNITION.md), [dataset/evaluation protocol](docs/PHASE21_3_DATASET_AND_EVALUATION.md), [benchmark results](docs/PHASE21_3_BENCHMARK_RESULTS.md), and [backend contract](docs/PHASE21_3_BACKEND_CONTRACT.md). Phase 21.3.1 artifacts: [reliability report](docs/PHASE21_3_1_RECOGNITION_RELIABILITY.md), [dataset audit](docs/PHASE21_3_1_DATASET_AUDIT.md), [evaluation validation](docs/PHASE21_3_1_EVALUATION_VALIDATION.md), and [clean baseline results](docs/PHASE21_3_1_BENCHMARK_RESULTS.md). Phase 22: [dataset pairing specification and pilot](docs/PHASE22_CONNECTIVITY_DATASET.md).

## Quick start

Requires Python 3.11+ (Python 3.11 is the CI baseline; Python 3.12 also works).

```bash
python -m pip install -e ".[dev,ui,vision,datasheet]"
circuitcare --netlist tests/fixtures/simple_kicad_netlist.xml --fault U1
python -m pytest -q
streamlit run src/circuitcare/ui.py
```

The [Phase 18 User Guide](docs/USER_GUIDE.md) walks through the dashboard. Report examples are in [`examples/phase18_demo/`](examples/phase18_demo/); regenerate them with `python scripts/build_phase18_demo_report.py`.

CI는 Python 3.11에서 `dev`, `vision`, `datasheet` 테스트 의존성을 설치합니다. PDF 테스트는 공식 데이터시트나 LLM 모델을 다운로드하지 않고 합성 PDF로 실행됩니다. Vision은 실행 시 선택 기능이지만, 현재 전체 테스트 수집에는 NumPy/OpenCV/Pillow가 필요합니다.

For the integrated dashboard, install the optional UI dependency and start Streamlit:

```bash
python -m pip install -e ".[ui]"
streamlit run src/circuitcare/ui.py
```

To enable the PCB component detector and editable box review inside the PCB fault-diagnosis screen, install its local model runtime too:

```bash
python -m pip install -e ".[ui,pcb-recognition]"
streamlit run src/circuitcare/ui.py
```

After adding a PCB photo, start the diagnosis workspace, open **PCB**, then expand **소자 탐지 · 박스 검수** under the selected photo. Detection boxes and labels can be reviewed, excluded, edited, added, and classified again inside the selected box. Review changes remain in the current session and can be exported as CSV; they do not alter training labels.

To enable the **데이터시트 AI** tab's local PDF text/table extraction, install the optional PDF readers too:

```bash
python -m pip install -e ".[ui,datasheet]"
streamlit run src/circuitcare/ui.py
```

The tab uses `pypdf` and `pdfplumber`; local LLM structuring reuses the loopback-only Ollama integration. Candidate values are never applied until their cited source, MPN-to-Reference mapping, conditions, and human approval are confirmed. See [Datasheet AI guide](docs/DATASHEET_AI_GUIDE.md) and [Phase 14 evaluation](docs/PHASE14_EVALUATION.md).

To enable the optional local PCB photo workflow (OpenCV, NumPy, and Pillow), install the vision extra too:

```bash
python -m pip install -e ".[ui,vision]"
streamlit run src/circuitcare/ui.py
```

The **PCB 사진 인식** tab can run optional local detector/OCR candidates and, separately, align a photo to KiCad Component Position CSV using a manually confirmed homography. Risk is attached to a detection only after the user confirms the matching KiCad reference and board identity. See the [PCB Vision guide](docs/PCB_VISION_GUIDE.md). The current detector benchmark on 13 source validation images / 196 boxes reports mAP@0.5 0.585 with a 0.001 AP prediction floor. Source labels are unreviewed and board IDs are absent, so this is not board-independent or CircuitCare design-matching validation. No matching real-board photo/design data is available. The Phase 11 CI hotfix passed on Python 3.11 in [GitHub Actions](https://github.com/alfred0105/CircuitCare-AI/actions/runs/41570036079) (29 tests).

The standalone local **CircuitCare AI · PCB 소자 인식** browser UI is packaged in [`apps/pcb-component-recognition/`](apps/pcb-component-recognition/README.md). It runs the selected YOLO26s locator + AI Doctor supplement + Arshia type classifier, then shows visual pin/pad candidates. Start it with the module README instructions; model weights stay outside Git and are loaded from `CIRCUITCARE_PCB_WEIGHTS`.

Optional AI setup: `python -m pip install -e ".[ui,vision,vision-ai,vision-ocr]"`, then `python scripts/prepare_pcb_detector.py`. The model exporter is installed separately by the user and is not part of the runtime package. See [setup and artifact provenance](docs/PCB_AI_DETECTION.md) and [evaluation limits](docs/PCB_AI_EVALUATION.md).

The **분석 개요** tab provides a Component–Net fault influence candidate map with separate connectivity, electrical-stress, and conditional inspection-priority views. The **AI 고장 가설 분석** tab proposes candidates from the explicit fault catalog and local Ollama, validates each proposal against circuit/SPICE mappings, and runs ngspice only after explicit user selection. Simulation fit applies only to the configured model and observations. U1 What-if controls recalculate Bayesian outputs from a `simulated_demo` virtual measurement; they do not change actual measurement records or rerun SPICE. See [Fault Hypothesis AI](docs/FAULT_HYPOTHESIS_AI.md), [Phase 15 evaluation](docs/PHASE15_EVALUATION.md), [Datasheet benchmark](docs/DATASHEET_EXTRACTION_BENCHMARK.md), the [fault map guide](docs/FAULT_MAP_GUIDE.md), and [What-if guide](docs/WHAT_IF_DIAGNOSIS.md).

The dashboard has circuit/fault, SPICE, rating risk, Bayesian diagnosis, local LLM, PCB photo, datasheet review, AI fault hypothesis, **수리 이력 및 AI 학습**, and **AI 성능 평가** tabs. The benchmark tab reads saved runs only and does not launch LLM or SPICE evaluation. See [Benchmark Methodology](docs/BENCHMARK_METHODOLOGY.md). Repair data stays in the ignored local `data/private/repair_feedback/repair_cases.sqlite3` database; it is not uploaded to GitHub. See [Repair Feedback guide](docs/REPAIR_FEEDBACK_GUIDE.md), [Prior Learning guide](docs/PRIOR_LEARNING_GUIDE.md), and [Phase 16 evaluation](docs/PHASE16_EVALUATION.md). It opens the saved 5 V→8 V regulator demo by default; the divider and RC charging examples can also be run with ngspice. Simulation status distinguishes `CACHED`, `RUNNING`, `SUCCESS`, `FAILED`, and `NOT_RUN`; failed reruns do not keep a previous live result active. User measurements and simulated demo measurements are distinguished. The saved U1 3/6/8 V Bayesian set is shown only in its matching regulator demo context; it is identified as stored scenario evidence, separate from the selected live run. Ollama is called only after an explicit user action. Without Ollama, deterministic advice and rule-based fault/datasheet candidates remain available.

For an offline presentation, install the package, ngspice, and (if desired) an Ollama model before disconnecting from the internet. Ollama models are not committed to this repository. Run `python scripts/demo_check.py` before the event to verify local prerequisites and cached data without repeating circuit simulations.

### Fixed demonstration sequence

1. **U1 overvoltage:** select the default cached regulator case. Compare 5 V baseline and 8 V fault, inspect C1 at 8 V / 6.3 V = 1.2698 utilization (`EXCEEDED`) and RLOAD at 16 mW / 125 mW = 0.128 (`WITHIN_LIMIT`). These are configured demonstration ratings, not datasheet-verified values.
2. **Bayesian example:** use the saved `simulated_demo` VOUT 7.9 V ± 0.2 V observation; the conditional 8 V hypothesis weight becomes approximately 1.0. This is not an instrument reading or a real failure probability.
3. **Independent resistor divider:** R1 changes from 1 kΩ to 10 kΩ; the saved/live ngspice VOUT changes from 2.5000 V to 0.4545 V.
4. **Independent RC charging:** RCHARGE changes from 10 kΩ to 100 kΩ; the committed transient has 545 samples across 0–50 ms, with VC at the final sample changing from 0.24325 V to 0.02488 V.

Stored results can be displayed when ngspice is unavailable. `scripts/demo_check.py` reports ngspice and Ollama availability separately and always identifies the deterministic fallback. See [the five-minute script](docs/DEMO_SCRIPT.md), [the release checklist](docs/RELEASE_CHECKLIST.md), and [poster content](docs/POSTER_CONTENT.md). To regenerate the poster figure, install the optional plotting extra with `python -m pip install -e ".[poster]"` and run `python scripts/make_poster_figures.py`.

To inspect parsed component metadata and pin-to-net connections, add `--show-pins`:

```bash
circuitcare --netlist tests/fixtures/simple_kicad_netlist.xml --fault U1 --show-pins
```

To list direct and indirect connectivity paths (two component hops by default), use `--show-paths`. Export the bipartite graph as DOT with `--export-dot`:

```bash
circuitcare --netlist tests/fixtures/simple_kicad_netlist.xml --fault U1 --show-paths --max-hops 2
circuitcare --netlist tests/fixtures/simple_kicad_netlist.xml --fault U1 --export-dot circuit_graph.dot
```

Net types are unknown unless explicitly supplied in a JSON mapping such as `{"GND":"ground","VIN":"power","VOUT":"power","SIGNAL":"signal"}` using `--net-types net-types.json`. `--exclude-net-types ground` filters ground paths from path traversal while retaining them in the graph and direct connectivity results. This filter does not imply that a real effect cannot travel through GND.

Generate a faulted SPICE circuit using the explicit KiCad-to-SPICE mapping and starter circuit:

```bash
circuitcare --netlist tests/fixtures/fault_demo_netlist.xml --fault R1 \
  --fault-type resistance_drift --fault-value 10k \
  --spice-base data/circuits/fault_demo.cir \
  --spice-mapping data/circuits/fault_demo_mapping.json \
  --output data/fault_scenarios
```

This writes `baseline.cir`, a faulted `.cir`, and `scenarios.json`. Supported scenarios include resistor open/short/drift, ideal regulator output rise/drop/loss, and mapped voltage-controlled switch stuck-open/stuck-on. These are stated approximations; scenario generation does not run a simulator. The committed `data/fault_scenarios/` files show resistor open, short, and 10 kΩ drift examples.

To run both baseline and faulted decks with ngspice and compare probe data, add `--simulate` and `--probes`:

```bash
circuitcare --netlist tests/fixtures/fault_demo_netlist.xml --fault R1 \
  --fault-type resistance_drift --fault-value 10k \
  --spice-base data/circuits/fault_demo.cir \
  --spice-mapping data/circuits/fault_demo_mapping.json \
  --output data/fault_scenarios/example_001 \
  --simulate --probes data/circuits/fault_demo_probes.json
```

ngspice must be installed (`brew install ngspice` on macOS). The runner saves logs and CSV samples; comparison and component stress summaries are JSON. The example uses a 5 V source and two 1 kΩ resistors: LOAD measures 2.5 V at baseline and about 0.455 V with R1 at 10 kΩ. Resistor current/power use the mapped terminals and actual netlist resistance; ideal voltage-source current comes from ngspice. Unsupported switch current/power are left null. These are simulated electrical changes, not damage or rating assessments.

The included sample intentionally connects U1 to J1, U2 and C3; R4 is present as a non-directly-connected comparison. A shared net is **a candidate influence path, not evidence of damage**. The CLI reports the selected failed component, total component count, direct candidates, shared nets, and connectivity-only evidence level.

To compare fault-case stress against explicit component ratings and create `risk_assessment.json`, add `--ratings` and `--assess-risk`:

```bash
circuitcare --netlist tests/fixtures/fault_demo_netlist.xml --fault U1 \
  --fault-type regulator_output_rise --fault-value 8 \
  --spice-base data/circuits/fault_demo.cir \
  --spice-mapping data/circuits/fault_demo_mapping.json \
  --output data/fault_scenarios/overvoltage_demo \
  --simulate --probes data/circuits/fault_demo_probes.json \
  --ratings data/component_specs/fault_demo_ratings.json --assess-risk
```

The demo raises the ideal 5 V output to 8 V. C1 is measured at 8 V against an explicitly labeled, unverified 6.3 V demo assumption (utilization 1.27, `EXCEEDED`). RLOAD dissipates 16 mW against a 125 mW demo assumption (`WITHIN_LIMIT`). Those rating values are illustrative and are not verified against component datasheets. The priority score is an explainable inspection ordering index, never a damage probability. Missing ratings or probes remain `UNKNOWN`; Q1 is excluded from physical ranking because its fixture pins do not map consistently to the SPICE switch terminals. Capacitor current, ESR loss, and heating are not calculated.

Run prior-only diagnosis from stored SPICE and risk JSON, then rerun with a measurement:

```bash
circuitcare-diagnose --hypotheses data/diagnosis/demo_hypotheses.json \
  --output data/diagnosis/prior_results
circuitcare-diagnose --hypotheses data/diagnosis/demo_hypotheses.json \
  --measurements data/diagnosis/demo_measurements.json \
  --output data/diagnosis/posterior_results
```

The three hypotheses reference saved 3 V, 6 V, and 8 V ngspice results; predicted node voltages are loaded from each `comparison.json`, not embedded in the diagnosis code. Priors are equal weights, and voltage measurement σ=0.2 V is a labeled demo assumption. The example 7.9 V observation is marked `simulated_demo`; it moves the 8 V hypothesis from 1/3 prior weight to approximately 1.0 conditional posterior. Before measurement, VOUT has the highest calculated information gain; after measuring VOUT it is excluded, and the remaining LOAD prediction is not informative enough to recommend another measurement. Outputs contain posterior history, measurement recommendations, and prior/posterior weighted inspection indices. These conditional values are not real-world failure probabilities, and measurement suggestions do not authorize live probing.

For evidence-backed repair guidance, install Ollama separately and pull a local model (for example, `ollama pull qwen3:4b-instruct`). The advisor never sends data to a cloud API. It checks the local Ollama model list, uses only evidence JSON from the diagnosis, inspection ranking, measurement recommendation, and optional risk/stress reports, then writes `repair_advice.json`, `evidence.json`, and `llm_explanation.json`:

```bash
circuitcare-advise \
  --diagnosis data/diagnosis/posterior_results/diagnosis.json \
  --inspection-ranking data/diagnosis/posterior_results/inspection_ranking.json \
  --measurement-recommendations data/diagnosis/posterior_results/measurement_recommendations.json \
  --model qwen3:4b-instruct \
  --question 'C1을 왜 먼저 점검해야 해?' \
  --output data/repair_advice/example
```

Use `--no-llm` to produce deterministic isolation, inspection, replacement, and post-repair checks without model inference. An installed local model can be selected with `--model`; Phase 08 was exercised with `gemma4:12b` on Ollama 0.40.3. Ollama 0.40.3 does not support structured output for this model, so the client retries once in plain-text mode and validates the returned JSON and evidence references locally. Model explanations are rejected on invalid or unsupported claims, leaving the deterministic advice available. Output labels simulation/demo assumptions and unknowns separately; the model is not allowed to assert damage or convert conditional posterior weights into real failure probabilities.

## Current implementation and next phase

Phases 01–10 provide the electrical-analysis workflow; Phase 11 adds manual PCB photo alignment, Phase 12 adds optional local detector/OCR review, Phase 13 adds interactive visualization and an isolated What-if preview, Phase 14 adds locally processed datasheets with evidence-gated rating review, and Phase 15 adds local fault proposals with physics checks. Electrical evaluation is in [the final evaluation record](docs/FINAL_EVALUATION.md), with the [Phase 09 summary](data/evaluations/phase09_summary.json); run `python scripts/evaluate.py` to regenerate the latter. PCB model evidence and limits are in [the Phase 12 evaluation](docs/PCB_AI_EVALUATION.md); datasheet evidence and limits are documented separately in [Phase 14 evaluation](docs/PHASE14_EVALUATION.md) and [the Phase 15 benchmark](docs/DATASHEET_EXTRACTION_BENCHMARK.md). Evaluation and timing values describe the recorded local environment and are not general performance guarantees. Only explicitly supplied ratings and supported measurements are evaluated. Posterior weights are conditional on the configured hypothesis set and do not estimate real-world failure probability or confirm damage.

The figures in [the poster content](docs/POSTER_CONTENT.md) are generated from committed JSON/CSV evidence by `python scripts/make_poster_figures.py`. Evaluation is limited to two independent small circuit topologies and a separate regulator Bayesian demo. There are no independent damaged-part labels, so damage-detection accuracy, Top-k damage coverage, and avoidable inspections are not claimed. The RSS reported by the evaluation excludes ngspice and Ollama child processes.

Phase 22 builds pin-level link-prediction graphs from the 4,957 wire-removed/original schematic pairs. It uses cKDTree for nearby cross-component pin candidates, NetworkX for pin nodes and candidate-link edges, and compares a compact GNN-ACLP-inspired model against LightGBM on project-grouped splits. See [the dataset and learning-curve results](docs/PHASE22_CONNECTIVITY_DATASET.md), [the experiment report](data/benchmarks/phase22_connectivity/learning_curve.json), and the [535 MB Mac training archive](https://github.com/alfred0105/CircuitCare-AI/releases/tag/connectivity-dataset-v1). The run extracted 4,948 graphs; test candidate recall averaged 0.807. Across 200, 500, 1,000, 2,000, and all 3,920 training schematics, LightGBM PR-AUC ranged from 0.271 to 0.278, while GNN PR-AUC rose from 0.194 to 0.220. This is an initial one-seed comparison, not a full GNN-ACLP paper reproduction.

## Scope and caution

- The first supported input is a **KiCad XML netlist** (e.g. exported using KiCad's schematic netlist export).
- Photo-derived electrical connectivity, external cloud APIs, and automated energization of damaged boards are **out of scope** for the competition MVP. Optional detector boxes remain visual candidates.
- Distinguish **known failed**, **potentially affected**, **measured abnormal**, and **confirmed damaged** components.
- Do not report calibrated failure percentages without real failure-rate/validation data.
- Tests and simulations are not substitutes for safety checks, isolation, and current-limited professional measurement.

## Structure

```text
src/circuitcare/       parser, graph, CLI
tests/                 offline parser/graph tests and KiCad XML fixture
docs/ROADMAP.md        delivery plan and phase status
docs/ARCHITECTURE.md   model assumptions, data contracts, evaluation
docs/PCB_VISION_GUIDE.md manual PCB photo registration, AI review, and overlay guide
docs/FAULT_MAP_GUIDE.md    interactive influence map and evidence guide
docs/WHAT_IF_DIAGNOSIS.md  isolated virtual measurement workflow
docs/PHASE13_EVALUATION.md hotfix, interaction, and regression record
docs/POSTER_CONTENT.md Korean poster text and data figure
docs/DEMO_SCRIPT.md    five-minute presentation and Q&A
docs/RELEASE_CHECKLIST.md offline preflight and fallback procedures
```

Future version: align a PCB photo with known EDA coordinates and overlay prioritized test/repair points.

## Attribution
This repository was repurposed from an unrelated Drosophila experiment on 2026-11-01. Old project files are preserved in the `archive/drosomath-before-circuitcare-2026-11-01` branch.
