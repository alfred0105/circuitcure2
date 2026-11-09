# Architecture and evidence policy

## Runtime stability boundaries

Graph traversal, PDF extraction, image preprocessing, and SPICE result parsing have explicit resource bounds or bounded caches. The dashboard selects one analysis area per rerun. Native ONNX/OpenCV/OCR code may still terminate the Python process; Python exception handling cannot recover a segmentation fault. Use the isolated diagnostics runner and preserve the OS crash report when available. Incomplete graph searches and failed simulations remain explicitly incomplete or failed.

## Workflow
1. Input: KiCad XML netlist, failed reference, **specified failure mode** and optional measurement evidence.
2. Parse components, references, pins, and nets into a typed representation.
3. Find candidate electrically connected components using a **bipartite** component/net graph; paths show potential propagation only.
4. Generate supported baseline/fault SPICE scenarios; never treat unsupported components as valid simulations.
5. Calculate pin-level changes and compare them with **part-specific** ratings and scenario exposure time.
6. Rank inspection candidates using interpretable features and/or lightweight local inference.
7. Suggest next safe diagnostic measurement; update ranking only when real measurements are provided.
8. Optionally align a user-supplied PCB photo with KiCad component-position coordinates using manually confirmed board corners and at least four component correspondences; render existing risk results only after explicit registration confirmation.
9. Optionally run a local ONNX PCB object detector and local OCR. Treat boxes and text as candidates; use design coordinates and heuristic scores only to suggest a review mapping. Require user confirmation for each detector-to-Reference link before risk overlay.
10. Show a Component–Net influence candidate map, scenario-matched stress/rating overlays, and separate conditional Bayesian inspection priorities. The U1 What-if preview reuses the existing diagnosis engine with `simulated_demo` input and never mutates real measurement state or reruns SPICE.
11. Optionally parse a local datasheet PDF, propose rating candidates with deterministic rules and a loopback-only Ollama model, verify quoted text/number/unit on the cited page, and require human MPN/reference and condition review before a per-metric rating source can enter the existing rating checker. Rating-only updates reuse saved SPICE stress and leave simulation, Bayesian hypotheses, real measurements, and confirmed PCB geometry unchanged.
12. Optionally generate deterministic fault candidates from explicit component-to-SPICE mappings and supported fault models, ask a local Ollama model for structured proposals, and fail closed through a validator before any proposal reaches the existing Fault Injector. The user explicitly selects a bounded set for ngspice. Predicted values are compared only with compatible observations; evidence ranking is model-fit ordering, and existing Bayesian inference is enabled only when every candidate predicts the same observed nodes with explicit uncertainty.
13. Run benchmark comparisons only through an explicit command. The evaluator loads public case inputs for candidate generation and scores separately stored labels afterward. It freezes code/input hashes, separates candidate accuracy from SPICE ranking, and resamples by topology groups. The dashboard displays saved benchmark outputs without starting model or simulation runs. See [Benchmark Methodology](BENCHMARK_METHODOLOGY.md).
14. A content-addressed Workspace identifies the active circuit, configured SPICE inputs, and fault selection. Dependency-scoped validators mark stored records `STALE` when their source inputs change. The overview shows the guided workflow and exports a current analysis snapshot as JSON, Markdown, standalone HTML, or a hash-verified ZIP bundle. Report export collects existing artifacts only; it does not rerun SPICE or invoke an LLM. See [Diagnostic workflow](DIAGNOSTIC_WORKFLOW.md) and [Reporting guide](REPORTING_GUIDE.md).

## Integrity requirements
- Electrical connectivity is not directional. Power flow and damaging fault propagation require device models, topology, current paths, and boundary conditions.
- 'Potentially affected' != 'damaged'. A pin exceeding an absolute maximum does not establish an empirical damage probability.
- Separate fault-present stress, stress-duration assumptions, device ratings, and verified physical damage.
- A simulator error is not evidence of a safe circuit.
- No ungrounded probabilities; use ordinal risk and explicit uncertainty until calibration data exist.
- Circuit-family split during evaluation (not random splits of scenarios from the same circuit). Compare graph-only and physics-only baselines.
- Local-first: no telemetry, cloud API, or remote dependency in the core pipeline.
- Streamlit analysis state is scoped to a context containing the circuit inputs, selected failed component, fault mode/parameter, and configuration file identities. A failed rerun cannot activate the previous live output as the new result.
- The saved U1 3/6/8 V Bayesian example is a separate configured scenario set. It is displayed only for its matching regulator demo context; other circuits and fault settings are labeled unsupported rather than borrowing its posterior.
- LLM answers are bound to the current analysis context, exact measurement set, and simulation artifact identity. Changing any of these invalidates the displayed answer, and the next explicit question rebuilds evidence from current supported inputs.
- `scripts/demo_check.py` checks local prerequisites, cached JSON, and the loopback Ollama endpoint; it does not rerun analyses or contact a cloud service.
- The optional `circuitcare-ai[vision]` extra supplies Pillow/OpenCV/NumPy. The PCB photo tool keeps image bytes in session memory, writes no original photo, treats contour detection as an unverified candidate, and requires explicit user confirmation before joining mapped component centers to a matching risk report. A saved overlay retains image/CSV hashes, board side, homography, calibration error, scenario identity, and source risk report. Reference-center markers are not package outlines; a photo alone never establishes circuit connectivity or damage.

## Demonstration data and provenance
- Regulator demo: committed ngspice comparison, stress/risk reports, and explicit 6.3 V C1 / 125 mW RLOAD example limits.
- Independent topology evaluation: saved resistor divider and RC charging comparisons generated by `scripts/evaluate.py`.
- Bayesian example: saved 3 V / 6 V / 8 V hypotheses with equal priors and a labeled `simulated_demo` 7.9 V ± 0.2 V VOUT input.
- Figures and presentation material are generated only from these committed JSON/CSV values. `WITHIN_LIMIT` applies only to the metrics evaluated against configured limits.

## Data model (bootstrap)
- `Component(ref, value, library_id)`
- `Connection(ref, pin, net)`
- `Circuit(components, connections)`
- `CandidateImpact(target_ref, shared_nets, evidence_level="connectivity_only")`

Planned extensions: `FaultScenario`, `ElectricalStress`, `Measurement`, `RiskAssessment`, `RepairRecommendation`, with provenance.

## Repair safety
Default to power-off inspection. Avoid powering damaged boards until basic isolation and short-circuit checks are complete. Physical test procedures should be scoped to suitable low-voltage laboratory circuits and safe, current-limited supplies.

## Optional PCB AI
- The optional `vision-ai` extra supplies ONNX Runtime CPU; `vision-ocr` supplies RapidOCR and ONNX Runtime. Neither is required for the electrical workflow. Model downloads and ONNX conversion are explicit user actions, weights are kept outside Git, and the Ultralytics exporter is not an application dependency.
- Detection confidence is a model score, not calibrated correctness or damage likelihood. OCR output preserves raw text; Reference-shaped tokens are not automatically corrected or confirmed. Position/OCR/class compatibility scores are heuristic ordering indices, not probabilities.
- A trained detector can return no detections or unknown/low-confidence classes. Unsupported or missing model artifacts produce `MODEL_UNAVAILABLE`; no fallback result is invented. Manual additions and corrections remain separately tagged as user-provided evidence.
- Detection-to-risk overlays require a current, successful scenario-matched risk result, a user assertion that the image/design/circuit identify the same board, and per-detection Reference confirmation. A public generic model evaluation does not validate CircuitCare design matching or real-world board registration.
- The Phase 13 map is a graph visualization, not a reproduction of KiCad schematic placement. Connectivity edges are undirected candidates. Electrical risk colors come only from an aligned SPICE/rating report; What-if priority bands come from Bayesian posterior-weighted inspection scores and are a separate view.
- What-if measurements have source `simulated_demo`, remain in context-scoped preview state, and are not inserted into real measurements. They can change conditional posterior/inspection scores but cannot edit stress data, ratings, simulation artifacts, repair history, or LLM answers. Model mismatch is shown before any top-hypothesis interpretation.
- The pinned detector source, dataset/model licensing distinctions, evaluation setup, and limitations are recorded in [PCB AI detection](PCB_AI_DETECTION.md) and [PCB AI evaluation](PCB_AI_EVALUATION.md).
- Datasheet PDFs are optional local inputs, never uploaded to a cloud model or committed. The approval adapter stores provenance per metric, preserves demo assumptions, and keeps unsupported rating kinds `unknown`. The workflow and one-document evaluation limits are in [Datasheet AI guide](DATASHEET_AI_GUIDE.md) and [Phase 14 evaluation](PHASE14_EVALUATION.md).
- Fault hypotheses are proposals, not findings. The local model cannot execute commands or write SPICE; its exact fault mode and parameters must match a deterministic preflight catalog. The simulator receives only existing supported fault-injection substitutions. Successful simulation and observation fit do not confirm physical damage. See [Fault Hypothesis AI](FAULT_HYPOTHESIS_AI.md).
- Phase 15 fault metrics use three injected synthetic cases and are development diagnostics. The four-family datasheet pilot informed parser changes; its post-fix score is also development-only. Neither result is an independent accuracy estimate; the separate evaluation reports record dataset versions, hashes, source PDFs, and limits.

- Repair feedback is local by default under ignored `data/private/repair_feedback/`. Synthetic, simulated, LLM-generated, unverified, duplicate, validation, and holdout cases cannot enter real-repair prior learning. Outcome verification does not itself prove unique root cause. Learned weights are conditional priors for a compatible hypothesis set, not real-world failure probabilities; no prior is activated without sufficient reviewed physical cases and explicit approval. Current real-case count is zero. See [Phase 16 evaluation](PHASE16_EVALUATION.md).

## Phase 18.1 diagnostic integrity

- Active measurement diagnosis and Bayesian What-if preview use separate hashes and storage. A preview is exported only by explicit selection and remains `PREVIEW_ONLY`.
- Measurement source codes are canonical; `user_measurement` means user reported. Zero voltage is a valid recorded value.
- Simulation cache identity includes content hashes for the netlist, deck/includes, mapping, probes, fault parameters, and settings. Legacy results without a verified manifest are retained but cannot be used as current evidence.
- Prior activation and rollback share source eligibility checks and commit status/event updates transactionally. Invalidated source cases cannot be restored.
- The report and bundle validators cross-check measurement/hypothesis, circuit/scenario, artifact, and archive context. See [Phase 18.1 hotfix record](PHASE18_1_RELIABILITY_HOTFIX.md).
