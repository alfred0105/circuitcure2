# Phase 21.0 — Technical Audit

## Scope and evidence rules

This inventory describes the checked repository at Phase 21.0. Implementation, automated tests, synthetic measurements, design-file checks, and physical-board validation are separate evidence levels. No capability is classified `VERIFIED_PHYSICAL`: the repository has no independent CircuitCare board bundle with matching photographs, KiCad PCB coordinates, and reference electrical measurements. `VERIFIED_DESIGN` means a tool or workflow was checked against design artifacts; it does not mean a physical board was validated. CI passing alone is not a validation level.

| Function | Main modules | Dependencies / flow | Current state and evidence | Validation level | Phase 21 role | Action |
|---|---|---|---|---|---|---|
| KiCad netlist parsing | `netlist.py`, `component_parser.py`, `models.py` | XML → Component/Connection | CLI and fixture coverage; README records KiCad export | VERIFIED_DESIGN | Support / input | Maintain |
| Component–Net graph | `graph.py`, `path_analyzer.py` | Circuit → bipartite graph and bounded paths | unit tests; paths are candidate influence only | VERIFIED_DESIGN | Core support | Maintain |
| Fault injection | `fault_models.py`, `fault_injector.py`, `scenario_generator.py` | explicit mapping → supported altered deck | resistor/regulator/switch scenarios documented | VERIFIED_SYNTHETIC | Core | Maintain |
| ngspice simulation | `spice_runner.py`, `stress_analyzer.py`, `simulation_results.py` | supported decks → sampled node/component results | divider, RC and demo runs saved; unsupported mappings fail closed | VERIFIED_SYNTHETIC | Core | Maintain; add board-relevant circuits |
| Secondary stress/risk | `risk_engine.py`, `rating_checker.py`, `impact_ranking.py` | simulated stress + explicit ratings | tests and illustrative demo; assumptions are not datasheet or damage evidence | VERIFIED_SYNTHETIC | Core | Maintain; calibrate only with evidence |
| Bayesian diagnosis | `diagnosis_ai.py`, `bayesian_engine.py`, `diagnosis_models.py` | hypotheses + observations → conditional posterior | synthetic scenario tests; weights are not field probabilities | VERIFIED_SYNTHETIC | Core | Maintain; assess identifiability |
| Measurement selection | `measurement_selector.py` | current hypothesis distributions → information gain | selector implemented; no matched policy-efficiency evaluation | IMPLEMENTED_UNEVALUATED | Core | Keep behavior; evaluate policies |
| Local LLM | `local_llm.py`, `llm_prompts.py`, `conversation.py`, `fault_ai/` | explicit local request → constrained candidate/explanation | synthetic case evaluation and local smoke runs documented; model-dependent | VERIFIED_SYNTHETIC | Supporting research | Retain behind evidence gates |
| Datasheet analysis | `datasheet/` | local PDF → cited candidate → human review | one-document smoke and limited development evaluations | PARTIAL | Support | Retain; do not present candidates as ratings before approval |
| PCB detection and OCR | `vision/`, `vision_ui.py` | local image → candidate boxes/text | public split 13 images / 196 objects: P .498, R .612, mAP@.5 .509; OCR one-image smoke | PARTIAL (public validation split; board split provenance unverified) | Core research input | Keep optional; obtain independent board split |
| Photo registration / identity | `vision/registration.py`, `reference_matcher.py`, `pcb_positions.py`, `overlay.py` | photo + Position CSV + human-confirmed correspondences | unit mechanics; no matching real CircuitCare photo/design bundle | IMPLEMENTED_UNEVALUATED | Core prerequisite | Validate with registered real board |
| Schematic viewer / overlays | `schematic/` | KiCad schematic export + geometry + evidence overlays | geometry, interaction, electrical-equivalence, overlay tests | VERIFIED_DESIGN | Support / explanation | Maintain; browser check in this phase |
| Repair advice / reports | `repair_advisor.py`, `report_generator.py`, workspace reporting | evidence records → advice and export | deterministic and local LLM flows; synthetic report tests | VERIFIED_SYNTHETIC | Support | Maintain |
| Repair feedback / prior learning | `repair_feedback/`, `repair_feedback_ui.py` | reviewed cases → gated conditional priors | synthetic workflow; no eligible real repair records | PARTIAL | Support | Retain local safeguards; no performance claim |
| Benchmark infrastructure | `benchmark/`, `fault_ai/evaluation.py`, `datasheet/evaluation.py` | frozen inputs/code → blind candidates → scoring | manifests, hashes, baselines, group bootstrap, saved run records | IMPLEMENTED; integrity tests | Core research support | Reuse; preserve split boundaries |
| Multiview reconstruction | `vision/reconstruction/` | images → registration, segmentation, feature candidates | synthetic fixtures and PRISM image pipeline | PARTIAL | Core research | Separate image-only and design-assisted tasks |
| PRISM adapter | `vision/reconstruction/prism_dataset.py` | public dataset → normalized images/schema | 12 boards / 24 sides processed; no calibrated image-mm truth | VERIFIED_PIPELINE_ONLY | Core research dataset | Do not claim electrical/geometry accuracy |
| Copper / pad / via candidates | `copper_segmentation.py`, `quality.py`, reconstruction models | raster → uncertain masks/features | HSV/adaptive candidate experiment; no valid precision/recall/IoU or pad/via F1 | IMPLEMENTED_UNEVALUATED | Core research | Require registered ground truth |
| Observed topology | `reconstruction/topology.py` | same-side visible mask → contact candidates | code and tests preserve inferred status | PARTIAL | Research only | Never promote automatically to electrical nets/SPICE |
| Dashboard / workflow | `ui.py`, `dashboard/`, `workspace/` | user inputs + provenance-gated outputs | Streamlit AppTests and integrity tests; responsive defects addressed in Phase 21.0 | VERIFIED_DESIGN | Support | Maintain evidence gates |

## Data flow and boundary

```text
PCB images → multiview registration → detector / segmentation / pad-via candidates
          → ObservedTopologyGraph (inferred, same-side visual contact only)

KiCad XML → Circuit(Component, Connection) → Component–Net graph
          → explicit supported fault scenario → ngspice → stress + configured ratings
          → conditional diagnosis / measurement recommendation → evidence report
```

`ObservedTopologyGraph` is defined in `vision/reconstruction/topology.py` and its reconstruction models. Repository search found no integration path that converts it into `circuitcare.graph`, a KiCad `Circuit`, or SPICE input. This is a useful safety boundary; do not bypass it. Photo overlays additionally require matching scenario and explicit registration/reference confirmation.

## Existing evaluation evidence

- **Phase 17:** 10 synthetic cases across 5 topology groups; repeated five times, so repetitions do not increase independent sample size. Rules exact precision .0495 / recall 1.0; LLM exact precision/recall 0; Hybrid showed no exact gain. Rules+SPICE strict Top-1 .30, Top-3 .40, MRR .4156; tied results require care.
- **Phase 17.1:** same 10-case holdout reused, not a new independent sample. Strict Top-1 .30 versus tie-aware 1.0, with ties in 40/50 repeated runs. Off-grid used 10 cases on five reused topology templates: exact Top-1/Top-3/MRR all 0; the correct hypothesis was absent from the candidate catalog.
- **Active measurement:** expected information gain is implemented. No same-case Random / Rule-based / EIG policy comparison measured diagnostic efficiency or number of measurements saved.
- **PCB detector:** source validation split has 13 images / 196 objects; board-level split independence is unestablished. No independent CircuitCare real-board detection-to-reference / physical-registration validation.
- **PRISM / segmentation:** 12 boards and 24 images processed, but no scale calibration or image-to-design registration. Candidate pixel fractions are not segmentation metrics; electrical connectivity and pad/via detection accuracy remain unverified.

Detailed source records: [Phase 17](PHASE17_EVALUATION.md), [Phase 17.1](PHASE17_1_EVALUATION_HOTFIX.md), [PCB AI](PCB_AI_EVALUATION.md), [PRISM](PHASE20_1_1_PRISM_BENCHMARK.md), [segmentation](PHASE20_2_SEGMENTATION_UPGRADE.md), and [benchmark methodology](BENCHMARK_METHODOLOGY.md).

## Risks and Phase 21 action

1. No real-board photo ↔ KiCad coordinate ↔ netlist reference set exists; reconstruction-to-electrical comparison is blocked.
2. Small synthetic diagnosis samples and reused topology templates limit generalization; holdout truth must remain sealed.
3. EIG has no policy-level efficiency evidence; freeze a matched evaluation protocol before changing the selector.
4. Public detector metrics do not establish board-level generalization or CircuitCare identity matching.
5. Visual connectivity remains a candidate representation and must not silently enter electrical simulation.
6. Existing modules are numerous but have clear bounded areas; Phase 21.0 makes no broad rewrite. Refactor only around demonstrated coupling or reliability problems.

Phase 21.1 can begin as a **data and calibration readiness** task. Electrical reconstruction performance work should not begin until a registered, independently split ground-truth set and an image-only / design-assisted protocol are available.
