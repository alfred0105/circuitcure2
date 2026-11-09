# Diagnostic workflow

## 1. Choose and validate a circuit

Start with one of the supplied examples in the **회로 데이터셋** selector, or select **KiCad XML 업로드** to load an XML netlist for connectivity review. Uploads are stored locally under the ignored `data/private/workspaces/netlists/` directory using a content-addressed filename; DTD/entity declarations are rejected. An uploaded netlist is graph-only until the user configures and validates its SPICE model and mapping. The workspace identifies the selected KiCad XML, SPICE deck, mapping, probes, ratings, and active fault configuration by content hash. `READY` confirms the local input files were read; it does not confirm that every component has a supported SPICE model.

## 2. Inspect the graph

The overview map and circuit tab show component/net connectivity candidates. A shared net is a possible path for influence, not proof of current direction, damage, or root cause.

## 3. Enter known faults or observed symptoms

The main sidebar uses a known component and fault mode for the supported demo workflow. For symptom-led fault hypotheses, use **AI 고장 가설 분석** to enter an observation and request deterministic rule candidates. Check **측정값을 기록함** to record a measurement; a checked 0.0 V is valid, while an unchecked widget default is not an observation. The source code is stored separately from its Korean display label. `user_measurement` means user reported, not independently verified. A local LLM proposal is optional and requires an explicit button press. Unsupported hypotheses are not sent to SPICE.

## 4. Verify selected cases

Review mapping and probe coverage. Press **ngspice 실행** to run the selected known scenario, or explicitly select validated hypotheses in the AI fault tab. Live runs record input and result fingerprints. A legacy cache without generation-time fingerprints is labeled `PROVENANCE_UNVERIFIED`, and a changed input/result is `STALE`; neither is used as current evidence. Missing ngspice leaves non-simulation workflow and report generation available.

## 5. Review evidence and uncertainty

Compare observations with the correct circuit and fault context. The active diagnosis uses its own observation-set hash. Bayesian What-if is a separate `simulated_demo` preview and never replaces the active result. It is included in a report only when explicitly selected, in a separate `PREVIEW_ONLY` section. Keep user-reported observations, simulated demo values, SPICE predictions, unverified AI proposals, and approved datasheet ratings distinct.

## 6. Export a report

The overview creates JSON, Markdown, standalone HTML, and an Evidence Bundle ZIP from the current snapshot. Report export does not run SPICE or an LLM. It includes circuit and input hashes, available results, evidence types, status, limitations, and reproducibility metadata.

## 7. Optionally record a repair

Use **수리 이력 및 AI 학습** only after actual repair work. Synthetic examples and unverified outcomes are not eligible to activate a learned prior. All repair feedback remains in the local ignored database by default.

## Optional tools

Ollama, PCB vision models, datasheet readers, and repair-history priors are optional. If they are not installed or available, the core netlist, connectivity, saved-result review, and deterministic report flow remain usable. No cloud API is called by the core workflow.
