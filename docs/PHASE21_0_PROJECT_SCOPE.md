# Phase 21.0 — Project Scope

## Product goal

The target product starts from multiview PCB photographs and a natural-language fault symptom, even when the user does not already know the failed component. It helps the repairer interpret symptoms, reconstruct visual structure, ground hypotheses in electrical models, identify possible secondary stress, and recommend useful measurements. The product must distinguish observed pixels, inferred connections, design connectivity, simulated stress, and physically confirmed outcomes. When connectivity or cause remains uncertain, it asks for a design file, confirmation, or measurement instead of presenting an inferred result as confirmed.

Competition claims focus on measurable contributions in: (1) PCB electrical connectivity reconstruction, (2) physics-informed fault-cause reasoning, and (3) active measurement that improves diagnostic efficiency. Feature count is not evidence of technical merit.

## Core research scope

- Multiview image registration, component/pad/via detection, copper candidates, and electrical-connectivity reconstruction.
- SPICE-supported fault scenarios and secondary electrical-stress analysis.
- Bayesian / physics-informed hypothesis ranking with uncertainty and model limits.
- Active measurement recommendation, evaluated against matched baselines.
- Natural-language symptom intake and evidence-gated fault overlays as interfaces to the core methods.
- Summary and detailed analysis views that expose source, uncertainty, and validation status.
- A user-centered workbench that connects photo intake, symptom input, analysis, PCB viewing, measurement updates, and repair guidance.

## Supporting capabilities

KiCad import and viewer, datasheet extraction and human review, local LLM explanations, OCR, repair recommendations, reports, and verified repair history remain available. They support the research workflow and must not delay its central data, modeling, or evaluation work. All preserve existing provenance and approval gates.

## Deferred capabilities

Full 3D reconstruction, automatic component purchasing, large-scale LLM training, automatic continual learning, exhaustive multi-fault enumeration, and live camera streaming are deferred. Phase 21.8 may examine a bounded set of compound-fault hypotheses; exhaustive multi-fault enumeration remains deferred. Existing code and public APIs are retained; advanced routes may be separated from the default workflow without breaking compatibility.

## Evidence labels

Use `VERIFIED_SYNTHETIC`, `VERIFIED_DESIGN`, `VERIFIED_PHYSICAL`, `IMPLEMENTED_UNEVALUATED`, `PARTIAL`, `BLOCKED`, or `DEPRECATED`. Apply labels per claim, dataset and task. CI success is software evidence only. A test against a KiCad design can verify design behavior, not physical board accuracy. Do not use `VERIFIED_PHYSICAL` without a traceable physical board, independent truth, and measurement procedure.

## Maintenance rules

- Preserve phase records, private/local user data, external datasets and public APIs.
- Reuse existing parser, graph, simulator, Bayesian, measurement and reporting modules.
- Keep observed visual topology separate from design electrical nets; do not auto-promote visual hypotheses into SPICE.
- Maintain fail-closed provenance gates, explicit user confirmations and unsupported-model reporting.
- Make focused changes when they improve a measured boundary or remove a confirmed defect; do not refactor for line count.
- Keep optional model/runtime dependencies optional and local-first.
