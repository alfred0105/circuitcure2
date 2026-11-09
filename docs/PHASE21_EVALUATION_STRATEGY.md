# Phase 21 — Evaluation Strategy

## Three competition claims and evaluation tracks

| Research claim | Primary task / unit | Required baseline | Primary metrics | Current readiness |
|---|---|---|---|---|
| PCB electrical connectivity reconstruction | held-out board / traceable net segment, not random pixels | image-only heuristic and design-assisted registration reported separately | component/reference match precision-recall; registered trace pixel PR/F1/IoU; pad/via detection F1; net/edge precision-recall; false short/open | Blocked: no calibrated image↔KiCad ground truth |
| Physics-informed fault-cause reasoning | independently held-out circuit topology and fault case | rules-only, Bayesian-only, SPICE-fit, and combined method | exact fault precision/recall, top-k, MRR, calibration/coverage; report identifiability and abstention | Synthetic development evidence exists; physical generalization not evaluated |
| Active measurement diagnostic efficiency | same case, same initial evidence, paired policy run | random and fixed rule-based; existing EIG; improved EIG only after frozen protocol | measurements to correct diagnosis, accuracy at fixed budget, diagnostic yield / cost, failure and abstention rates | Not evaluated; EIG code exists |

Phase 21.8 adds an integration task, not a fourth headline research claim. Evaluate (a) natural-language symptom field/module structuring against annotated symptoms, (b) fault-candidate accuracy against diagnosis truth, and (c) secondary-impact candidate accuracy against separately labeled affected components. Score those outputs independently; end-to-end user completion belongs to the usability evaluation in Phase 21.9. Never score an LLM explanation as proof of physical damage.

## Existing evidence interpretation

Phase 17 measured 10 synthetic cases across five topology groups, with five repeated method passes. The independent unit remains 10 cases / five groups. Rules exact precision was .0495 and recall 1; LLM exact precision/recall were both zero; Hybrid did not improve exact performance. SPICE Rules strict Top-1 was .30, Top-3 .40, MRR .4156. Phase 17.1 reused that holdout: strict Top-1 .30, tie-aware Top-1 1.0, with ties in 40/50 runs. Its off-grid evaluation used 10 faults on five reused templates, with exact Top-1, Top-3 and MRR all zero because the true candidate was not proposed. These are synthetic diagnosis results, not hardware accuracy.

Detector P .498 / R .612 / mAP@.5 .509 comes from a public validation split of 13 images / 196 objects; board-independent split provenance is unknown. PRISM processed 12 boards / 24 sides but lacks image-mm calibration and registered ground truth. Therefore no precision/recall/IoU or electrical connectivity score can be derived from its candidate pixel fractions. Measurement-policy benefit is `NOT_EVALUATED`.

## Dataset and leakage policy

Every experiment must identify data source/version/hash, split, unit of independence, topology/board grouping, truth access boundary, and whether design metadata was used. Split by board or circuit topology before generating views, augmentations, scenarios, or measurements. Keep public development/validation data separate from an unreleased final evaluation set and independent external/physical data. Repeated runs, multiple views, and multiple faults on one topology are not independent boards/topologies.

Freeze input manifest, candidate-generation code/model versions, and baseline configuration before opening held-out truth. Candidate generation receives only permitted public inputs. An independent scorer accesses labels only after candidates are serialized. Any evaluation used to tune a model or rule becomes development data; do not relabel it as a fresh holdout. Record image-only and design-assisted tracks separately, and disclose design-assisted inputs (KiCad outlines, positions, netlists, references) explicitly.

Physical validation requires traceable board identity/revision, photographs with view/side metadata, image↔design calibration points and residuals, independently reviewed component/pad/via/connectivity truth, and measurement conditions. Use separate validation points not used to fit registration. A PRISM raster without scale/reference alignment is not sufficient electrical ground truth.

## Measurement policy comparison

Use identical cases, initial priors, observation model, available probe set and stopping rule for Random, Rule-based, existing EIG, and any improved EIG policy. Pair policy runs by case and seed; disclose whether the simulated oracle uses exact model parameters. Primary outcome: measurements to a uniquely correct diagnosis under a predeclared threshold, with unresolved/tied cases reported as failures or abstentions according to a frozen rule. Also report exact diagnosis at fixed measurement budgets (e.g. 1/2/3), cumulative measurement cost, invalid/unavailable probes, and calibration. Aggregate by independent case/topology, not repeated policy trials. Report paired differences and group-level uncertainty; do not infer real repair-time savings from simulated measurements.

## Common experiment record

Use [`../evaluation/experiment_record.schema.json`](../evaluation/experiment_record.schema.json) for new records. It captures task, dataset and split identity, code/model versions, baselines, metrics, limitations and reproducibility. Historical files need not be migrated. Missing evidence is `NOT_EVALUATED`, never silently represented as zero.

Minimum reproducibility record: command/configuration, code commit, data manifest and hashes, model artifact hash, environment and optional runtime versions, random seeds, start/end timestamps, hardware, metric definitions, exclusions, raw per-case outputs, and scorer version. Preserve an immutable run record and document any failed or exploratory runs.
