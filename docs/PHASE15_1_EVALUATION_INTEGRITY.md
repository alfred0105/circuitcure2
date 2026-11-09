# Phase 15.1 — Evaluation integrity hotfix

## Status

This hotfix changes evaluation validity, not the fault or datasheet inference engines. Existing Phase 15 behavior is retained. All data in this report is development-only. The fault cases are three synthetic SPICE-injected examples previously used during development; the four datasheets were also used to tune extraction. No independent holdout is available.

## Fault prompt boundary

The old call passed answer-bearing `case_id` values such as `REGULATOR_U1_RISE_8V`, and described each input as a “known injected fault.” Those prompt fields invalidated the earlier LLM proposal and ranking scores. They are preserved in the results archive under `LEAKED_DEVELOPMENT_RESULT`.

The new `build_blind_diagnostic_input` constructs its output from allowlisted public diagnostic fields. The LLM receives an anonymous `case_0001`-style ID, component references and public circuit metadata, measured observations and uncertainty, and supported component/fault pairs. The evaluator does not pass the internal case ID, paths, evaluation notes, source tags, ground truth, truth parameter values, or fault injection logs. Candidate parameters are omitted from the LLM catalog; the server-side validator still checks output against the supported full catalog. For example, the symptom text is `Observed VOUT = 8.0 V (measurement uncertainty ±0.2 V); mode: final_sample`.

Tests cover answer-bearing IDs, gold fields and parameters, preservation of measurements, invariant prompts across different private labels, excluded paths/debug/source data, captured model messages, and no ground-truth injection into LLM-only ranking. These are code-boundary tests, not evidence of independent dataset validity.

## Fault results

The three repeated local runs and full run metadata are saved at `evaluation/fault_hypothesis/results/leakage_free_development_metrics.json`. Proposal quality is reported separately from diagnostic ranking. LLM-only ranking contains only the hypotheses actually proposed by the LLM; the combined method uses the union of rule and LLM proposals. Missing or failed simulations remain visible and contribute to coverage and simulation status; the injected truth is never inserted to improve ranking.

The leakage-free run used Python 3.12.14, Gemma 4 12B digest `c7597fc90b86aff6965317f37dd5465402284e26382d9b4bf2dfe3ef1491f4b6`, prompt `fault-hypothesis-v2`, and ngspice 47. It was recorded at code revision `0abfd79a04b65ceeb337e0e3d944e65bed0e8c0a`.

| Method | Proposal candidates / correct | Precision | Recall | Valid proposal rate | Top-1 / Top-3 | P@3 | Ranking coverage | SPICE attempted / success | Mean LLM latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rules only | 43 / 3 | 0.0698 | 1.00 | 1.00 | 1.00 / 1.00 | 0.333 | 1.00 | 129 / 129 | — |
| Local LLM only | 3–4 / 0 per run | 0.00 | 0.00 | 0.25–0.333 | 0.00 / 0.00 | 0.00 | 0.333 | 3 / 3 eligible | 5.10 s per call |
| Rules + LLM + validator | 45–46 / 3 | 0.0638–0.0667 | 1.00 | 0.9348–0.9556 | 1.00 / 1.00 | 0.333 | 1.00 | 129 / 129 | 5.10 s per call |

Three LLM runs generated 11 LLM-only candidates in total, none matching the exact labeled fault hypothesis; only 3 were executable and successfully simulated. Combined Top-K equals the rules result because the deterministic candidates were retained; this does not demonstrate LLM ranking improvement. Across combined repetitions, eight proposals were unsupported by the validator. All 129 eligible simulations succeeded. The metric denominators and per-run results are in the JSON. The old Phase 15 LLM result remains `LEAKED_DEVELOPMENT_RESULT`; this new record is `LEAKAGE_FREE_DEVELOPMENT_RESULT`, but both use development examples. `holdout_status` remains `INDEPENDENT_HOLDOUT_NOT_EVALUATED`.

## Datasheet result scope

Each of the four documents has a manually verified target-rating label, not a complete rating inventory. The evaluation scope names the rating key and uses available MPN/variant, row, column, and source-anchor constraints. Only candidates within that scope enter target precision. Other rating types are excluded; candidates with insufficient evidence are unverified; wrong variants and wrong units are separately classified. Candidate-level human adjudication is absent, so unsupported-claim and hallucination metrics are `NOT_EVALUATED`. Automated source-check rejection is reported separately and is not treated as hallucination.

The previous 0.0833 and 0.0741 precision figures counted every unannotated candidate as a false positive. Their document-wide interpretation is invalid and preserved with status `INVALID_EVALUATION`. `ground_truth_complete` is false, so complete-document precision, recall and F1 are `NOT_EVALUATED`. Corrected target-scope results are in `evaluation/datasheet_extraction/results/corrected_scope_development_metrics.json`; they are development-pilot measurements, not a claim of complete extraction accuracy. The final run used Gemma 4 12B digest `c7597fc90b86aff6965317f37dd5465402284e26382d9b4bf2dfe3ef1491f4b6`, code revision `e83538bee387678e17c8d8b0f0c23fc2f6bd2ca8`, and 31.69 seconds total model processing time (7.92 seconds per PDF on average).

## Holdout and next evaluation

The three fault examples remain under the development split; anonymizing them does not make them independent. `evaluation/fault_hypothesis/holdout/README.md` records `INDEPENDENT_HOLDOUT_NOT_EVALUATED`. A future holdout must use independently prepared unseen topologies and confirmed labels, and must not be used for prompt, parser, or scoring tuning. Datasheet full-document evaluation requires a human-verified exhaustive inventory; target-rating evaluation requires complete, explicit scope labels.

## Verification

The full test suite passes: **87 passed**, including the original 73 tests and new evaluation-integrity coverage. `compileall` and `git diff --check` also completed successfully. Local Gemma and ngspice runs are recorded above. No independent holdout performance claim is made.
