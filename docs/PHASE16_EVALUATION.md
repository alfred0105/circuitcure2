# Phase 16 Evaluation — Repair Feedback and Prior Adaptation

## Status

| Item | Status |
|---|---|
| Repair Feedback Implementation | `COMPLETE` |
| Synthetic Workflow Test | `VERIFIED` |
| Real Repair Cases | `0` |
| Eligible Real Repair Cases | `0` |
| Active Learned Prior | `NONE` |
| Real-Repair Accuracy | `NOT_EVALUATED` |
| Independent Improvement Evaluation | `NOT_EVALUATED` |

The local private database was inspected before finalization and contained zero repair cases and zero prior model records. Test fixtures use temporary databases and synthetic examples only; they are not real repairs and are not installed as active priors.

## What was implemented

The phase adds local SQLite case history, immutable diagnosis snapshots, observation/intervention/test and verification records, evidence-aware eligibility, exact and suspected duplicate handling, case search/export/backup/deletion, an explainable smoothed prior estimator, versioned prior proposals with human approval and rollback, optional prior injection into the existing Bayesian update, same-evidence comparison, and holdout/leakage checks in evaluation utilities. The Streamlit tab handles empty state without creating a prior.

## Minimum verification performed

- Full existing and new suite: `118 passed`.
- Streamlit main dashboard AppTest: no exceptions; the new repair history tab is present.
- Python bytecode compilation: passed.
- `git diff --check`: passed.
- Temporary fixture experiments cover empty database behavior, synthetic exclusion, five-record estimator behavior, Bayesian update with optional priors, and model lifecycle/rollback. The temporary records are not real-world evidence.

## Evaluation protocol and interpretation

The evaluation utilities keep development, validation, and independent holdout splits separate and detect repeated case IDs and shared board instances across partitions. Cases used to estimate a prior cannot also count as independent evidence of improvement. Metrics supported by the utility include Top-1/Top-3 accuracy, reciprocal rank, negative log-likelihood, Brier score, calibration error when enough labeled examples exist, diagnostic coverage, and unsupported-case rate.

No independent physical holdout exists, so none of these metrics is reported as measured real-repair performance. `INFORMATION_GAIN` is also left `NOT_EVALUATED` absent a compatible safety-qualified model. Any posterior remains conditional on the supplied hypothesis set and is not a real-world component failure probability.

## Data leakage and provenance safeguards

Synthetic, simulated, LLM-predicted, unverified, duplicate, validation, and holdout records cannot enter eligible real-repair prior training. Physical source, measurements, root-cause review, intervention, passed post-repair checks, and compatible before/after evidence are required. The original diagnosis snapshot stays unchanged when a later comparison is saved. Every prior proposal captures its source case IDs and configuration. Case edits/deletion invalidate dependent learned priors.

## Limitations and next evidence needed

The minimum sample count and `alpha=5` smoothing are configured defaults, not empirically optimized values. Selection bias, imperfect root-cause attribution, and limited circuit-family coverage remain. Real improvement evaluation requires independently collected physical repair cases with reviewed labels, comparable hypothesis sets, and board-instance/time-based holdout separation. No repair action safety is guaranteed by the software.
