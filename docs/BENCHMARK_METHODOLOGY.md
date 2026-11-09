# Benchmark Methodology

## Data boundaries

Each case has a public diagnostic record and a separately stored private ground-truth mapping. The LLM adapter receives only an allowlisted anonymous case ID, public circuit components, observations, symptoms, and supported component/fault-mode pairs. Candidate parameters, truth labels, internal IDs, split seeds, and paths are excluded from its payload. Scoring runs only after candidate generation. Public payloads are checked immediately before inference.

Cases derived from one circuit topology share a `topology_group`; the integrity validator rejects groups that cross Development, Validation, or Holdout splits. Dataset manifests record SHA-256 hashes for public records and private ground truth. New benchmark results are saved to unique run files; collisions fail instead of overwriting earlier runs.

The Phase 17 synthetic holdout builder creates five new topology groups with two ngspice-injected faults per group. The source circuit and observation are public inputs; the injected component/mode/parameter are kept in the separate truth file. This is an independent synthetic split relative to the previously used Phase 15 examples, but it does not model field repair prevalence or hardware effects outside the configured SPICE models.

## Methods and fairness

- **Rules Only:** existing explicit fault catalog and existing preflight validator.
- **LLM Only:** the installed local model proposes from the same supported component/fault-mode catalog; exact fault parameters are not shown. Its output passes through the existing validator.
- **Hybrid:** union of Rules and LLM proposals with duplicate removal.
- **Rules + SPICE / Hybrid + SPICE:** each method uses the same maximum simulation count per case, same probes, same observations, and same residual ranking function. Candidate generation and residual ranking are reported separately.

The existing fault catalog parameter grid constrains executable proposals. It is not an unrestricted LLM baseline. LLM calls are optional and explicitly requested; the UI never runs them.

## Metrics

Component, fault mode, and exact hypothesis use distinct matching levels. Exact hypotheses require exact normalized parameter equality; numeric relative error is also reported separately with denominator `max(abs(target), tolerance, 1e-15)`. The current relative-error summary selects the first proposal with matching component and fault mode; it is not a best-candidate error and may be nonzero even when another proposal is an exact match. Use the exact-hypothesis precision/recall for exact parameter matching. Unsupported ground truth is excluded from supported-only accuracy and counted separately. Precision, recall, F1, Top-1/Top-3, mean reciprocal rank, candidate counts, validation rates, simulation success/failure, coverage, wall-clock runtime, and simulation counts are stored where available. A missing metric is `NOT_EVALUATED`, not zero.

SPICE end-to-end ranking counts a candidate-generation miss as a miss. Conditional ranking is separately restricted to cases where the true hypothesis was generated and simulated successfully. One-node and multiple-node strata are reported separately. The existing holdout contains only the configured node counts; absent strata remain unevaluated.

## Statistics and repeats

LLM repeats measure output stability, not additional independent fault cases. Mean, median, standard deviation, minimum, and maximum are recorded for runtime and repeat-level metrics. Failed LLM calls may have missing latency when the adapter raises before returning timing metadata. Cluster bootstrap resamples topology groups, not individual cases. At least two groups are required; with few groups its interval is labeled exploratory and is not a population guarantee. Paired method comparisons use the same cases. No p-value or statistically significant improvement claim is made from the small synthetic dataset.

## Execution and reproducibility

1. Commit/freeze benchmark code and record its Git SHA.
2. Run `python scripts/build_phase17_synthetic_holdout.py` once; retain the manifest hashes.
3. Validate the split and hashes with `load_dataset_bundle` and the integrity tests.
4. Run the desired baselines explicitly, for example:

   ```bash
   python scripts/run_phase17_benchmark.py --dataset evaluation/phase17/holdout --run-llm --model gemma4:12b --llm-runs 5 --simulate --spice-budget 40
   ```
5. Keep every result JSON and generate the Markdown report with its exact run ID.

Record Python, OS, Git SHA, dataset and ground-truth hashes, split, prompt version, LLM name/digest, ngspice version, seed, simulation budget, and timing. Ollama token counts and process-wide peak RSS are not reported unless the runtime exposes them. `tracemalloc` measures Python-managed allocation only.

## Scope exclusions

Synthetic SPICE injection is not physical fault validation. Results do not establish actual repair accuracy, real component failure probability, safety of probe locations, broad topology generalization, datasheet extraction accuracy, PCB reference matching, or benefit from Phase 16 repair priors. Those tracks require separate physical and untouched datasets.

## Phase 17.1 metric schema

The original Phase 17 run uses its original scoring and remains immutable. New and rescored runs use `phase17-metrics-v2`: deduplicate at each identity level before one-to-one scoring; preserve raw-candidate metrics and duplicate counts; and report selected-candidate, oracle-best, and legacy-first parameter errors separately. Oracle-best is post-hoc and cannot be interpreted as the system's selected value. If no explicit selection/rank exists, selected error is `NOT_EVALUATED`.

Run status, per-method status, split, and metric schema must agree; methods not run are `NOT_EVALUATED`. LLM wall time includes failed attempts measured externally around the call. SPICE strict Top-1 remains distinct from tie-aware Top-1; a true candidate in a tied best set is reported as ambiguous, not uniquely identified. Observed node-count strata are descriptive unless identical cases are separately evaluated under different measurement sets. See [Phase 17.1 results](PHASE17_1_EVALUATION_HOTFIX.md) for dataset hashes and observed values.
