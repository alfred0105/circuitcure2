# Phase 17 Evaluation

## Outcome

- Benchmark infrastructure: implemented; the dashboard reads saved results and does not start evaluations.
- Evaluation: `MEASURED_INDEPENDENT_SYNTHETIC_HOLDOUT`.
- Dataset: `circuitcare-phase17-independent-synthetic-v1`; 10 supported cases across five new topology groups.
- Every case had an executable ground-truth hypothesis. The independent sample size is 10 cases / five topology groups.
- Rules, local LLM, Hybrid, Rules + SPICE, and Hybrid + SPICE ran five repeated passes on the same 10 public cases.
- Real repair accuracy, physical repair outcomes, and real component damage probability: `NOT_EVALUATED`.
- Full immutable run record: [`phase17_20261103T080936_c04b602f.json`](../evaluation/phase17/results/phase17_20261103T080936_c04b602f.json).

The holdout was generated and evaluated against frozen source SHA `699d22f6f8e6f8a1ac2d5354581f7bb83e2a3fd4`. Its five topology groups and ten injected-fault cases are independent of the reused Phase 15 development examples. This is topology-separated synthetic evidence, not a field-repair sample.

## Methods and results

See the [baseline comparison](BASELINE_COMPARISON.md) for the complete precision/recall, Top-K, MRR, runtime, candidate count, and SPICE table.

- Rules found an exact target in every case (exact parameter-level recall 1.0000) but proposed 202 candidates per repeated 10-case run, yielding exact precision 0.0495.
- LLM Only achieved component precision 0.3467 and recall 0.9200. Exact fault-mode-plus-parameter precision and recall were both 0.0000. Two of its 50 calls produced malformed JSON.
- Hybrid retained exact recall 1.0000 through the Rules candidates, while exact precision was 0.0490. The measured Hybrid did not demonstrate an exact-diagnosis gain over Rules.
- Rules + SPICE and Hybrid + SPICE each achieved Top-1 0.3000, Top-3 0.4000, and MRR 0.4156 over the five repeated passes. This is 50 repeated case-runs over only 10 independent cases.
- Both SPICE methods completed 1,010/1,010 attempted simulations with no recorded simulation failures.
- The LLM method averaged 225.5918 seconds per repeated 10-case batch among recorded durations. The runner does not retain latency for malformed responses, so total LLM and Hybrid runtime is understated by those two failed-call durations.

Exact hypothesis scoring includes exact normalized parameter equality. The result record's separate `parameter_error` summary uses the first proposal with matching component and fault mode, not the closest or selected proposal. Because this can report a nonzero error even when an exact candidate is present, that summary is not interpreted as a best-candidate parameter error. Exact hypothesis precision/recall are used for parameter-level conclusions.

## Leakage, reproducibility, and records

- Public-input audit: `PASSED`; private ground truth was not passed to proposal generation and was opened only after proposal and SPICE processing completed.
- 50 LLM calls were attempted across five repetitions; the call payload was audited immediately before inference. The two failed-call exceptions did not persist their individual pre-call hashes in the per-case result rows.
- Public dataset SHA-256: `ee70b06c71d312e36f455a0cdda46492f60f3ae728a83cbb505de4cca2a3b85e`.
- Private ground-truth SHA-256: `62c376f0d8582bfe4bbe4fee7d77a7e63ddc62fec1535d2e1e1e52eb1d12182b`.
- Git SHA: `699d22f6f8e6f8a1ac2d5354581f7bb83e2a3fd4`.
- Python / platform: `3.12.14` / `macOS-27.0.1-arm64-arm-64bit`.
- LLM: `gemma4:12b`, digest `c7597fc90b86aff6965317f37dd5465402284e26382d9b4bf2dfe3ef1491f4b6`; prompt `fault-hypothesis-v2`.
- Simulator: ngspice 47. LLM temperature 0.1; maximum generation 1,300 tokens; JSON mode requested; per-case SPICE cap 40.
- Process-wide peak RSS was 36,388,864 bytes; it is not isolated by method. Ollama token usage was unavailable. Warmup was zero and cache state was uncontrolled.

The run-level record marks the independent synthetic holdout correctly. Its per-method status fields retain a legacy `MEASURED_SYNTHETIC_DEVELOPMENT_REUSED` label; this conflicts with the run-level status and manifest, but does not alter the stored scoring values. Treat the run-level status, dataset manifest, and this report as authoritative for the split.

## Uncertainty and scope

Topology-cluster bootstrap used 2,000 resamples over five groups. The intervals are labeled exploratory and are unstable at this small group count. Repeated runs measure stability and are not independent cases. No p-value or statistical-significance claim is made.

Only the included measurement-node configuration was evaluated. Multi-node measurement-policy comparisons, open-set or unsupported-fault rejection, false-confidence/abstention evaluation, a new datasheet holdout, physical PCB reference matching, and a real repair-prior comparison remain `NOT_EVALUATED`. Synthetic SPICE injection does not capture board variation, parasitics, repair history, all component models, or physical damage behavior.

## Next steps

Phase 17 synthetic benchmark completion is recorded. To estimate real repair performance, collect verified physical fault/repair cases with independent boards and labels, freeze another untouched split, and repeat the same evaluation before making real-world accuracy claims.

## Phase 17.1 metric correction

The original run above remains unchanged and preserves its original scoring definition. The Phase 17.1 rerun reuses the same holdout and is not a new independent result. It applies `phase17-metrics-v2`, which deduplicates predictions and truths separately at component, component/fault-mode, and exact-hypothesis levels, records raw candidate precision and duplicate counts separately, and distinguishes selected, oracle-best, and legacy-first parameter errors. The corrected report, new off-grid evaluation, SPICE tie analysis, runtime accounting, and limits are documented in [Phase 17.1 evaluation integrity hotfix](PHASE17_1_EVALUATION_HOTFIX.md).
