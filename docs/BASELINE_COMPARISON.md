# Phase 17 Baseline Comparison

Run: `phase17_20261103T080936_c04b602f` · [full run record](../evaluation/phase17/results/phase17_20261103T080936_c04b602f.json)

This is an independent **synthetic** holdout: 10 cases in five topology groups. Each method was repeated five times over those same cases; repetitions are not additional independent cases. Candidate metrics are averaged across the five runs. SPICE ranking metrics therefore summarize 50 repeated case-runs, while the independent sample size remains 10.

| Method | Component P / R | Fault mode P / R | Exact hypothesis P / R | SPICE Top-1 / Top-3 | MRR | Proposal runtime per 10-case run | SPICE runs / successes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rules Only | 0.0495 / 1.0000 | 0.0495 / 1.0000 | 0.0495 / 1.0000 | — | — | 0.0072 s | — |
| LLM Only | 0.3467 / 0.9200 | 0.0377 / 0.1000 | 0.0000 / 0.0000 | — | — | 225.5918 s* | — |
| Hybrid | 0.0490 / 1.0000 | 0.0490 / 1.0000 | 0.0490 / 1.0000 | — | — | 225.5990 s* | — |
| Rules + SPICE | — | — | — | 0.3000 / 0.4000 | 0.4156 | 0.0072 s | 1,010 / 1,010 |
| Hybrid + SPICE | — | — | — | 0.3000 / 0.4000 | 0.4156 | 225.5990 s* | 1,010 / 1,010 |

P = precision; R = recall; MRR = mean reciprocal rank. Proposal runtimes are the mean per 10-case batch across five repetitions. SPICE simulation time averaged 7.6364 s per Rules batch and 6.2504 s per Hybrid batch. The equal per-case cap was 40 simulations; all valid proposals fit within the cap. Both methods completed all recorded simulations without a SPICE failure.

\* The LLM runtime summary omits the two failed calls' latency because the adapter did not return a duration on malformed JSON. Forty-eight of 50 calls returned parseable JSON. The two failures occurred on case `p17h_0009` in repetitions 1 and 2. Thus the recorded LLM and Hybrid runtimes understate total wall time for those batches.

## Interpretation

- Rules generated 202 candidates per repeated 10-case run. It found each supported exact target at least once (exact recall 1.0), but precision was only 0.0495 because it produced many false candidates.
- LLM Only improved mean component precision to 0.3467 and component recall was 0.9200, but exact fault-mode-plus-parameter precision and recall were both 0. Its mean candidate count was 26.6 per repeated 10-case run.
- Hybrid retained 1.0 exact recall from Rules, but precision (0.0490) was slightly below Rules. These results show no demonstrated exact-diagnosis gain from adding the LLM in this setup; LLM generation also added substantial latency.
- SPICE ranking put an exact target first in 30% of repeated case-runs and in the top three in 40%. Since the test set has only 10 independent cases and five topology groups, these repeated-run proportions are descriptive and not a broad accuracy guarantee.
- Exact hypothesis matching requires exact normalized parameter equality. The stored relative parameter-error field selects the first proposal with the same component and fault mode, rather than the closest or selected exact proposal. It is therefore not used here as a best-candidate parameter error; exact-match precision/recall are the relevant parameter-level metrics for this run.
- The bootstrap intervals in the evaluation report resample only five topology groups and are exploratory. They do not establish population-level confidence.
- All results are generated from synthetic ngspice circuits. They do not measure physical repair accuracy or real-world damage probability.

See [benchmark methodology](BENCHMARK_METHODOLOGY.md) for input boundaries, scoring definitions, uncertainty, and exclusions.

> **Metric version note:** This table is the original Phase 17 report and is retained as recorded. Corrected unique-level metrics and the reused-holdout v2 rerun are in [Phase 17.1 evaluation integrity hotfix](PHASE17_1_EVALUATION_HOTFIX.md). The changed component precision is a scoring correction, not a system-performance gain.
