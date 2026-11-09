# Phase 17.1 Evaluation Integrity Hotfix

## Scope and records

Phase 17.1 corrects candidate-level scoring, parameter-error reporting, run metadata, failed-call timing, and SPICE tie reporting. The metric schema is `phase17-metrics-v2`. Source was frozen at `064ec91eef9b7d38aef58fbaca6ca26500c8d6d5` for the new dataset. The original Phase 17 manifest, private truth, and result JSON were preserved unchanged.

| Record | Run / dataset | Interpretation |
|---|---|---|
| Original Phase 17 | `phase17_20261103T080936_c04b602f` | Original frozen metrics and predictions; unchanged |
| Existing-holdout v2 rerun | [`phase17_20261103T103812_87517c46`](../evaluation/phase17_1/results/phase17_20261103T103812_87517c46.json) | Same 10-case holdout re-evaluated with corrected metrics; explicitly not a fresh independent result |
| Off-grid v2 | [`phase17_20261103T105709_5bfcd352`](../evaluation/phase17_1/results/phase17_20261103T105709_5bfcd352.json) | New synthetic parameter set, reused topology templates; not a new-topology or physical evaluation |

The old holdout still has public SHA-256 `ee70b06c71d312e36f455a0cdda46492f60f3ae728a83cbb505de4cca2a3b85e` and private-truth SHA-256 `62c376f0d8582bfe4bbe4fee7d77a7e63ddc62fec1535d2e1e1e52eb1d12182b`. The new off-grid dataset is `circuitcare-phase17-1-off-grid-v2`, with public SHA-256 `451f9a0fe661602ab16736ed0ff9ca8cd2955a0e6834a3f94ea6c17d4e1d68d4` and private-truth SHA-256 `d8ef2fac227ecee403def32040914c1fe31d1c82add705ae764d8eb3043f0ce7`. It contains 10 off-grid faults over five reused topology templates and zero new topologies. Resistor factors are 1.37 and 0.73; regulator offsets are +0.7 V and -0.7 V. The candidate grid and original holdout were not changed.

## Corrected candidate metrics

Precision and recall are now computed after one-to-one matching and deduplication at each identity level: component, component/fault mode, and exact normalized hypothesis. Raw proposal precision/counts and duplicate counts remain separately available. Thus the old holdout Rules component precision is 0.2941 after unique-component scoring while raw candidate precision remains 0.0495; this is a metric-definition correction, not an algorithm improvement.

| Dataset / method | Unique component P/R | Unique mode P/R | Exact hypothesis P/R | Raw candidates per batch |
|---|---:|---:|---:|---:|
| Existing holdout / Rules | 0.2941 / 1.0000 | 0.0980 / 1.0000 | 0.0495 / 1.0000 | 202 |
| Existing holdout / LLM Only | 0.4251 / 0.9000 | 0.0391 / 0.1000 | 0 / 0 | 25.6 mean |
| Existing holdout / Hybrid | 0.2941 / 1.0000 | 0.0980 / 1.0000 | 0.0491 / 1.0000 | 203.6 mean |
| Off-grid / Rules | 0.2941 / 1.0000 | 0.0980 / 1.0000 | 0 / 0 | 202 |
| Off-grid / LLM Only | 0.4394 / 1.0000 | 0.0709 / 0.2000 | 0 / 0 | 28.2 mean |
| Off-grid / Hybrid | 0.2941 / 1.0000 | 0.0980 / 1.0000 | 0 / 0 | 203.6 mean |

The Rules batch contains 34 unique components, 102 unique component/mode candidates, and 202 exact candidates; 168 component and 100 mode duplicates are removed when calculating unique-level metrics. Off-grid exact recall is zero because the supported proposal catalog does not include the exact injected parameter.

Parameter error is split into explicitly selected-candidate, oracle-best-candidate, and legacy-first-candidate metrics. A missing selection rank is `NOT_EVALUATED`. On the existing holdout, Rules oracle-best mean relative error is 0 and legacy-first mean is 1.2. On off-grid, the corresponding means are 0.32221 and 0.89207. Oracle-best is a post-hoc optimistic bound using the answer; it is not system selection performance. LLM parameter error is not evaluated where no matching supported candidate exists.

## Candidate-grid and SPICE analysis

All 10 independent old-holdout truths are `IN_CATALOG_GRID` (50/50 repeated case-runs); none is unsupported. The old exact recall therefore measures coverage and ranking over a grid that already contains the truth, not discovery of unknown continuous parameters. In the new off-grid dataset, 0/10 truths are in-grid and 10/10 are `OUT_OF_CATALOG_GRID`; unsupported faults are 0/10. This is a limited parameter-generalization check on reused circuit templates.

| Dataset / method | Strict Top-1 | Tie-aware Top-1 | Top-3 | MRR | SPICE runs |
|---|---:|---:|---:|---:|---:|
| Existing holdout / Rules + SPICE | 0.30 | 1.00 | 0.40 | 0.4156 | 1,010/1,010 successful |
| Existing holdout / Hybrid + SPICE | 0.30 | 1.00 | 0.40 | 0.4156 | 1,010/1,010 successful |
| Off-grid / Rules + SPICE | 0 | 0 | 0 | 0 | 1,010/1,010 successful |
| Off-grid / Hybrid + SPICE | 0 | 0 | 0 | 0 | 1,010/1,010 successful |

The tie tolerance and per-case candidate ranks/residuals are included in each result JSON. On the existing holdout, 40/50 repeated case-runs have residual ties/identifiability ambiguity; the true candidate is in the best-residual set, so tie-aware Top-1 is 1.0, but strict Top-1 remains 0.30. This indicates that the measured final scalar output often does not uniquely distinguish the correct in-grid fault. For off-grid, the correct exact candidate was never proposed or simulated (`CORRECT_CANDIDATE_NOT_SIMULATED`); 40/50 runs have ties among the best *incorrect catalog candidates*. Those candidate ties do not make the true diagnosis ambiguous because the true hypothesis is absent. `ambiguous_case_count` is therefore 0 for the off-grid truth, while tied-best-candidate count is 40/50.

Existing observed node strata are 2 one-node cases and 8 two-node cases. Off-grid has the same counts. Same-case measurement-count intervention is `NOT_EVALUATED`; these descriptive strata do not establish that adding a measurement improves performance. Paired method differences are descriptive on repeated shared cases and show no exact-recall gain for Hybrid over Rules in either dataset.

## LLM runtime and failures

The rerun captures wall time around every attempted local-model call, including malformed responses. On the reused holdout, 48/50 calls succeeded and 2/50 returned malformed JSON; mean successful-call latency was 14.9439 s, mean attempted-call latency 15.1526 s, total LLM wall time 757.6304 s, and total benchmark wall time 805.4567 s. The failed calls were in repeated runs of `p17h_0009`; individual timings and parse errors are present in the run record. Off-grid LLM calls succeeded 50/50; mean attempted/successful latency was 19.9129 s, total LLM wall time 995.645 s, and total benchmark wall time 1,051.98 s. Model-reported timing is retained separately from measured wall time.

## Statistical and practical limits

Each dataset has 10 independent cases and five topology groups, repeated five times. Repetitions are not additional independent cases. Topology-cluster bootstrap intervals use only five groups and are exploratory. Node strata and paired comparisons are descriptive; no significance or causal claim is made. The new dataset reuses topology templates and tests only the selected off-grid parameter values. Neither dataset measures physical damage, repair accuracy, real-world failure probability, or broad unseen-topology generalization. SPICE completion is a simulator execution measure, not diagnosis accuracy.

The saved-results evaluation UI now reads the Phase 17.1 results and presents corrected unique/raw metrics, ties, and runtime metadata. It does not start benchmark runs. A first off-grid build attempt was rejected by the runner's blinded-ID format check before prompt/truth loading or scoring; it produced no evaluation result. The builder was corrected, and only the valid v2 dataset and run are retained.
