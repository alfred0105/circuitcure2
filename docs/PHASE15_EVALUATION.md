# Phase 15 evaluation — historical record

> **Historical metrics below are invalid as blinded model performance.** The pre-hotfix LLM prompt included answer-bearing case IDs and described cases as known injected faults. Original files are preserved under `evaluation/fault_hypothesis/results/leaked_development_result/` and tagged `LEAKED_DEVELOPMENT_RESULT`. Do not use those numbers as model accuracy.

## Scope and status

Two independent tracks are maintained: generated fault hypotheses and datasheet rating extraction. The fault benchmark is a **measured development set of three simulator-injected cases**, not real repairs and not an independent holdout. The datasheet benchmark uses four official-source documents from four component families; its first run exposed parser issues, and those same documents informed fixes. Its current post-fix measurements are therefore development-pilot metrics, not unseen-holdout accuracy. See [the separate datasheet report](DATASHEET_EXTRACTION_BENCHMARK.md).

## Fault hypothesis run

The pre-hotfix local run used Python 3.12.14, macOS arm64, Ollama `gemma4:12b` (digest `c7597fc90b86aff6965317f37dd5465402284e26382d9b4bf2dfe3ef1491f4b6`), and ngspice 47. It covered the 5 V→8 V regulator source example, divider R1 1 kΩ→10 kΩ, and RC RCHARGE 10 kΩ→100 kΩ. Observations were generated from simulation at the final transient sample. These details describe the archived run, not a valid blinded evaluation.

| Method | Candidates | Correct | Precision | Recall | Top-1 / Top-3 | Executable | Sim success |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule catalog | 43 | 3 | 0.0698 | 1.00 | 1.00 / 1.00 | 1.00 | 43/43 (1.00) |
| Local LLM only | 4 | 3 | 0.7500 | 1.00 | 1.00 / 1.00 | 1.00 | 4/4 (1.00) |
| Rules + local LLM + validator | 43 | 3 | 0.0698 | 1.00 | 1.00 / 1.00 | 1.00 | 43/43 (1.00) |

The table is retained verbatim for audit context and is classified `LEAKED_DEVELOPMENT_RESULT`; every listed LLM accuracy value is invalid because the prompt leaked the case label.

### Historical leaked run observations (invalid for model evaluation)

Candidate precision is affected by the intentionally broad deterministic parameter catalog. In this run, Top-3 hit rate was 1.00 while mean precision@3 was 0.333 for the rule and combined methods. Local LLM generation took 5.70 s, 8.10 s, and 13.98 s across the three cases (mean 9.26 s); mean ngspice runtime was about 0.021 s per attempted rule/combined candidate and 0.019 s for LLM-only candidates. A prior local run returned a different number of proposals and missed one injected value, illustrating model output variability. These small synthetic values are not representative accuracy.

All 43 explicit catalog simulations in the combined method succeeded. For each synthetic case, its injected truth ranked first under the residual fit. For the known U1 8 V observation, the matching candidate predicted 8.0 V, residual 0 V and z=0; the 6 V alternative predicted 6.0 V, residual +2.0 V and z=10 with σ=0.2 V. Similar comparisons are present in the JSON output for divider and RC cases. The consistency label refers only to these simulated observations and the configured model.

### Findings and limitations

- The LLM-only run included all three injected faults plus one extra candidate. The model can vary across repeated calls.
- No hallucinated references were produced in this run; unsupported proposal rate was 0 for these constrained proposals. This does not test unbounded circuits.
- The combined pipeline retained the deterministic proposal catalog and therefore had the same candidate precision/recall as rules alone on this dataset.
- No physical repair label, measured board data, independent topology, or failure-probability calibration is available.
- Not-supported classes include arbitrary IC internal failures and component types with no explicit supported model or SPICE mapping.
- Bayesian reuse is conditional: identical predicted node coverage and explicit observation uncertainty are required. It uses the existing equal-prior engine and never treats LLM confidence as a prior.

## Reproduction

```bash
.venv/bin/python scripts/evaluate_fault_hypotheses.py --run-llm --model gemma4:12b --llm-runs 3 --simulate
```

See [Phase 15.1 integrity hotfix](PHASE15_1_EVALUATION_INTEGRITY.md) and `evaluation/fault_hypothesis/results/leakage_free_development_metrics.json` for the answer-free rerun. Proposal quality and simulation-based ranking quality are separate. The same three previously used synthetic development scenarios remain non-independent; `holdout_status` is `INDEPENDENT_HOLDOUT_NOT_EVALUATED`.
