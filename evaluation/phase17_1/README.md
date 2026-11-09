# Phase 17.1 evaluation data

This directory contains only the new synthetic off-grid dataset v2 and the two Phase 17.1 result records. It does not replace or modify `evaluation/phase17/`.

## `off_grid_v2`

- Dataset ID: `circuitcare-phase17-1-off-grid-v2`
- 10 injected faults; five reused topology templates; zero new topologies
- Public data SHA-256: `451f9a0fe661602ab16736ed0ff9ca8cd2955a0e6834a3f94ea6c17d4e1d68d4`
- Private ground truth SHA-256: `d8ef2fac227ecee403def32040914c1fe31d1c82add705ae764d8eb3043f0ce7`
- Candidate grid was frozen before evaluation; all ten truths are off-grid and supported by the existing fault model.

The `results/` directory contains a v2-metric reevaluation of the existing Phase 17 holdout (not a fresh independent result) and the off-grid evaluation. See [the Phase 17.1 report](../../docs/PHASE17_1_EVALUATION_HOTFIX.md) for interpretation and limits. The private ground-truth file is committed for project reproducibility; do not pass it to candidate generation or prompt construction.
