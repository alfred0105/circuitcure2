# Phase 17 benchmark assets

`development/` contains an answer-separated repackaging of the three previously used Phase 15 simulator-injected examples. It remains a reused development set, never an independent holdout.

`scripts/build_phase17_synthetic_holdout.py` generates five new circuit topologies and two injected-fault cases per topology. It saves public inputs separately from `private_ground_truth.json`; the manifest records dataset hashes and the code-freeze Git SHA. The data is synthetic ngspice output, not physical repair evidence. Once the holdout has been evaluated, do not tune on it or rerun a changed method and call the outcome independent.

`validation/` is reserved; no validation split is claimed until one is actually created and frozen. `results/` uses append-only, uniquely named JSON run records. `scripts/run_phase17_benchmark.py` is explicit and never runs from the dashboard.
