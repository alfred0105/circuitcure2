# Pre-freeze exploratory runs

These local Phase 17 dry runs were executed before the benchmark implementation was committed and before a code fingerprint was fixed. They are not accepted as the final Phase 17 evaluation and are excluded from the dashboard loader.

`phase17_20261103T072822_91193a59.json` was a Rules-only diagnostic run before the runner status correction; its empty LLM-only and Hybrid entries are invalid and must not be interpreted as measured zero accuracy. The later records are development-only LLM/SPICE experiments on the reused Phase 15 set. The frozen, reportable evaluation is stored at the top level of `results/` after the Phase 17 code commit.
