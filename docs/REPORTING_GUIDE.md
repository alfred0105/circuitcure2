# Reporting guide

The overview's report controls export a snapshot of the current workspace. Export is read-only and never reruns circuit analysis.

## Report files

- **JSON** is the structured snapshot for later inspection or processing.
- **Markdown** summarizes circuit identity, measurements and sources, hypotheses, SPICE comparison, conditional diagnosis ranking, risk review, evidence, limitations, and Git revision.
- **HTML** is a self-contained UTF-8 page with offline tables for available measurements, hypotheses, risk rows, and evidence.
- **Evidence Bundle ZIP** contains those three outputs and a manifest with SHA-256 hashes. Each report evidence row says whether its file is included or reference-only. Raw PCB photographs, original datasheet PDFs, private repair database files, and external paths are excluded by default.

## Evidence labels

The report differentiates `user_measurement` (user reported), `simulation`, `simulated_demo`, imported, and unknown sources. A 0.0 V reading is valid when explicitly recorded. Active diagnosis and What-if preview hashes are checked separately; an opted-in preview remains `PREVIEW_ONLY`. A cache without a matching input/result fingerprint is `PROVENANCE_UNVERIFIED` or `STALE` and is excluded from the active report. `LIVE_SIMULATION` and verified `CACHED_SIMULATION` still do not prove physical component damage. Conditional Bayesian weights and inspection scores are not failure probabilities.

Every report records the circuit hash and configured input hashes when available. Findings refer to evidence IDs. The validator checks measurement/hypothesis context, circuit and scenario consistency, stale cache status, and damage-confirmation claims. The bundle verifier cross-checks evidence references against included files, recomputes file hashes, and rejects duplicate or unsafe paths and oversized archives.

## Review before sharing

Inspect the circuit name and hash, active fault, observation source, scenario ID, rating notes, model status, and limitations. Private data is stored locally. Review an export before sharing it; do not include original photos, source PDFs, or repair history by default.
