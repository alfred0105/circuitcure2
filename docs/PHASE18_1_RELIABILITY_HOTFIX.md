# Phase 18.1 Reliability & Evidence Integrity Hotfix

## Scope and preserved data

This hotfix hardens diagnostic context and evidence handling. It does not add an inference model or change the Phase 17/17.1 evaluation datasets, results, `v0.1.0-mvp` tag, or release. Physical repair validation remains `NOT_EVALUATED`.

## Changes

- Active, user reported observations are hashed independently from optional what-if preview observations and posterior. The default report contains only the active diagnosis. A user can opt in to a separate `PREVIEW_ONLY` section; its measurement hash and `simulated_demo` source are validated independently.
- The Fault AI measurement form now has an explicit “record measurement” checkbox. An unchecked numeric widget default is missing data; a checked 0.0 V is retained as a valid observation. Nonfinite values and negative uncertainty are rejected by the canonical measurement validator. Negative voltage remains valid.
- Measurement source codes are standardized as `user_measurement`, `simulation`, `simulated_demo`, `imported_measurement`, and `unknown`. Legacy Korean labels are mapped explicitly and retained as `source_original`. A user reported measurement is not treated as independently verified physical evidence.
- Simulation fingerprints cover netlist, SPICE deck and included model files, component mapping, probes, fault parameters, and simulation settings. A live run records its fingerprint and result file hashes. A cached result without a verifiable manifest is `PROVENANCE_UNVERIFIED`; a mismatch or changed result file is `STALE`, and neither is used in the active analysis or report. The existing legacy cache files are retained unchanged.
- Report validation checks measurement and hypothesis set hashes, circuit identity, comparison/risk scenario and component/fault context, cached simulation provenance, and damage-confirmation claims. Invalid prior source evidence can no longer be reactivated by activation or rollback; eligibility checks and state/event writes run in a single SQLite transaction.
- Evidence ZIP exports record which evidence artifacts are included, their archive path and digest, and which sources are reference-only. Bundle verification cross-checks manifest hashes, report references, circuit/scenario identity, duplicate paths, path traversal, and compressed/expanded size limits.
- Markdown and offline HTML now show measurement provenance, hypotheses and validation status, SPICE comparisons, conditional posterior rows, risk candidates, and inspection details when those values exist. Structured path fields, known private fields, and common absolute path strings are redacted from report snapshots.

## Regression and integration coverage

Automated regression tests cover zero/missing measurements, canonical and legacy sources, active-vs-preview measurement hashes, same-size input content changes, tampered cached result files, cross-scenario risk, report-to-bundle artifact references, invalidated prior rollback, and rollback transaction failure. The existing full test suite remains in place.

Validation results, Python versions, CI links, and limitations are recorded below after the final push:

| Check | Result |
|---|---|
| Full local test suite | Python 3.12: `161 passed`; no test removed or skipped on this host |
| Python 3.11 GitHub Actions | Pass · [run 41754936208](https://github.com/alfred0105/CircuitCare-AI/actions/runs/41754936208) |
| Python 3.12 GitHub Actions | Pass · [run 41754936208](https://github.com/alfred0105/CircuitCare-AI/actions/runs/41754936208) |
| Static checks | `compileall` and `git diff --check` passed |
| Live ngspice scenario | ngspice 47: R1 10 kΩ and U1 output-rise 8 V each completed baseline and fault runs; temporary output only |
| Simulation manifest | Live R1 manifest verified, then correctly changed to `STALE` after altering `comparison.json` |
| UI / AppTest | AppTest passed with all 11 tabs; live Streamlit smoke on localhost:8503 showed `PROVENANCE_UNVERIFIED`, withheld legacy risk data, and report controls |
| Phase 17 and 17.1 evaluation data | No tracked evaluation/data files changed |
| `v0.1.0-mvp` tag and release | Preserved unchanged; no new release created |

## Known limits

- Older cached simulations do not have generation-time fingerprints. They remain on disk for reference but cannot be presented as verified current results until a matching manifest is produced by a new live run.
- A reported measurement source reflects the operator’s declaration; it is not an independent instrument attestation.
- Scenario comparison can only verify fields actually carried in simulation artifacts and manifests. Missing provenance is `NOT_EVALUATED` or rejected, not assumed to match.
- Physical damage, repair outcomes, real PCB reference matching, and generalization across independent datasheets remain `NOT_EVALUATED`.
