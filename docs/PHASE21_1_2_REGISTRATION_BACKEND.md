# Phase 21.1.2 — Registration Validation and Backend Stabilization

## Result

**Backend implementation: complete. PRISM regression: complete. Independent accuracy validation: BLOCKED.**

The reusable registration interface is `circuitcare.vision.reconstruction.registration_service.register(RegistrationRequest(...))`. It validates the request, detects image anchors, extracts KiCad-derived design anchors, performs bounded automatic matching or manual calibration, estimates a transform, evaluates supplied held-out points, and returns a versioned serializable record. `require_verified_transform()` rejects transforms unless they are independently verified and have the expected coordinate frames, board side, finite matrix, and invertibility. Automatic design-assisted estimates cannot be promoted to verified without disjoint reviewed validation points and a predeclared pixel threshold.

The record schema is `circuitcare.registration.v1`. It includes board identity and side, mode/status, matrix, source/target frame and units, candidate/inlier counts, independent-validation metrics, provenance, error code/detail, runtime, and resource limits/usage. Resource-limit exhaustion is a failure with no transform. The same service is used by `scripts/calibrate_prism.py`; it can be called directly by future API or UI adapters without adding a web server.

## Independent validation status

PRISM provides paired board images and KiCad-derived reference geometry, but the pinned dataset metadata does not provide independent reviewed image-to-design correspondences or a physical scale/registration reference. The automatically matched through-hole/via anchors participate in transform estimation and therefore cannot also serve as held-out truth. No independent validation set was found for any of the 12 boards / 24 views.

| Measure | Result |
|---|---:|
| PRISM images | 24 |
| Images with automatic matching candidates | 23 |
| Transform estimates generated | 16 |
| Transforms rejected | 8 |
| Independently verified transforms | 0 |
| Held-out RMSE / P95 | Not available; no independent points |
| Projected PRISM reference geometry | 0 views; correctly withheld |

The PRISM validation utility reports 24 checked views and zero verified views. Ground-truth projection reports `NOT_PROJECTED_REGISTRATION_NOT_VERIFIED`. TOP and BOTTOM are processed separately; BOTTOM mirror candidates are searched rather than forced. These are implementation safeguards, not independent proof of orientation accuracy. The validation acceptance condition requiring one to three real independently anchored boards remains **BLOCKED** pending reviewed image/design point pairs, ideally measured from the physical board or a separately authored reference not used by matching.

## Resource behavior and regression comparison

Default caps: 4,000,000 image pixels; 128 design and image anchors; 1,000,000 anchor-pair products; 200,000 transform hypotheses; 10 retained candidates; 20 seconds per service request. OpenCV RANSAC uses a stable seed derived from input coordinates. Pair products are rejected before materializing oversized pair arrays; timeout/hypothesis/pair limits return failure, never a successful partial registration.

The largest PRISM input was 2,127,600 pixels. The largest pair-product workload was 615,825, whose ratio array is estimated at 4,926,600 bytes (~4.70 MiB). This is only the bounded ratio-array estimate, **not total process RSS**; image decoding, OpenCV, and Python object memory are not represented by that figure. Per-board elapsed timings and input fingerprints are recorded in `data/benchmarks/phase21_1_2/runtime_comparison.json`.

| Same-host measurement, 24 views | Baseline at `396f8f8` | Phase 21.1.2 |
|---|---:|---:|
| Matcher total | 21.280 s | 20.952 s |
| Matcher median | 0.0264 s | 0.0184 s |
| Matcher P95 | 3.175 s | 3.077 s |
| Matcher maximum | 8.458 s | 8.529 s |
| Generated / rejected | 16 / 8 | 16 / 8 |

The matcher-only timings are comparable; service timings additionally include image detection and geometry extraction (21.038 s total, 8.533 s maximum). The data shows similar runtime and unchanged transform counts, not a broad speedup. The saved baseline match records and current run used PRISM commit `9189c87bbec6995b745b809db99cef6d3269401f`; the comparison records the CircuitCare base SHA, dirty-worktree state, host, Python/OpenCV versions, and per-input hashes. Earlier historical reports of 21 generated / 3 failed belong to an earlier experiment stage; the current pinned run is 16 / 8. The earlier record is retained in its historical report.

## Reproduction

With the pinned PRISM checkout at `data/external/prism`:

```bash
python scripts/calibrate_prism.py --all --mode AUTO_REGISTRATION --artifact-dir data/benchmarks/phase21_1_2
python scripts/validate_registration.py --registration data/benchmarks/phase21_1_2/summary.json --output data/benchmarks/phase21_1_2/validation/summary.json
python scripts/build_pcb_reference.py --all --registration data/benchmarks/phase21_1_2/summary.json --output-dir data/benchmarks/phase21_1_2/projected_references
```

The first command records schema, SHA-256 input fingerprints, code base SHA and whether the worktree was dirty. Anchor/match/overlay intermediates are ignored by Git; compact result summaries and the runtime comparison are retained. CI preserves the Python-version, ngspice, stress, native OCR and crash-injection jobs. Pushes cancel superseded runs; pip caches are enabled; documentation-only pushes/PRs skip those jobs. The full PRISM job is opt-in via `workflow_dispatch` so routine CI does not fetch/benchmark the external dataset.

## Verification and remaining gate

Focused service/coordinate/matching tests cover serialization, TOP/BOTTOM and frame validation, reviewed held-out verification, duplicate/overlap prevention, unverified downstream blocking, deterministic estimates and resource-limit failures. The full project suite and CI must pass before release. No PCB AI model, UI, SPICE, or Bayesian implementation was changed.

Phase 21.2 may begin building and reviewing Ground Truth, but registration-derived labels must remain withheld until at least one representative physical board has a separately reviewed calibration/validation bundle and acceptable held-out metrics. This phase does not establish electrical connectivity accuracy.
