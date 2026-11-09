# Phase 21.1 — Dataset Foundation

## Bundle format

`PCBDatasetBundle` (schema `pcb-dataset-bundle-v1`) records board ID/revision, dataset source/commit, repository and dataset license metadata, source paths and SHA-256 hashes, images, design artifacts, calibration, and reference availability. Optional or missing artifacts are recorded as `NOT_AVAILABLE`; source files are referenced in place rather than copied into the CircuitCare repository.

Generate manifests with:

```bash
python scripts/build_pcb_bundle.py --all
```

By default manifests are written to ignored `data/benchmarks/phase21_1/bundles/`. PRISM inputs remain at ignored `data/external/prism/`.

## PRISM source and inventory

- Source: [KamMalinski/PRISM](https://github.com/KamMalinski/PRISM)
- Pinned revision: `9189c87bbec6995b745b809db99cef6d3269401f`
- Cases: Cascade, Connector, Crossover, Divider, Elements, Filter, Floating, Groundplane, Meander, Round, Simple, Transistors.
- Inputs: each case has TOP/BOTTOM PNG images and a matching `.kicad_pcb` and `.kicad_sch`.
- Evaluation references include geometry, components and expected netlist JSON, loaded separately from image inference.
- KiCad Position CSV is not supplied by the adapter; manifest marks it `NOT_AVAILABLE`.
- These are controlled rendered examples, not photographs of physical boards. No physical revision, measurement data or independent photo/design calibration truth is supplied.

The pinned repository has an AGPL-3.0 `LICENSE.txt`; the dataset README describes the examples but does not provide a separate dataset-specific license grant. Whether every dataset artifact is covered and what redistribution of derived assets permits remain **unresolved**. Manifests record `dataset_specific_license=NOT_STATED` and `redistribution_status=REVIEW_REQUIRED`; this project does not mirror the large source bundle into its Git history.

## Actual PCB bundle requirements

For a physical board, collect a stable board ID and revision; original front/back photos with orientation metadata; matching KiCad PCB, schematic/netlist and Position CSV where available; source hashes; at least four well-spread calibration correspondences plus separate validation points; layer and bottom-view conventions; and independent adjudication sources for electrical claims. Add continuity measurements and conditions for pad/pin-to-net truth. Missing evidence limits the uses of a bundle rather than silently excluding it.

## Completeness states

`AVAILABLE` means a source artifact exists and has been hashed. It does not mean the artifact has been validated against a photo or physical board. Missing design identity, revision, registration or electrical truth remains explicit in the manifest and validation report.
