# PCB pin connectivity experiment

Run date: 2026-11-09  
Input: `data/external/pcb-photo-cad-2026-11-09/pin_only_training` only. Prior schematic/Phase 22 training records were not loaded.

## Data and split

- 193 deduplicated PCB CAD boards inspected; 190 had usable cross-component connection labels and 3 were excluded.
- 153 train, 19 validation, 18 test boards, grouped by source repository.
- Test covers 5,514 eligible cross-component same-net pin pairs.
- Inputs contain PCB pad positions, component placements, pad geometry and heuristic component class. The label is same-net membership, which can represent an indirect electrical path, not necessarily one copper segment.

## Candidate-generation audit

The candidate graph does not add truth pairs. Same-component pin pairs are excluded from both candidates and the truth denominator.

| Radius / nearest pins | Validation candidate recall, micro | Mean candidates per pin |
| --- | ---: | ---: |
| 25 mm / 8 | 62.69% | 58.5 |
| 50 mm / 16 | 89.57% | 95.4 |
| 100 mm / 32 | 99.986% | 104.2 |
| 150 mm / 64 | 100.00% | 104.2 |

The validation rule chose 100 mm / 32: it is within 0.5 percentage points of the best recall and costs slightly fewer candidates than 150 mm / 64. The selected setting had 99.989% train, 99.986% validation and 100% test candidate recall. Test recall is reported after the validation-based selection and was not used to choose the setting.

## Held-out test results

| Model | Device | Candidate PR-AUC | End-to-end Recall@5, micro | Oracle Recall@5 | Net-pair P / R / F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| LightGBM | CPU | 0.0840 | 10.06% | 35.19% | 6.37% / 99.98% / 11.97% |
| GNN | NVIDIA RTX 4060 Ti CUDA | 0.0664 | 7.38% | 35.19% | 5.10% / 38.65% / 9.02% |

LightGBM remains the stronger ranking baseline. Its high net-pair recall and very low precision show that connected-component transitive closure over-merges predicted nets. The oracle Recall@5 is the ceiling for the current top-five metric after candidate generation; it is not model accuracy.

## Changes made

- Replaced schematic coordinates with pad coordinates from Eagle/KiCad PCB CAD.
- Kept all eligible same-net truth pairs separately from the geometry-only candidate graph, so candidate recall measures actual coverage.
- Applied the same cross-component rule to candidates, labels, ranking denominators and oracle recall.
- Added validation-only candidate-radius selection, oracle Recall@5, and validation-calibrated transitive net reconstruction metrics.
- Trained only on PCB-derived pin data. LightGBM used CPU because the installed Windows build has no CUDA learner; the GNN trained on the RTX 4060 Ti.

## Limits

This is a CAD-to-CAD PCB experiment, not a photograph-to-netlist evaluation. It does not yet use image-to-pad registration, visible copper trace evidence, or via segmentation. The 18-board held-out test set is small. The same-net target also differs from a literal direct wire segment. Net reconstruction needs a merge strategy that controls transitive over-merging before this can be treated as a practical circuit-restoration result.

## Artifacts

- Metrics: `data/benchmarks/pcb_pin_connectivity/pcb_pin_connectivity_results.json`
- LightGBM model: `data/benchmarks/pcb_pin_connectivity/models/lightgbm_pcb_pin_only.txt`
- CUDA GNN weights: `data/benchmarks/pcb_pin_connectivity/models/gnn_cuda_pcb_pin_only.pt`
- Experiment runner: `scripts/benchmark_pcb_pin_connectivity.py`
- Live progress page: `http://127.0.0.1:8766/`
