# PCB pin connectivity: review follow-up run

Run date: 2026-11-09. This run uses only the pin-only PCB CAD archive; no Phase 22 schematic graphs were loaded. The repository-grouped split remains 153 training, 19 validation, and 18 test boards (190 labeled boards total).

## Candidate audit correction

The earlier report called `E/N` “mean candidates per pin.” For an undirected candidate graph, average pin degree is `2E/N`. The earlier 104.2 was therefore half the true average degree. This run reports both.

| Radius / k | Validation candidate recall | Correct mean degree `2E/N` |
|---|---:|---:|
| 25 mm / 8 | 62.69% | 116.95 |
| 50 mm / 16 | 89.57% | 190.72 |
| 100 mm / 32 | 99.986% | 208.39 |
| 150 mm / 64 | 100.00% | 208.44 |

The selected 100 mm / 32 graph is within 0.5 percentage points of the best validation recall and slightly sparser than 150 mm / 64. Test candidate recall is 100%, but each pin still has about 208 candidate neighbors on average; coverage does not imply the classifier can identify the right neighbors.

## Held-out test comparison

| Model | Candidate PR-AUC | ROC-AUC | End-to-end Recall@5 |
|---|---:|---:|---:|
| Random score | 0.0627 | 0.4978 | 4.43% |
| Distance only | 0.0889 | 0.5730 | 8.61% |
| LightGBM geometry/type | 0.0840 | 0.5819 | 10.06% |
| LightGBM + value/footprint/pad text | 0.0853 | 0.5875 | 9.97% |
| Edge-aware GNN + hashed pad features (CUDA) | 0.0687 | 0.5337 | 8.12% |

The feature ablation is mixed: hashed values, footprints, pad names/numbers, and pad shapes slightly improve LightGBM PR-AUC and ROC-AUC, but do not improve Recall@5. The GNN uses a separate 16-nearest-neighbor message graph instead of aggregating over the much denser candidate graph; it improves over the prior GNN run, but still trails LightGBM and the distance score on PR-AUC.

## Net reconstruction strategies

Parameters were selected on validation only. Mutual Top-K gives the highest pairwise F1 for the trained models, but the resulting clusters still over-merge:

| Model / cluster strategy | Net-pair P | Net-pair R | Net-pair F1 | Predicted pairs / true pairs |
|---|---:|---:|---:|---:|
| LightGBM semantic / mutual Top-12 | 6.79% | 95.59% | 12.68% | 77,614 / 5,514 |
| GNN semantic / mutual Top-12 | 7.11% | 91.26% | 13.19% | 70,791 / 5,514 |
| LightGBM semantic / constrained merge | 10.13% | 11.52% | 10.78% | 6,268 / 5,514 |

The constrained merge sharply reduces predicted-pair volume by requiring cross-cluster support and blocking explicit ground-versus-supply pad-name conflicts. It rejects many merges, but its recall is too low and it still creates false-merge clusters. The random baseline also gets 11.80% net-pair F1 under a validation-calibrated threshold-union strategy by producing 87,931 pairs. That result shows pairwise F1 alone can reward oversized, low-confidence nets; report prediction volume and false merges alongside it.

## Changes and limits

- Corrected the candidate-degree formula and candidate-selection budget in the runner.
- Added random and distance-only baselines, semantic feature ablation, and threshold-union / mutual Top-K / constrained clustering comparisons.
- Applied the same cross-component rule to candidate and truth pairs. Same-net membership remains the target; it is not a direct copper-segment label.
- Did not use original CAD tracks or vias as features because they reveal the net target and cannot stand in for traces detected from a photograph. No image-to-pad registration or photo-derived trace evidence was evaluated.
- The held-out test contains 18 boards. Results are exploratory, not a stable real-world accuracy claim.

## Artifacts

- Results: `data/benchmarks/pcb_pin_connectivity/pcb_pin_connectivity_review2_results.json`
- Runner: `scripts/benchmark_pcb_pin_connectivity_review2.py`
- Geometry LightGBM: `data/benchmarks/pcb_pin_connectivity/models/lightgbm_pcb_geometry_review2.txt`
- Semantic LightGBM: `data/benchmarks/pcb_pin_connectivity/models/lightgbm_pcb_semantic_review2.txt`
- CUDA GNN: `data/benchmarks/pcb_pin_connectivity/models/gnn_cuda_pcb_semantic_review2.pt`
- Live progress page: `http://127.0.0.1:8766/`
