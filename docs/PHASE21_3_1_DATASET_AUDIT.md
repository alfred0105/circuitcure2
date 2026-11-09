# Phase 21.3.1 — Public PCB Dataset Audit

## Source and license

Source repository: [wangshnegqi/pcb-component-detection-dataset-yolo](https://github.com/wangshnegqi/pcb-component-detection-dataset-yolo), commit `0d7ae7310fa607389779b6132285336ad73e96e1`.

- Source declares PCB images and annotations under **CC BY 4.0** and model/repository assets under **Apache-2.0**. Those are separate licenses; dataset redistribution must retain attribution and indicate changes.
- The repository documents real PCB/electronics board photos with YOLO boxes and ten classes: resistor, capacitor, diode, LED, IC, connector, switch/button, header, crystal, Debug/TP.
- Its tracked split contains 117 training images/label files (3,104 boxes) and 13 validation images/label files (196 boxes). The source tree includes nine empty training label files and one empty validation label file. Those empty files are not reviewed proof of a negative image.
- Board IDs, revision IDs, camera view, and TOP/BOTTOM labels are not supplied. The source's `train`/`val` split is image-level, not verified board-level.
- Source filenames contain 11 `train3_*` families with explicit flip/rotation suffix variants, all within the training split. They are likely derived image groups by naming, but this alone does not establish source image or board identity.
- The source repository includes training metrics and a validation command/configuration for the checkpoint. Its validation split is therefore not a sealed holdout; the selected checkpoint may already reflect validation use.

## Board identity and duplicates

No Board ID is assigned to the 130 source images. The local sparse checkout contains the 13 validation photos only. SHA-256 and pHash search on those local 13 images found no exact or close duplicate candidates at pHash distance ≤6. This check did **not** compare training photos, so it cannot rule out cross-split same-board or near-duplicate imagery. Filename families provide candidate related views/derivatives for human review only.

The audit result is: **board count unresolved, board-side metadata absent, same-board split leakage not testable**. The source images remain excluded from claims of board-independent performance until board relationships are adjudicated. Do not interpret the 13 image count as 13 distinct PCBs.

## Class count

| Source class | Train boxes | Validation boxes |
|---|---:|---:|
| Resistor | 829 | 52 |
| Capacitor | 1,164 | 59 |
| Diode | 43 | 2 |
| LED | 88 | 15 |
| IC | 542 | 32 |
| Connector | 138 | 9 |
| Switch/button | 53 | 6 |
| Header | 183 | 8 |
| Crystal | 51 | 4 |
| Debug/test point | 13 | 9 |
| **Total** | **3,104** | **196** |

The tracked training image files were not downloaded into the project workspace; label counts were audited from source Git objects. Image dimensions and visual duplicate candidates were measured only for the 13 locally available validation images.

## Review manifest

[`public_pcb_component_validation_manifest.json`](../data/datasets/public_pcb_component_validation_manifest.json) records the source commit, CC BY 4.0 declaration, split, image/label hashes, source paths, box/class labels, and unresolved board/side status. Its 196 annotations are all `DRAFT_ANNOTATION`; the blank label is `UNANNOTATED`. Annotation instance IDs are image-local line identifiers, not physical component identities.

Current readiness: **`BLOCKED_BOARD_ID_UNRESOLVED`**, **`BLOCKED_REVIEWED_GROUND_TRUTH_UNAVAILABLE`**, and **`BLOCKED_SEALED_TEST_UNAVAILABLE`**. See [annotation review packet instructions](PHASE21_3_1_RECOGNITION_RELIABILITY.md#tools-added).
