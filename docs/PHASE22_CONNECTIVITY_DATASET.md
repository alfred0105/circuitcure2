# Phase 22 — Schematic Connectivity Dataset

## Objective

Build paired examples for estimating electrical connectivity from a schematic
that still shows components and their pins but has no drawn electrical
connections. Keep the original design and its netlist as the answer. Never
rewrite source data in place.

The fixed component taxonomy is `IC`, `LED`, `battery`, `buzzer`,
`capacitor`, `connector`, `diode`, `display`, `emi_filter`, `ferrite_bead`,
`fuse`, `inductor`, `jumper`, `oscillator`, `potentiometer`, `relay`,
`resistor`, `switch`, and `transistor`.

## Source fit

| Source | Role | Constraints |
|---|---|---|
| [bshada/open-schematics](https://huggingface.co/datasets/bshada/open-schematics) | Primary input. Contains KiCad schematic source, rendered schematic image, PCB artifacts, component/library metadata and a `name` field. | 87.9k rows; Hugging Face reports 317 GB total and CC BY 4.0. A bounded acquisition produced 5,046 unique KiCad schematics (about 693 MiB). Preserve `name` as a project group so sibling schematics stay in the same split. |
| [AbijahKaj/kicad-netlist-sft-dataset](https://huggingface.co/datasets/AbijahKaj/kicad-netlist-sft-dataset) | Structured target-netlist supplement and cross-check. `netlist_json` includes components and pin-to-net membership. | 100,179 rows, Apache-2.0 declared. One exact project match was used for the netlist-format pilot. Many records derive from open-schematics, so deduplicate by source/design before combining. |
| [digailab/PCB-Bench](https://github.com/digailab/PCB-Bench) | Optional external evaluation/context source for PCB routing reasoning and real PCB project artifacts. | 174 projects are PCB design comprehension cases; its primary tasks do not provide a clean schematic-image/wire-free input and component-to-net answer pair. Do not count QA/CQ records as connectivity training pairs. Inspect each artifact and its license/provenance before conversion. |

## Pair contract

Each accepted source schematic produces:

- `inputs/<id>.kicad_sch`: a byte-preserving copy with root-level `wire`,
  `bus`, `bus_entry`, `junction`, `no_connect`, `label`, `global_label`, and
  `hierarchical_label` records removed. Component symbols and their pin
  definitions remain.
- `targets/<id>.json`: when a netlist is supplied/exported, a graph of the
  19-class components and original nets with component reference and pin.
- `targets/<id>.kicad_sch`: in `original-schematic` mode, an untouched copy of
  the complete wired source schematic as the ground-truth target.
- `originals/<id>.netlist.json`: full exported original connectivity for audit
  when using netlist mode.
- `records.jsonl`: source hashes, labels, source/license attribution, group,
  split and output paths.

Labels and global/hierarchical labels are removed because they can disclose
connectivity even when wires are absent. Junction dots are removed for the same
reason. `no_connect` markers are removed so the input does not directly expose
negative pin labels. In `netlist-json` mode, the netlist is exported from the
untouched source schematic via `kicad-cli`; the wire-removed input is never used
to create its own target. In `original-schematic` mode, the target is a copy of
the original wired schematic.

Components not classifiable into the 19 labels remain visible in the input. In
`netlist-json` mode they are omitted as target nodes, and a target net is
retained only when it connects at least two labelled components. The
`original-schematic` target keeps every original component and connection.
Unknown library mappings are left unlabelled rather than guessed into a class.
The first-pass mapping uses symbol library ID/value/description and reference
prefix; review the class distribution and ambiguous mappings before training.

## Preparation

For direct netlist export from each source schematic, install KiCad 8 or newer
and make `kicad-cli` available on PATH. A matched target supplied through
`--netlist-map` does not require KiCad CLI. Fetch a bounded subset of schematic
source text with:

```bash
python scripts/fetch_open_schematics.py --limit 100 --page-size 25
```

The fetcher saves no images or PCB files and records each source project and
hash in `source_manifest.jsonl`. Use `--limit 0` only when intentionally
acquiring the entire source (Hugging Face reports 317 GB for the dataset).
Then prepare the clean-input/original-target pairs:

```bash
python scripts/prepare_connectivity_pairs.py data/external/open-schematics-sample \
  --source-name bshada/open-schematics \
  --source-url https://huggingface.co/datasets/bshada/open-schematics \
  --license CC-BY-4.0 \
  --output data/external/connectivity_pairs
```

Use `--limit 100` for an initial pass. The output folder is ignored by Git.
The script skips designs that do not have at least two taxonomy-labelled
components. In `netlist-json` mode it also skips designs without a net joining
at least two labelled components.
Flat input folders fall back to filename-based split groups; this is marked in
the manifest and is not adequate evidence of leakage-free board/project splits.
For a paired source-to-source dataset without KiCad CLI/netlist matches, use
`--target-format original-schematic`; this preserves the wired original as the
target while creating the wire-free input. The pair is lossless design
supervision, while JSON netlist targets remain the preferred training format
for a graph-prediction model.

When an independently sourced target netlist is available, provide a JSON map
whose keys exactly match paths relative to the source directory and whose
values are either `netlist_json` strings or parsed netlist objects. This allows
the source and target datasets to be recorded separately:

```bash
python scripts/prepare_connectivity_pairs.py data/external/open-schematics \
  --netlist-map data/external/kicad-netlist-sft/netlist-map.json \
  --source-name bshada/open-schematics --license CC-BY-4.0 \
  --target-source-name AbijahKaj/kicad-netlist-sft-dataset \
  --target-license Apache-2.0 \
  --output data/external/connectivity_pairs
```

### Pilot and batch completed

One matching sample was retrieved from each Hugging Face source using the
datasets-server rows API. `TiebeDeclercq/Uart-programmer` matched exactly by
GitHub project URL. The resulting local-only pair has 39 taxonomy-labelled
components, 28 target nets and 72 component-pin nodes. Nine of the 19 classes
appear in this single sample. A scan of the generated input confirmed that
root-level wires, net labels and junctions were absent. The source file,
target map and generated pair are in ignored `data/external/` paths and are not
committed. This is a conversion smoke check, not a dataset-quality or model
evaluation result.

The bounded batch contains **5,046 unique source schematics** (726,551,030
bytes; 692.9 MiB) and produced **4,957 wire-free/original pairs**. The 89
unpaired designs each had fewer than two components mapped to the 19-class
taxonomy. Across accepted samples, class counts are in the generated
`summary.json`; the largest are resistor (36,439), capacitor (31,679),
connector (22,144), switch (16,342), and diode (13,352). These are component
instance counts, not sample counts. The paired artifacts occupy about 1.26
GiB. Source and pair artifacts remain in ignored local data directories.

Batch audit found all 19 classes represented, verified 4,957 inputs and 4,957
original targets, and found zero remaining root-level wire, bus, junction,
no-connect or net-label records in the inputs. The 4,955 project groups do not
cross the generated train/validation/test split. The split is project-grouped,
but its labels are automatically mapped and have not had manual review.

The current input artifact is a wire-free KiCad schematic source file. It keeps
component symbols and their pin definitions. The batch target is an untouched
copy of the original wired schematic. KiCad CLI 10.0.7 is now installed in the
user environment; the link-prediction pipeline exports each original target to
KiCad XML at extraction time, leaving the stored source pairs unchanged.
Rendering and validating a wire-free image view, manually reviewing the
automatic component classes, and scaling beyond this bounded batch remain open
work.

## Pin-level link prediction and baseline comparison

`scripts/benchmark_connectivity_link_prediction.py` exports the original
schematic netlist with `kicad-cli`, parses symbol-library pin coordinates, and
applies the schematic instance rotation and mirror transform. Each pin is a
NetworkX node. A cKDTree finds nearby pins from different components; the
symmetrized k-nearest-neighbor links within the configured radius become
candidate edges. Edge labels are positive when both `(reference, pin)` endpoints
belong to the same original net. Multi-pin nets therefore create pairwise
positive links; this is a link-prediction proxy for shared-net membership, not
a claim that every positive pin pair has a direct drawn wire between them.

Candidate recall is measured before model evaluation. A positive pin pair that
was not proposed by cKDTree cannot be predicted by either model, so report
candidate recall alongside test metrics. A sweep over all 4,948 extracted
graphs gave mean per-schematic candidate recall of about 0.50 at `k=32`,
`100 mm`; 0.71 at `k=64`, `200 mm`; and 0.80 at `k=96`, `250 mm`. The larger
setting produced about 6,200 candidate edges per schematic on average, so the
default uses `k=96`, radius `250 mm` as a recall/density tradeoff. Distances use
KiCad schematic millimeters. This sweep is descriptive; final settings should
be chosen on validation projects, not test projects.

Both models use the existing project-grouped `train`/`validation`/`test`
manifest splits and the exact same candidate edges. LightGBM gets endpoint
class/pin features and relative geometry. The compact PyTorch model aggregates
messages over the candidate graph and scores candidate links; it is an
independent GNN-ACLP-inspired baseline, not a reproduction of the paper's SEAL
model or reported results. Evaluation emphasizes PR-AUC because positive links
are sparse; threshold metrics are also reported. LightGBM training caps sampled
negative edges to control memory, while test metrics include all test candidates.

### Learning-curve experiment

All five runs used the same seed, 15 epochs, graph mini-batches of 8, the same
nested training sample, and the same 513 held-out test projects (3,258,373
candidate edges). The 200-graph sample is reused in each larger run. Test
candidate recall stays at 0.807 because the candidate graph and test set do not
change.

| Training schematics | LightGBM PR-AUC | GNN PR-AUC | LightGBM ROC-AUC | GNN ROC-AUC |
|---:|---:|---:|---:|---:|
| 200 (baseline) | 0.271 | 0.194 | 0.789 | 0.771 |
| 500 (Experiment A) | 0.274 | 0.199 | 0.790 | 0.777 |
| 1,000 (Experiment B) | 0.277 | 0.199 | 0.793 | 0.779 |
| 2,000 (Experiment C) | 0.278 | 0.203 | 0.796 | 0.783 |
| 3,920 (Experiment D, full train split) | 0.277 | 0.220 | 0.796 | 0.790 |

LightGBM has the highest PR-AUC at every size and levels off near 0.277 by
1,000 examples. GNN PR-AUC rises from 0.194 to 0.220 using all 3,920 training
schematics, but remains below LightGBM. The final LightGBM PR-AUC is slightly
lower than its 2,000-example result, so more data did not improve both models
monotonically. The 0.5-threshold F1 and precision/recall are recorded in
`learning_curve.json`; threshold 0.5 was held fixed for this comparison.

These are initial learning-curve results from one seed. LightGBM uses up to 20
negative candidates per positive (capped at 1,000 negatives per schematic); GNN
uses all candidate links in the sampled training schematics. Treat the trend as
indicative, not as a statistical significance claim.

### Ranking, hard-negative, net-aware, and ensemble experiments

The follow-up used the same released graph archive and fixed project-grouped
split: 3,920 training schematics, 515 validation schematics, and 513 test
schematics (3,258,373 test candidate edges). All candidate models below were
evaluated on the same full test candidate set. The random seed was 42 and
LightGBM models used 250 trees.

Hard negatives were mined from three-fold project-grouped out-of-fold scores,
so each training schematic's hard-negative scores came from a model that did
not train on its project. H1 used 75% random and 25% hard negatives; H2 used
50% each. N1 approximately equalized positive sample mass per net. N2 capped
sampled positive pairs at 64 per net. The ranker used one query per anchor pin,
the `lambdarank` objective, and at most 50 sampled negatives per query. The
CUDA GNN used all candidate links, 15 epochs, and graph batches of 8. Ensemble
scores were endpoint-local rank normalized; alpha was selected from 0.25, 0.5,
0.75, and 0.9 on validation Pin Precision@1 only.

| Experiment | PR-AUC | Pin Precision@1 | End-to-end Recall@5 |
|---|---:|---:|---:|
| H0 LightGBM baseline | **0.2785** | 0.3219 | **0.1131** |
| H1 random 75% / hard 25% | 0.2468 | 0.2980 | 0.1048 |
| H2 random 50% / hard 50% | 0.2124 | 0.2822 | 0.0983 |
| N1 net-weighted positives | 0.0928 | 0.2799 | 0.0906 |
| N2 cap 64 positive pairs per net | 0.2408 | 0.3305 | 0.1123 |
| LightGBM Ranker | 0.2103 | **0.3485** | 0.1111 |
| GNN on CUDA | 0.2133 | 0.2687 | 0.0896 |
| Rank-normalized ensemble (alpha 0.9) | 0.1972 | 0.3298 | 0.1136 |

Pin Precision@K is macro averaged over anchor pins that have at least one
positive in the generated candidate set. End-to-end Recall@5 counts all
directed ground-truth pin pairs in its denominator, including pairs omitted by
candidate generation. Candidate recall on test is 0.8065 as a per-schematic
macro average, but 0.6482 when weighted by the number of truth pairs. Keep
these two candidate-coverage views distinct when comparing experiments.

The Ranker produced the highest Pin Precision@1, while H0 retained the highest
PR-AUC and end-to-end Recall@5 among the classifier/ranker/GNN models. N2
slightly exceeded H0 on Pin Precision@1 but not on the other two metrics. Hard
negative mining and N1 reduced all three metrics. These single-seed results do
not establish statistical significance. Adaptive candidate generation reduced
the average test candidates from 6,365 to 3,958, while per-schematic macro
candidate recall fell from 0.8065 to 0.6846; this setting is not an improvement
under the current recall/density tradeoff.

PyTorch GNN training used CUDA on an NVIDIA GeForce RTX 4060 Ti. The installed
Windows LightGBM wheel has no GPU learner, so the classifier and Ranker trained
on CPU. The report and experiment script are
`data/benchmarks/phase22_connectivity/recommendation_experiments.json` and
`scripts/benchmark_connectivity_recommendations.py`. A local progress page is
available from `scripts/connectivity_progress_dashboard.py`.

The GNN-ACLP paper describes port-level prediction with a SEAL framework and
its official implementation is GPL-3.0. This repository uses a separate
implementation and does not copy that code. See the [paper](https://doi.org/10.1016/j.engappai.2025.113035)
and [official repository](https://github.com/guanyuanpeterpan/GNN-ACLP) for
the reference method and license.

Run the full graph extraction and comparison:

```bash
pip install -e ".[connectivity]"
python scripts/benchmark_connectivity_link_prediction.py extract --workers 6
python scripts/benchmark_connectivity_link_prediction.py train
python scripts/benchmark_connectivity_link_prediction.py sweep --sizes 200 500 1000 2000 3920
```

KiCad CLI can be passed explicitly with `--kicad-cli PATH`; the script also
searches common Windows install paths and `KICAD_CLI`. Intermediate graph JSONL
and extraction failures are local-only under ignored
`data/benchmarks/phase22_connectivity/`. The compact comparison report and
extraction summary are kept as reviewable benchmark outputs. Training also
writes the GNN state dictionary and LightGBM text model under ignored
`models/phase22_connectivity/`.
The `sweep` command nests the same seeded 200-graph baseline subset into each
larger training-size run, trains GNNs with bounded graph mini-batches, and
writes the consolidated learning curve to `learning_curve.json`.

### Mac training archive

Download [`circuitcare-connectivity-training-2026-11-08.tar.gz`](https://github.com/alfred0105/CircuitCare-AI/releases/download/connectivity-dataset-v1/circuitcare-connectivity-training-2026-11-08.tar.gz)
from the [connectivity-dataset-v1 Release](https://github.com/alfred0105/CircuitCare-AI/releases/tag/connectivity-dataset-v1).
The 535,115,641-byte archive contains 4,948 ready-to-train pin graphs, pair
records, source inventory, and result summaries. Verify the download with the
accompanying `.sha256` asset; the archive digest is
`85a09edf9a3c67c4f0ffb8f962c19029a90ccc6dea57ef2e6d7787597a30fd8c`.
It does not contain the original KiCad schematic files or trained model
weights. The archive README has the extraction and Mac training commands.

The repository does not download the full 317 GB source dataset. The completed
bounded acquisition contains 5,046 unique schematic files in ignored
`data/external/open-schematics-5000/`, with a SHA-256 inventory. The generated
wire-free/original pairs are in ignored
`data/external/connectivity_pairs-5000-complete/`. Retain source URLs and hashes
in every derived sample. Avoid publishing source artifacts or generated copies
unless the applicable source license allows redistribution.

## Evaluation safeguards

- Split by source project/repository before sample generation, not randomly by
  schematic file. Sibling boards and derived variants must stay together.
- Keep the SFT dataset's overlapping open-schematics-derived records out of
  validation/test unless deduplication has been performed at project/design
  level.
- Record the source revision, exact license, SHA-256 and conversion tool
  version. A license on the dataset wrapper does not automatically settle the
  licensing of every upstream design artifact.
- Audit class counts, unmatched classes, empty-net counts, disconnected pins,
  and duplicate design hashes before training. The initial label mapper is a
  proposal and requires manual review on a stratified sample.
- This dataset evaluates schematic design connectivity. It does not establish
  connectivity reconstruction accuracy from physical PCB photographs.
