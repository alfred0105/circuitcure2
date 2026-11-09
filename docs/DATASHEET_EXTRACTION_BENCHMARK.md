# Datasheet extraction benchmark — historical record and corrected scope

## Status and corpus

The evaluation schema and scoring pipeline are implemented. Four official manufacturer PDFs were obtained locally and manually labeled: one resistor family, one capacitor MPN, one diode, and one regulator IC family. There are three manufacturers across those four documents, four verified target labels, and one target rating per family. PDFs are intentionally outside Git; the manifest records source URL, revision, SHA-256 and page metadata. Phase 14's YAGEO development PDF is excluded. An alternate TDK capacitor PDF retrieval returned HTTP 403 and is recorded as not acquired.

The corpus was first evaluated, then used to improve table extraction and row/column evidence handling. It remains a development pilot, not an independent holdout. The previous full-candidate precision and unsupported-claim values below are **invalid interpretations** because only one selected target rating per PDF was labeled. The original outputs are preserved and marked `INVALID_EVALUATION` in `evaluation/datasheet_extraction/results/pre_integrity_hotfix/` and `comparison.json`. The earlier pre-table-context result remains in `pre_table_context_fix_development_pilot.json`.

## Methods and post-fix results

The old figures below are shown only to preserve the historic report. They must not be interpreted as document-wide extraction quality or hallucination rates. A correct, unlabeled rating was incorrectly treated like a false positive under the former denominator.

| Historical metric (invalid interpretation) | Rules only | Rules + local LLM |
|---|---:|---:|
| Candidates | 48 | 54 |
| Candidate precision | 0.0833 | 0.0741 |
| Unsupported claim rate (`1 - precision`) | 0.9167 | 0.9259 |
| Source-verification rejection rate | 0.3333 | 0.3519 |

Corrected target-rating metrics are recorded in `evaluation/datasheet_extraction/results/corrected_scope_development_metrics.json`. The scope is one manually labeled target rating per document, constrained by rating key plus source anchor and available model/table context. Full-document precision/recall/F1 are `NOT_EVALUATED` because exhaustive inventories are unavailable. Unsupported claims and hallucinations are `NOT_EVALUATED` without candidate-level human adjudication. Source-verification rejection remains a separate automated-check metric.

| Corrected development-pilot metric | Rules only | Rules + local LLM |
|---|---:|---:|
| All extracted candidates | 48 | 54 |
| Candidates in target scope | 4 | 5 |
| Target rating precision | 1.00 (4/4) | 0.80 (4/5) |
| Target rating recall | 1.00 (4/4) | 1.00 (4/4) |
| Value accuracy | 1.00 (4/4) | 1.00 (4/4) |
| Unit accuracy | 1.00 (4/4) | 1.00 (4/4) |
| Source-check rejection | 16/48 (0.3333) | 19/54 (0.3519) |
| Unsupported-claim / hallucination rate | `NOT_EVALUATED` | `NOT_EVALUATED` |
| Complete-document precision/recall/F1 | `NOT_EVALUATED` | `NOT_EVALUATED` |

The corrected target scores use four labeled targets across four previously used PDFs. They are development-only and do not measure full-document extraction. The final run used Gemma 4 12B digest `c7597fc90b86aff6965317f37dd5465402284e26382d9b4bf2dfe3ef1491f4b6`, code revision `e83538bee387678e17c8d8b0f0c23fc2f6bd2ca8`, and recorded 31.69 seconds total model processing time (7.92 seconds per document on average). Document hashes, candidate counts, denominators, per-document latency, and scopes are recorded in the result JSON. The earlier runtime and precision narrative in the historical record is not a like-for-like model comparison; the denominator definition changed.

## Error patterns and review controls

- Table rows now retain a candidate's row label, model variant, column, selected cell, and table coordinates. The Streamlit review view exposes this context along with the cited text and page table.
- Multi-column resistor family tables can still produce surplus candidates, including an empty/repeated model cell when PDF extraction merges rows. These candidates are not treated as verified facts solely because the numeric quote is found.
- Diode continuous/peak distinctions and temperature conditions are kept separate. Adjacent-row or neighboring-model values need row/column evidence and are not silently combined.
- The LM7805 product feature “up to 1.5 A” lacks a complete operating condition in its text excerpt; the candidate remains `CONDITION_INCOMPLETE` and cannot be applied without human review and a written condition rationale.
- `page_evidence_accuracy` measures page plus quote agreement against the human label; `table_context_accuracy` measures row/model/column/cell association where the label is table-based. Fields not applicable to a label are `null`, not counted as correct.

## Reproduction and limits

Place local PDFs named by `document_id` in a folder and run:

```bash
python scripts/evaluate_datasheet_extraction.py --documents-dir /path/to/local/pdfs --run-llm --model gemma4:12b
```

The benchmark verifies each PDF's SHA-256 against the manifest and records model, digest, prompt version, commit, operating system, timing, and result status. Missing documents remain unevaluated. The PDFs are not committed. The current sample has one document per family and does not cover multiple models per vendor, multiple vendors per family, scans/OCR, or a genuine untouched holdout. No claim is made that extracted values match a physical PCB Reference without human confirmation. The four documents have target-rating labels only; `ground_truth_complete` is false.
