# Datasheet AI guide

## Setup

PDF processing is optional and is not imported by the core CLI. Install the UI and PDF extras when needed:

```bash
python -m pip install -e ".[ui,datasheet]"
streamlit run src/circuitcare/ui.py
```

The **데이터시트 AI** tab uses `pypdf` for page text and `pdfplumber` for table rows. The upload is limited to 25 MiB and 200 pages. Files stay in Streamlit session memory; CircuitCare stores only extraction/review metadata and a SHA-256 digest under the ignored `data/private/datasheet_ai/` directory. No PDF is uploaded to a cloud model. Password-protected documents are rejected. Empty-text/scanned documents show `OCR_REQUIRED`; datasheet OCR is not implemented in this phase.

## Review workflow

1. Select a KiCad Reference; inspect its KiCad Value separately.
2. Enter the manufacturer, real MPN, and optional document revision, then upload the manufacturer's PDF.
3. Review document metadata, SHA-256, pages, extracted text/tables, and rule-based candidates.
4. Optionally select an installed Ollama model and request local JSON structuring. No call occurs until the user presses the button.
5. Check the exact PDF page and quoted source text. The application compares quote, normalized number, and physical unit to the cited page. The local LLM output remains a proposal.
6. Confirm that the MPN belongs to the selected KiCad Reference and that the datasheet's temperature, duration, derating and electrical conditions are appropriate. A review decision is recorded with original and edited candidate data.
7. Only an approved `SOURCE_MATCHED` rating with an accepted Reference/MPN match and an explicit condition confirmation can be applied. Reassessment uses the saved SPICE stress, then updates the risk results visible in the fault map and PCB risk overlay. The previous risk report and SPICE stress remain unchanged. Applied ratings can be revoked; audit events remain appended.

## Supported rating mapping

The initial adapter supports voltage (`V`, `mV`, `kV`), current (`A`, `mA`, `µA`), and power (`W`, `mW`, `µW`) converted to SI units. A negative voltage candidate preserves its signed source text and polarity while the existing absolute-peak voltage comparison receives its magnitude; the reviewer still has to confirm the terminal and operating context. It keeps absolute maximum, peak/pulse, and continuous meanings distinct. Unsupported kinds retain `unknown` in the existing Rating Engine and do not produce a safe classification. Resistor power can be evaluated only for mapped resistor models. Capacitor dissipation is not inferred from a voltage rating.

Rule-based extraction is deterministic candidate discovery from table rows and nearby text. Local Ollama extraction structures only pages where candidates were found, up to five pages and a bounded text size. The prompt treats PDF contents as untrusted document data. The source validator does not execute document content, follow links, or accept a model-provided page/value without a same-page quote, value and unit match.

Evidence states: `SOURCE_MATCHED`, `SOURCE_AMBIGUOUS`, `SOURCE_MISMATCH`, `SOURCE_NOT_FOUND`, and `CONDITION_INCOMPLETE`. `SOURCE_MATCHED` verifies the cited text and value/unit; it does not prove that a physical board contains the named MPN or that all operating conditions match. Human approval is still required.

## Stored evidence and limits

Review history and applied/revoked events append to JSON files under `data/private/datasheet_ai/`, which is ignored by Git. Records include document digest, page index, MPN, Reference, source text, conditions, extraction method/model, original proposal, user edits, decision and timestamp. The uploaded PDF itself is not saved.

The UI's Reference-match and condition controls are explicit user attestations. They do not independently inspect a board or measure temperature. Risk labels compare the existing simulation stress with a selected rating. They are not physical failure probabilities or confirmation of damage. Bayesian hypotheses are not changed by a rating-only reassessment; their existing context must remain separate. See [Phase 14 evaluation](PHASE14_EVALUATION.md) for a real local extraction and its limitations.
