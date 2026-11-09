# Phase 14 evaluation record

## Implementation status

Implemented the optional local PDF flow, rule and table candidate discovery, local Ollama JSON structuring, same-page source/value/unit checks, append-only human review records, guarded Rating Engine adapter, and before/after risk comparison using saved SPICE stress. PDFs are not committed. The UI uses an explicit MPN-to-Reference confirmation and a separate operating-condition confirmation. The adapter preserves the original demo source and stores datasheet provenance per rating metric.

## Real local document smoke evaluation

- Manufacturer source: [YAGEO RC0805FR-071KL manufacturer product specification](https://yageogroup.com/component-documentation/download/specsheet/RC0805FR-071KL). The temporary local PDF used for the run was not committed. This one-page product sheet identifies a YAGEO 1 kΩ, 0805 thick-film resistor and lists 0.125 W at 70°C and voltage characteristics.
- Document SHA-256: `bb56b02867c38d70288d56487494497f2323d708427c77977addff7f20c23331`; 56,390 bytes; 1 page; text extracted; `pdfplumber` found 4 tables.
- Rule extraction found the `Power` row candidate `0.125 W (70C)`. The page quote, number, and unit match, while the use duration is not explicit; the current workflow labels this `CONDITION_INCOMPLETE` and requires a reviewer note before applying it.
- Local LLM: installed Ollama `gemma4:12b` (MLX runner, 7.7 GB). A real local inference proposed the 0.125 W rating and voltage candidates. The explicit 150 V continuous working-voltage text passed the source/value/unit check. Two 300 V-style candidates were rejected because the cited text did not state the unit with the number; the validator did not inherit a unit from nearby prose.
- One Streamlit AppTest uploaded the PDF, extracted its table/text, displayed the candidate, and ran the local Ollama action without UI exceptions. This is a one-document workflow smoke check, not an extraction-accuracy benchmark.
- Latest UI extraction call took 15.74 seconds in this local session. This single warm-model observation is not a performance benchmark.
- The exact manufacturer MPN is present in the PDF and its 1 kΩ nominal value matches the demo `RLOAD` KiCad Value. The physical RLOAD-to-MPN association is not known from the repository; no actual Reference confirmation was made. The production UI therefore still requires an explicit user assertion.

## Rating-engine integration check

A unit-level reassessment used an explicitly labeled test-only approval record derived from the extracted 0.125 W candidate. Existing `check_component_ratings` and `assess_component_risk` functions recomputed the RLOAD assessment from the cached stress; `RLOAD` remained `WITHIN_LIMIT`, rank 2, while its source changed from `demo_assumption` to `datasheet_user_approved`. The result is expected because the documented value (0.125 W) equals the existing demo assumption. No approval was written to the user's local registry. The cached `component_stress.json` SHA-256 remained `1ebbf5b0f8af505bdae3201aec22725f36129727a2263a6b27dc402ae9d255ca` before and after. The comparison did not rerun SPICE.

## Automated verification

- `.venv/bin/python -m pytest -q`: **62 passed** locally with the optional datasheet extra installed.
- `compileall` over source, scripts, and tests completed; `git diff --check` passed.
- The baseline CircuitCare CLI still loaded the five-component sample and returned U1 connectivity candidates.
- Streamlit AppTest loaded all eight tabs, uploaded the official PDF, showed `CONDITION_INCOMPLETE` for the 70°C power row, executed a real local Ollama extraction, and showed `SOURCE_MISMATCH` for a 300 value whose unit was only implied by context. No app exceptions occurred.
- GitHub Actions: [Tests run 41581184319](https://github.com/alfred0105/CircuitCare-AI/actions/runs/41581184319) passed for implementation commit `0236d1ab91ccfb4abbe8925b3aff9852797ed45b`.

## Accuracy, boundaries, next work

- General extraction accuracy: **NOT EVALUATED**. One datasheet is insufficient for precision, recall, condition extraction, or a rules-vs-LLM comparison. Candidate source acceptance in this one run is only a smoke observation, not a benchmark.
- The PDF is a one-page product specification, not a diverse multi-page data-sheet corpus. Datasheet OCR, complex merged-table semantics, RMS/peak conversions, derating calculation, and automatic package/temperature compatibility remain unsupported.
- A `SOURCE_MATCHED` result does not prove the selected board physically contains that MPN. A human must verify Reference mapping and operating-condition suitability. Unknown conditions keep the risk result unknown unless explicitly resolved by the reviewer.
- Test-only reassessment verifies the adapter and engine boundary; a real user-approved rating has not yet been applied to the demo board.
- Phase 15 should add a licensed, independently reviewed multi-family data set (resistors, capacitors, diodes and ICs), benchmark rules-only versus rules-plus-LLM, improve table-cell condition linking, and validate MPN-to-Reference workflow on a known board. Store no source PDF in Git unless its license expressly permits redistribution.
