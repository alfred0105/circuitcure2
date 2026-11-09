# Phase 19.1.1 — Schematic Reliability and Fixture Validation

## Root causes addressed

- The schematic Inspector previously inferred verified SPICE output from the presence of `comparison.json`. It now displays a verified result only when the existing `AnalysisSnapshot` reports `SUCCESS` or `CACHED_SIMULATION` together with `VERIFIED` provenance. SVG export state is shown separately.
- The first identity reader used substring-oriented scanning and only inspected direct child sheets. It now parses balanced KiCad S-expressions, extracts actual `Sheet file` properties, traverses project-contained child sheets with depth/file limits, separates sheet paths and units, deduplicates legitimate reused identities, and rejects UUID and reference/unit collisions.
- Reference identity does not prove electrical equivalence. The UI and identity API keep that distinction explicit; the Divider test independently verifies the exported netlist.

## Divider fixture and KiCad validation

Added `tests/fixtures/divider.kicad_sch`, using the installed KiCad 10 symbol-library definitions for `Connector_Generic:Conn_01x02` and `Device:R`. The drawing represents J1 pin 1 to R1 pin 1 on VIN, R1 pin 2 to R2 pin 1 on VOUT, and J1 pin 2 to R2 pin 2 on GND.

Local KiCad CLI used: `/Applications/KiCad.app/Contents/MacOS/kicad-cli`, version `10.0.6`.

CLI SVG export and XML netlist export both succeeded. `test_divider_schematic_matches_xml_netlist` compares the exported and checked-in XML Component Reference and Value sets and the normalized pin-to-net groups. Exported KiCad local-net names `/VIN`, `/VOUT`, `/GND` are normalized by removing one leading slash and match the existing `VIN`, `VOUT`, `GND` names. The legacy XML has no `libsource` entries, so it cannot attest library IDs; the new schematic uses the standard KiCad library IDs `Connector_Generic:Conn_01x02` and `Device:R`. No legacy netlist fields were rewritten.

The Divider case now points at the validated fixture. Other cases retain the connectivity fallback. Missing or unsafe child sheets produce an incomplete-project warning and prevent SVG export.

## Identity coverage

Tests cover single/multiple hierarchical paths, nested sheets, units, reused UUIDs across distinct sheet paths, conflicting UUIDs, duplicate Reference/unit collisions, missing and traversal child sheets, malformed S-expressions, and escaped quotes/parentheses. `VERIFIED_REFERENCE` confirms only a unique, exact Reference identity; electrical pin/net equivalence is separately covered by the KiCad CLI integration check.

## SVG exporter coverage

Mocked tests cover CLI absence, success, nonzero exit, timeout, missing/malformed SVG, scripts, external SVG references, oversized SVG, cache reuse/invalidation after parent and child source changes, multiple page files, and external child sheets. `close_fds=False`, argv invocation, timeout, output-size/page limits, and content-addressed caching remain in place. The real CLI integration test is marked `integration`; when KiCad CLI is absent it reports a skip reason beginning `NOT_VERIFIED`.

## UI and simulation integrity

The Inspector accepts only the `AnalysisSnapshot` status and integrity fields. A matching existing JSON file alone cannot show a verified result. No schematic view invokes simulation or changes diagnosis state. The Divider has a verified static schematic; the Regulator and RC cases continue to use their existing connectivity fallback.

The local Streamlit UI was opened with the Divider case and the “회로 및 고장” view. The schematic image and fixture path rendered, and the provenance-unverified demo result remained labeled unverified. The browser was refreshed after restarting the development server so it loaded the current source.

## Verification results

- `python -m pytest -q -m "not integration and not stress and not crash_injection"`: 227 passed, 36 deselected.
- `python -m pytest -q -m integration -rs`: 22 passed, including the real KiCad Divider export and netlist comparison.
- `python -m pytest -q -m stress -rs`: 11 passed.
- Ruff: passed on all changed Python files.
- Mypy: passed on `identity.py`, `exporter.py`, `schematic_workspace.py`, and `case_loader.py`.
- `git diff --check`: passed before commit.

The default and stress groups emitted existing `zipfile` duplicate-name and pandas `reset_index(inplace=...)` deprecation warnings; all corresponding tests passed. No broad mypy claim is made for `ui.py` or the whole repository.

## Limits and phase assessment

- The legacy divider XML has no library ID metadata, so library identity cannot be compared retroactively.
- Hierarchical sheet traversal is limited to 8 levels and 32 project files.
- Visual identity and netlist equivalence are distinct checks; no symbol-click selection or overlay is included.
- The tested local environment used Python 3.12 and KiCad CLI 10.0.6; the project’s preferred Python 3.11 environment was not separately exercised here.

With the real Divider KiCad CLI integration and local UI render passing, Phase 19.1.1's schematic reliability and fixture validation hotfix is complete. Phase 19.2 can begin on interactive selection and view synchronization; electrical stress and diagnosis algorithms remain outside that phase.
