# Phase 19.1 — KiCad Schematic Workspace Foundation

## Existing structure

The application already loads KiCad XML netlists into `Circuit` objects and provides eleven Streamlit analysis areas. XML data is the source for component and net connectivity; the existing Graphviz output is explicitly a connectivity view. Simulation, risk, diagnosis, and workspace fingerprints remain on their existing paths.

## New structure and behavior

- `circuitcare.schematic.identity` reads placed symbol references, UUIDs, unit numbers, library IDs, values, and instance sheet paths from a `.kicad_sch` file. A reference match is reported as `VERIFIED_REFERENCE`; absent/ambiguous matches remain `UNVERIFIED_MAPPING`. This verifies the identifier only, not electrical equivalence between schematic wires and the XML netlist.
- `circuitcare.schematic.exporter` invokes KiCad CLI with an argv list, timeout, size/page bounds, project-contained paths, output XML validation, and a content-addressed SVG cache. Cache identity includes source and related sheet bytes, CLI version, requested page, and renderer version. Existing simulation fingerprints are unchanged.
- `schematic_workspace` adds left navigation, a central KiCad SVG canvas, and a right component inspector. When no schematic is available or export fails, it labels and displays the existing connectivity graph as a fallback. Rendering does not launch ngspice or Ollama or mutate diagnostic data.
- Uploaded schematics are size-limited and stored by content hash inside the project workspace. Child sheets outside the project root or missing child sheets cause an incomplete-project state.

## KiCad export and fixture provenance

The available KiCad CLI is `/Applications/KiCad.app/Contents/MacOS/kicad-cli`, version `10.0.6`. The supported command syntax was checked with `kicad-cli sch export svg --help`; an actual KiCad-generated SVG export succeeded using a schematic file bundled with the installed KiCad application.

There was no `.kicad_sch` in this repository. `tests/fixtures/identity_minimal.kicad_sch` is a small local identity-parser fixture, not the existing Divider circuit and not a circuit-validation fixture. The existing Divider XML contains J1, R1, and R2. Since no matching schematic source was present, no relationship between that XML and the fixture is claimed. The UI therefore keeps the connectivity fallback and only offers schematic identity information for an explicitly supplied source. A matching, electrically verified Divider schematic remains required for the full Phase 19.1 acceptance criteria.

## Limits

- Static SVG display only; no symbol click, pan/zoom, or fault-location overlay.
- Reference equality does not establish that schematic nets match XML nets.
- The current exporter detects direct child-sheet references and bounds them; deeply nested project discovery is not yet supported.
- KiCad CLI integration is optional in CI and must be reported as unavailable/unverified if absent.
- Arbitrary external projects are not claimed as fully supported.

## Verification

CLI version/help and a KiCad-owned schematic SVG export were checked locally. Unit tests cover reference extraction, unmatched references, and project-root path rejection. Full regression and static checks are recorded with the Phase 19.1 commit.

## Phase 19.2 prerequisites

First add a real Divider `.kicad_sch` that exports successfully and whose exported KiCad netlist is reconciled against `tests/fixtures/divider_netlist.xml`, including reference, value, pin, and net checks. Only then should interactive symbol selection and schematic-to-analysis selection synchronization be considered.
