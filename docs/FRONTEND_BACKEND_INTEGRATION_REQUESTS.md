# Frontend / Backend Integration Requests

Status: frontend requests after F1.1 review of backend `origin/main` at `8f68af7`/`29d710e`; no backend changes made.

F1 now creates a session-only UI project draft behind `CIRCUITCARE_UI_V2=1`. It does not persist projects or run diagnosis. The existing legacy UI remains the default without the flag.

1. Specify project and board identity for projects created from uploaded photos and optional circuit files.
2. Specify intake persistence and supported attachment types/limits, including multiple photos and front/back/detail roles.
3. Specify measurement update and diagnosis refresh inputs/outputs, including `probe_id`, `node_id`, unit, source, timestamp, analysis context, project and board identity. Preserve zero as a valid value.
4. Specify verified component identity mapping between PCB and schematic, with an explicit status and evidence reference.
5. Specify provenance and confidence semantics for diagnosis candidates, recommended measurements and secondary impacts. Define when `CONFIRMED` is valid.
6. Confirm callable service boundaries and optional runtime dependencies for UI-triggered diagnosis refresh.

Current UI work can proceed using frontend-owned models and an explicit demo-only adapter. No request requires edits to diagnosis, SPICE, Bayesian, reconstruction, training or ground-truth engines.

## Photo-first direction / Phase 21.1.2

1. Add or document a multi-photo analysis entry point that accepts project/board IDs and each source `photo_id`, hash, side, capture type and original dimensions, without requiring full KiCad board reconstruction.
2. Define how an unspecified `capture_type` is handled by backend processing. Current `collect_photos()` accepts four known capture types and otherwise defaults omitted values to `full_view`; the frontend adapter deliberately withholds unknown types rather than making that assumption.
3. Define an image-only analysis/registration path or explicitly state which analyses require design-reference geometry. Current Phase 21.1.2 registration service requires `design_reference_geometry`; photo-only input must not fabricate it.
4. Specify per-photo registration results keyed by original photo ID, including frames, board side, verification status, provenance, and resource failures. An automatic estimate without independent held-out validation must remain unverified and cannot drive a verified overlay.
5. Specify component detection and reference-mapping output contracts so detected visual components remain separate from design-confirmed references.
6. Confirm project and board persistence/identity semantics. F1.1 IDs and image bytes are session-local and do not provide durable storage.

These are integration requests for later phases. F1.1 only creates/validates intake records and exercises the metadata adapter; it does not call registration, component detection, topology reconstruction or diagnosis.

## F2 / Phase 21.2 adapter boundary

The latest `origin/main` adds `MultiImageComponentMapService.ingest/build_map/fault_overlay` and `PhotoInput`. The frontend keeps this optional and does not import backend code on its own branch. F2 defines a local request DTO and conservative mappings: `front/back → TOP/BOTTOM`; `full_view/close_up/oblique → OVERVIEW/CLOSEUP/ANGLED`. `unknown` and `marking_detail` are withheld, and `full_view` is not promoted to `REFERENCE`. Preserve `frontend_photo_id`, content hash and board ID because backend `photo_id` is derived independently.

F3 should add a service adapter on a branch containing the backend interface, call bounded ingest/map APIs, persist the frontend↔backend photo-ID map, and display only returned per-photo/canonical artifacts with registration provenance. The F2 workbench displays one original photo at a time and labels registration as not run. Backend detection without a configured model yields no detections; do not substitute mock boxes. Canonical mosaics, component overlays and fault overlays are not part of F2.

F2.1 adds a local-only `PREVIEW_ONLY` session project with zero photos. The normal diagnostic factory remains strict and requires an image. Viewer geometry is source-image pixel based, but responsive display bounds remain explicitly unverified; the backend must not consume the requested display scale as an overlay transform. KiCad SVG display uses the existing local exporter/viewer only and does not create a project or circuit model on the backend.
