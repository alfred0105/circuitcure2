# Repair Feedback Guide

## Purpose

Repair Feedback stores the diagnosis state, physical observations, repair actions, and post-repair evidence as a local case record. It keeps repair outcome separate from root-cause confirmation so that a board that works after service does not automatically label a replaced part as the unique cause.

## Local storage and privacy

The default database is `data/private/repair_feedback/repair_cases.sqlite3`. The private data directory is ignored by Git. SQLite schema version 1 stores cases, diagnosis snapshots, observations, inspection events, interventions, post-repair tests, verification events, prior versions, and prior activation events. Events are append-only; the original diagnosis snapshot is protected from edits. Explicit deletion is supported. JSON export supports full and anonymized output, and a database backup can be created locally.

No case data is uploaded by the application. The anonymized export removes case and board identifiers, notes, photos, reviewer details, source case IDs, and local simulation paths, while pseudonymizing circuit-family identifiers.

## Case lifecycle

A case begins as `UNVERIFIED`, with unknown outcome and `NOT_ELIGIBLE` training status. The UI records reported symptoms, circuit identity/hash, source type, physical measurements and conditions, repair interventions, post-repair tests, and a diagnosis snapshot when available. Evidence review records repair outcome and root-cause review as distinct decisions. Available case and verification states are represented explicitly in the data model rather than collapsed into a single success flag.

Before/after measurements are comparable only when they refer to the same node and unit, include compatible test conditions and a positive uncertainty, and have physical measurement provenance. The comparison reports measured change and whether the after value falls in a supplied expected range. It does not claim that a particular repair caused the change.

## Duplicate handling and eligibility

An identical physical event is rejected. A likely repeat on the same board, circuit, and cause within the configured time window is saved for review and excluded from training pending resolution. Independent boards can remain distinct cases. Eligibility is recalculated after review or case changes.

Only reviewed, explicitly approved physical repair cases can qualify. A case must have a comparable circuit family and hypothesis set, a diagnosis snapshot containing the verified hypothesis, real measurement evidence, a documented intervention, a passed real post-repair test, comparable before/after evidence, and a reviewed root cause. Synthetic, simulated, LLM-predicted, unverified, unresolved duplicate, holdout, and multi-component ambiguous cases are excluded. User-reported records require review. Eligibility is not a claim about population-level failure frequency.

## UI

Open the Streamlit dashboard and select **수리 이력 및 AI 학습**. The tab provides case registration, verification, history/search, local export/backup/deletion, prior proposal review, and default-versus-learned diagnosis comparison. With no verified cases, it displays `No Verified Repair Cases` and does not create an active learned prior.

For details of the estimator, approval, and rollback, see [Prior Learning Guide](PRIOR_LEARNING_GUIDE.md). For current data and evaluation status, see [Phase 16 Evaluation](PHASE16_EVALUATION.md).
