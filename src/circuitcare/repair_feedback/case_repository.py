"""SQLite persistence for local repair cases, append-only reviews, and prior versions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from .case_validator import assert_valid_case
from .models import utc_now


class RepairCaseRepository:
    CURRENT_SCHEMA_VERSION = 1

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > self.CURRENT_SCHEMA_VERSION:
                raise RuntimeError(f"Database schema {version} is newer than supported schema {self.CURRENT_SCHEMA_VERSION}.")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS repair_cases (
                    case_id TEXT PRIMARY KEY,
                    circuit_id TEXT NOT NULL,
                    circuit_hash TEXT NOT NULL,
                    data_source TEXT NOT NULL,
                    case_status TEXT NOT NULL,
                    verification_status TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS repair_cases_circuit_idx ON repair_cases(circuit_id, circuit_hash);
                CREATE TABLE IF NOT EXISTS case_events (
                    event_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES repair_cases(case_id) ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS case_events_case_idx ON case_events(case_id, created_at);
                CREATE TABLE IF NOT EXISTS diagnosis_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES repair_cases(case_id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS observations (
                    observation_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES repair_cases(case_id) ON DELETE CASCADE,
                    phase TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS inspection_events (
                    inspection_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES repair_cases(case_id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS repair_interventions (
                    intervention_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES repair_cases(case_id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS post_repair_tests (
                    test_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES repair_cases(case_id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS verification_events (
                    verification_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES repair_cases(case_id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS prior_models (
                    version TEXT PRIMARY KEY,
                    group_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS prior_activation_events (
                    activation_id TEXT PRIMARY KEY,
                    version TEXT,
                    action TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
            """)
            db.execute(f"PRAGMA user_version = {self.CURRENT_SCHEMA_VERSION}")

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)

    def create_case(self, case: dict[str, Any]) -> None:
        case = dict(case)
        assert_valid_case(case)
        now = utc_now()
        snapshot = case.get("diagnosis_snapshot")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing_rows = db.execute("SELECT payload_json FROM repair_cases").fetchall()
            if existing_rows:
                from .case_similarity import find_duplicate_cases
                existing_cases = [json.loads(row[0]) for row in existing_rows]
                duplicate = find_duplicate_cases(case, existing_cases)
                if duplicate["status"] == "DUPLICATE_CASE":
                    # Let the database enforce duplicate Case IDs; reject repeated physical events here.
                    if not any(row.get("case_id") == case["case_id"] for row in existing_cases):
                        raise ValueError(f"Duplicate repair event matches {duplicate['matched_case_id']}.")
                if duplicate["status"] == "REVIEW_REQUIRED":
                    case["duplicate_review"] = duplicate
                    case["training_eligibility"] = {"status": "REVIEW_REQUIRED", "reasons": ["Possible duplicate physical event requires review."]}
                    assert_valid_case(case)
            db.execute("INSERT INTO repair_cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (case["case_id"], case["circuit_id"], case["circuit_hash"], case["data_source"],
                 case["case_status"], case["verification_status"], case["outcome"], case["schema_version"],
                 case.get("created_at", now), case.get("updated_at", now), self._json(case)))
            self._insert_event(db, case["case_id"], "CASE_CREATED", {"data_source": case["data_source"]})
            if case.get("duplicate_review"):
                self._insert_event(db, case["case_id"], "DUPLICATE_REVIEW_REQUIRED", case["duplicate_review"])
            if isinstance(snapshot, dict) and snapshot:
                snapshot_id = snapshot.get("snapshot_id", f"snapshot_{uuid4().hex}")
                db.execute("INSERT INTO diagnosis_snapshots VALUES (?, ?, ?, ?)",
                    (snapshot_id, case["case_id"], now, self._json(snapshot)))
            for row in case.get("observations", []):
                db.execute("INSERT INTO observations VALUES (?, ?, ?, ?, ?)",
                    (f"obs_{uuid4().hex}", case["case_id"], row.get("phase", "before"), now, self._json(row)))
            for table, key, rows in (("inspection_events", "inspection_id", case.get("inspection_events", [])),
                                     ("repair_interventions", "intervention_id", case.get("interventions", [])),
                                     ("post_repair_tests", "test_id", case.get("post_repair_tests", []))):
                for row in rows:
                    db.execute(f"INSERT INTO {table} VALUES (?, ?, ?, ?)",
                        (f"{key}_{uuid4().hex}", case["case_id"], now, self._json(row)))

    @staticmethod
    def _insert_event(db: sqlite3.Connection, case_id: str, event_type: str, payload: dict[str, Any]) -> None:
        db.execute("INSERT INTO case_events VALUES (?, ?, ?, ?, ?)",
            (f"event_{uuid4().hex}", case_id, event_type, utc_now(), RepairCaseRepository._json(payload)))

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM repair_cases WHERE case_id = ?", (case_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def list_cases(self, *, status: str | None = None, component_ref: str | None = None,
                   fault_type: str | None = None, circuit_family: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT payload_json FROM repair_cases ORDER BY created_at DESC, case_id").fetchall()
        cases = [json.loads(row[0]) for row in rows]
        return [case for case in cases
                if (status is None or case.get("verification_status") == status)
                and (circuit_family is None or case.get("circuit_family") == circuit_family)
                and (component_ref is None or any(i.get("component_ref") == component_ref for i in case.get("interventions", [])) or (case.get("root_cause") or {}).get("component_ref") == component_ref)
                and (fault_type is None or (case.get("root_cause") or {}).get("fault_type") == fault_type)]

    def update_case(self, case: dict[str, Any], *, event_type: str, event_payload: dict[str, Any]) -> None:
        assert_valid_case(case)
        case = dict(case)
        case["updated_at"] = utc_now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old_row = db.execute("SELECT payload_json FROM repair_cases WHERE case_id=?", (case["case_id"],)).fetchone()
            if old_row is None:
                raise KeyError(f"Unknown repair case: {case['case_id']}")
            previous = json.loads(old_row[0])
            if previous.get("diagnosis_snapshot") != case.get("diagnosis_snapshot"):
                raise ValueError("The original diagnosis snapshot is immutable; save a new diagnosis as a separate record.")
            result = db.execute("UPDATE repair_cases SET case_status=?, verification_status=?, outcome=?, updated_at=?, payload_json=? WHERE case_id=?",
                (case["case_status"], case["verification_status"], case["outcome"], case["updated_at"], self._json(case), case["case_id"]))
            if result.rowcount != 1:
                raise KeyError(f"Unknown repair case: {case['case_id']}")
            self._insert_event(db, case["case_id"], event_type, event_payload)
            self._invalidate_priors_using_case(db, case["case_id"], reason="source_case_updated")
            if event_type == "EVIDENCE_REVIEWED":
                db.execute("INSERT INTO verification_events VALUES (?, ?, ?, ?)",
                    (f"verify_{uuid4().hex}", case["case_id"], case["updated_at"], self._json(event_payload)))

    def add_event(self, case_id: str, event_type: str, payload: dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM repair_cases WHERE case_id=?", (case_id,)).fetchone():
                raise KeyError(f"Unknown repair case: {case_id}")
            self._insert_event(db, case_id, event_type, payload)

    def list_events(self, case_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT event_id,event_type,created_at,payload_json FROM case_events WHERE case_id=? ORDER BY created_at,event_id", (case_id,)).fetchall()
        return [{"event_id": row[0], "event_type": row[1], "created_at": row[2], "payload": json.loads(row[3])} for row in rows]

    def add_diagnosis_snapshot(self, case_id: str, snapshot: dict[str, Any]) -> None:
        if not snapshot.get("snapshot_id"):
            raise ValueError("Every reanalysis snapshot requires a unique snapshot_id.")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM repair_cases WHERE case_id=?", (case_id,)).fetchone():
                raise KeyError(f"Unknown repair case: {case_id}")
            created_at = snapshot.get("created_at", utc_now())
            db.execute("INSERT INTO diagnosis_snapshots VALUES (?, ?, ?, ?)",
                (snapshot["snapshot_id"], case_id, created_at, self._json(snapshot)))
            self._insert_event(db, case_id, "DIAGNOSIS_REANALYSIS_SAVED", {
                "snapshot_id": snapshot["snapshot_id"], "snapshot_type": snapshot.get("snapshot_type", "REANALYSIS"),
                "prior_model_version": snapshot.get("prior_model_version")})

    def list_diagnosis_snapshots(self, case_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT payload_json FROM diagnosis_snapshots WHERE case_id=? ORDER BY created_at,snapshot_id", (case_id,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def delete_case(self, case_id: str) -> bool:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM repair_cases WHERE case_id=?", (case_id,)).fetchone():
                return False
            self._invalidate_priors_using_case(db, case_id, reason="source_case_deleted")
            return db.execute("DELETE FROM repair_cases WHERE case_id=?", (case_id,)).rowcount == 1

    def _invalidate_priors_using_case(self, db: sqlite3.Connection, case_id: str, *, reason: str) -> None:
        rows = db.execute("SELECT version,payload_json FROM prior_models").fetchall()
        for row in rows:
            model = json.loads(row[1])
            if case_id not in model.get("training_case_ids", []):
                continue
            model["status"] = "INVALIDATED_SOURCE_CHANGE"
            model["invalidation"] = {"reason": reason, "at": utc_now()}
            if reason == "source_case_deleted":
                model["training_case_ids"] = [item for item in model.get("training_case_ids", []) if item != case_id]
                model["training_case_count"] = len(model["training_case_ids"])
                model["prior_weights"] = None
                model["counts"] = {}
            db.execute("UPDATE prior_models SET status=?, payload_json=? WHERE version=?",
                (model["status"], self._json(model), row[0]))
            db.execute("INSERT INTO prior_activation_events VALUES (?, ?, ?, ?, ?)",
                (f"prior_event_{uuid4().hex}", row[0], "INVALIDATED_SOURCE_CHANGE", utc_now(), self._json(model["invalidation"])))

    def save_prior_model(self, model: dict[str, Any]) -> None:
        version = model["version"]
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT INTO prior_models VALUES (?, ?, ?, ?, ?)",
                (version, model["group_id"], model.get("status", "PROPOSED"), model.get("created_at", utc_now()), self._json(model)))

    def get_prior_model(self, version: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM prior_models WHERE version=?", (version,)).fetchone()
        return json.loads(row[0]) if row else None

    def list_prior_models(self, *, group_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as db:
            if group_id:
                rows = db.execute("SELECT payload_json FROM prior_models WHERE group_id=? ORDER BY created_at DESC", (group_id,)).fetchall()
            else:
                rows = db.execute("SELECT payload_json FROM prior_models ORDER BY created_at DESC").fetchall()
        return [json.loads(row[0]) for row in rows]

    def update_prior_model(self, version: str, *, status: str) -> dict[str, Any]:
        model = self.get_prior_model(version)
        if not model:
            raise KeyError(f"Unknown prior model: {version}")
        model["status"] = status
        model["updated_at"] = utc_now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE prior_models SET status=?, payload_json=? WHERE version=?", (status, self._json(model), version))
        return model

    def record_prior_event(self, *, version: str | None, action: str, payload: dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT INTO prior_activation_events VALUES (?, ?, ?, ?, ?)",
                (f"prior_event_{uuid4().hex}", version, action, utc_now(), self._json(payload)))

    def list_prior_events(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT activation_id,version,action,created_at,payload_json FROM prior_activation_events ORDER BY created_at,activation_id").fetchall()
        return [{"activation_id": row[0], "version": row[1], "action": row[2], "created_at": row[3], "payload": json.loads(row[4])} for row in rows]

    def export_json(self, *, anonymize: bool = False) -> dict[str, Any]:
        cases = self.list_cases()
        if anonymize:
            family_map: dict[str, str] = {}
            for index, case in enumerate(cases, 1):
                case["case_id"] = f"anonymous_{index:05d}"
                case["circuit_id"] = f"anonymous_circuit_{index:05d}"
                case["circuit_hash"] = f"anonymous_hash_{index:05d}"
                family = case.get("circuit_family")
                if family:
                    family_map.setdefault(family, f"anonymous_family_{len(family_map) + 1:05d}")
                    case["circuit_family"] = family_map[family]
                case.pop("board_instance_id", None)
                case.pop("source_case_id", None)
                case.pop("created_by", None)
                case.pop("photos", None)
                case.pop("private_notes", None)
                case.pop("reported_symptoms", None)
                case.pop("inspection_events", None)
                snapshot = case.get("diagnosis_snapshot")
                if isinstance(snapshot, dict):
                    snapshot.pop("simulation_sources", None)
                    snapshot.pop("local_paths", None)
                    for row in snapshot.get("hypothesis_definitions", []):
                        if isinstance(row, dict):
                            row.pop("simulation_source", None)
                            row.pop("risk_assessment_source", None)
                for intervention in case.get("interventions", []):
                    if isinstance(intervention, dict):
                        intervention.pop("note", None)
                        intervention.pop("reason", None)
                review = case.get("training_review")
                if isinstance(review, dict):
                    review.pop("reviewer", None)
                    review.pop("note", None)
        return {"schema_version": self.CURRENT_SCHEMA_VERSION, "cases": cases}

    def backup(self, target: str | Path) -> None:
        destination = Path(target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as source, sqlite3.connect(destination) as backup:
            source.backup(backup)
