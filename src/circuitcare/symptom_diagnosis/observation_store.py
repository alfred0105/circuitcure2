"""Local SQLite persistence for a diagnosis session and its evidence."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class ObservationStore:
    def __init__(self, path: Path):
        self.path = path

    def save(self, *, diagnosis_id: str, project_id: str, board_id: str | None,
             status: str, result: dict[str, Any], observations: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=10) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS diagnosis_sessions (
                    diagnosis_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, board_id TEXT,
                    status TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, result_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS diagnosis_observations (
                    observation_id TEXT NOT NULL, diagnosis_id TEXT NOT NULL,
                    code TEXT NOT NULL, value_json TEXT, status TEXT NOT NULL,
                    source TEXT NOT NULL, component_id TEXT, confidence REAL,
                    conditions_json TEXT, evidence_quote TEXT, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (diagnosis_id, observation_id),
                    FOREIGN KEY (diagnosis_id) REFERENCES diagnosis_sessions(diagnosis_id) ON DELETE CASCADE
                );
            """)
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("""
                INSERT INTO diagnosis_sessions(diagnosis_id, project_id, board_id, status, result_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(diagnosis_id) DO UPDATE SET board_id=excluded.board_id,
                status=excluded.status, updated_at=CURRENT_TIMESTAMP, result_json=excluded.result_json
            """, (diagnosis_id, project_id, board_id, status,
                  json.dumps(result, ensure_ascii=False, allow_nan=False)))
            connection.execute("DELETE FROM diagnosis_observations WHERE diagnosis_id=?", (diagnosis_id,))
            for index, row in enumerate(observations):
                observation_id = str(row.get("id") or f"{row.get('code', 'unknown')}:{index}")
                connection.execute("""
                    INSERT INTO diagnosis_observations(
                        observation_id, diagnosis_id, code, value_json, status, source,
                        component_id, confidence, conditions_json, evidence_quote
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (observation_id, diagnosis_id, str(row.get("code", "unknown")),
                      json.dumps(row.get("value"), ensure_ascii=False),
                      str(row.get("status", "unknown")), str(row.get("source", "unknown")),
                      row.get("component_id"), row.get("confidence"),
                      json.dumps(row.get("measurement_conditions"), ensure_ascii=False),
                      row.get("evidence_quote")))

    def get(self, diagnosis_id: str) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        with sqlite3.connect(self.path, timeout=5) as connection:
            connection.row_factory = sqlite3.Row
            session = connection.execute(
                "SELECT * FROM diagnosis_sessions WHERE diagnosis_id=?", (diagnosis_id,)
            ).fetchone()
            if session is None:
                return None
            observations = connection.execute(
                "SELECT * FROM diagnosis_observations WHERE diagnosis_id=? ORDER BY updated_at, observation_id",
                (diagnosis_id,),
            ).fetchall()
        return {"session": dict(session), "observations": [dict(row) for row in observations]}
