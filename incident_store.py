from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


class IncidentStore:
    def __init__(self, db_path: str = "data/incidents.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    source TEXT,
                    frame_index INTEGER,
                    track_id INTEGER,
                    hazard_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    confidence REAL,
                    details_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_incidents_created_at ON incidents(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents(severity)")

    def add_many(self, source: str, incidents: Iterable[Dict[str, Any]]) -> int:
        rows = []
        now = datetime.now(timezone.utc).isoformat()
        for item in incidents:
            rows.append(
                (
                    now,
                    source,
                    item.get("frame_index"),
                    item.get("track_id"),
                    item.get("type", "UNKNOWN"),
                    item.get("severity", "LOW"),
                    float(item.get("confidence", 0.0)),
                    json.dumps(item, ensure_ascii=False),
                )
            )
        if not rows:
            return 0
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO incidents
                (created_at, source, frame_index, track_id, hazard_type, severity, confidence, details_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [dict(r) for r in rows]

    def summary(self) -> Dict[str, Any]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
            by_severity = conn.execute(
                "SELECT severity, COUNT(*) AS count FROM incidents GROUP BY severity ORDER BY count DESC"
            ).fetchall()
            by_hazard = conn.execute(
                "SELECT hazard_type, COUNT(*) AS count FROM incidents GROUP BY hazard_type ORDER BY count DESC"
            ).fetchall()
        return {
            "total": total,
            "by_severity": [dict(r) for r in by_severity],
            "by_hazard": [dict(r) for r in by_hazard],
        }
