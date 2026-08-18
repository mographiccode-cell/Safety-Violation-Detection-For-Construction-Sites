from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


class ZoneStore:
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
                CREATE TABLE IF NOT EXISTS safety_zones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    zone_type TEXT NOT NULL DEFAULT 'RESTRICTED',
                    points_json TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    UNIQUE(camera_id, name)
                )
                """
            )

    @staticmethod
    def _validate_points(points: List[List[float]]) -> List[List[float]]:
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError("A polygon zone requires at least 3 points")
        cleaned = []
        for p in points:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                raise ValueError("Each zone point must be [x, y]")
            x, y = float(p[0]), float(p[1])
            # Stored coordinates are normalized so the same zone works at any resolution.
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("Zone coordinates must be normalized between 0 and 1")
            cleaned.append([x, y])
        return cleaned

    def upsert(self, camera_id: str, name: str, points: List[List[float]], zone_type: str = "RESTRICTED", enabled: bool = True) -> Dict[str, Any]:
        camera_id = (camera_id or "default").strip()
        name = (name or "Restricted Zone").strip()
        zone_type = (zone_type or "RESTRICTED").upper().strip()
        points = self._validate_points(points)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO safety_zones(camera_id, name, zone_type, points_json, enabled)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(camera_id, name) DO UPDATE SET
                    zone_type=excluded.zone_type,
                    points_json=excluded.points_json,
                    enabled=excluded.enabled
                """,
                (camera_id, name, zone_type, json.dumps(points), int(bool(enabled))),
            )
        return self.get(camera_id, name)

    def get(self, camera_id: str, name: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM safety_zones WHERE camera_id=? AND name=?",
                (camera_id, name),
            ).fetchone()
        if not row:
            raise KeyError(name)
        return self._row(row)

    def list(self, camera_id: str = "default", enabled_only: bool = False) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM safety_zones WHERE camera_id=?"
        params = [camera_id]
        if enabled_only:
            sql += " AND enabled=1"
        sql += " ORDER BY id"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row(r) for r in rows]

    def delete(self, zone_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM safety_zones WHERE id=?", (int(zone_id),))
        return cur.rowcount > 0

    @staticmethod
    def _row(row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "camera_id": row["camera_id"],
            "name": row["name"],
            "zone_type": row["zone_type"],
            "points": json.loads(row["points_json"]),
            "enabled": bool(row["enabled"]),
        }
