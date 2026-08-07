"""SQLite-backed inspection history for the manufacturing application."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


VALID_INSPECTION_STATUSES = frozenset({"PASS", "FAIL", "WRONG_INPUT"})


class HistoryManager:
    """Persist inspection sessions and per-angle evidence in SQLite.

    A connection is opened per operation so the manager is safe to call from
    FastAPI's worker threads. WAL mode and a busy timeout allow readers and the
    single local writer to overlap without an application-level file lock.
    """

    def __init__(self, db_file: str | os.PathLike[str] | None = None):
        repository_root = Path(__file__).resolve().parents[2]
        default_path = repository_root / "runtime-data" / "inspections_history.db"
        configured = db_file or os.getenv("JERRYSCAN_HISTORY_PATH") or default_path
        self.db_path = Path(configured).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_sessions = int(os.getenv("JERRYSCAN_HISTORY_MAX_SESSIONS", "10000"))
        if self.max_sessions <= 0:
            raise ValueError("JERRYSCAN_HISTORY_MAX_SESSIONS must be positive")
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @contextmanager
    def _connection(self):
        conn = self._get_conn()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    overall_status TEXT NOT NULL
                        CHECK (overall_status IN ('PASS', 'FAIL', 'WRONG_INPUT')),
                    model_name TEXT
                );

                CREATE TABLE IF NOT EXISTS angle_results (
                    session_id TEXT NOT NULL,
                    angle_id TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    PRIMARY KEY (session_id, angle_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_status
                    ON sessions(overall_status);
                CREATE INDEX IF NOT EXISTS idx_sessions_newest
                    ON sessions(timestamp DESC, id DESC);
                """
            )

    @staticmethod
    def _validate_status(status: str) -> str:
        if status not in VALID_INSPECTION_STATUSES:
            allowed = ", ".join(sorted(VALID_INSPECTION_STATUSES))
            raise ValueError(f"Unknown inspection status {status!r}; expected one of {allowed}")
        return status

    @staticmethod
    def _validate_angles(angles_results: Dict[str, Dict[str, Any]]) -> None:
        if not isinstance(angles_results, dict) or not angles_results:
            raise ValueError("angles_results must be a non-empty object")
        for angle_id, result in angles_results.items():
            if not isinstance(angle_id, str) or not angle_id:
                raise ValueError("Every angle id must be a non-empty string")
            if not isinstance(result, dict):
                raise ValueError(f"Result for angle {angle_id!r} must be an object")

    def save_session(
        self,
        angles_results: Dict[str, Dict[str, Any]],
        overall_status: str,
        model_name: Optional[str] = None,
    ) -> str:
        """Save one new inspection session and return its generated UUID."""

        self._validate_status(overall_status)
        self._validate_angles(angles_results)
        session_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            self._insert(conn, session_id, timestamp, overall_status, model_name, angles_results)
            self._prune(conn)
        return session_id

    @classmethod
    def _validated_import(cls, session: Dict[str, Any]) -> tuple[str, str, str, Optional[str], Dict[str, Dict[str, Any]]]:
        if not isinstance(session, dict):
            raise ValueError("Each legacy session must be an object")
        session_id = session.get("id")
        timestamp = session.get("timestamp")
        status = session.get("overall_status")
        angles = session.get("angles")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Legacy session id must be a non-empty string")
        if not isinstance(timestamp, str) or not timestamp:
            raise ValueError(f"Legacy session {session_id!r} has no valid timestamp")
        cls._validate_status(status)
        cls._validate_angles(angles)
        model_name = session.get("model_name")
        if model_name is not None and not isinstance(model_name, str):
            raise ValueError(f"Legacy session {session_id!r} has an invalid model_name")
        return session_id, timestamp, status, model_name, angles

    def import_session(self, session: Dict[str, Any]) -> bool:
        """Import one legacy session idempotently."""

        inserted, _ = self.import_sessions([session])
        return inserted == 1

    def import_sessions(self, sessions: List[Dict[str, Any]]) -> tuple[int, int]:
        """Atomically import validated legacy sessions.

        Identical IDs are skipped, while conflicting IDs abort the entire
        transaction. Returns ``(inserted, skipped)``.
        """

        if not isinstance(sessions, list):
            raise ValueError("Legacy history must be a JSON array")
        validated = [self._validated_import(session) for session in sessions]
        seen: dict[str, tuple[str, str, Optional[str], Dict[str, Dict[str, Any]]]] = {}
        for session_id, timestamp, status, model_name, angles in validated:
            evidence = (timestamp, status, model_name, angles)
            if session_id in seen and seen[session_id] != evidence:
                raise ValueError(f"Legacy JSON contains conflicting session id {session_id!r}")
            seen[session_id] = evidence

        inserted = 0
        skipped = 0
        with self._connection() as conn:
            for session_id, timestamp, status, model_name, angles in validated:
                existing = conn.execute(
                    "SELECT timestamp, overall_status, model_name FROM sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
                if existing is None:
                    self._insert(conn, session_id, timestamp, status, model_name, angles)
                    inserted += 1
                    continue

                expected = (timestamp, status, model_name)
                actual = (
                    existing["timestamp"],
                    existing["overall_status"],
                    existing["model_name"],
                )
                if actual != expected or self._load_angles(conn, session_id) != angles:
                    raise ValueError(
                        f"Legacy session id {session_id!r} conflicts with existing data"
                    )
                skipped += 1
            self._prune(conn)
        return inserted, skipped

    @staticmethod
    def _insert(
        conn: sqlite3.Connection,
        session_id: str,
        timestamp: str,
        status: str,
        model_name: Optional[str],
        angles: Dict[str, Dict[str, Any]],
    ) -> None:
        conn.execute(
            "INSERT INTO sessions (id, timestamp, overall_status, model_name) VALUES (?, ?, ?, ?)",
            (session_id, timestamp, status, model_name),
        )
        conn.executemany(
            "INSERT INTO angle_results (session_id, angle_id, result_json) VALUES (?, ?, ?)",
            [
                (session_id, angle_id, json.dumps(result, separators=(",", ":")))
                for angle_id, result in angles.items()
            ],
        )

    def _prune(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            DELETE FROM sessions
            WHERE id IN (
                SELECT id FROM sessions
                ORDER BY timestamp DESC, id DESC
                LIMIT -1 OFFSET ?
            )
            """,
            (self.max_sessions,),
        )

    def get_history(self, status: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Return newest sessions first, optionally filtered by status."""

        if status is not None:
            self._validate_status(status)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        with self._connection() as conn:
            if status is None:
                rows = conn.execute(
                    """SELECT id, timestamp, overall_status, model_name
                       FROM sessions ORDER BY timestamp DESC, id DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT id, timestamp, overall_status, model_name
                       FROM sessions WHERE overall_status = ?
                       ORDER BY timestamp DESC, id DESC LIMIT ?""",
                    (status, limit),
                ).fetchall()
            return [self._row_to_session(conn, row) for row in rows]

    def _load_angles(self, conn: sqlite3.Connection, session_id: str) -> Dict[str, Dict]:
        rows = conn.execute(
            """SELECT angle_id, result_json FROM angle_results
               WHERE session_id = ? ORDER BY angle_id""",
            (session_id,),
        ).fetchall()
        return {row["angle_id"]: json.loads(row["result_json"]) for row in rows}

    def _row_to_session(self, conn: sqlite3.Connection, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "overall_status": row["overall_status"],
            "model_name": row["model_name"],
            "angles": self._load_angles(conn, row["id"]),
        }

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Return one complete session, or ``None`` when it does not exist."""

        with self._connection() as conn:
            row = conn.execute(
                """SELECT id, timestamp, overall_status, model_name
                   FROM sessions WHERE id = ?""",
                (session_id,),
            ).fetchone()
            return self._row_to_session(conn, row) if row is not None else None

    def get_stats(self) -> Dict[str, Any]:
        """Aggregate decisions in SQL; wrong inputs never count as defects."""

        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN overall_status = 'PASS' THEN 1 ELSE 0 END) AS passes,
                    SUM(CASE WHEN overall_status = 'FAIL' THEN 1 ELSE 0 END) AS faults,
                    SUM(CASE WHEN overall_status = 'WRONG_INPUT' THEN 1 ELSE 0 END) AS wrong_inputs
                FROM sessions
                """
            ).fetchone()
        passes = int(row["passes"] or 0)
        faults = int(row["faults"] or 0)
        decision_count = passes + faults
        return {
            "total": int(row["total"]),
            "decision_count": decision_count,
            "passes": passes,
            "faults": faults,
            "wrong_inputs": int(row["wrong_inputs"] or 0),
            "pass_rate": (passes / decision_count) * 100 if decision_count else None,
        }
