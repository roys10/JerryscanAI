
import os
import json
import uuid
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional


class HistoryManager:
    def __init__(self, db_file: str = "inspections_history.db"):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = os.path.join(self.base_dir, db_file)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                overall_status TEXT NOT NULL,
                model_name TEXT
            );

            CREATE TABLE IF NOT EXISTS angle_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                angle_id TEXT NOT NULL,
                result_json TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(overall_status);
            CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON sessions(timestamp);
            CREATE INDEX IF NOT EXISTS idx_angle_results_session ON angle_results(session_id);
        """)
        conn.commit()
        conn.close()

    def save_session(self, angles_results: Dict[str, Dict], overall_status: str, model_name: Optional[str] = None) -> str:
        """Saves a full Jerrycan inspection session."""
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO sessions (id, timestamp, overall_status, model_name) VALUES (?, ?, ?, ?)",
                (session_id, timestamp, overall_status, model_name),
            )
            for angle_id, result in angles_results.items():
                conn.execute(
                    "INSERT INTO angle_results (session_id, angle_id, result_json) VALUES (?, ?, ?)",
                    (session_id, angle_id, json.dumps(result)),
                )
            conn.commit()
        finally:
            conn.close()

        return session_id

    def get_history(self, status: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Retrieves inspection history with optional filtering. Returns summaries without full image data for performance."""
        conn = self._get_conn()
        try:
            if status:
                rows = conn.execute(
                    "SELECT id, timestamp, overall_status, model_name FROM sessions WHERE overall_status = ? ORDER BY timestamp DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, timestamp, overall_status, model_name FROM sessions ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()

            results = []
            for row in rows:
                session = {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "overall_status": row["overall_status"],
                    "model_name": row["model_name"],
                    "angles": self._load_angles(conn, row["id"]),
                }
                results.append(session)
            return results
        finally:
            conn.close()

    def _load_angles(self, conn: sqlite3.Connection, session_id: str) -> Dict[str, Dict]:
        """Loads angle results for a session."""
        rows = conn.execute(
            "SELECT angle_id, result_json FROM angle_results WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        return {row["angle_id"]: json.loads(row["result_json"]) for row in rows}

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Gets a single session by ID."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT id, timestamp, overall_status, model_name FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "overall_status": row["overall_status"],
                "model_name": row["model_name"],
                "angles": self._load_angles(conn, session_id),
            }
        finally:
            conn.close()

    def get_stats(self) -> Dict:
        """Calculates aggregated statistics using SQL."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT COUNT(*) as total, SUM(CASE WHEN overall_status = 'PASS' THEN 1 ELSE 0 END) as passes FROM sessions"
            ).fetchone()

            total = row["total"]
            passes = row["passes"] or 0
            fails = total - passes

            return {
                "total": total,
                "passes": passes,
                "fails": fails,
                "pass_rate": (passes / total) * 100 if total > 0 else 0,
            }
        finally:
            conn.close()
