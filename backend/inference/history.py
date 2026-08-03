
import os
import json
import uuid
import tempfile
import threading
from datetime import datetime, timezone
from typing import List, Dict, Optional

class HistoryManager:
    def __init__(self, history_file: str | None = None):
        # Runtime evidence never belongs in the tracked backend source tree.
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_path = os.path.join(os.path.dirname(self.base_dir), "runtime-data", "inspections_history.json")
        self.history_path = history_file or os.getenv("JERRYSCAN_HISTORY_PATH", default_path)
        os.makedirs(os.path.dirname(os.path.abspath(self.history_path)), exist_ok=True)
        self._lock = threading.RLock()
        self.max_sessions = int(os.getenv("JERRYSCAN_HISTORY_MAX_SESSIONS", "10000"))
        if self.max_sessions <= 0:
            raise ValueError("JERRYSCAN_HISTORY_MAX_SESSIONS must be positive")
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not os.path.exists(self.history_path):
            self._save([])

    def _load(self):
        with self._lock, open(self.history_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save(self, data):
        with self._lock:
            directory = os.path.dirname(os.path.abspath(self.history_path))
            handle, temporary = tempfile.mkstemp(dir=directory, suffix='.tmp')
            try:
                with os.fdopen(handle, 'w', encoding='utf-8') as stream:
                    json.dump(data, stream, indent=4)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.history_path)
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    def save_session(self, angles_results: Dict[str, Dict], overall_status: str, model_name: Optional[str] = None) -> str:
        """
        Saves a full Jerrycan inspection session.
        """
        session_id = str(uuid.uuid4())
        session = {
            "id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": overall_status,
            "model_name": model_name,
            "angles": angles_results
        }

        with self._lock:
            data = self._load()
            data.insert(0, session) # Most recent first
            del data[self.max_sessions:]
            self._save(data)
        
        return session_id

    def get_history(self, status: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """
        Retrieves inspection history with optional filtering.
        """
        data = self._load()
        
        if status:
            data = [s for s in data if s["overall_status"] == status]
        
        return data[:limit]

    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Gets a single session by ID.
        """
        data = self._load()
        
        for s in data:
            if s["id"] == session_id:
                return s
        return None

    def get_stats(self) -> Dict:
        """
        Calculates aggregated statistics.
        """
        data = self._load()
        
        total = len(data)
        if total == 0:
            return {
                "total": 0, "decision_count": 0, "pass_rate": None,
                "passes": 0, "faults": 0, "reviews": 0, "shadow": 0,
                "system_errors": 0, "other": 0,
            }
        counts = {
            status: len([session for session in data if session.get("overall_status") == status])
            for status in ("PASS", "FAIL", "REVIEW", "SHADOW", "SYSTEM_ERROR")
        }
        decision_count = counts["PASS"] + counts["FAIL"]
        known = sum(counts.values())
        
        return {
            "total": total,
            "decision_count": decision_count,
            "passes": counts["PASS"],
            "faults": counts["FAIL"],
            "reviews": counts["REVIEW"],
            "shadow": counts["SHADOW"],
            "system_errors": counts["SYSTEM_ERROR"],
            "other": total - known,
            "pass_rate": (counts["PASS"] / decision_count) * 100 if decision_count else None,
        }
