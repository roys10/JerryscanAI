
import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional

class HistoryManager:
    def __init__(self, history_file: str = "inspections_history.json"):
        # Put history file in backend root or dedicated data dir
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.history_path = os.path.join(self.base_dir, history_file)
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not os.path.exists(self.history_path):
            with open(self.history_path, 'w') as f:
                json.dump([], f)

    def save_session(self, angles_results: Dict[str, Dict], overall_status: str, model_name: Optional[str] = None) -> str:
        """
        Saves a full Jerrycan inspection session.
        """
        session_id = str(uuid.uuid4())
        session = {
            "id": session_id,
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "model_name": model_name,
            "angles": angles_results
        }

        with open(self.history_path, 'r+') as f:
            data = json.load(f)
            data.insert(0, session) # Most recent first
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
        
        return session_id

    def get_history(self, status: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """
        Retrieves inspection history with optional filtering.
        """
        with open(self.history_path, 'r') as f:
            data = json.load(f)
        
        if status:
            data = [s for s in data if s["overall_status"] == status]
        
        return data[:limit]

    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Gets a single session by ID.
        """
        with open(self.history_path, 'r') as f:
            data = json.load(f)
        
        for s in data:
            if s["id"] == session_id:
                return s
        return None

    def get_stats(self) -> Dict:
        """
        Calculates aggregated statistics.
        """
        with open(self.history_path, 'r') as f:
            data = json.load(f)
        
        total = len(data)
        if total == 0:
            return {"total": 0, "pass_rate": 0, "fails": 0, "passes": 0}
            
        passes = len([s for s in data if s["overall_status"] == "PASS"])
        fails = total - passes
        
        return {
            "total": total,
            "passes": passes,
            "fails": fails,
            "pass_rate": (passes / total) * 100
        }
