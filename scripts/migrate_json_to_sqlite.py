"""
One-time migration script: converts inspections_history.json to SQLite database.

Usage:
    python scripts/migrate_json_to_sqlite.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from inference.history import HistoryManager


def migrate():
    backend_dir = os.path.join(os.path.dirname(__file__), "..", "backend")
    json_path = os.path.join(backend_dir, "inspections_history.json")

    if not os.path.exists(json_path):
        print("No inspections_history.json found — nothing to migrate.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    if not data:
        print("JSON file is empty — nothing to migrate.")
        return

    print(f"Found {len(data)} sessions to migrate...")

    manager = HistoryManager()

    # Check if DB already has data
    stats = manager.get_stats()
    if stats["total"] > 0:
        print(f"Database already has {stats['total']} sessions. Skipping migration to avoid duplicates.")
        print("Delete the .db file first if you want to re-migrate.")
        return

    # Insert sessions (they are stored most-recent-first in JSON, so reverse for chronological insert)
    conn = manager._get_conn()
    try:
        for i, session in enumerate(reversed(data)):
            conn.execute(
                "INSERT INTO sessions (id, timestamp, overall_status, model_name) VALUES (?, ?, ?, ?)",
                (
                    session["id"],
                    session["timestamp"],
                    session["overall_status"],
                    session.get("model_name"),
                ),
            )
            for angle_id, result in session.get("angles", {}).items():
                conn.execute(
                    "INSERT INTO angle_results (session_id, angle_id, result_json) VALUES (?, ?, ?)",
                    (session["id"], angle_id, json.dumps(result)),
                )
            if (i + 1) % 100 == 0:
                print(f"  Migrated {i + 1}/{len(data)} sessions...")
                conn.commit()

        conn.commit()
    finally:
        conn.close()

    final_stats = manager.get_stats()
    print(f"Migration complete! {final_stats['total']} sessions in database.")
    print(f"  Passes: {final_stats['passes']}, Fails: {final_stats['fails']}, Pass rate: {final_stats['pass_rate']:.1f}%")
    print(f"\nYou can now safely rename/archive inspections_history.json.")


if __name__ == "__main__":
    migrate()
