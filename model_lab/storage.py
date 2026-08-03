from __future__ import annotations

import json
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .settings import LabSettings


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        # Windows can briefly deny replacement while another thread, antivirus,
        # or an indexer has the previous JSON file open. Preserve atomic writes,
        # but tolerate these transient locks instead of failing a long benchmark.
        for attempt in range(8):
            try:
                os.replace(name, path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(0.05 * (2**attempt), 0.5))
    finally:
        Path(name).unlink(missing_ok=True)


class ComparisonStore:
    def __init__(self, settings: LabSettings) -> None:
        self.settings = settings
        settings.ensure_writable_directories()

    def create(
        self,
        config: dict[str, Any],
        samples: list[dict[str, Any]],
        evaluation_snapshot: dict[str, Any] | None = None,
    ) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        comparison_id = f"{stamp}-{uuid.uuid4().hex[:8]}"
        folder = self.path(comparison_id)
        folder.mkdir(parents=True)
        atomic_json(folder / "comparison.json", config)
        atomic_json(folder / "samples.json", samples)
        if evaluation_snapshot is not None:
            atomic_json(folder / "evaluation_snapshot.json", evaluation_snapshot)
        self.write_status(comparison_id, "queued", completed=0, total=0)
        return comparison_id

    def path(self, comparison_id: str) -> Path:
        if not re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", comparison_id):
            raise ValueError("Invalid comparison ID")
        return self.settings.results_dir / comparison_id

    def read_json(self, comparison_id: str, filename: str) -> Any:
        return json.loads((self.path(comparison_id) / filename).read_text(encoding="utf-8"))

    def write_status(self, comparison_id: str, state: str, **details: Any) -> None:
        value = {
            "comparison_id": comparison_id,
            "state": state,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **details,
        }
        atomic_json(self.path(comparison_id) / "status.json", value)

    def append_result(self, comparison_id: str, row: dict[str, Any]) -> None:
        path = self.path(comparison_id) / "sample_results.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def results(self, comparison_id: str) -> list[dict[str, Any]]:
        path = self.path(comparison_id) / "sample_results.jsonl"
        if not path.is_file():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def latest_results(self, comparison_id: str) -> list[dict[str, Any]]:
        latest: dict[tuple[str, str], dict[str, Any]] = {}
        for row in self.results(comparison_id):
            latest[(row["model_id"], row["sample_id"])] = row
        return list(latest.values())

    def list(self) -> list[dict[str, Any]]:
        comparisons = []
        for folder in sorted(self.settings.results_dir.iterdir(), reverse=True):
            status = folder / "status.json"
            config = folder / "comparison.json"
            if folder.is_dir() and status.is_file() and config.is_file():
                comparisons.append(
                    {**json.loads(config.read_text(encoding="utf-8")),
                     **json.loads(status.read_text(encoding="utf-8"))}
                )
        return comparisons
