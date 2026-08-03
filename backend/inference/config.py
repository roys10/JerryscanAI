"""Runtime alert settings with environment-only credentials."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


class ConfigManager:
    """Persist non-secret alert settings outside the tracked source tree."""

    def __init__(self, config_path: str | None = None):
        backend_dir = Path(__file__).resolve().parents[1]
        project_root = backend_dir.parent
        load_dotenv(backend_dir / ".env")
        configured = config_path or os.getenv("JERRYSCAN_CONFIG_PATH")
        self.config_path = (
            Path(configured).expanduser()
            if configured
            else backend_dir / "config.json"
        ).resolve()
        self._lock = threading.RLock()
        self.default_config = {
            "smtp": {
                "server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
                "port": int(os.getenv("SMTP_PORT", "587")),
                "user": os.getenv("SMTP_USER", ""),
            },
            "alerts": [
                {
                    "id": "default-streak",
                    "name": "Failure Streak",
                    "type": "consecutive_fails",
                    "threshold": 3,
                    "emails": [],
                    "webhook_url": "",
                    "enabled": True,
                }
            ],
        }
        self.config = self._load()

    @staticmethod
    def _without_secrets(data: dict[str, Any]) -> dict[str, Any]:
        sanitized = copy.deepcopy(data)
        smtp = sanitized.get("smtp")
        if isinstance(smtp, dict):
            smtp.pop("password", None)
            smtp.pop("password_configured", None)
        return sanitized

    def _load(self) -> dict[str, Any]:
        if not self.config_path.exists():
            value = copy.deepcopy(self.default_config)
            self._save(value)
            return value
        try:
            stored = json.loads(self.config_path.read_text(encoding="utf-8"))
            if not isinstance(stored, dict):
                raise ValueError("settings must be a JSON object")
            merged = copy.deepcopy(self.default_config)
            merged.update(self._without_secrets(stored))
            merged["smtp"] = {
                **self.default_config["smtp"],
                **self._without_secrets(stored).get("smtp", {}),
            }
            return merged
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            print(f"Alert settings could not be loaded ({type(exc).__name__}); using safe defaults.")
            return copy.deepcopy(self.default_config)

    def _save(self, data: dict[str, Any]) -> None:
        sanitized = self._without_secrets(data)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=self.config_path.parent, suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(sanitized, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.config_path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def get_all(self) -> dict[str, Any]:
        """Return public settings; never return the SMTP credential."""
        with self._lock:
            public = self._without_secrets(self.config)
            public["smtp"]["password_configured"] = bool(os.getenv("SMTP_PASSWORD"))
            return public

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            value = copy.deepcopy(self.config.get(key, default))
        if key == "smtp" and isinstance(value, dict):
            value.update(
                {
                    "server": os.getenv("SMTP_SERVER", value.get("server", "")),
                    "port": int(os.getenv("SMTP_PORT", str(value.get("port", 587)))),
                    "user": os.getenv("SMTP_USER", value.get("user", "")),
                    "password": os.getenv("SMTP_PASSWORD", ""),
                }
            )
        return value

    def update(self, new_settings: dict[str, Any]) -> dict[str, Any]:
        """Persist settings after stripping any submitted secret field."""
        if not isinstance(new_settings, dict):
            raise ValueError("Settings must be a JSON object")
        sanitized = self._without_secrets(new_settings)
        with self._lock:
            self.config.update(sanitized)
            if "smtp" in sanitized:
                self.config["smtp"] = {
                    **self.default_config["smtp"],
                    **sanitized["smtp"],
                }
            self._save(self.config)
            return self.get_all()
