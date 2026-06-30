import json
import os
from typing import Dict, Any
from dotenv import load_dotenv


# SMTP settings can be supplied via environment variables. These take
# precedence over anything in config.json so that credentials — especially the
# password — are sourced from the environment and never need to be stored on
# disk. See backend/.env.example.
SMTP_ENV_VARS = {
    "server": "SMTP_SERVER",
    "port": "SMTP_PORT",
    "user": "SMTP_USER",
    "password": "SMTP_PASSWORD",
}

DEFAULT_SMTP = {
    "server": "smtp.gmail.com",
    "port": 587,
    "user": "",
    "password": "",
}


class ConfigManager:
    """Manages persistent system configuration via a local JSON file.

    Non-secret settings (alert rules, SMTP host/port/user) are persisted to
    config.json. The SMTP password is read from the environment and is never
    written to disk.
    """

    def __init__(self, config_path: str = "config.json"):
        # Default to the backend directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Load environment variables from .env specifically in the backend folder
        load_dotenv(os.path.join(base_dir, ".env"))
        self.config_path = os.path.join(base_dir, config_path)

        # Default configuration (SMTP values overlaid from the environment)
        self.default_config = {
            "smtp": self._smtp_from_env(dict(DEFAULT_SMTP)),
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
    def _smtp_from_env(smtp: Dict[str, Any]) -> Dict[str, Any]:
        """Overlay SMTP settings with environment variables when present.

        Environment variables win over file/UI values so credentials can be
        supplied securely at runtime instead of being committed to disk.
        """
        result = dict(smtp)
        for field, env_name in SMTP_ENV_VARS.items():
            value = os.getenv(env_name)
            if value:
                result[field] = int(value) if field == "port" else value
        return result

    def _load(self) -> Dict[str, Any]:
        """Loads configuration from disk, creating defaults if missing."""
        if not os.path.exists(self.config_path):
            self._save(self.default_config)
            return self.default_config.copy()

        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
                # Merge file data over defaults, prioritising user data
                merged = self.default_config.copy()
                merged.update(data)

                # Environment variables are the source of truth for SMTP settings
                merged["smtp"] = self._smtp_from_env(merged.get("smtp", {}))

                return merged
        except Exception as e:
            print(f"Error loading config: {e}. using defaults.")
            return self.default_config.copy()

    def _save(self, data: Dict[str, Any]):
        """Saves the configuration to disk, never persisting the SMTP password."""
        try:
            to_save = dict(data)
            smtp = to_save.get("smtp")
            if isinstance(smtp, dict):
                # Keep the secret out of the on-disk file; it comes from the env.
                to_save["smtp"] = {k: v for k, v in smtp.items() if k != "password"}
            with open(self.config_path, 'w') as f:
                json.dump(to_save, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get_all(self) -> Dict[str, Any]:
        """Returns the entire configuration dictionary."""
        return self.config.copy()

    def get(self, key: str, default: Any = None) -> Any:
        """Gets a specific configuration value."""
        return self.config.get(key, default)

    def update(self, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Updates settings and persists them to disk."""
        # Update our in-memory config with whatever the frontend sent
        self.config.update(new_settings)
        # Re-apply env precedence so UI input can't override env-supplied creds
        self.config["smtp"] = self._smtp_from_env(self.config.get("smtp", {}))
        self._save(self.config)
        return self.config
