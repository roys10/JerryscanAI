import json
import os
from typing import Dict, Any

class ConfigManager:
    """Manages persistent system configuration via a local JSON file."""
    
    def __init__(self, config_path: str = "config.json"):
        # Default to the backend directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(base_dir, config_path)
        
        # Default configuration
        self.default_config = {
            "alert_threshold": 3,
            "alert_email_address": "",
            "alert_webhook_url": "",
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_user": "",
            "smtp_password": "",
            "alert_pass_rate_threshold": 90,
            "alert_pass_rate_window": 50
        }
        
        self.config = self._load()

    def _load(self) -> Dict[str, Any]:
        """Loads configuration from disk, creating defaults if missing."""
        if not os.path.exists(self.config_path):
            self._save(self.default_config)
            return self.default_config.copy()
            
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
                # Merge with defaults to ensure all keys exist
                merged = self.default_config.copy()
                merged.update(data)
                return merged
        except Exception as e:
            print(f"Error loading config: {e}. using defaults.")
            return self.default_config.copy()

    def _save(self, data: Dict[str, Any]):
        """Saves the current configuration to disk."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get_all(self) -> Dict[str, Any]:
        """Returns the entire configuration dictionary."""
        return self.config.copy()
        
    def get(self, key: str, default: Any = None) -> Any:
        """Gets a specific configuration value."""
        return self.config.get(key, default)

    def update(self, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Updates multiple settings and persists them to disk."""
        for key, value in new_settings.items():
            if key in self.default_config:
                self.config[key] = value
        
        self._save(self.config)
        return self.config
