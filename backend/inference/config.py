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
            "smtp": {
                "server": "smtp.gmail.com",
                "port": 587,
                "user": "",
                "password": ""
            },
            "alerts": [
                {
                    "id": "default-streak",
                    "name": "Failure Streak",
                    "type": "consecutive_fails",
                    "threshold": 3,
                    "emails": [],
                    "webhook_url": "",
                    "enabled": True
                }
            ]
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
                # For a professional system, we merge keys but prioritizing user data
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
        """Updates settings and persists them to disk."""
        # Update our in-memory config with whatever the frontend sent
        self.config.update(new_settings)
        self._save(self.config)
        return self.config
