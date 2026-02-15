from typing import Dict, Optional
from .core import JerryScanPadimModel
import os

class JerryScanModelManager:
    def __init__(self):
        self.models: Dict[str, JerryScanPadimModel] = {}
        self.default_model_name = "default"

    def load_model(self, name: str, path: str, model_type: str = "padim"):
        """
        Loads a model and registers it under the given name.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model checkpoint not found at {path}")
        
        print(f"Loading model '{name}' from {path}...")
        if model_type == "padim":
            self.models[name] = JerryScanPadimModel(path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"Model '{name}' loaded.")

    def get_model(self, name: Optional[str] = None) -> JerryScanPadimModel:
        """
        Retrieves a model by name. If name is None, returns the default model.
        """
        if name is None:
            name = self.default_model_name
        
        if name not in self.models:
             # Fallback if only one model exists and no name provided
            if len(self.models) == 1:
                return next(iter(self.models.values()))
            raise KeyError(f"Model '{name}' not found. Available models: {list(self.models.keys())}")
        
        return self.models[name]
