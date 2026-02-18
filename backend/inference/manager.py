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

    def load_all_models(self, models_dir: str):
        """
        Scans the directory for .ckpt files and loads them.
        The filename (without extension) becomes the model name.
        """
        if not os.path.exists(models_dir):
            print(f"Models directory not found at {models_dir}")
            return

        print(f"Scanning for models in {models_dir}...")
        count = 0
        for filename in os.listdir(models_dir):
            if filename.endswith(".ckpt"):
                name = os.path.splitext(filename)[0] # e.g., "front.ckpt" -> "front"
                path = os.path.join(models_dir, filename)
                try:
                    self.load_model(name, path)
                    count += 1
                except Exception as e:
                    print(f"Failed to load model '{name}': {e}")
        
        print(f"Loaded {count} models from {models_dir}")

    def get_model(self, name: Optional[str] = None) -> JerryScanPadimModel:
        """
        Retrieves a model by name.
        """
        print(name)
        if name is None:
             # If no name provided, try 'default' or fall back to single model
             if "default" in self.models:
                 return self.models["default"]
             if len(self.models) == 1:
                 return next(iter(self.models.values()))
             if len(self.models) > 0:
                 # If multiple models exist but none requested, return the first one?
                 # Or raise error to be specific.
                 return next(iter(self.models.values()))
             
             raise KeyError("No models loaded.")
        
        if name not in self.models:
            raise KeyError(f"Model '{name}' not found.")
        
        return self.models[name]
