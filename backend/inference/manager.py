from typing import Dict, Optional, List
from .core import JerryScanAnomalibModel
import os

class JerryScanModelManager:
    def __init__(self):
        # Structure: { "model_name": { "angle_id": JerryScanAnomalibModel } }
        self.models: Dict[str, Dict[str, JerryScanAnomalibModel]] = {}

    def load_model(self, model_name: str, angle_id: str, path: str):
        """
        Loads a single angle model into a specific model set.
        Architecture is auto-detected from the checkpoint.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model checkpoint not found at {path}")
        
        if model_name not in self.models:
            self.models[model_name] = {}

        print(f"Loading model '{model_name}' - '{angle_id}' from {path}...")
        # JerryScanAnomalibModel auto-detects Padim vs PatchCore
        self.models[model_name][angle_id] = JerryScanAnomalibModel(path)
        
    def load_all_models(self, base_models_dir: str):
        """
        Scans base_models_dir for subdirectories. 
        Each subdirectory name is a model set.
        Inside each subdirectory, .ckpt files are loaded as angles.
        """
        if not os.path.exists(base_models_dir):
            print(f"Models directory not found at {base_models_dir}")
            return

        # Look for subdirectories (Model Sets)
        for entry in os.listdir(base_models_dir):
            model_set_path = os.path.join(base_models_dir, entry)
            if os.path.isdir(model_set_path):
                model_name = entry
                
                # Load .ckpt files in this set
                for filename in os.listdir(model_set_path):
                    if filename.endswith(".ckpt"):
                        angle_id = os.path.splitext(filename)[0]
                        ckpt_path = os.path.join(model_set_path, filename)
                        try:
                            # Auto-detection happens inside load_model -> JerryScanAnomalibModel
                            self.load_model(model_name, angle_id, ckpt_path)
                        except Exception as e:
                            print(f"Failed to load angle '{angle_id}' for set '{model_name}': {e}")

        print(f"Loaded {len(self.models)} model sets: {list(self.models.keys())}")

    def get_model_names(self) -> List[str]:
        """Returns list of available model sets."""
        return sorted(list(self.models.keys()))

    def get_model(self, angle_id: str, model_name: Optional[str] = None) -> JerryScanAnomalibModel:
        """
        Retrieves a model for a specific angle from a specific model set.
        """
        if not self.models:
            raise KeyError("No models loaded in the system.")

        # Determine which model set to use
        target_set = model_name
        if target_set is None:
            target_set = self.get_model_names()[0]

        if target_set not in self.models:
            raise KeyError(f"Model set '{target_set}' not found.")
        
        if angle_id not in self.models[target_set]:
            raise KeyError(f"Angle '{angle_id}' not found in model set '{target_set}'.")
        
        return self.models[target_set][angle_id]
