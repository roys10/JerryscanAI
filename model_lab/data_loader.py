import os
import glob
from typing import List, Dict, Optional
import numpy as np
from PIL import Image

class LabDataLoader:
    """Helper to load image datasets for benchmarking."""
    
    def __init__(self, dataset_path: str, angle: str):
        self.dataset_path = os.path.join(dataset_path, angle)
        self.categories = ["normal", "fault"]
        
    def get_samples(self) -> List[Dict]:
        """
        Scans researchers' dataset structure.
        Expected:
        path/angle/normal/*.png
        path/angle/fault/*.png
        path/angle/ground_truth/*.png
        """
        samples = []
        
        for cat in self.categories:
            cat_dir = os.path.join(self.dataset_path, cat)
            if not os.path.exists(cat_dir):
                continue
                
            # Find all images
            for ext in ["*.jpg", "*.png", "*.jpeg", "*.bmp"]:
                for img_path in glob.glob(os.path.join(cat_dir, ext)):
                    filename = os.path.basename(img_path)
                    
                    sample = {
                        "path": img_path,
                        "filename": filename,
                        "label": cat,
                        "is_anomaly": cat == "fault",
                        "mask_path": None
                    }
                    
                    # Look for ground truth mask
                    if cat == "fault":
                        gt_dir = os.path.join(self.dataset_path, "ground_truth")
                        if os.path.exists(gt_dir):
                            mask_p = os.path.join(gt_dir, filename)
                            if os.path.exists(mask_p):
                                sample["mask_path"] = mask_p
                    
                    samples.append(sample)
        
        return samples

    def load_image(self, path: str):
        return Image.open(path).convert("RGB")

    def load_mask(self, path: str):
        if not path:
            return None
        return np.array(Image.open(path).convert("L"))
