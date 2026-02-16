
import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from anomalib.data import PredictDataset
from anomalib.models import Padim
from torchvision.transforms import v2
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
ckpt_path = "model.ckpt"
image_path = "cli_predictions/G01-251224-094208-433.bmp"

# --- Helper Classes ---
class DictDot(dict):
    """Dict that supports dot access (required for Anomalib models)."""
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

def collate_fn(batch):
    """Custom collate function to stack images and preserve metadata."""
    # Stack images
    images = torch.stack([item.image for item in batch])
    
    # Create dict with all fields
    batch_dict = {"image": images}
    
    # Copy other fields as lists if available
    if len(batch) > 0:
        keys = batch[0].__dataclass_fields__.keys()
        for k in keys:
            if k == "image": continue
            batch_dict[k] = [getattr(item, k) for item in batch]
            
    return DictDot(batch_dict)

def main():
    print(f"Loading model from {ckpt_path}...")
    try:
        model = Padim.load_from_checkpoint(ckpt_path)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Preprocessing (Exact CLI Match) ---
    print(f"Preparing data for {image_path}...")
    # Standard ImageNet normalization + Resize to 256x256 (Padim default)
    transform = v2.Compose([
        v2.Resize((256, 256), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        dataset = PredictDataset(path=image_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        return

    # --- Inference ---
    print("Running inference...")
    try:
        batch = next(iter(dataloader))
        # Move to model device if necessary (default CPU for this script)
        outputs = model.predict_step(batch, 0)
        # outputs is the updated batch object
    except Exception as e:
        print(f"Error during inference: {e}")
        return

    # --- Post-Processing (CLI Parity) ---
    print("Post-processing...")
    
    # 1. Extract Anomaly Map
    # predict_step updates batch in-place and returns None (or dict.update result)
    if isinstance(batch, dict) or isinstance(batch, DictDot):
        anomaly_map = batch.anomaly_map
        pred_score = batch.pred_score
    else:
        # Fallback if batch is something else
        anomaly_map = batch.anomaly_map
        pred_score = batch.pred_score

    if isinstance(anomaly_map, torch.Tensor):
        anomaly_map = anomaly_map.cpu().numpy()
    if isinstance(pred_score, torch.Tensor):
        pred_score = pred_score.item()

    # Squeeze batch dim -> [256, 256]
    if anomaly_map.ndim == 4: anomaly_map = anomaly_map[0, 0]
    elif anomaly_map.ndim == 3: anomaly_map = anomaly_map[0]

    # 2. Global Normalization (Crucial for correct colors)
    # Try to get stats from various locations
    if hasattr(model, 'pixel_min') and hasattr(model, 'pixel_max'):
        min_val = model.pixel_min.cpu().numpy()
        max_val = model.pixel_max.cpu().numpy()
    elif hasattr(model, 'image_min') and hasattr(model, 'image_max'):
        min_val = model.image_min.cpu().numpy()
        max_val = model.image_max.cpu().numpy()
    elif hasattr(model, 'post_processor') and hasattr(model.post_processor, 'pixel_min'):
         min_val = model.post_processor.pixel_min.cpu().numpy()
         max_val = model.post_processor.pixel_max.cpu().numpy()

    if min_val is not None and max_val is not None:
        print(f"Global Normalization stats: Min={min_val:.4f}, Max={max_val:.4f}")
        # Standard normalization
        anomaly_map_norm = (anomaly_map - min_val) / (max_val - min_val + 1e-6)
        anomaly_map_norm = np.clip(anomaly_map_norm, 0, 1)
    else:
        print("WARNING: No global stats found. Fallback to local normalization.")
        # Fallback local norm
        anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-6)

    # 3. Gaussian Blur (Smoother contours)
    # CLI applies sigma=4 blur
    anomaly_map_norm = cv2.GaussianBlur(anomaly_map_norm, (0, 0), sigmaX=4, sigmaY=4)

    # 4. Thresholding
    threshold = 0.5
    if hasattr(model, 'pixel_threshold'):
        threshold = model.pixel_threshold.value.item()
        print(f"Using Trained Threshold: {threshold:.4f}")
    elif hasattr(model, 'image_threshold'):
        threshold = model.image_threshold.value.item()
        print(f"Using Trained Threshold: {threshold:.4f}")

    pred_mask = (anomaly_map_norm > threshold).astype(np.uint8)

    # --- Convert Score to Percentage (0-100%) ---
    # Normalize the raw score using the same global stats
    if min_val is not None and max_val is not None:
        norm_score = (pred_score - min_val) / (max_val - min_val + 1e-6)
    else:
        # Fallback if no stats 
        norm_score = min(pred_score, 1.0)
    
    score_percentage = np.clip(norm_score, 0, 1) * 100

    print("-" * 30)
    print(f"Prediction: {'BAD' if norm_score > threshold else 'GOOD'}")
    print(f"Anomaly Score: {score_percentage:.2f}% (Raw: {pred_score:.4f})")
    print(f"Threshold Limit: {threshold * 100:.2f}%")
    print("-" * 30)

    # --- Visualization ---
    # Load original image for display
    orig_bgr = cv2.imread(image_path)
    if orig_bgr is None:
        print("Could not read original image for visualization.")
        return
        
    h, w = orig_bgr.shape[:2]
    
    # Resize map/mask to original image size
    anomaly_map_up = cv2.resize(anomaly_map_norm, (w, h), interpolation=cv2.INTER_LINEAR)
    pred_mask_up = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 1. Heatmap
    # Normalize 0-1 -> 0-255
    heat = (anomaly_map_up * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay_heat = cv2.addWeighted(orig_bgr, 0.6, heat, 0.4, 0)

    # 2. Contour
    mask255 = (pred_mask_up * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_mask = orig_bgr.copy()
    cv2.drawContours(overlay_mask, contours, -1, (0, 0, 255), 3)

    # 3. Save/Show
    def add_title(img, text):
        out = img.copy()
        cv2.putText(out, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        return out

    final_cat = np.hstack([
        add_title(orig_bgr, "Original"),
        add_title(overlay_heat, "Anomaly Map"),
        add_title(overlay_mask, "Pred Mask")
    ])
    
    cv2.imwrite("output_result.png", final_cat)
    print("Saved result to output_result.png")

if __name__ == "__main__":
    main()
