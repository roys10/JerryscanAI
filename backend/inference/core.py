
import torch
import numpy as np
import cv2
import base64
from anomalib.models import Padim
from torchvision.transforms import v2
from PIL import Image
import io

class DictDot(dict):
    """Dict that supports dot access (required for Anomalib models)."""
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

class JerryScanPadimModel:
    def __init__(self, ckpt_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading Padim model from {ckpt_path}...")
        
        # Load model
        self.model = Padim.load_from_checkpoint(ckpt_path).to(self.device).eval()
        
        # Preprocessing (Exact Match to CLI/PredictDataset)
        # Resize to 256x256 and Normalize (ImageNet stats)
        self.transform = v2.Compose([
            v2.Resize((256, 256), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image_bytes: bytes) -> dict:
        # 1. Decode Image
        nparr = np.frombuffer(image_bytes, np.uint8)
        original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if original is None: raise ValueError("Could not decode image")
        
        # Convert to tensor for transform
        # v2 transforms expect [C, H, W] or [H, W, C] depending... 
        # Safest is to convert to PIL or torch Tensor [3, H, W]
        img_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        # Convert to tensor [3, H, W] scaled 0-255 (uint8)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1) 
        
        # Apply Transform
        # v2.Resize expects tensor [C, H, W]
        input_tensor = self.transform(tensor).unsqueeze(0).to(self.device) # [1, 3, 256, 256]

        # 2. Prepare Batch (for predict_step)
        batch = DictDot({"image": input_tensor})
        
        # 3. Inference
        with torch.no_grad():
            outputs = self.model.predict_step(batch, 0)

        # 4. Extract Results
        # predict_step updates batch in-place
        if isinstance(batch, dict) or isinstance(batch, DictDot):
            anomaly_map = batch.anomaly_map
            pred_score = batch.pred_score
        else:
             # Fallback
            anomaly_map = batch.anomaly_map
            pred_score = batch.pred_score

        if isinstance(anomaly_map, torch.Tensor):
            anomaly_map = anomaly_map.cpu().numpy()
        if isinstance(pred_score, torch.Tensor):
            pred_score = pred_score.item()

        # Squeeze [1, 1, 256, 256] -> [256, 256]
        if anomaly_map.ndim == 4: anomaly_map = anomaly_map[0, 0]
        elif anomaly_map.ndim == 3: anomaly_map = anomaly_map[0]

        # 5. Post-Processing (Global Norm -> Blur -> Threshold)
        
        # Global Normalization
        # Try to get stats from various locations (Robust)
        min_val, max_val = None, None
        if hasattr(self.model, 'pixel_min') and hasattr(self.model, 'pixel_max'):
            min_val = self.model.pixel_min.cpu().numpy()
            max_val = self.model.pixel_max.cpu().numpy()
        elif hasattr(self.model, 'image_min') and hasattr(self.model, 'image_max'):
            min_val = self.model.image_min.cpu().numpy()
            max_val = self.model.image_max.cpu().numpy()
        elif hasattr(self.model, 'post_processor') and hasattr(self.model.post_processor, 'pixel_min'):
             min_val = self.model.post_processor.pixel_min.cpu().numpy()
             max_val = self.model.post_processor.pixel_max.cpu().numpy()

        if min_val is not None and max_val is not None:
            # print(f"Global Norm: {min_val} - {max_val}")
            anomaly_map_norm = (anomaly_map - min_val) / (max_val - min_val + 1e-6)
            anomaly_map_norm = np.clip(anomaly_map_norm, 0, 1)
        else:
            # Fallback
            sys_min, sys_max = anomaly_map.min(), anomaly_map.max()
            anomaly_map_norm = (anomaly_map - sys_min) / (sys_max - sys_min + 1e-6)

        # Gaussian Blur (Standard Anomalib Step)
        anomaly_map_norm = cv2.GaussianBlur(anomaly_map_norm, (0, 0), sigmaX=4, sigmaY=4)
        
        # Thresholding
        threshold = 0.5
        if hasattr(self.model, 'pixel_threshold'):
            threshold = self.model.pixel_threshold.value.item()
        elif hasattr(self.model, 'image_threshold'):
            threshold = self.model.image_threshold.value.item()
            
        pred_mask = (anomaly_map_norm > threshold).astype(np.uint8)

        # --- Score Percentage Calculation (0-100%) ---
        if min_val is not None and max_val is not None:
             norm_score = (pred_score - min_val) / (max_val - min_val + 1e-6)
        else:
             norm_score = min(pred_score, 1.0)
        
        score_percentage = float(np.clip(norm_score, 0, 1) * 100)
        threshold_percentage = float(threshold * 100)

        # 6. Visualization
        h, w = original.shape[:2]
        
        # Resize map/mask to original
        anomaly_map_up = cv2.resize(anomaly_map_norm, (w, h), interpolation=cv2.INTER_LINEAR)
        pred_mask_up = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Heatmap
        heatmap_u8 = (anomaly_map_up * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        heatmap_overlay = cv2.addWeighted(original, 0.6, colormap, 0.4, 0)

        # Segmentation (Red Contour)
        seg_overlay = original.copy()
        mask255 = (pred_mask_up * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(seg_overlay, contours, -1, (0, 0, 255), 3)

        return {
            "status": "FAIL" if norm_score > threshold else "PASS", 
            "score": pred_score,
            "score_percentage": score_percentage,
            "threshold_percentage": threshold_percentage,
            "heatmap_image": self._encode(heatmap_overlay),
            "segmentation_image": self._encode(seg_overlay),
            "original_image": self._encode(original)
        }

    def _encode(self, img):
        _, buffer = cv2.imencode('.jpg', img)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
