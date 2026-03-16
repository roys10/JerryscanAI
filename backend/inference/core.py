
import torch
import numpy as np
import cv2
import base64
from anomalib.models import Padim, Patchcore
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

class JerryScanAnomalibModel:
    def __init__(self, ckpt_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Try loading as Padim, then fallback to Patchcore
        try:
            self.model = Padim.load_from_checkpoint(ckpt_path).to(self.device).eval()
        except Exception:
            try:
                self.model = Patchcore.load_from_checkpoint(ckpt_path).to(self.device).eval()
            except Exception as e:
                print(f"Failed to load {ckpt_path} as Padim or Patchcore. Error: {e}")
                raise ValueError(f"Unsupported model architecture: {ckpt_path}")
        
        # Preprocessing (Exact Match to CLI/PredictDataset)
        self.transform = v2.Compose([
            v2.Resize((256, 256), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image_bytes: bytes) -> dict:
        # 1. Decode Image 
        try:
             image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
             original = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
             print(f"Error loading image: {e}")
             raise ValueError("Could not decode image")
        
        # 2. Preprocess 
        tensor_img = v2.functional.to_image(image_pil)
        input_tensor_float = v2.functional.to_dtype(tensor_img, torch.float32, scale=True)
        input_tensor = self.transform(input_tensor_float).unsqueeze(0).to(self.device)

        # 3. Prepare Batch
        batch = DictDot({"image": input_tensor})
        
        # 4. Inference
        with torch.no_grad():
            outputs = self.model.predict_step(batch, 0)

        # 5. Extract Results (Safe retrieval from outputs or original batch)
        data = outputs if isinstance(outputs, (dict, DictDot)) else batch
        anomaly_map = data.get("anomaly_map")
        pred_score = data.get("pred_score")

        if isinstance(anomaly_map, torch.Tensor):
            anomaly_map = anomaly_map.cpu().numpy()
        if isinstance(pred_score, torch.Tensor):
            pred_score = pred_score.item()

        # Reshape to 2D
        if anomaly_map.ndim == 4: anomaly_map = anomaly_map[0, 0]
        elif anomaly_map.ndim == 3: anomaly_map = anomaly_map[0]

        # 6. Post-Processing (Global Norm -> Blur -> Threshold)
        min_val, max_val = None, None
        
        # Robust Retrieval of Normalization Stats
        # Stats can be in model root or post_processor depending on version/architecture
        targets = [self.model]
        if hasattr(self.model, 'post_processor'):
            targets.insert(0, self.model.post_processor) # Prefer post_processor
            
        for obj in targets:
            # Check for pixel stats first, then image stats
            for attr_prefix in ['pixel', 'image']:
                min_attr = f"{attr_prefix}_min"
                max_attr = f"{attr_prefix}_max"
                if hasattr(obj, min_attr) and hasattr(obj, max_attr):
                    v_min = getattr(obj, min_attr)
                    v_max = getattr(obj, max_attr)
                    # Convert Tensors to numpy
                    if isinstance(v_min, torch.Tensor): v_min = v_min.cpu().numpy()
                    if isinstance(v_max, torch.Tensor): v_max = v_max.cpu().numpy()
                    
                    # Store and break if found valid pair
                    min_val, max_val = v_min, v_max
                    break
            if min_val is not None: break

        if min_val is not None and max_val is not None:
            anomaly_map_norm = (anomaly_map - min_val) / (max_val - min_val + 1e-6)
            anomaly_map_norm = np.clip(anomaly_map_norm, 0, 1)
        else:
            # Local Normalization fallback (Only as a last resort)
            sys_min, sys_max = anomaly_map.min(), anomaly_map.max()
            anomaly_map_norm = (anomaly_map - sys_min) / (sys_max - sys_min + 1e-6)

        anomaly_map_norm = cv2.GaussianBlur(anomaly_map_norm, (0, 0), sigmaX=4, sigmaY=4)
        
        threshold = 0.5
        if hasattr(self.model, 'pixel_threshold'):
            threshold = self.model.pixel_threshold.value.item()
        elif hasattr(self.model, 'image_threshold'):
            threshold = self.model.image_threshold.value.item()
            
        pred_mask = (anomaly_map_norm > threshold).astype(np.uint8)

        # Score Percentage 
        if min_val is not None and max_val is not None:
             norm_score = (pred_score - min_val) / (max_val - min_val + 1e-6)
        else:
             sys_min, sys_max = anomaly_map.min(), anomaly_map.max()
             norm_score = (pred_score - sys_min) / (sys_max - sys_min + 1e-6)
        
        score_percentage = float(np.clip(norm_score, 0, 1) * 100)
        threshold_percentage = float(threshold * 100)

        # 7. Visualization
        h, w = original.shape[:2]
        anomaly_map_up = cv2.resize(anomaly_map_norm, (w, h), interpolation=cv2.INTER_LINEAR)
        pred_mask_up = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        heatmap_u8 = (anomaly_map_up * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        heatmap_overlay = cv2.addWeighted(original, 0.6, colormap, 0.4, 0)

        seg_overlay = original.copy()
        mask255 = (pred_mask_up * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(seg_overlay, contours, -1, (0, 0, 255), 3)

        return {
            "status": "FAIL" if score_percentage > threshold_percentage else "PASS", 
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
