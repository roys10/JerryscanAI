
import torch
import numpy as np
import cv2
import base64
from anomalib.models import Padim

class JerryScanPadimModel:
    def __init__(self, ckpt_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading Padim model from {ckpt_path}...")
        
        # Load model with correct precision
        self.model = Padim.load_from_checkpoint(ckpt_path).float().to(self.device).eval()
        
        # Preprocessing constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.image_size = (256, 256)

    def predict(self, image_bytes: bytes) -> dict:
        # 1. Preprocess
        nparr = np.frombuffer(image_bytes, np.uint8)
        original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if original is None: raise ValueError("Could not decode image")
        
        h, w = original.shape[:2]
        img_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, self.image_size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        norm = (resized - self.mean) / self.std
        tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        # 2. Inference
        with torch.no_grad():
            output = self.model(tensor)

        # 3. Extract & Normalize Anomaly Map
        raw_map = output.anomaly_map[0, 0].detach().cpu().numpy()
        pred_score = float(output.pred_score.reshape(-1)[0].item())
        pred_label = bool(output.pred_label.reshape(-1)[0].item())

        # Resize to original
        anomaly_map = cv2.resize(raw_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalize using model stats (Critical for Padim)
        if hasattr(self.model, 'normalization_metrics') or hasattr(self.model, 'min_max'):
            metrics = getattr(self.model, 'normalization_metrics', self.model.min_max)
            min_val = metrics.min.cpu().numpy()
            max_val = metrics.max.cpu().numpy()
            anomaly_map = (anomaly_map - min_val) / (max_val - min_val + 1e-6)
        
        anomaly_map = np.clip(anomaly_map, 0, 1)

        # --- Gaussian Blur (Standard Anomalib Step) ---
        # Anomalib applies blur (sigma=4) to Padim maps to smooth noise before thresholding
        anomaly_map = cv2.GaussianBlur(anomaly_map, (0, 0), sigmaX=4, sigmaY=4)

        # 4. Thresholding (Mask Generation)
        threshold = 0.5
        if hasattr(self.model, 'pixel_threshold'):
            threshold = self.model.pixel_threshold.value.item()
        elif hasattr(self.model, 'image_threshold'):
            threshold = self.model.image_threshold.value.item()
            
        mask = (anomaly_map > threshold).astype(np.uint8)

        # 5. Visualization
        # Heatmap Overlay
        heatmap_u8 = (anomaly_map * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        heatmap_overlay = cv2.addWeighted(original, 0.6, colormap, 0.4, 0)

        # Segmentation Overlay (Red Contour)
        seg_overlay = original.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(seg_overlay, contours, -1, (0, 0, 255), 2)

        return {
            "status": "FAIL" if pred_label else "PASS",
            "score": pred_score,
            "heatmap_image": self._encode(heatmap_overlay),
            "segmentation_image": self._encode(seg_overlay),
            "original_image": self._encode(original)
        }

    def _encode(self, img):
        _, buffer = cv2.imencode('.jpg', img)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
