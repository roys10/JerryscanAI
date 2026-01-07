import os
import cv2
import torch
import numpy as np
from anomalib.models import Padim

ckpt_path = r"results\Padim\jerryscan_singleangle\latest\weights\lightning\model.ckpt"
image_path = r"cli_prediction\G01-251224-094208-433.bmp"#"..\JerryscanAI_dataset_singleangle\test\good\G01-251224-163802-928.bmp"
out_dir = r"predictions\example1"
os.makedirs(out_dir, exist_ok=True)

# ---- load original image ----
orig_bgr = cv2.imread(image_path)
if orig_bgr is None:
    raise FileNotFoundError(image_path)

h, w = orig_bgr.shape[:2]

# ---- preprocess (resnet18 pretrained) ----
IMAGE_SIZE = 256
rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
norm = (resized - mean) / std

x = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float()

# ---- load model ----
model = Padim.load_from_checkpoint(ckpt_path).float().eval()

# ---- inference ----
with torch.no_grad():
    output = model(x)

pred_label = bool(output.pred_label.reshape(-1)[0].item())
pred_score = float(output.pred_score.reshape(-1)[0].item())

print("Prediction:", "BAD (anomalous)" if pred_label else "GOOD (normal)")
print("Score:", pred_score)

# ---- get maps ----
anomaly_map = output.anomaly_map[0, 0].detach().cpu().numpy()          # [256, 256]
pred_mask = output.pred_mask[0, 0].detach().cpu().numpy().astype(np.uint8)  # [256, 256] 0/1

# upscale to original size for visualization
anomaly_map_up = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)
pred_mask_up = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

# ---- panel 2: heatmap overlay ----
heat = cv2.normalize(anomaly_map_up, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)  # BGR
overlay_heat = cv2.addWeighted(orig_bgr, 0.6, heat, 0.4, 0)

# ---- panel 3: mask outline (red) ----
mask255 = (pred_mask_up * 255).astype(np.uint8)
contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

overlay_mask = orig_bgr.copy()
cv2.drawContours(overlay_mask, contours, -1, (0, 0, 255), 3)  # red in BGR

# ---- add titles ----
def title(img, text):
    out = img.copy()
    cv2.putText(out, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    return out

p1 = title(orig_bgr, "Image")
p2 = title(overlay_heat, "Image + Anomaly Map")
p3 = title(overlay_mask, "Image + Pred Mask")

# ---- combine into one wide image ----
combined = np.concatenate([p1, p2, p3], axis=1)

# ---- save ----
out_path = os.path.join(out_dir, "prediction_triptych.png")
cv2.imwrite(out_path, combined)

print("Saved:", out_path)
