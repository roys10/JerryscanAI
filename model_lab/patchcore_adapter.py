from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from training.datasets.create_dataset_manifest import sha256_file


PATCHCORE_TRANSFORM_CONTRACT = {
    "resize": "square",
    "interpolation": "bilinear",
    "normalization": "imagenet",
}


@dataclass
class PatchCoreOutput:
    raw_image_score: float
    raw_anomaly_map: np.ndarray
    image_threshold: float | None
    prediction: str | None
    inference_ms: float
    pixel_display_scale: str
    transform_contract: dict[str, str]


class PatchCoreAdapter:
    family = "patchcore"

    def __init__(self, model_contract: dict[str, Any]) -> None:
        if model_contract.get("family") != self.family:
            raise ValueError("PatchCoreAdapter only accepts PatchCore contracts")
        import anomalib
        import torch
        from anomalib.models import Patchcore
        from torchvision.transforms import v2

        if not hasattr(anomalib, "PrecisionType"):
            anomalib.PrecisionType = str
        self.contract = model_contract
        self.torch = torch
        self.v2 = v2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = Path(model_contract["checkpoint_path"])
        if sha256_file(checkpoint) != model_contract["checkpoint_sha256"]:
            raise ValueError("Registered PatchCore checkpoint hash does not match")
        self.model = Patchcore.load_from_checkpoint(
            checkpoint, map_location=self.device, weights_only=False
        ).to(self.device).eval()
        size = int(model_contract.get("metadata", {}).get("model", {}).get("image_size", 256))
        self.transform = v2.Compose(
            [
                v2.Resize(
                    (size, size),
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @staticmethod
    def _value(data: Any, name: str) -> Any:
        if isinstance(data, dict):
            return data.get(name)
        return getattr(data, name, None)

    def predict(self, image_path: Path) -> PatchCoreOutput:
        image = Image.open(image_path).convert("RGB")
        tensor = self.v2.functional.to_image(image)
        tensor = self.v2.functional.to_dtype(
            tensor, self.torch.float32, scale=True
        )
        tensor = self.transform(tensor).unsqueeze(0).to(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model(tensor)
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - started) * 1000
        score = self._value(output, "pred_score")
        anomaly_map = self._value(output, "anomaly_map")
        if score is None or anomaly_map is None:
            raise ValueError("PatchCore output lacks pred_score or anomaly_map")
        if isinstance(score, self.torch.Tensor):
            score = float(score.detach().reshape(-1)[0].cpu())
        if isinstance(anomaly_map, self.torch.Tensor):
            anomaly_map = anomaly_map.detach().cpu().numpy()
        anomaly_map = np.asarray(anomaly_map).squeeze()
        if anomaly_map.ndim != 2:
            raise ValueError(f"Unexpected anomaly-map shape: {anomaly_map.shape}")
        threshold = self.contract.get("image_threshold")
        prediction = None if threshold is None else (
            "fault" if float(score) > float(threshold) else "normal"
        )
        return PatchCoreOutput(
            raw_image_score=float(score),
            raw_anomaly_map=anomaly_map.astype(np.float32),
            image_threshold=float(threshold) if threshold is not None else None,
            prediction=prediction,
            inference_ms=inference_ms,
            pixel_display_scale="per_map_display_only",
            transform_contract=dict(PATCHCORE_TRANSFORM_CONTRACT),
        )


def save_anomaly_visualization(raw_map: np.ndarray, destination: Path) -> None:
    """Save a review image; per-map scaling is explicitly not a metric score."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    low, high = float(raw_map.min()), float(raw_map.max())
    normalized = (raw_map - low) / (high - low + 1e-12)
    heatmap = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if not cv2.imwrite(str(destination), heatmap):
        raise OSError(f"Could not write anomaly map: {destination}")
