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
    "source": "checkpoint_pre_processor.transform",
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
        if not self.model.pre_processor or not self.model.pre_processor.transform:
            raise ValueError("PatchCore checkpoint lacks its training preprocessor")
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                self.model.pre_processor.transform,
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
        tensor = self.transform(tensor).unsqueeze(0).to(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            # Call the inner torch model. The outer Anomalib module would apply
            # its pre_processor and post_processor again, clipping raw outputs.
            output = self.model.model(tensor)
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - started) * 1000
        score = self._value(output, "pred_score")
        anomaly_map = self._value(output, "anomaly_map")
        if score is None or anomaly_map is None:
            raise ValueError("PatchCore output lacks pred_score or anomaly_map")
        if isinstance(score, self.torch.Tensor):
            score = float(score.detach().reshape(-1)[0].cpu())
        if not np.isfinite(float(score)):
            raise ValueError("PatchCore produced a non-finite raw image score")
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


def save_anomaly_visualization(
    raw_map: np.ndarray,
    model_input: Path,
    heatmap_destination: Path,
    overlay_destination: Path,
) -> dict[str, Any]:
    """Save display-only map and aligned overlay without changing raw values."""
    raw_map = np.asarray(raw_map, dtype=np.float32)
    finite = np.isfinite(raw_map)
    warnings: list[str] = []
    if not finite.any():
        low = high = None
        normalized = np.zeros(raw_map.shape, dtype=np.float32)
        warnings.append("anomaly_map_has_no_finite_values")
    else:
        low = float(raw_map[finite].min())
        high = float(raw_map[finite].max())
        display_map = np.where(finite, raw_map, low)
        if not finite.all():
            warnings.append("anomaly_map_nonfinite_values_replaced_for_display")
        if high - low <= 1e-12:
            normalized = np.zeros(raw_map.shape, dtype=np.float32)
            warnings.append("anomaly_map_is_constant")
        else:
            normalized = (display_map - low) / (high - low)

    original = cv2.imread(str(model_input), cv2.IMREAD_COLOR)
    if original is None:
        raise OSError(f"Could not read model input for overlay: {model_input}")
    normalized = cv2.resize(
        normalized, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    heatmap = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    heatmap_destination.parent.mkdir(parents=True, exist_ok=True)
    overlay_destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(heatmap_destination), heatmap):
        raise OSError(f"Could not write display heatmap: {heatmap_destination}")
    if not cv2.imwrite(str(overlay_destination), overlay):
        raise OSError(f"Could not write display overlay: {overlay_destination}")
    return {
        "normalization": "per_map_minmax_display_only",
        "aligned_to": "actual_model_input",
        "raw_finite_min": low,
        "raw_finite_max": high,
        "warnings": warnings,
    }
