"""Small, fail-closed runtime for one local PatchCore model folder.

The folder is the deployment unit for the current project.  It contains a
human-readable ``model.json`` contract and the user supplies the large model
artifacts beside it.  There is deliberately no signing, release registry, or
automatic model discovery in this local runtime.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from training.preprocessing.runtime import create_backend, process_single_image


MANIFEST_FILENAME = "model.json"
SUPPORTED_BACKENDS = {"raw_letterbox", "fixed_crop", "rembg"}


class ModelFolderError(RuntimeError):
    """The selected local model folder is incomplete or inconsistent."""


class InspectionReviewError(RuntimeError):
    """An image cannot safely receive a model result."""

    def __init__(
        self,
        stage: str,
        code: str,
        detail: str,
        *,
        quality_flags: list[str] | None = None,
    ) -> None:
        super().__init__(detail)
        self.stage = stage
        self.code = code
        self.detail = detail
        self.quality_flags = list(quality_flags or [])


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelFolderError(f"Missing {label}: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelFolderError(f"Could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelFolderError(f"{label} must contain one JSON object: {path}")
    return value


def _direct_child(folder: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ModelFolderError(f"{label} must be a non-empty file name")
    relative = Path(value)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name != value:
        raise ModelFolderError(f"{label} must name a file directly inside the model folder")
    return folder / relative


def _positive_size(value: Any, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in value)
    ):
        raise ModelFolderError(f"{label} must be a two-item list of positive integers")
    return int(value[0]), int(value[1])


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ModelFolderError(f"{label} must be a positive integer")
    return value


def _artifact_identity(contract: Any, label: str) -> tuple[str, str, int]:
    if not isinstance(contract, dict):
        raise ModelFolderError(f"{label} must be an object with file, sha256, and size_bytes")
    filename = contract.get("file")
    digest = str(contract.get("sha256", "")).lower()
    try:
        digest_bytes = bytes.fromhex(digest)
    except ValueError as exc:
        raise ModelFolderError(f"{label}.sha256 must be a 64-character SHA-256") from exc
    if len(digest_bytes) != 32 or len(digest) != 64:
        raise ModelFolderError(f"{label}.sha256 must be a 64-character SHA-256")
    size = _positive_integer(contract.get("size_bytes"), f"{label}.size_bytes")
    return str(filename or ""), digest, size


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_artifact(path: Path, expected_sha256: str, expected_size: int, label: str) -> None:
    if not path.is_file():
        raise ModelFolderError(f"Missing {label}: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ModelFolderError(
            f"{label} size mismatch: expected {expected_size} bytes, found {actual_size}"
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ModelFolderError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, found {actual_sha256}"
        )


@dataclass(frozen=True)
class LocalModelManifest:
    """Validated, resolved contract for one local model directory."""

    folder: Path
    model_id: str
    family: str
    angle: str
    image_size: int
    original_width: int
    original_height: int
    checkpoint: Path
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    metadata: Path
    preprocessing_id: str
    preprocessing_config: dict[str, Any]
    preprocessing_weight: Path | None
    preprocessing_weight_sha256: str | None
    preprocessing_weight_size_bytes: int | None
    exploratory_threshold: float | None
    max_image_bytes: int

    @classmethod
    def load(
        cls, folder: str | Path, *, require_artifacts: bool = True
    ) -> "LocalModelManifest":
        selected = Path(folder).expanduser().resolve()
        if not selected.is_dir():
            raise ModelFolderError(f"Selected model folder does not exist: {selected}")

        document = _json_object(selected / MANIFEST_FILENAME, "model manifest")
        if document.get("schema_version") != "1.0":
            raise ModelFolderError("model.json schema_version must be '1.0'")

        model = document.get("model")
        input_contract = document.get("input")
        artifacts = document.get("artifacts")
        preprocessing = document.get("preprocessing")
        if not isinstance(model, dict) or not isinstance(artifacts, dict):
            raise ModelFolderError("model.json requires object fields 'model' and 'artifacts'")
        if not isinstance(input_contract, dict):
            raise ModelFolderError("model.json requires an object field 'input'")
        if not isinstance(preprocessing, dict):
            raise ModelFolderError("model.json requires an object field 'preprocessing'")

        model_id = str(model.get("id", ""))
        family = str(model.get("family", "")).casefold()
        angle = str(model.get("angle", ""))
        image_size = model.get("image_size")
        if model_id != selected.name:
            raise ModelFolderError(
                f"Model id {model_id!r} must match selected folder name {selected.name!r}"
            )
        if family != "patchcore":
            raise ModelFolderError("Only PatchCore model folders are currently supported")
        if angle != "G01":
            raise ModelFolderError("The current manufacturing runtime supports only angle G01")
        if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size <= 0:
            raise ModelFolderError("model.image_size must be a positive integer")

        original_width = _positive_integer(input_contract.get("width"), "input.width")
        original_height = _positive_integer(input_contract.get("height"), "input.height")

        checkpoint_file, checkpoint_sha256, checkpoint_size = _artifact_identity(
            artifacts.get("checkpoint"), "artifacts.checkpoint"
        )
        checkpoint = _direct_child(selected, checkpoint_file, "artifacts.checkpoint.file")
        metadata = _direct_child(selected, artifacts.get("metadata"), "artifacts.metadata")

        preprocessing_id = str(preprocessing.get("id", ""))
        config = preprocessing.get("config")
        if not preprocessing_id or not isinstance(config, dict):
            raise ModelFolderError("preprocessing requires a non-empty id and object config")
        config = dict(config)
        if config.get("preprocessing_id") != preprocessing_id:
            raise ModelFolderError("Preprocessing id and inline config preprocessing_id differ")
        backend = str(config.get("backend", ""))
        if backend not in SUPPORTED_BACKENDS:
            raise ModelFolderError(f"Unsupported preprocessing backend: {backend or '<missing>'}")
        _positive_size(config.get("output_size"), "preprocessing.config.output_size")

        weight = cls._resolve_preprocessing_weight(
            selected, preprocessing, backend, require_artifacts=require_artifacts
        )
        weight_sha256: str | None = None
        weight_size: int | None = None
        if weight is not None:
            _, weight_sha256, weight_size = _artifact_identity(
                preprocessing.get("weight"), "preprocessing.weight"
            )
            config["model_dir"] = str(weight.parent)
            config["model_filename"] = weight.name

        threshold_value = document.get("exploratory_threshold")
        threshold: float | None
        if threshold_value is None:
            threshold = None
        elif (
            not isinstance(threshold_value, (int, float))
            or isinstance(threshold_value, bool)
            or not math.isfinite(float(threshold_value))
        ):
            raise ModelFolderError("exploratory_threshold must be null or a finite number")
        else:
            threshold = float(threshold_value)

        max_bytes = document.get("max_image_bytes", 25 * 1024 * 1024)
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
            raise ModelFolderError("max_image_bytes must be a positive integer")

        manifest = cls(
            folder=selected,
            model_id=model_id,
            family=family,
            angle=angle,
            image_size=image_size,
            original_width=original_width,
            original_height=original_height,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_size_bytes=checkpoint_size,
            metadata=metadata,
            preprocessing_id=preprocessing_id,
            preprocessing_config=config,
            preprocessing_weight=weight,
            preprocessing_weight_sha256=weight_sha256,
            preprocessing_weight_size_bytes=weight_size,
            exploratory_threshold=threshold,
            max_image_bytes=max_bytes,
        )
        if require_artifacts:
            manifest.validate_artifacts()
        return manifest

    @staticmethod
    def _resolve_preprocessing_weight(
        folder: Path,
        preprocessing: dict[str, Any],
        backend: str,
        *,
        require_artifacts: bool,
    ) -> Path | None:
        weight = preprocessing.get("weight")
        if backend != "rembg":
            if weight not in (None, {}):
                raise ModelFolderError("Only rembg preprocessing may declare a weight")
            return None
        if not isinstance(weight, dict):
            raise ModelFolderError("rembg preprocessing requires a weight object")

        local = _direct_child(folder, weight.get("file"), "preprocessing.weight.file")
        if local.is_file() and local.stat().st_size > 0:
            return local

        shared_value = weight.get("shared_fallback")
        if not isinstance(shared_value, str) or not shared_value.strip():
            raise ModelFolderError(f"Missing rembg weight: {local}")
        shared = (folder / shared_value).resolve()
        models_root = folder.parent.resolve()
        try:
            shared.relative_to(models_root)
        except ValueError as exc:
            raise ModelFolderError(
                "preprocessing.weight.shared_fallback must stay inside the models directory"
            ) from exc
        if not require_artifacts and not shared.is_file():
            return local
        if not shared.is_file() or shared.stat().st_size <= 0:
            raise ModelFolderError(
                f"Missing rembg weight. Copy u2net.onnx to {local} or provide {shared}"
            )
        return shared

    def validate_artifacts(self) -> None:
        _verify_artifact(
            self.checkpoint,
            self.checkpoint_sha256,
            self.checkpoint_size_bytes,
            "PatchCore checkpoint",
        )
        if self.preprocessing_weight is not None:
            assert self.preprocessing_weight_sha256 is not None
            assert self.preprocessing_weight_size_bytes is not None
            _verify_artifact(
                self.preprocessing_weight,
                self.preprocessing_weight_sha256,
                self.preprocessing_weight_size_bytes,
                "preprocessing weight",
            )
        if not self.metadata.is_file() or self.metadata.stat().st_size <= 0:
            raise ModelFolderError(f"Missing or empty metadata: {self.metadata}")

        metadata = _json_object(self.metadata, "training metadata")
        model = metadata.get("model")
        dataset = metadata.get("dataset")
        if not isinstance(model, dict) or not isinstance(dataset, dict):
            raise ModelFolderError("Training metadata requires model and dataset objects")
        if str(model.get("class", "")).casefold() != "patchcore":
            raise ModelFolderError("Training metadata is not for PatchCore")
        expected = {
            "model.angle": (model.get("angle"), self.angle),
            "model.model_set": (model.get("model_set"), self.model_id),
            "model.image_size": (model.get("image_size"), self.image_size),
            "dataset.preprocessing_id": (
                dataset.get("preprocessing_id"),
                self.preprocessing_id,
            ),
        }
        mismatches = [
            f"{name}={actual!r} (expected {wanted!r})"
            for name, (actual, wanted) in expected.items()
            if actual != wanted
        ]
        if mismatches:
            raise ModelFolderError("Training metadata mismatch: " + "; ".join(mismatches))
        binding = metadata.get("checkpoint_artifact")
        if binding is not None:
            if not isinstance(binding, dict):
                raise ModelFolderError("Training metadata checkpoint_artifact must be an object")
            metadata_sha256 = str(binding.get("sha256", "")).lower()
            metadata_size = binding.get("size_bytes")
            if (
                metadata_sha256 != self.checkpoint_sha256
                or metadata_size != self.checkpoint_size_bytes
            ):
                raise ModelFolderError(
                    "Training metadata checkpoint_artifact differs from authoritative model.json"
                )


class RawPatchCoreEngine:
    """Load one checkpoint and expose its raw image score and anomaly map."""

    def __init__(self, checkpoint: Path) -> None:
        import anomalib
        import torch
        from anomalib.models import Patchcore
        from torchvision.transforms import v2

        if not hasattr(anomalib, "PrecisionType"):
            anomalib.PrecisionType = str
        self.torch = torch
        self.v2 = v2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Patchcore.load_from_checkpoint(
            checkpoint, map_location=self.device, weights_only=False
        ).to(self.device).eval()
        if not self.model.pre_processor or not self.model.pre_processor.transform:
            raise RuntimeError("PatchCore checkpoint lacks its training preprocessor")
        self.transform = v2.Compose(
            [v2.ToDtype(torch.float32, scale=True), self.model.pre_processor.transform]
        )

    @staticmethod
    def _value(data: Any, name: str) -> Any:
        if isinstance(data, dict):
            return data.get(name)
        return getattr(data, name, None)

    def predict(self, image: Image.Image) -> tuple[float, np.ndarray, float]:
        tensor = self.v2.functional.to_image(image.convert("RGB"))
        tensor = self.transform(tensor).unsqueeze(0).to(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.model(tensor)
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - started) * 1000
        score = self._value(output, "pred_score")
        anomaly_map = self._value(output, "anomaly_map")
        if score is None or anomaly_map is None:
            raise RuntimeError("PatchCore output lacks pred_score or anomaly_map")
        if isinstance(score, self.torch.Tensor):
            score = float(score.detach().reshape(-1)[0].cpu())
        if isinstance(anomaly_map, self.torch.Tensor):
            anomaly_map = anomaly_map.detach().cpu().numpy()
        anomaly_map = np.asarray(anomaly_map, dtype=np.float32).squeeze()
        if anomaly_map.ndim != 2:
            raise RuntimeError(f"Unexpected PatchCore anomaly-map shape: {anomaly_map.shape}")
        if not math.isfinite(float(score)) or not np.isfinite(anomaly_map).all():
            raise RuntimeError("PatchCore produced non-finite raw output")
        return float(score), anomaly_map, inference_ms

    def close(self) -> None:
        self.model = None
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()


def _encode_image(image: Image.Image, *, format: str = "JPEG") -> str:
    output = io.BytesIO()
    if format == "JPEG":
        image.convert("RGB").save(output, format=format, quality=90)
        mime = "image/jpeg"
    else:
        image.save(output, format=format)
        mime = "image/png"
    payload = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _display_overlay(anomaly_map: np.ndarray, model_input: Image.Image) -> str:
    low = float(anomaly_map.min())
    high = float(anomaly_map.max())
    if high - low <= 1e-12:
        normalized = np.zeros_like(anomaly_map, dtype=np.float32)
    else:
        normalized = (anomaly_map - low) / (high - low)
    rgb = np.asarray(model_input.convert("RGB"))
    normalized = cv2.resize(
        normalized, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    heatmap_bgr = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb, 0.6, heatmap_rgb, 0.4, 0)
    return _encode_image(Image.fromarray(overlay))


class LocalPatchCoreRuntime:
    """Reusable preprocessing and PatchCore inference for one selected folder."""

    def __init__(
        self,
        manifest: LocalModelManifest,
        *,
        engine_factory: Callable[[Path], Any] = RawPatchCoreEngine,
        preprocessor_factory: Callable[[dict[str, Any]], Any] = create_backend,
    ) -> None:
        self.manifest = manifest
        self.model_id = manifest.model_id
        self.angle = manifest.angle
        self.preprocessing_id = manifest.preprocessing_id
        self.mode = "shadow"
        try:
            self.preprocessor = preprocessor_factory(manifest.preprocessing_config)
            self.engine = engine_factory(manifest.checkpoint)
        except Exception as exc:
            raise ModelFolderError(f"Could not load selected model folder: {exc}") from exc
        self._gate = threading.BoundedSemaphore(1)

    def _decode(self, image_bytes: bytes, angle: str) -> Image.Image:
        if angle != self.angle:
            raise InspectionReviewError(
                "input", "camera_angle_mismatch", f"Selected model accepts {self.angle}, not {angle}"
            )
        if not image_bytes:
            raise InspectionReviewError("input", "empty_image", "Camera image is empty")
        if len(image_bytes) > self.manifest.max_image_bytes:
            raise InspectionReviewError("input", "image_too_large", "Camera image is too large")
        try:
            with Image.open(io.BytesIO(image_bytes)) as opened:
                opened.load()
                expected_size = (
                    self.manifest.original_width,
                    self.manifest.original_height,
                )
                if opened.size != expected_size:
                    raise InspectionReviewError(
                        "input",
                        "image_size_mismatch",
                        (
                            f"Expected original G01 image {expected_size[0]}x{expected_size[1]}, "
                            f"received {opened.width}x{opened.height}. Upload the original camera image."
                        ),
                    )
                return opened.convert("RGB").copy()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InspectionReviewError(
                "input", "image_decode_failed", "Could not decode camera image"
            ) from exc

    def predict(self, image_bytes: bytes, angle: str) -> dict[str, Any]:
        total_started = time.perf_counter()
        image = self._decode(image_bytes, angle)
        with self._gate:
            try:
                model_input, mask, metrics = process_single_image(
                    image,
                    self.manifest.preprocessing_config,
                    backend_instance=self.preprocessor,
                )
            except Exception as exc:
                raise InspectionReviewError(
                    "preprocessing", "preprocessing_failed", str(exc)
                ) from exc
            quality_flags = list(metrics.get("quality_flags", []))
            if metrics.get("status") != "ok":
                raise InspectionReviewError(
                    "preprocessing",
                    "preprocessing_qa_failed",
                    "Preprocessing quality checks failed",
                    quality_flags=quality_flags,
                )
            try:
                raw_score, anomaly_map, inference_ms = self.engine.predict(model_input)
            except Exception as exc:
                raise InspectionReviewError(
                    "inference", "patchcore_inference_failed", str(exc)
                ) from exc

        threshold = self.manifest.exploratory_threshold
        exploratory_decision = None
        if threshold is not None:
            exploratory_decision = (
                "EXPLORATORY_FAULT" if raw_score > threshold else "EXPLORATORY_NORMAL"
            )
        return {
            "status": "SHADOW",
            "system_status": "SHADOW",
            "decision": "UNDECIDED",
            "exploratory_decision": exploratory_decision,
            "model_id": self.model_id,
            "preprocessing_id": self.preprocessing_id,
            "angle": self.angle,
            "mode": self.mode,
            "raw_image_score": raw_score,
            "score": raw_score,
            "image_threshold": threshold,
            "threshold_rule": (
                "exploratory_fault_if_score_gt_threshold" if threshold is not None else None
            ),
            "anomaly_map_shape": list(anomaly_map.shape),
            "quality_flags": quality_flags,
            "preprocessing": metrics,
            "timings_ms": {
                "inference_ms": round(float(inference_ms), 3),
                "total_ms": round((time.perf_counter() - total_started) * 1000, 3),
            },
            "display_contract": "per_map_minmax_display_only_never_used_for_decision",
            "heatmap_image": _display_overlay(anomaly_map, model_input),
            "segmentation_image": _encode_image(mask, format="PNG") if mask is not None else None,
            "original_image": _encode_image(image),
            "model_input_image": _encode_image(model_input),
        }

    def review_result(
        self, error: InspectionReviewError, *, system_error: bool = False
    ) -> dict[str, Any]:
        status = "SYSTEM_ERROR" if system_error else "REVIEW"
        return {
            "status": status,
            "system_status": status,
            "decision": "UNDECIDED",
            "exploratory_decision": None,
            "model_id": self.model_id,
            "preprocessing_id": self.preprocessing_id,
            "angle": self.angle,
            "mode": self.mode,
            "raw_image_score": None,
            "score": None,
            "image_threshold": self.manifest.exploratory_threshold,
            "quality_flags": error.quality_flags,
            "error": {"stage": error.stage, "code": error.code, "detail": error.detail},
            "heatmap_image": None,
            "segmentation_image": None,
            "original_image": None,
            "model_input_image": None,
        }

    def close(self) -> None:
        close = getattr(self.engine, "close", None)
        if close:
            close()
