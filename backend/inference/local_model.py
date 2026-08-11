"""Small, fail-closed runtime for one local multi-angle PatchCore model folder.

The folder is the deployment unit for the current project.  It contains a
human-readable ``model.json`` contract and the user supplies the large model
artifacts beside it.  There is deliberately no signing, release registry, or
automatic model discovery in this local runtime.
"""

from __future__ import annotations

import base64
import hashlib
import gc
import io
import json
import logging
import math
import os
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
LOGGER = logging.getLogger(__name__)


class ModelFolderError(RuntimeError):
    """The selected local model folder is incomplete or inconsistent."""

    def __init__(
        self,
        message: str,
        *,
        configured_angles: tuple[str, ...] = (),
        unavailable_angles: dict[str, dict[str, str]] | None = None,
    ) -> None:
        super().__init__(message)
        self.configured_angles = configured_angles
        self.unavailable_angles = dict(unavailable_angles or {})


def _angle_unavailable(stage: str, exc: BaseException) -> dict[str, str]:
    """Return a stable, JSON-safe explanation for one unavailable angle."""
    return {
        "stage": stage,
        "error_type": type(exc).__name__,
        "detail": str(exc),
    }


class InspectionInputError(RuntimeError):
    """The submitted image cannot safely be inspected."""

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


class InferenceRuntimeError(RuntimeError):
    """The model was loaded but failed while performing inference."""


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
class AngleModelArtifact:
    """One camera angle's checkpoint, metadata, and decision contract."""

    angle: str
    checkpoint: Path
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    metadata: Path
    decision_threshold: float
    threshold_provenance: str


def _decision_threshold(contract: Any, label: str) -> tuple[float, str]:
    if not isinstance(contract, dict):
        raise ModelFolderError(f"{label} must be an object")
    if contract.get("score") != "raw_patchcore_image_score":
        raise ModelFolderError(f"{label}.score must be 'raw_patchcore_image_score'")
    if contract.get("rule") != "fail_if_score_greater_than_or_equal":
        raise ModelFolderError(
            f"{label}.rule must be 'fail_if_score_greater_than_or_equal'"
        )
    value = contract.get("value")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ModelFolderError(f"{label}.value must be a positive finite number")
    provenance = contract.get("provenance", "unspecified")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ModelFolderError(f"{label}.provenance must be a non-empty string")
    return float(value), provenance.strip()


@dataclass(frozen=True)
class LocalModelManifest:
    """Validated, resolved contract for one local model-set directory."""

    folder: Path
    model_id: str
    display_name: str
    family: str
    image_size: int
    original_width: int
    original_height: int
    configured_angles: tuple[str, ...]
    angles: dict[str, AngleModelArtifact]
    unavailable_angles: dict[str, dict[str, str]]
    preprocessing_id: str
    preprocessing_config: dict[str, Any]
    preprocessing_weight: Path | None
    preprocessing_weight_sha256: str | None
    preprocessing_weight_size_bytes: int | None
    max_image_bytes: int

    @property
    def required_angles(self) -> tuple[str, ...]:
        return tuple(self.angles)

    # Compatibility properties for existing schema-1.0 single-angle folders.
    @property
    def primary_angle(self) -> AngleModelArtifact:
        return next(iter(self.angles.values()))

    @property
    def angle(self) -> str:
        return self.primary_angle.angle

    @property
    def checkpoint(self) -> Path:
        return self.primary_angle.checkpoint

    @property
    def checkpoint_sha256(self) -> str:
        return self.primary_angle.checkpoint_sha256

    @property
    def checkpoint_size_bytes(self) -> int:
        return self.primary_angle.checkpoint_size_bytes

    @property
    def metadata(self) -> Path:
        return self.primary_angle.metadata

    @property
    def decision_threshold(self) -> float:
        return self.primary_angle.decision_threshold

    @classmethod
    def load(
        cls, folder: str | Path, *, require_artifacts: bool = True
    ) -> "LocalModelManifest":
        selected = Path(folder).expanduser().resolve()
        if not selected.is_dir():
            raise ModelFolderError(f"Selected model folder does not exist: {selected}")

        document = _json_object(selected / MANIFEST_FILENAME, "model manifest")
        schema_version = document.get("schema_version")
        if schema_version not in {"1.0", "1.1"}:
            raise ModelFolderError("model.json schema_version must be '1.0' or '1.1'")

        model = document.get("model")
        input_contract = document.get("input")
        preprocessing = document.get("preprocessing")
        if not isinstance(model, dict):
            raise ModelFolderError("model.json requires an object field 'model'")
        if not isinstance(input_contract, dict):
            raise ModelFolderError("model.json requires an object field 'input'")
        if not isinstance(preprocessing, dict):
            raise ModelFolderError("model.json requires an object field 'preprocessing'")

        model_id = str(model.get("id", ""))
        display_name_value = model.get("display_name", model_id)
        if not isinstance(display_name_value, str) or not display_name_value.strip():
            raise ModelFolderError("model.display_name must be a non-empty string")
        display_name = display_name_value.strip()
        family = str(model.get("family", "")).casefold()
        image_size = model.get("image_size")
        if model_id != selected.name:
            raise ModelFolderError(
                f"Model id {model_id!r} must match selected folder name {selected.name!r}"
            )
        if family != "patchcore":
            raise ModelFolderError("Only PatchCore model folders are currently supported")
        if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size <= 0:
            raise ModelFolderError("model.image_size must be a positive integer")

        original_width = _positive_integer(input_contract.get("width"), "input.width")
        original_height = _positive_integer(input_contract.get("height"), "input.height")

        unavailable_angles: dict[str, dict[str, str]] = {}
        if schema_version == "1.0":
            artifacts = document.get("artifacts")
            angle = str(model.get("angle", ""))
            if not isinstance(artifacts, dict) or not angle:
                raise ModelFolderError(
                    "schema-1.0 model.json requires model.angle and object artifacts"
                )
            threshold, provenance = _decision_threshold(
                document.get("decision_threshold"), "decision_threshold"
            )
            checkpoint_file, checkpoint_sha256, checkpoint_size = _artifact_identity(
                artifacts.get("checkpoint"), "artifacts.checkpoint"
            )
            angles = {
                angle: AngleModelArtifact(
                    angle=angle,
                    checkpoint=_direct_child(
                        selected, checkpoint_file, "artifacts.checkpoint.file"
                    ),
                    checkpoint_sha256=checkpoint_sha256,
                    checkpoint_size_bytes=checkpoint_size,
                    metadata=_direct_child(
                        selected, artifacts.get("metadata"), "artifacts.metadata"
                    ),
                    decision_threshold=threshold,
                    threshold_provenance=provenance,
                )
            }
            configured_angles = (angle,)
        else:
            if model.get("angle") is not None or document.get("artifacts") is not None:
                raise ModelFolderError(
                    "schema-1.1 stores angle-specific artifacts under the angles object"
                )
            angle_contracts = document.get("angles")
            if not isinstance(angle_contracts, dict) or not angle_contracts:
                raise ModelFolderError("schema-1.1 model.json requires a non-empty angles object")
            configured_angles = tuple(angle_contracts)
            angles = {}
            for angle, angle_contract in angle_contracts.items():
                if (
                    not isinstance(angle, str)
                    or not angle.startswith("G")
                    or not angle[1:].isdigit()
                    or not isinstance(angle_contract, dict)
                ):
                    raise ModelFolderError(f"Invalid angle contract: {angle!r}")
                try:
                    checkpoint_file, checkpoint_sha256, checkpoint_size = _artifact_identity(
                        angle_contract.get("checkpoint"), f"angles.{angle}.checkpoint"
                    )
                    threshold, provenance = _decision_threshold(
                        angle_contract.get("decision_threshold"),
                        f"angles.{angle}.decision_threshold",
                    )
                    angles[angle] = AngleModelArtifact(
                        angle=angle,
                        checkpoint=_direct_child(
                            selected,
                            checkpoint_file,
                            f"angles.{angle}.checkpoint.file",
                        ),
                        checkpoint_sha256=checkpoint_sha256,
                        checkpoint_size_bytes=checkpoint_size,
                        metadata=_direct_child(
                            selected,
                            angle_contract.get("metadata"),
                            f"angles.{angle}.metadata",
                        ),
                        decision_threshold=threshold,
                        threshold_provenance=provenance,
                    )
                except ModelFolderError as exc:
                    unavailable_angles[angle] = _angle_unavailable(
                        "contract_validation", exc
                    )

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

        try:
            weight = cls._resolve_preprocessing_weight(
                selected, preprocessing, backend, require_artifacts=require_artifacts
            )
        except ModelFolderError as exc:
            raise ModelFolderError(
                str(exc),
                configured_angles=configured_angles,
                unavailable_angles={
                    angle: _angle_unavailable("shared_preprocessing_artifact", exc)
                    for angle in configured_angles
                },
            ) from exc
        weight_sha256: str | None = None
        weight_size: int | None = None
        if weight is not None:
            _, weight_sha256, weight_size = _artifact_identity(
                preprocessing.get("weight"), "preprocessing.weight"
            )
            config["model_dir"] = str(weight.parent)
            config["model_filename"] = weight.name

        max_bytes = document.get("max_image_bytes", 25 * 1024 * 1024)
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
            raise ModelFolderError("max_image_bytes must be a positive integer")

        manifest = cls(
            folder=selected,
            model_id=model_id,
            display_name=display_name,
            family=family,
            image_size=image_size,
            original_width=original_width,
            original_height=original_height,
            configured_angles=configured_angles,
            angles=angles,
            unavailable_angles=unavailable_angles,
            preprocessing_id=preprocessing_id,
            preprocessing_config=config,
            preprocessing_weight=weight,
            preprocessing_weight_sha256=weight_sha256,
            preprocessing_weight_size_bytes=weight_size,
            max_image_bytes=max_bytes,
        )
        if require_artifacts:
            if schema_version == "1.0":
                # Retain the strict behavior of existing single-angle bundles.
                manifest.validate_artifacts()
            else:
                manifest.validate_shared_artifacts()
                for angle, angle_artifact in tuple(manifest.angles.items()):
                    try:
                        manifest.validate_angle_artifacts(angle_artifact)
                    except ModelFolderError as exc:
                        manifest.unavailable_angles[angle] = _angle_unavailable(
                            "artifact_validation", exc
                        )
                        manifest.angles.pop(angle)
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
        self.validate_shared_artifacts()
        for angle_artifact in self.angles.values():
            self.validate_angle_artifacts(angle_artifact)

    def validate_shared_artifacts(self) -> None:
        """Validate artifacts shared by every configured camera angle."""
        if self.preprocessing_weight is not None:
            assert self.preprocessing_weight_sha256 is not None
            assert self.preprocessing_weight_size_bytes is not None
            _verify_artifact(
                self.preprocessing_weight,
                self.preprocessing_weight_sha256,
                self.preprocessing_weight_size_bytes,
                "preprocessing weight",
            )

    def validate_angle_artifacts(self, angle_artifact: AngleModelArtifact) -> None:
        """Validate one angle without making sibling angles unusable."""
        _verify_artifact(
            angle_artifact.checkpoint,
            angle_artifact.checkpoint_sha256,
            angle_artifact.checkpoint_size_bytes,
            f"{angle_artifact.angle} PatchCore checkpoint",
        )
        self._validate_angle_metadata(angle_artifact)

    def _validate_angle_metadata(self, angle_artifact: AngleModelArtifact) -> None:
        if not angle_artifact.metadata.is_file() or angle_artifact.metadata.stat().st_size <= 0:
            raise ModelFolderError(f"Missing or empty metadata: {angle_artifact.metadata}")

        metadata = _json_object(angle_artifact.metadata, "training metadata")
        model = metadata.get("model")
        dataset = metadata.get("dataset")
        if not isinstance(model, dict) or not isinstance(dataset, dict):
            raise ModelFolderError("Training metadata requires model and dataset objects")
        if str(model.get("class", "")).casefold() != "patchcore":
            raise ModelFolderError("Training metadata is not for PatchCore")
        expected = {
            "model.angle": (model.get("angle"), angle_artifact.angle),
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
                metadata_sha256 != angle_artifact.checkpoint_sha256
                or metadata_size != angle_artifact.checkpoint_size_bytes
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
        self.model = None
        self.device_fallback_reason: str | None = None
        self._load_on_preferred_device(checkpoint, Patchcore)

    @staticmethod
    def _normalize_transform_interpolation(transform: Any, interpolation_mode: Any) -> int:
        """Repair string interpolation values saved by newer Torchvision builds.

        Some G02-G04 checkpoints were produced with a Torch/Torchvision build
        that serialized ``Resize.interpolation`` as ``"bilinear"``.  Older
        supported runtimes require the equivalent ``InterpolationMode`` enum.
        Converting the representation preserves the trained resize operation;
        it does not alter model weights, inputs, scores, or thresholds.
        """
        modes = {
            str(mode.value).casefold(): mode
            for mode in interpolation_mode
        }
        modes.update({mode.name.casefold(): mode for mode in interpolation_mode})
        converted = 0
        pending = [transform]
        visited: set[int] = set()
        while pending:
            module = pending.pop()
            identity = id(module)
            if identity in visited:
                continue
            visited.add(identity)
            value = getattr(module, "interpolation", None)
            if isinstance(value, str):
                replacement = modes.get(value.casefold())
                if replacement is not None:
                    module.interpolation = replacement
                    converted += 1

            # Torchvision v2 Compose keeps its children in a plain list rather
            # than registering them as torch modules, while other containers
            # use ``_modules``. Traverse both representations.
            children = getattr(module, "transforms", ())
            if isinstance(children, (list, tuple)):
                pending.extend(children)
            registered = getattr(module, "_modules", None)
            if isinstance(registered, dict):
                pending.extend(registered.values())
        return converted

    def _initialize_on_device(self, checkpoint: Path, patchcore_class: Any, device: Any) -> None:
        """Load and warm up the checkpoint on one concrete Torch device."""
        self.device = device
        self.model = None
        if device.type == "cpu":
            available_threads = max(1, os.cpu_count() or 1)
            target_threads = min(12, available_threads)
            if self.torch.get_num_threads() < target_threads:
                self.torch.set_num_threads(target_threads)
        self.model = patchcore_class.load_from_checkpoint(
            checkpoint, map_location=device, weights_only=False
        ).to(device).eval()
        if not self.model.pre_processor or not self.model.pre_processor.transform:
            raise RuntimeError("PatchCore checkpoint lacks its training preprocessor")
        from torchvision.transforms import InterpolationMode

        converted = self._normalize_transform_interpolation(
            self.model.pre_processor.transform,
            InterpolationMode,
        )
        if converted:
            LOGGER.info(
                "Normalized %d checkpoint interpolation value(s) for Torchvision compatibility",
                converted,
            )
        self.pixel_display_bounds = self._pixel_display_bounds()
        self.transform = self.v2.Compose(
            [
                self.v2.ToDtype(self.torch.float32, scale=True),
                self.model.pre_processor.transform,
            ]
        )
        self.warmup_ms = self._warm_up()

    def _load_on_preferred_device(self, checkpoint: Path, patchcore_class: Any) -> None:
        """Prefer CUDA, falling back to CPU if CUDA initialization is unusable."""
        cpu = self.torch.device("cpu")
        if not self.torch.cuda.is_available():
            self._initialize_on_device(checkpoint, patchcore_class, cpu)
            return

        cuda = self.torch.device("cuda")
        try:
            self._initialize_on_device(checkpoint, patchcore_class, cuda)
            return
        except Exception as cuda_error:
            self.model = None
            self.torch.cuda.empty_cache()
            self.device_fallback_reason = (
                f"{type(cuda_error).__name__}: {cuda_error}"
            )
            LOGGER.warning(
                "CUDA PatchCore initialization failed; falling back to CPU: %s",
                self.device_fallback_reason,
            )

        try:
            self._initialize_on_device(checkpoint, patchcore_class, cpu)
        except Exception as cpu_error:
            raise RuntimeError(
                "CUDA PatchCore initialization failed "
                f"({self.device_fallback_reason}); CPU fallback also failed "
                f"({type(cpu_error).__name__}: {cpu_error})"
            ) from cpu_error

    def _warm_up(self) -> float:
        """Pay one-time Torch/kernel initialization cost during API startup."""
        probe = self.v2.functional.to_image(Image.new("RGB", (256, 256), "black"))
        tensor = self.transform(probe).unsqueeze(0).to(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            self.model.model(tensor)
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()
        return (time.perf_counter() - started) * 1000

    def _pixel_display_bounds(self) -> tuple[float, float] | None:
        """Read the training-wide pixel scale saved by Anomalib."""
        owners = (getattr(self.model, "post_processor", None), self.model)
        for owner in owners:
            if owner is None or not (
                hasattr(owner, "pixel_min") and hasattr(owner, "pixel_max")
            ):
                continue
            low = getattr(owner, "pixel_min")
            high = getattr(owner, "pixel_max")
            if hasattr(low, "value"):
                low = low.value
            if hasattr(high, "value"):
                high = high.value
            if isinstance(low, self.torch.Tensor):
                low = low.detach().reshape(-1)[0].cpu().item()
            if isinstance(high, self.torch.Tensor):
                high = high.detach().reshape(-1)[0].cpu().item()
            low = float(low)
            high = float(high)
            if math.isfinite(low) and math.isfinite(high) and high > low:
                return low, high
        return None

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


_MAIN_COMPATIBILITY_THRESHOLD = 0.5
QUALITY_FAILURE_BOUNDARY_PERCENTAGE = 70.0


def _normalize_anomaly_map(
    anomaly_map: np.ndarray,
    display_bounds: tuple[float, float] | None,
) -> tuple[np.ndarray, str]:
    """Normalize a pixel map for display without changing the raw model output."""
    if display_bounds is not None:
        low, high = display_bounds
        normalized = np.clip((anomaly_map - low) / (high - low), 0, 1)
        contract = "checkpoint_pixel_minmax_display_only_never_used_for_decision"
    else:
        low = float(anomaly_map.min())
        high = float(anomaly_map.max())
        if high - low <= 1e-12:
            normalized = np.zeros_like(anomaly_map, dtype=np.float32)
        else:
            normalized = (anomaly_map - low) / (high - low)
        contract = "per_map_minmax_fallback_display_only_never_used_for_decision"
    normalized = cv2.GaussianBlur(
        normalized.astype(np.float32), (0, 0), sigmaX=4, sigmaY=4
    )
    return normalized, contract


def _defect_localization_mask(normalized: np.ndarray) -> tuple[np.ndarray, float | None]:
    """Return a focused, display-only mask for the strongest anomalous pixels.

    The checkpoint's 0.5 pixel cutoff first defines the candidate anomaly
    region. Two nested Otsu separations then isolate the high-anomaly core: the
    first removes the broad outer halo and the second separates the remaining
    core from its softer edge. Both cutoffs are derived from the current map's
    score distribution rather than from a defect-example-specific constant.

    This mask is used only to draw the operator overlay. It never changes the
    raw image score, the configured decision threshold, or PASS/FAIL.
    """
    initial_candidate = normalized > _MAIN_COMPATIBILITY_THRESHOLD
    if not np.any(initial_candidate):
        return np.zeros_like(initial_candidate, dtype=np.uint8), None

    focused_threshold = _MAIN_COMPATIBILITY_THRESHOLD
    for _ in range(2):
        candidate = normalized > focused_threshold
        candidate_values = np.clip(normalized[candidate] * 255, 0, 255).astype(np.uint8)
        if candidate_values.size < 2 or np.unique(candidate_values).size < 2:
            break
        otsu_threshold, _ = cv2.threshold(
            candidate_values.reshape(-1, 1),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        # Move to the next 8-bit bin because OpenCV assigns values equal to
        # its Otsu threshold to the lower population.
        refined_threshold = max(
            focused_threshold,
            min(1.0, (float(otsu_threshold) + 1.0) / 255.0),
        )
        if refined_threshold <= focused_threshold:
            break
        focused_threshold = refined_threshold

    focused = (normalized > focused_threshold).astype(np.uint8)
    # Quantization can make Otsu's threshold equal the map maximum. Always
    # preserve the actual peak so the display locator cannot erase its target.
    peak = np.unravel_index(int(np.argmax(normalized)), normalized.shape)
    focused[peak] = 1
    return focused, focused_threshold


def _display_artifacts(
    anomaly_map: np.ndarray,
    model_input: Image.Image,
    display_bounds: tuple[float, float] | None,
    *,
    show_defect_contours: bool,
) -> tuple[str, str, str]:
    """Reproduce main's heatmap and contour view as separate UI artifacts."""
    normalized, contract = _normalize_anomaly_map(anomaly_map, display_bounds)
    rgb = np.asarray(model_input.convert("RGB"))
    normalized_up = cv2.resize(
        normalized, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    heatmap_bgr = cv2.applyColorMap(
        (normalized_up * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb, 0.6, heatmap_rgb, 0.4, 0)

    # origin/main called this a segmentation overlay.  Keep the actual
    # preprocessing mask in ``segmentation_image`` and expose defect contours
    # independently so the UI can show both without changing their meaning.
    # PASS results suppress contours so the operator view cannot contradict
    # the configured image-level decision.
    anomaly_mask, localization_threshold = _defect_localization_mask(normalized)
    if not show_defect_contours:
        anomaly_mask.fill(0)
    anomaly_mask_up = cv2.resize(
        anomaly_mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    contours, _ = cv2.findContours(
        anomaly_mask_up * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    defect_overlay = rgb.copy()
    cv2.drawContours(defect_overlay, contours, -1, (255, 0, 0), 3)
    return (
        _encode_image(Image.fromarray(overlay)),
        _encode_image(Image.fromarray(defect_overlay)),
        (
            contract
            + "; defect_localization=display_only_nested_otsu_above_checkpoint_pixel_cutoff"
            + (
                f"@{localization_threshold:.4f}"
                if localization_threshold is not None
                else "@none"
            )
        ),
    )


def _quality_score_percentage(raw_score: float, raw_threshold: float) -> float:
    """Map the raw anomaly scale to the operator's monotonic quality index."""
    quality = 100 - (
        (100 - QUALITY_FAILURE_BOUNDARY_PERCENTAGE)
        * raw_score
        / raw_threshold
    )
    return float(np.clip(quality, 0, 100))


class LocalPatchCoreRuntime:
    """Reusable shared preprocessing plus one PatchCore engine per angle."""

    def __init__(
        self,
        manifest: LocalModelManifest,
        *,
        engine_factory: Callable[[Path], Any] = RawPatchCoreEngine,
        preprocessor_factory: Callable[[dict[str, Any]], Any] = create_backend,
    ) -> None:
        self.manifest = manifest
        self.model_id = manifest.model_id
        self.model_display_name = manifest.display_name
        self.configured_angles = manifest.configured_angles
        self.preprocessing_id = manifest.preprocessing_id
        self.engines: dict[str, Any] = {}
        self.unavailable_angles = dict(manifest.unavailable_angles)
        try:
            self.preprocessor = preprocessor_factory(manifest.preprocessing_config)
        except Exception as exc:
            raise ModelFolderError(
                f"Could not load shared preprocessing: {exc}",
                configured_angles=self.configured_angles,
                unavailable_angles={
                    angle: _angle_unavailable("shared_preprocessing", exc)
                    for angle in self.configured_angles
                },
            ) from exc

        for angle, artifact in manifest.angles.items():
            try:
                self.engines[angle] = engine_factory(artifact.checkpoint)
            except Exception as exc:
                self.unavailable_angles[angle] = _angle_unavailable(
                    "engine_initialization", exc
                )
                LOGGER.exception("Could not initialize %s PatchCore engine", angle)
                # Failed constructors can leave large Torch objects waiting for
                # collection. Release them before attempting the next angle so
                # one OOM does not automatically prevent every later angle.
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:  # pragma: no cover - cleanup must be best effort
                    LOGGER.debug("Could not clear accelerator cache", exc_info=True)

        self.available_angles = tuple(self.engines)
        self.required_angles = self.available_angles
        if not self.available_angles:
            close = getattr(self.preprocessor, "close", None)
            if close:
                close()
            details = "; ".join(
                f"{angle}: {reason['detail']}"
                for angle, reason in self.unavailable_angles.items()
            )
            raise ModelFolderError(
                "No configured angle could be loaded"
                + (f" ({details})" if details else ""),
                configured_angles=self.configured_angles,
                unavailable_angles=self.unavailable_angles,
            )
        self.angle = self.available_angles[0]  # schema-1.0 compatibility
        self.engine = self.engines[self.angle]  # schema-1.0 compatibility
        self.inference_devices = {
            angle: str(getattr(engine, "device", "unknown"))
            for angle, engine in self.engines.items()
        }
        self.device_fallback_reasons = {
            angle: getattr(engine, "device_fallback_reason", None)
            for angle, engine in self.engines.items()
        }
        unique_devices = set(self.inference_devices.values())
        self.inference_device = (
            next(iter(unique_devices)) if len(unique_devices) == 1 else "mixed"
        )
        reasons = [reason for reason in self.device_fallback_reasons.values() if reason]
        self.device_fallback_reason = "; ".join(reasons) if reasons else None
        # The shared preprocessor is protected because some supported backends
        # keep mutable per-image state.  Once preprocessing finishes, distinct
        # angle engines may infer concurrently.  Each individual checkpoint is
        # still single-flight, so two requests cannot enter the same engine at
        # once.  This pipelines U2Net work and uses adequately provisioned
        # CPU/GPU hardware without changing any model output.
        self._preprocessing_gate = threading.BoundedSemaphore(1)
        self._angle_gates = {
            angle: threading.BoundedSemaphore(1) for angle in self.required_angles
        }

    def _decode(self, image_bytes: bytes, angle: str) -> Image.Image:
        if angle not in self.engines:
            reason = self.unavailable_angles.get(angle)
            if reason is not None:
                raise InspectionInputError(
                    "model",
                    "camera_angle_unavailable",
                    f"{angle} is configured but unavailable: {reason['detail']}",
                )
            raise InspectionInputError(
                "input",
                "camera_angle_mismatch",
                f"Selected model accepts {', '.join(self.available_angles)}, not {angle}",
            )
        if not image_bytes:
            raise InspectionInputError("input", "empty_image", "Camera image is empty")
        if len(image_bytes) > self.manifest.max_image_bytes:
            raise InspectionInputError("input", "image_too_large", "Camera image is too large")
        try:
            with Image.open(io.BytesIO(image_bytes)) as opened:
                opened.load()
                expected_size = (
                    self.manifest.original_width,
                    self.manifest.original_height,
                )
                if opened.size != expected_size:
                    raise InspectionInputError(
                        "input",
                        "image_size_mismatch",
                        (
                            f"Expected original {angle} image {expected_size[0]}x{expected_size[1]}, "
                            f"received {opened.width}x{opened.height}. Upload the original camera image."
                        ),
                    )
                return opened.convert("RGB").copy()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InspectionInputError(
                "input", "image_decode_failed", "Could not decode camera image"
            ) from exc

    def predict(self, image_bytes: bytes, angle: str) -> dict[str, Any]:
        total_started = time.perf_counter()
        image = self._decode(image_bytes, angle)
        angle_artifact = self.manifest.angles[angle]
        engine = self.engines[angle]
        with self._preprocessing_gate:
            try:
                model_input, mask, metrics = process_single_image(
                    image,
                    self.manifest.preprocessing_config,
                    backend_instance=self.preprocessor,
                )
            except Exception as exc:
                raise InspectionInputError(
                    "preprocessing", "preprocessing_failed", str(exc)
                ) from exc
            quality_flags = list(metrics.get("quality_flags", []))
            if metrics.get("status") != "ok":
                raise InspectionInputError(
                    "preprocessing",
                    "preprocessing_qa_failed",
                    "Preprocessing quality checks failed",
                    quality_flags=quality_flags,
                )
        with self._angle_gates[angle]:
            try:
                raw_score, anomaly_map, inference_ms = engine.predict(model_input)
            except Exception as exc:
                raise InferenceRuntimeError(
                    f"PatchCore inference failed: {type(exc).__name__}: {exc}"
                ) from exc

        display_bounds = getattr(engine, "pixel_display_bounds", None)
        threshold = angle_artifact.decision_threshold
        decision = "FAIL" if raw_score >= threshold else "PASS"
        quality_score_percentage = _quality_score_percentage(raw_score, threshold)
        heatmap_image, defect_overlay_image, display_contract = _display_artifacts(
            anomaly_map,
            model_input,
            display_bounds,
            show_defect_contours=decision == "FAIL",
        )
        return {
            "status": decision,
            "decision": decision,
            "model_id": self.model_id,
            "model_display_name": self.model_display_name,
            "preprocessing_id": self.preprocessing_id,
            "angle": angle,
            "raw_image_score": raw_score,
            "score": raw_score,
            "quality_score_percentage": quality_score_percentage,
            "image_threshold": threshold,
            "quality_failure_boundary_percentage": QUALITY_FAILURE_BOUNDARY_PERCENTAGE,
            "threshold_score": "raw_patchcore_image_score",
            "threshold_rule": "fail_if_score_greater_than_or_equal",
            "decision_contract": "configured_raw_patchcore_image_score",
            "quality_score_contract": "relative_quality_zero_raw_is_100_threshold_is_70",
            "anomaly_map_shape": list(anomaly_map.shape),
            "quality_flags": quality_flags,
            "preprocessing": metrics,
            "timings_ms": {
                "inference_ms": round(float(inference_ms), 3),
                "total_ms": round((time.perf_counter() - total_started) * 1000, 3),
            },
            "display_contract": display_contract,
            "heatmap_image": heatmap_image,
            "defect_overlay_image": defect_overlay_image,
            "segmentation_image": _encode_image(mask, format="PNG") if mask is not None else None,
            "original_image": _encode_image(image),
            "model_input_image": _encode_image(model_input),
        }

    def wrong_input_result(
        self, error: InspectionInputError, angle: str | None = None
    ) -> dict[str, Any]:
        selected_angle = angle if angle in self.manifest.angles else self.angle
        return {
            "status": "WRONG_INPUT",
            "decision": None,
            "model_id": self.model_id,
            "model_display_name": self.model_display_name,
            "preprocessing_id": self.preprocessing_id,
            "angle": selected_angle,
            "raw_image_score": None,
            "score": None,
            "quality_score_percentage": None,
            "image_threshold": self.manifest.angles[selected_angle].decision_threshold,
            "quality_failure_boundary_percentage": QUALITY_FAILURE_BOUNDARY_PERCENTAGE,
            "threshold_score": "raw_patchcore_image_score",
            "threshold_rule": "fail_if_score_greater_than_or_equal",
            "quality_flags": error.quality_flags,
            "error": {"stage": error.stage, "code": error.code, "detail": error.detail},
            "heatmap_image": None,
            "defect_overlay_image": None,
            "segmentation_image": None,
            "original_image": None,
            "model_input_image": None,
        }

    def close(self) -> None:
        for engine in self.engines.values():
            close = getattr(engine, "close", None)
            if close:
                close()
        close = getattr(self.preprocessor, "close", None)
        if close:
            close()
