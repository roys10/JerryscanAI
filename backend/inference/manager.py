"""Select and own exactly one local multi-angle model-folder runtime."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Callable

from .local_model import (
    InferenceRuntimeError,
    InspectionInputError,
    LocalModelManifest,
    LocalPatchCoreRuntime,
)


class ModelNotReadyError(RuntimeError):
    """No validated local model is available for inspection."""


class ModelSelectionError(ValueError):
    """The caller requested a model other than the selected local model."""


class JerryScanModelManager:
    """Load only the folder named by the caller or ``JERRYSCAN_MODEL_FOLDER``."""

    def __init__(
        self,
        model_folder: str | Path | None = None,
        *,
        runtime_factory: Callable[[LocalModelManifest], Any] = LocalPatchCoreRuntime,
    ) -> None:
        configured = model_folder or os.getenv("JERRYSCAN_MODEL_FOLDER")
        self.model_folder = (
            Path(configured).expanduser().resolve() if configured else None
        )
        self.runtime_factory = runtime_factory
        self.runtime: Any | None = None
        self.startup_error: str | None = None
        self.configured_angles: tuple[str, ...] = ()
        self.unavailable_angles: dict[str, dict[str, str]] = {}
        self._condition = threading.Condition(threading.RLock())
        self._reload_lock = threading.Lock()
        self._inflight: dict[int, int] = {}

    @property
    def ready(self) -> bool:
        with self._condition:
            return self.runtime is not None

    def load_selected(self) -> None:
        """Validate and load the explicitly selected folder.

        The old runtime remains usable if a manual reload fails.  On initial
        startup, a missing or invalid artifact leaves the service not ready.
        """
        with self._reload_lock:
            if self.model_folder is None:
                message = "JERRYSCAN_MODEL_FOLDER is not set. Point it to one model directory."
                with self._condition:
                    self.startup_error = message
                raise RuntimeError(message)
            try:
                manifest = LocalModelManifest.load(self.model_folder)
                candidate = self.runtime_factory(manifest)
            except Exception as exc:
                with self._condition:
                    self.startup_error = f"{type(exc).__name__}: {exc}"
                    self.configured_angles = tuple(
                        getattr(exc, "configured_angles", ())
                    )
                    self.unavailable_angles = dict(
                        getattr(exc, "unavailable_angles", {})
                    )
                raise
            with self._condition:
                previous = self.runtime
                self.runtime = candidate
                self.startup_error = None
                self.configured_angles = tuple(
                    getattr(candidate, "configured_angles", manifest.configured_angles)
                )
                self.unavailable_angles = dict(
                    getattr(candidate, "unavailable_angles", {})
                )
                if previous is not None:
                    previous_key = id(previous)
                    while self._inflight.get(previous_key, 0) > 0:
                        self._condition.wait()
            if previous is not None:
                close = getattr(previous, "close", None)
                if close:
                    close()

    def get_model_names(self) -> list[str]:
        with self._condition:
            return [self.runtime.model_id] if self.runtime is not None else []

    def get_required_angles(self) -> list[str]:
        with self._condition:
            if self.runtime is None:
                raise ModelNotReadyError(self.startup_error or "No local model folder is loaded")
            return list(
                getattr(self.runtime, "available_angles", self.runtime.manifest.required_angles)
            )

    def get_configured_angles(self) -> list[str]:
        with self._condition:
            if self.runtime is not None:
                return list(
                    getattr(
                        self.runtime,
                        "configured_angles",
                        self.runtime.manifest.configured_angles,
                    )
                )
            return list(self.configured_angles)

    def get_unavailable_angles(self) -> dict[str, dict[str, str]]:
        with self._condition:
            if self.runtime is not None:
                return dict(getattr(self.runtime, "unavailable_angles", {}))
            return dict(self.unavailable_angles)

    def _not_ready_message(self) -> str:
        with self._condition:
            return self.startup_error or "No local model folder is loaded"

    def inspect(
        self, angle: str, image_bytes: bytes, *, requested_model: str | None = None
    ) -> dict[str, Any]:
        with self._condition:
            runtime = self.runtime
            if runtime is not None:
                runtime_key = id(runtime)
                self._inflight[runtime_key] = self._inflight.get(runtime_key, 0) + 1
        if runtime is None:
            raise ModelNotReadyError(self._not_ready_message())
        try:
            if requested_model and requested_model != runtime.model_id:
                raise ModelSelectionError(
                    f"Requested model {requested_model!r} is not the selected model "
                    f"{runtime.model_id!r}"
                )
            return runtime.predict(image_bytes, angle)
        except InspectionInputError as exc:
            return runtime.wrong_input_result(exc, angle)
        except (ModelSelectionError, InferenceRuntimeError):
            raise
        except Exception as exc:
            raise InferenceRuntimeError(
                f"Unexpected inference failure: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            with self._condition:
                self._inflight[runtime_key] -= 1
                if self._inflight[runtime_key] <= 0:
                    self._inflight.pop(runtime_key, None)
                self._condition.notify_all()

    def health(self) -> dict[str, Any]:
        with self._condition:
            runtime = self.runtime
            error = self.startup_error
        if runtime is None:
            return {
                "status": "not_ready",
                "ready_for_inference": False,
                "ready_for_decisions": False,
                "model_folder": str(self.model_folder) if self.model_folder else None,
                "error": error or "No local model folder is loaded",
                "configured_angles": list(self.configured_angles),
                "available_angles": [],
                "required_angles": [],
                "supported_angles": [],
                "unavailable_angles": dict(self.unavailable_angles),
                "coverage": "none",
            }
        configured_angles = list(runtime.configured_angles)
        available_angles = list(runtime.available_angles)
        unavailable_angles = dict(runtime.unavailable_angles)
        partial = bool(unavailable_angles)
        return {
            "status": "degraded" if partial else "ready",
            "ready_for_inference": True,
            "ready_for_decisions": not partial,
            "model_folder": str(runtime.manifest.folder),
            "model_id": runtime.model_id,
            "display_name": getattr(
                runtime, "model_display_name", runtime.model_id
            ),
            "preprocessing_id": runtime.preprocessing_id,
            "configured_angles": configured_angles,
            "available_angles": available_angles,
            # Backward-compatible names: batch input is required only for the
            # angles whose engines actually loaded.
            "required_angles": available_angles,
            "supported_angles": available_angles,
            "unavailable_angles": unavailable_angles,
            "coverage": "partial" if partial else "full",
            "inference_device": getattr(runtime, "inference_device", "unknown"),
            "inference_devices": getattr(runtime, "inference_devices", {}),
            "device_fallback_reason": getattr(
                runtime, "device_fallback_reason", None
            ),
            "decision_thresholds": {
                angle: {
                    "score": "raw_patchcore_image_score",
                    "value": artifact.decision_threshold,
                    "rule": "fail_if_score_greater_than_or_equal",
                    "provenance": artifact.threshold_provenance,
                }
                for angle, artifact in runtime.manifest.angles.items()
                if angle in runtime.available_angles
            },
            "quality_score": {
                "failure_boundary_percentage": 70.0,
                "rule": "fail_if_quality_less_than_or_equal",
                "meaning": "relative_operator_index_not_probability",
            },
        }
