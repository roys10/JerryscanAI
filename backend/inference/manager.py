"""Select and own exactly one local model-folder runtime."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Callable

from .local_model import (
    InspectionReviewError,
    LocalModelManifest,
    LocalPatchCoreRuntime,
)


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
                raise
            with self._condition:
                previous = self.runtime
                self.runtime = candidate
                self.startup_error = None
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

    def _not_ready_result(self, angle: str) -> dict[str, Any]:
        with self._condition:
            detail = self.startup_error or "No local model folder is loaded"
        return {
            "status": "REVIEW",
            "system_status": "REVIEW",
            "decision": "UNDECIDED",
            "exploratory_decision": None,
            "model_id": None,
            "preprocessing_id": None,
            "angle": angle,
            "mode": "shadow",
            "raw_image_score": None,
            "score": None,
            "image_threshold": None,
            "quality_flags": [],
            "error": {
                "stage": "startup",
                "code": "model_folder_not_ready",
                "detail": detail,
            },
            "heatmap_image": None,
            "segmentation_image": None,
            "original_image": None,
            "model_input_image": None,
        }

    def inspect(
        self, angle: str, image_bytes: bytes, *, requested_model: str | None = None
    ) -> dict[str, Any]:
        with self._condition:
            runtime = self.runtime
            if runtime is not None:
                runtime_key = id(runtime)
                self._inflight[runtime_key] = self._inflight.get(runtime_key, 0) + 1
        if runtime is None:
            return self._not_ready_result(angle)
        try:
            if requested_model and requested_model != runtime.model_id:
                return runtime.review_result(
                    InspectionReviewError(
                        "routing",
                        "inactive_model_requested",
                        f"Requested model {requested_model!r} is not the selected folder",
                    )
                )
            return runtime.predict(image_bytes, angle)
        except InspectionReviewError as exc:
            return runtime.review_result(exc)
        except Exception as exc:
            return runtime.review_result(
                InspectionReviewError(
                    "inference",
                    "unexpected_runtime_failure",
                    f"{type(exc).__name__}: {exc}",
                ),
                system_error=True,
            )
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
            }
        return {
            "status": "shadow_ready",
            "ready_for_inference": True,
            "ready_for_decisions": False,
            "model_folder": str(runtime.manifest.folder),
            "model_id": runtime.model_id,
            "preprocessing_id": runtime.preprocessing_id,
            "angle": runtime.angle,
            "mode": runtime.mode,
            "exploratory_threshold": runtime.manifest.exploratory_threshold,
        }
