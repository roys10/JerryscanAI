from __future__ import annotations

import json
import csv
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.datasets.create_dataset_manifest import sha256_file
from training.preprocessing.runtime import canonical_json_hash, resolve_live_config

from .settings import LabSettings


SAFE_ID = re.compile(r"[^a-zA-Z0-9_.-]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: str) -> str:
    result = SAFE_ID.sub("-", value.strip()).strip("-.")
    if not result:
        raise ValueError("Model name must contain a letter or number")
    return result


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


class ModelRegistry:
    """Persistent PatchCore bundle registry; checkpoints are never guessed."""

    def __init__(self, settings: LabSettings) -> None:
        self.settings = settings
        settings.ensure_writable_directories()
        if not settings.registry_file.exists():
            _atomic_json(settings.registry_file, {"schema_version": "1.0", "models": []})

    def _read(self) -> dict[str, Any]:
        return json.loads(self.settings.registry_file.read_text(encoding="utf-8"))

    def list(self) -> list[dict[str, Any]]:
        return sorted(self._read()["models"], key=lambda model: model["display_name"])

    def get(self, model_id: str) -> dict[str, Any]:
        for model in self.list():
            if model["id"] == model_id:
                return model
        raise KeyError(f"Unknown model: {model_id}")

    def _save(self, model: dict[str, Any]) -> dict[str, Any]:
        data = self._read()
        data["models"] = [item for item in data["models"] if item["id"] != model["id"]]
        data["models"].append(model)
        _atomic_json(self.settings.registry_file, data)
        return model

    def import_checkpoint(
        self,
        checkpoint: Path,
        *,
        display_name: str | None = None,
        metadata_path: Path | None = None,
        preprocessing_config_path: Path | None = None,
        derivative_root: Path | None = None,
        image_threshold: float | None = None,
        angle: str | None = None,
        manifest_sha256: str | None = None,
        existing_model_id: str | None = None,
        created_at_utc: str | None = None,
        calibration: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.is_file() or checkpoint.suffix.lower() != ".ckpt":
            raise ValueError(f"PatchCore checkpoint does not exist: {checkpoint}")
        inferred_metadata = checkpoint.with_name(f"{checkpoint.stem}.metadata.json")
        metadata_path = (metadata_path or inferred_metadata).expanduser().resolve()
        metadata: dict[str, Any] = {}
        issues: list[str] = []
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            issues.append("training_metadata_required")

        family = str(metadata.get("model", {}).get("class", "")).lower()
        if family and family != "patchcore":
            raise ValueError("Model Lab currently supports PatchCore checkpoints only")
        preprocessing_id = str(
            metadata.get("dataset", {}).get("preprocessing_id", "")
        )
        if preprocessing_config_path is None and preprocessing_id:
            candidate = self.settings.preprocessing_configs_dir / f"{preprocessing_id}.json"
            preprocessing_config_path = candidate if candidate.is_file() else None
        config: dict[str, Any] = {}
        if preprocessing_config_path is not None:
            preprocessing_config_path = preprocessing_config_path.expanduser().resolve()
            if preprocessing_config_path.is_file():
                config = json.loads(preprocessing_config_path.read_text(encoding="utf-8"))
                config_id = str(config.get("preprocessing_id", ""))
                if preprocessing_id and preprocessing_id != config_id:
                    raise ValueError("Training metadata and preprocessing config IDs differ")
                preprocessing_id = config_id
            else:
                issues.append("preprocessing_config_required")
        else:
            issues.append("preprocessing_config_required")
        if not preprocessing_id:
            issues.append("preprocessing_id_required")

        manifest_hash = str(
            manifest_sha256
            if manifest_sha256 is not None
            else metadata.get("dataset", {}).get("manifest_sha256", "")
        )
        resolved_angle = str(
            angle if angle is not None else metadata.get("model", {}).get("angle", "")
        )
        if not manifest_hash:
            issues.append("manifest_sha256_required")
        if not resolved_angle:
            issues.append("camera_angle_required")
        if image_threshold is not None and calibration is None:
            calibration = {
                "method": "manual_unverified",
                "source_manifest_sha256": manifest_hash or None,
                "source_split": None,
                "sample_ids_sha256": None,
                "target_fpr": None,
                "recorded_at_utc": _utc_now(),
            }
        calibration_issue = image_threshold is None
        if calibration_issue:
            issues.append("image_threshold_missing_metrics_will_be_limited")
        elif calibration and calibration.get("method") == "manual_unverified":
            issues.append("image_threshold_manual_unverified")
        checkpoint_hash = sha256_file(checkpoint)
        model_id = existing_model_id or _safe_id(
            display_name or checkpoint.parent.name or checkpoint.stem
        )
        existing = next((item for item in self.list() if item["id"] == model_id), None)
        if existing and existing.get("checkpoint_sha256") != checkpoint_hash:
            if existing_model_id:
                raise ValueError("Cannot replace a registry ID with another checkpoint")
            model_id = f"{model_id}-{checkpoint_hash[:8]}"
            collision = next((item for item in self.list() if item["id"] == model_id), None)
            if collision and collision.get("checkpoint_sha256") != checkpoint_hash:
                raise ValueError(f"Model registry ID collision: {model_id}")
        preprocessing_model_sha256 = None
        live_backend = None
        if config:
            try:
                live_config = resolve_live_config(config, self.settings.preprocessing_configs_dir)
                live_backend = live_config.get("backend")
                model_filename = live_config.get("model_filename")
                model_dir = Path(str(live_config.get("model_dir", "")))
                if model_filename:
                    if not model_dir.is_absolute():
                        model_dir = (Path.cwd() / model_dir).resolve()
                    model_path = model_dir / str(model_filename)
                    if model_path.is_file():
                        preprocessing_model_sha256 = sha256_file(model_path)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
        if preprocessing_model_sha256 is None and derivative_root:
            derivative_manifest = derivative_root.expanduser().resolve() / "derivative_manifest.csv"
            if derivative_manifest.is_file():
                with derivative_manifest.open(newline="", encoding="utf-8") as stream:
                    model_hashes = {
                        row.get("model_sha256", "")
                        for row in csv.DictReader(stream)
                        if row.get("model_sha256")
                    }
                if len(model_hashes) == 1:
                    preprocessing_model_sha256 = next(iter(model_hashes))
        if preprocessing_model_sha256 is None and config:
            try:
                expected_hash = str(
                    resolve_live_config(config, self.settings.preprocessing_configs_dir).get(
                        "expected_model_sha256", ""
                    )
                )
                preprocessing_model_sha256 = expected_hash or None
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
        if live_backend in {"rembg", "sam2"} and not preprocessing_model_sha256:
            issues.append("preprocessing_model_sha256_required")
        hard_issues = [
            issue for issue in issues
            if not issue.startswith("image_threshold_")
        ]
        status = "incomplete" if hard_issues else (
            "ready_uncalibrated" if calibration_issue else "ready"
        )
        model = {
            "schema_version": "1.0",
            "id": model_id,
            "display_name": display_name or model_id,
            "family": "patchcore",
            "status": status,
            "issues": issues,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "metadata_path": str(metadata_path) if metadata_path.is_file() else None,
            "metadata": metadata,
            "angle": resolved_angle or None,
            "manifest_sha256": manifest_hash or None,
            "preprocessing_id": preprocessing_id or None,
            "preprocessing_config_path": (
                str(preprocessing_config_path) if preprocessing_config_path and preprocessing_config_path.is_file() else None
            ),
            "preprocessing_config_sha256": canonical_json_hash(config) if config else None,
            "preprocessing_model_sha256": preprocessing_model_sha256,
            "derivative_root": str(derivative_root.expanduser().resolve()) if derivative_root else None,
            "image_threshold": image_threshold,
            "calibration": calibration,
            "created_at_utc": created_at_utc or _utc_now(),
        }
        return self._save(model)

    def update_contract(self, model_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        current = self.get(model_id)
        allowed = {
            "display_name",
            "angle",
            "manifest_sha256",
            "preprocessing_config_path",
            "derivative_root",
            "image_threshold",
            "calibration",
        }
        unknown = set(updates) - allowed
        if unknown:
            raise ValueError(f"Unsupported model fields: {sorted(unknown)}")
        checkpoint = Path(current["checkpoint_path"])
        metadata = Path(current["metadata_path"]) if current.get("metadata_path") else None
        return self.import_checkpoint(
            checkpoint,
            display_name=str(updates.get("display_name", current["display_name"])),
            metadata_path=metadata,
            preprocessing_config_path=Path(
                updates.get("preprocessing_config_path")
                or current.get("preprocessing_config_path")
                or "missing"
            ),
            derivative_root=(
                Path(updates.get("derivative_root") or current["derivative_root"])
                if updates.get("derivative_root") or current.get("derivative_root")
                else None
            ),
            image_threshold=(
                updates["image_threshold"]
                if "image_threshold" in updates
                else current.get("image_threshold")
            ),
            angle=(updates["angle"] if "angle" in updates else current.get("angle")),
            manifest_sha256=(
                updates["manifest_sha256"]
                if "manifest_sha256" in updates
                else current.get("manifest_sha256")
            ),
            existing_model_id=current["id"],
            created_at_utc=current.get("created_at_utc"),
            calibration=(
                updates["calibration"] if "calibration" in updates else (
                    current.get("calibration")
                    if "image_threshold" not in updates
                    or updates.get("image_threshold") == current.get("image_threshold")
                    else None
                )
            ),
        )

    def discover(self, root: Path) -> list[dict[str, Any]]:
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Models directory does not exist: {root}")
        imported = []
        for checkpoint in sorted(root.rglob("*.ckpt")):
            imported.append(self.import_checkpoint(checkpoint))
        return imported
