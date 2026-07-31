from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image

from training.datasets.create_dataset_manifest import sha256_file
from training.preprocessing.runtime import (
    canonical_json_hash,
    create_backend,
    process_single_image,
    resolve_live_config,
    save_png_atomic,
)

from .settings import LabSettings


class PreprocessingResolver:
    """Provide a model's declared input from verified cache or original pixels."""

    def __init__(self, model: dict[str, Any], settings: LabSettings) -> None:
        self.model = model
        self.settings = settings
        config_path = Path(model["preprocessing_config_path"])
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.config_hash = canonical_json_hash(self.config)
        if self.config_hash != model["preprocessing_config_sha256"]:
            raise ValueError("Registered preprocessing config changed on disk")
        self.live_config = resolve_live_config(
            self.config, settings.preprocessing_configs_dir
        )
        self.backend = None
        self._derivative_rows: dict[str, dict[str, str]] | None = None

    def _load_external_manifest(self) -> dict[str, dict[str, str]]:
        if self._derivative_rows is not None:
            return self._derivative_rows
        root_value = self.model.get("derivative_root")
        if not root_value:
            self._derivative_rows = {}
            return self._derivative_rows
        manifest = Path(root_value) / "derivative_manifest.csv"
        if not manifest.is_file():
            self._derivative_rows = {}
            return self._derivative_rows
        with manifest.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self._derivative_rows = {row["parent_sample_id"]: row for row in rows}
        return self._derivative_rows

    @staticmethod
    def _source_bytes(source: Path, expected_hash: str) -> bytes:
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError("Original image hash does not match frozen manifest")
        return data

    def _verified_external(
        self, sample: dict[str, Any], manifest_hash: str
    ) -> tuple[Path, Path | None, dict[str, Any]] | None:
        row = self._load_external_manifest().get(sample["sample_id"])
        root_value = self.model.get("derivative_root")
        if not row or not root_value:
            return None
        expected_model_hash = self.model.get("preprocessing_model_sha256")
        if self.live_config.get("backend") in {"rembg", "sam2"} and not expected_model_hash:
            return None
        expected = {
            "status": "ok",
            "parent_sample_id": sample["sample_id"],
            "split": sample["split"],
            "label": sample["label"],
            "source_sha256": sample["source_sha256"],
            "parent_manifest_sha256": manifest_hash,
            "preprocessing_id": self.model["preprocessing_id"],
            "config_sha256": self.model["preprocessing_config_sha256"],
            "model_sha256": expected_model_hash or "",
        }
        if any(row.get(key, "") != str(value) for key, value in expected.items()):
            return None
        root = Path(root_value)
        output = root / row["output_relpath"]
        if not output.is_file() or sha256_file(output) != row.get("output_sha256"):
            return None
        mask = root / row["mask_relpath"] if row.get("mask_relpath") else None
        if mask and (not mask.is_file() or sha256_file(mask) != row.get("mask_sha256")):
            return None
        return output, mask, {
            "source": "verified_derivative",
            "processing_ms": float(row.get("processing_ms") or 0),
            "quality_flags": [flag for flag in row.get("quality_flags", "").split(";") if flag],
            "model_sha256": row.get("model_sha256", ""),
        }

    def _generated_cache(
        self, sample: dict[str, Any], manifest_hash: str
    ) -> tuple[Path, Path | None, dict[str, Any]] | None:
        cache = (
            self.settings.cache_dir
            / manifest_hash
            / self.model["preprocessing_config_sha256"]
            / sample["sample_id"]
        )
        metadata_path = cache / "provenance.json"
        output = cache / "input.png"
        if not metadata_path.is_file() or not output.is_file():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "source_sha256": sample["source_sha256"],
            "manifest_sha256": manifest_hash,
            "config_sha256": self.model["preprocessing_config_sha256"],
            "model_sha256": self.model.get("preprocessing_model_sha256") or "",
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            return None
        if sha256_file(output) != metadata.get("output_sha256"):
            return None
        mask = cache / "mask.png"
        if mask.is_file() and sha256_file(mask) != metadata.get("mask_sha256"):
            return None
        return output, mask if mask.is_file() else None, {
            **metadata,
            "source": "generated_cache",
            "processing_ms": 0.0,
        }

    def resolve(
        self,
        *,
        source: Path,
        sample: dict[str, Any],
        manifest_hash: str,
        force_live: bool,
    ) -> tuple[Path, Path | None, dict[str, Any]]:
        data = self._source_bytes(source, sample["source_sha256"])
        if not force_live:
            cached = self._verified_external(sample, manifest_hash)
            if cached is None:
                cached = self._generated_cache(sample, manifest_hash)
            if cached is not None:
                return cached
        image = Image.open(io.BytesIO(data)).convert("RGB")
        if self.backend is None:
            self.backend = create_backend(self.live_config)
            actual_model_hash = self.backend.model_sha256
            registered = self.model.get("preprocessing_model_sha256")
            if registered and actual_model_hash != registered:
                raise ValueError("Preprocessing model artifact hash changed")
        output_image, mask_image, provenance = process_single_image(
            image,
            self.config,
            config_dir=self.settings.preprocessing_configs_dir,
            backend_instance=self.backend,
        )
        cache = (
            self.settings.cache_dir
            / manifest_hash
            / self.model["preprocessing_config_sha256"]
            / sample["sample_id"]
        )
        output = cache / "input.png"
        mask = cache / "mask.png"
        save_png_atomic(output_image, output)
        if mask_image is not None:
            save_png_atomic(mask_image, mask)
        metadata = {
            **provenance,
            "source_sha256": sample["source_sha256"],
            "manifest_sha256": manifest_hash,
            "config_sha256": self.model["preprocessing_config_sha256"],
            "model_sha256": provenance.get("model_sha256", ""),
            "output_sha256": sha256_file(output),
            "mask_sha256": sha256_file(mask) if mask.is_file() else "",
        }
        (cache / "provenance.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        return output, mask if mask.is_file() else None, {
            **metadata,
            "source": "executed_live",
        }

    def process_untracked(
        self, image: Image.Image, output_dir: Path
    ) -> tuple[Path, Path | None, dict[str, Any]]:
        """Always preprocess an uploaded original; never use benchmark cache."""
        if self.backend is None:
            self.backend = create_backend(self.live_config)
        output_image, mask_image, provenance = process_single_image(
            image,
            self.config,
            config_dir=self.settings.preprocessing_configs_dir,
            backend_instance=self.backend,
        )
        output = output_dir / "model_input.png"
        mask = output_dir / "mask.png"
        save_png_atomic(output_image, output)
        if mask_image is not None:
            save_png_atomic(mask_image, mask)
        return output, mask if mask.is_file() else None, {
            **provenance,
            "source": "executed_live",
        }
