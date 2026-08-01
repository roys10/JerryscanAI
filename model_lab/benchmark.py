from __future__ import annotations

import json
import hashlib
import shutil
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from training.datasets.create_dataset_manifest import sha256_file, sha256_manifest
from training.datasets.materialize_dataset_split import load_manifest_rows
from training.preprocessing.runtime import canonical_json_hash

from .datasets import (
    safe_source_path,
    scan_exploratory_folder,
    select_exploratory_samples,
    select_samples,
)
from .metrics import calculate_metrics
from .patchcore_adapter import PatchCoreAdapter, save_anomaly_visualization
from .preprocessing import PreprocessingResolver
from .registry import ModelRegistry
from .settings import LabSettings
from .storage import ComparisonStore, atomic_json


def paired_model_summaries(
    rows: list[dict[str, Any]], model_ids: list[str], expected_sample_count: int
) -> tuple[dict[str, Any], int]:
    successful_by_model = {
        model_id: {
            row["sample_id"]
            for row in rows
            if row["model_id"] == model_id and row["status"] == "completed"
        }
        for model_id in model_ids
    }
    paired_ids = set.intersection(*successful_by_model.values())
    summaries = {
        model_id: {
            **calculate_metrics(
                [
                    row for row in rows
                    if row["model_id"] == model_id and row["sample_id"] in paired_ids
                ]
            ),
            "paired_sample_count": len(paired_ids),
            "excluded_unpaired_count": expected_sample_count - len(paired_ids),
            "model_error_count": sum(
                row["model_id"] == model_id and row["status"] == "error"
                for row in rows
            ),
        }
        for model_id in model_ids
    }
    return summaries, len(paired_ids)


class BenchmarkEngine:
    def __init__(
        self, settings: LabSettings, registry: ModelRegistry, store: ComparisonStore
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.store = store

    def prepare(self, request: dict[str, Any]) -> str:
        model_ids = list(dict.fromkeys(request["model_ids"]))
        if not 1 <= len(model_ids) <= 4:
            raise ValueError("Choose between one and four distinct PatchCore models")
        mode = str(request.get("dataset_mode", "exploratory_folder"))
        if (
            mode == "official_manifest"
            and request.get("split", "val") == "test"
            and request.get("locked_test_confirmation") != "RUN LOCKED TEST"
        ):
            raise ValueError("Locked test requires the exact confirmation: RUN LOCKED TEST")
        models = [self.registry.get(model_id) for model_id in model_ids]
        for model in models:
            if model["family"] != "patchcore":
                raise ValueError("Only PatchCore is supported")
            if model["status"] == "incomplete":
                raise ValueError(f"Model {model['id']} is incomplete: {model['issues']}")
        model_angles = {model["angle"] for model in models}
        if len(model_angles) != 1:
            raise ValueError("Selected models must use the same camera angle")
        selected_angle = next(iter(model_angles))
        snapshots = json.loads(json.dumps(models))
        for snapshot in snapshots:
            snapshot["contract_sha256"] = canonical_json_hash(snapshot)
        training_hashes = sorted(
            {model["manifest_sha256"] for model in models if model.get("manifest_sha256")}
        )
        warnings = []
        if len(training_hashes) > 1:
            warnings.append("Selected models were trained from different manifests")

        source_root = Path(request["source_root"]).expanduser().resolve()
        seed = int(request.get("seed", 42))
        requested_count = request.get("image_count")
        if requested_count is not None:
            requested_count = int(requested_count)
        split = "exploratory"
        manifest: Path | None = None
        is_locked_test = False
        if mode == "exploratory_folder":
            label_mode = str(request.get("label_mode", "unlabeled"))
            population, manifest_hash = scan_exploratory_folder(
                source_root, camera_angle=selected_angle, label_mode=label_mode
            )
            samples = select_exploratory_samples(
                population, count=requested_count, seed=seed
            )
            evaluation_snapshot = {
                "schema_version": "1.0",
                "mode": mode,
                "manifest_sha256": manifest_hash,
                "source_root": str(source_root),
                "label_mode": label_mode,
                "population": population,
                "selected_sample_ids": [sample["sample_id"] for sample in samples],
                "selection_seed": seed,
                "selection_policy": (
                    "all_sorted" if requested_count is None
                    else "deterministic_random_without_replacement_then_sorted"
                ),
            }
        elif mode == "official_manifest":
            manifest_value = request.get("manifest")
            if not manifest_value:
                raise ValueError("Official benchmark mode requires a frozen manifest")
            manifest = Path(manifest_value).expanduser().resolve()
            split = str(request.get("split", "val"))
            if split not in {"train", "val", "test"}:
                raise ValueError("Split must be train, val, or test")
            is_locked_test = split == "test"
            if is_locked_test and request.get("locked_test_confirmation") != "RUN LOCKED TEST":
                raise ValueError("Locked test requires the exact confirmation: RUN LOCKED TEST")
            manifest_rows = load_manifest_rows(manifest)
            split_count = sum(row.split == split for row in manifest_rows)
            count = split_count if requested_count is None else requested_count
            if is_locked_test and count != split_count:
                raise ValueError(f"Locked test must evaluate the full split ({split_count} images)")
            manifest_hash = sha256_manifest(manifest)
            samples = select_samples(manifest, split=split, count=count, seed=seed)
            angles = {sample["camera_angle"] for sample in samples}
            if len(angles) != 1:
                raise ValueError("A comparison sample set must contain exactly one camera angle")
            if angles != {selected_angle}:
                raise ValueError("Official sample angle must match every selected model")
            label_mode = "manifest_labels"
            evaluation_snapshot = {
                "schema_version": "1.0",
                "mode": mode,
                "manifest": str(manifest),
                "manifest_sha256": manifest_hash,
                "split": split,
                "selected_samples": samples,
                "selection_seed": seed,
                "selection_policy": (
                    "all_sorted" if count == split_count
                    else "deterministic_random_without_replacement_then_sorted"
                ),
            }
        else:
            raise ValueError("Dataset mode must be exploratory_folder or official_manifest")
        config = {
            "schema_version": "1.0",
            "name": request.get("name") or f"PatchCore comparison ({len(samples)} images)",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_ids": model_ids,
            "source_root": str(source_root),
            "dataset_mode": mode,
            "manifest": str(manifest) if manifest else None,
            "manifest_sha256": manifest_hash,
            "split": split,
            "label_mode": label_mode,
            "image_count": len(samples),
            "seed": seed,
            "selection_policy": evaluation_snapshot["selection_policy"],
            "force_live_preprocessing": bool(request.get("force_live_preprocessing", False)),
            "locked_test_confirmation_recorded": is_locked_test,
            "training_manifest_sha256s": training_hashes,
            "warnings": warnings,
            "model_contracts": snapshots,
        }
        comparison_id = self.store.create(config, samples, evaluation_snapshot)
        if is_locked_test:
            lock_path = self.settings.workspace / "locked_test_record.json"
            sample_hash = hashlib.sha256(
                "\n".join(sample["sample_id"] for sample in samples).encode()
            ).hexdigest()
            lock_record = {
                    "schema_version": "1.0",
                    "comparison_id": comparison_id,
                    "manifest_sha256": manifest_hash,
                    "full_test_sample_ids_sha256": sample_hash,
                    "model_contract_sha256": {
                        model["id"]: model["contract_sha256"] for model in snapshots
                    },
                    "created_at_utc": config["created_at_utc"],
                    "reset_policy": "administrative filesystem action outside Model Lab UI",
                }
            try:
                with lock_path.open("x", encoding="utf-8") as stream:
                    json.dump(lock_record, stream, indent=2)
                    stream.write("\n")
            except FileExistsError as exc:
                shutil.rmtree(self.store.path(comparison_id))
                raise ValueError(
                    "Locked test was already evaluated; administrative reset is required"
                ) from exc
        return comparison_id

    def run(self, comparison_id: str) -> None:
        config = self.store.read_json(comparison_id, "comparison.json")
        samples = self.store.read_json(comparison_id, "samples.json")
        contracts = {model["id"]: model for model in config["model_contracts"]}
        existing = self.store.latest_results(comparison_id)
        completed_keys = {
            (row["model_id"], row["sample_id"])
            for row in existing
            if row.get("status") == "completed"
        }
        total = len(config["model_ids"]) * len(samples)
        self.store.write_status(
            comparison_id, "running", completed=len(completed_keys), total=total
        )
        try:
            for model_id in config["model_ids"]:
                model = contracts[model_id]
                self._verify_contract(model)
                resolver = PreprocessingResolver(model, self.settings)
                adapter = PatchCoreAdapter(model)
                for sample in samples:
                    key = (model_id, sample["sample_id"])
                    if key in completed_keys:
                        continue
                    row = self._evaluate_one(
                        comparison_id, config, model, resolver, adapter, sample
                    )
                    self.store.append_result(comparison_id, row)
                    completed_keys.add(key)
                    self.store.write_status(
                        comparison_id,
                        "running",
                        completed=len(completed_keys),
                        total=total,
                        current_model_id=model_id,
                    )
                del adapter
            rows = self.store.latest_results(comparison_id)
            summaries, paired_count = paired_model_summaries(
                rows, config["model_ids"], len(samples)
            )
            atomic_json(self.store.path(comparison_id) / "summary.json", summaries)
            final_state = "completed" if paired_count == len(samples) else "incomplete"
            self.store.write_status(
                comparison_id,
                final_state,
                completed=len(completed_keys),
                total=total,
                paired_sample_count=paired_count,
                excluded_unpaired_count=len(samples) - paired_count,
            )
        except Exception as exc:
            (self.store.path(comparison_id) / "failure.log").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            self.store.write_status(
                comparison_id,
                "failed",
                completed=len(completed_keys),
                total=total,
                error=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _verify_contract(model: dict[str, Any]) -> None:
        expected_contract_hash = model.get("contract_sha256")
        contract = {key: value for key, value in model.items() if key != "contract_sha256"}
        if canonical_json_hash(contract) != expected_contract_hash:
            raise ValueError(f"Model contract snapshot is corrupt: {model['id']}")
        checkpoint = Path(model["checkpoint_path"])
        if not checkpoint.is_file() or sha256_file(checkpoint) != model["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed after comparison creation: {model['id']}")
        config_path = Path(model["preprocessing_config_path"])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if canonical_json_hash(config) != model["preprocessing_config_sha256"]:
            raise ValueError(f"Preprocessing config changed after comparison creation: {model['id']}")

    def _evaluate_one(
        self,
        comparison_id: str,
        config: dict[str, Any],
        model: dict[str, Any],
        resolver: PreprocessingResolver,
        adapter: PatchCoreAdapter,
        sample: dict[str, Any],
    ) -> dict[str, Any]:
        source = safe_source_path(Path(config["source_root"]), sample["source_relpath"])
        base = {
            "schema_version": "1.0",
            "comparison_id": comparison_id,
            "model_id": model["id"],
            "sample_id": sample["sample_id"],
            "split": sample["split"],
            "label": sample["label"],
            "source_sha256": sample["source_sha256"],
            "original_path": str(source),
            "status": "error",
            "error": None,
        }
        try:
            input_path, mask_path, provenance = resolver.resolve(
                source=source,
                sample=sample,
                manifest_hash=config["manifest_sha256"],
                force_live=config["force_live_preprocessing"],
            )
            output = adapter.predict(input_path)
            asset_root = self.store.path(comparison_id) / "assets" / model["id"]
            asset_root.mkdir(parents=True, exist_ok=True)
            input_copy = asset_root / "inputs" / f"{sample['sample_id']}.png"
            input_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, input_copy)
            mask_copy = None
            if mask_path:
                mask_copy = asset_root / "masks" / f"{sample['sample_id']}.png"
                mask_copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mask_path, mask_copy)
            raw_map = asset_root / "raw_maps" / f"{sample['sample_id']}.npy"
            raw_map.parent.mkdir(parents=True, exist_ok=True)
            np.save(raw_map, output.raw_anomaly_map, allow_pickle=False)
            heatmap = asset_root / "heatmaps" / f"{sample['sample_id']}.png"
            overlay = asset_root / "overlays" / f"{sample['sample_id']}.png"
            visualization = save_anomaly_visualization(
                output.raw_anomaly_map, input_copy, heatmap, overlay
            )
            preprocessing_ms = float(provenance.get("processing_ms", 0))
            return {
                **base,
                "status": "completed",
                "preprocessing_id": model["preprocessing_id"],
                "preprocessing_source": provenance["source"],
                "preprocessing_config_sha256": model["preprocessing_config_sha256"],
                "preprocessing_model_sha256": provenance.get("model_sha256", ""),
                "quality_flags": provenance.get("quality_flags", []),
                "input_asset": str(input_copy.relative_to(self.store.path(comparison_id))),
                "mask_asset": str(mask_copy.relative_to(self.store.path(comparison_id))) if mask_copy else None,
                "raw_anomaly_map_asset": str(raw_map.relative_to(self.store.path(comparison_id))),
                "heatmap_asset": str(heatmap.relative_to(self.store.path(comparison_id))),
                "heatmap_overlay_asset": str(overlay.relative_to(self.store.path(comparison_id))),
                "visualization": visualization,
                "raw_image_score": output.raw_image_score,
                "image_threshold": output.image_threshold,
                "prediction": output.prediction,
                "preprocessing_ms": preprocessing_ms,
                "inference_ms": output.inference_ms,
                "total_ms": preprocessing_ms + output.inference_ms,
                "pixel_display_scale": output.pixel_display_scale,
                "model_input_transform": output.transform_contract,
            }
        except Exception as exc:
            return {**base, "error": f"{type(exc).__name__}: {exc}"}


class BenchmarkRunner:
    """One sequential worker prevents competing PatchCore banks on the GPU."""

    def __init__(self, engine: BenchmarkEngine) -> None:
        self.engine = engine
        self._lock = threading.Lock()

    def start(self, comparison_id: str) -> None:
        def target() -> None:
            with self._lock:
                self.engine.run(comparison_id)

        threading.Thread(target=target, daemon=True, name=f"lab-{comparison_id}").start()
