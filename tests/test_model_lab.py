import csv
import json
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from model_lab.datasets import (
    safe_source_path,
    scan_exploratory_folder,
    select_exploratory_samples,
    select_samples,
)
from model_lab.benchmark import BenchmarkEngine, paired_model_summaries
from model_lab.metrics import calculate_metrics
from model_lab.patchcore_adapter import (
    PATCHCORE_TRANSFORM_CONTRACT,
    PatchCoreAdapter,
    save_anomaly_visualization,
)
from model_lab.preprocessing import PreprocessingResolver
from model_lab.registry import ModelRegistry
from model_lab.settings import LabSettings
from model_lab.storage import ComparisonStore
from training.datasets.create_dataset_manifest import build_rows, sha256_file, sha256_manifest, write_manifest
from training.preprocessing.runtime import canonical_json_hash, resolve_live_config


def settings_for(root: Path) -> LabSettings:
    workspace = root / "lab"
    return LabSettings(
        workspace=workspace,
        registry_file=workspace / "registry.json",
        imports_dir=workspace / "imports",
        cache_dir=workspace / "cache",
        results_dir=workspace / "results",
        preprocessing_configs_dir=root / "configs",
    )


class ModelLabMetricTests(unittest.TestCase):
    def test_normal_only_defect_metrics_are_unavailable_not_zero(self):
        rows = [
            {
                "status": "completed",
                "label": "normal",
                "raw_image_score": score,
                "prediction": "fault" if score > 0.5 else "normal",
                "preprocessing_ms": 1,
                "inference_ms": 2,
                "total_ms": 3,
                "quality_flags": [],
            }
            for score in (0.1, 0.2, 0.8)
        ]
        metrics = calculate_metrics(rows)
        self.assertIsNone(metrics["auroc"]["value"])
        self.assertFalse(metrics["recall"]["available"])
        self.assertEqual(metrics["false_positives"]["value"], 1)
        self.assertAlmostEqual(metrics["false_positive_rate"]["value"], 1 / 3)

    def test_auroc_uses_unclipped_raw_scores(self):
        rows = []
        for label, score, prediction in (
            ("normal", 100.0, "normal"),
            ("normal", 101.0, "normal"),
            ("fault", 102.0, "fault"),
            ("fault", 103.0, "fault"),
        ):
            rows.append(
                {
                    "status": "completed",
                    "label": label,
                    "raw_image_score": score,
                    "prediction": prediction,
                    "preprocessing_ms": 1,
                    "inference_ms": 2,
                    "total_ms": 3,
                    "quality_flags": [],
                }
            )
        self.assertEqual(calculate_metrics(rows)["auroc"]["value"], 1.0)

    def test_fpr_is_unavailable_without_normal_denominator(self):
        rows = [{
            "status": "completed", "label": "fault", "raw_image_score": 1.0,
            "prediction": "fault", "preprocessing_ms": 1, "inference_ms": 2,
            "total_ms": 3, "quality_flags": [],
        }]
        self.assertFalse(calculate_metrics(rows)["false_positive_rate"]["available"])

    def test_unlabeled_data_disables_all_class_metrics_with_clear_reason(self):
        rows = [{
            "status": "completed", "label": "unlabeled", "raw_image_score": 0.2,
            "prediction": "normal", "preprocessing_ms": 1, "inference_ms": 2,
            "total_ms": 3, "quality_flags": [],
        }]
        metrics = calculate_metrics(rows)
        for name in ("false_positive_rate", "auroc", "f1", "precision", "recall"):
            self.assertFalse(metrics[name]["available"])
            self.assertIn("unlabeled", metrics[name]["reason"].lower())


class ModelLabDatasetTests(unittest.TestCase):
    def test_sample_selection_is_shared_and_reproducible(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for index in range(8):
                Image.new("L", (8, 10), index).save(
                    source / f"G01-260203-1200{index:02d}-{index:03d}.bmp"
                )
            rows = build_rows(
                source,
                {"2026-02-03": "val"},
                label="normal",
                include_hash=True,
            )
            manifest = root / "split.csv"
            write_manifest(manifest, rows, overwrite=False)
            first = select_samples(manifest, split="val", count=4, seed=42)
            second = select_samples(manifest, split="val", count=4, seed=42)
            other = select_samples(manifest, split="val", count=4, seed=7)
            self.assertEqual(first, second)
            self.assertNotEqual(
                [row["sample_id"] for row in first],
                [row["sample_id"] for row in other],
            )

    def test_exploratory_folder_snapshot_is_hashed_stable_and_unlabeled(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            Image.new("RGB", (8, 10), "white").save(root / "b.png")
            Image.new("RGB", (8, 10), "black").save(root / "nested" / "a.jpg")
            first, first_hash = scan_exploratory_folder(
                root, camera_angle="G01", label_mode="unlabeled"
            )
            second, second_hash = scan_exploratory_folder(
                root, camera_angle="G01", label_mode="unlabeled"
            )
            self.assertEqual(first, second)
            self.assertEqual(first_hash, second_hash)
            self.assertEqual({row["label"] for row in first}, {"unlabeled"})
            self.assertEqual(len({row["sample_id"] for row in first}), 2)
            self.assertEqual(
                select_exploratory_samples(first, count=None, seed=42), first
            )
            self.assertEqual(
                select_exploratory_samples(first, count=1, seed=42),
                select_exploratory_samples(first, count=1, seed=42),
            )
            Image.new("RGB", (8, 10), "red").save(root / "b.png")
            _, changed_hash = scan_exploratory_folder(
                root, camera_angle="G01", label_mode="unlabeled"
            )
            self.assertNotEqual(first_hash, changed_hash)

    def test_exploratory_verified_normal_is_explicit_not_folder_inferred(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "fault"; root.mkdir()
            Image.new("RGB", (8, 10), "white").save(root / "image.png")
            unlabeled, _ = scan_exploratory_folder(
                root, camera_angle="G01", label_mode="unlabeled"
            )
            normal, _ = scan_exploratory_folder(
                root, camera_angle="G01", label_mode="verified_normal"
            )
            self.assertEqual(unlabeled[0]["label"], "unlabeled")
            self.assertEqual(normal[0]["label"], "normal")


class SharedPreprocessingRuntimeTests(unittest.TestCase):
    def test_patchcore_runtime_matches_training_bilinear_resize(self):
        self.assertEqual(PATCHCORE_TRANSFORM_CONTRACT["interpolation"], "bilinear")

    def test_patchcore_adapter_uses_inner_model_and_preserves_raw_outputs(self):
        import torch
        from torchvision.transforms import v2

        calls = {"transform": 0, "outer": 0, "pre": 0, "post": 0, "inner": 0}

        def transform(tensor):
            calls["transform"] += 1
            return tensor.to(torch.float32) / 255

        class Inner:
            def __call__(self, _tensor):
                calls["inner"] += 1
                return SimpleNamespace(
                    pred_score=torch.tensor([0.375], dtype=torch.float32),
                    anomaly_map=torch.tensor([[[[0.1, 0.2], [0.4, 0.9]]]]),
                )

        class Outer:
            model = Inner()

            def __call__(self, tensor):
                calls["outer"] += 1
                calls["pre"] += 1
                result = self.model(tensor)
                calls["post"] += 1
                return result

        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "input.png"
            Image.new("RGB", (8, 8), "white").save(image_path)
            adapter = PatchCoreAdapter.__new__(PatchCoreAdapter)
            adapter.torch = torch
            adapter.v2 = v2
            adapter.device = torch.device("cpu")
            adapter.transform = transform
            adapter.model = Outer()
            adapter.contract = {"image_threshold": None}
            output = adapter.predict(image_path)

        self.assertEqual(
            calls,
            {"transform": 1, "outer": 0, "pre": 0, "post": 0, "inner": 1},
        )
        self.assertAlmostEqual(output.raw_image_score, 0.375)
        self.assertTrue(np.allclose(output.raw_anomaly_map, [[0.1, 0.2], [0.4, 0.9]]))
        self.assertGreater(float(output.raw_anomaly_map.max()), float(output.raw_anomaly_map.min()))

    def test_display_overlay_warns_for_nonfinite_or_constant_map_without_mutation(self):
        import numpy as np

        with TemporaryDirectory() as directory:
            root = Path(directory)
            model_input = root / "input.png"
            Image.new("RGB", (20, 10), "white").save(model_input)
            raw_map = np.asarray([[np.nan, 2.0], [2.0, 2.0]], dtype=np.float32)
            original = raw_map.copy()
            metadata = save_anomaly_visualization(
                raw_map, model_input, root / "heatmap.png", root / "overlay.png"
            )
            self.assertTrue(np.array_equal(raw_map, original, equal_nan=True))
            self.assertIn("anomaly_map_nonfinite_values_replaced_for_display", metadata["warnings"])
            self.assertIn("anomaly_map_is_constant", metadata["warnings"])
            with Image.open(root / "overlay.png") as overlay:
                self.assertEqual(overlay.size, (20, 10))

    def test_black_recomposition_resolves_to_full_live_parent_pipeline(self):
        with TemporaryDirectory() as directory:
            config_dir = Path(directory)
            parent = {
                "preprocessing_id": "rembg_gray",
                "backend": "rembg",
                "model_name": "u2net",
                "model_dir": "models",
                "model_filename": "u2net.onnx",
                "output_size": [100, 100],
                "background_value": 128,
                "mask_threshold": 128,
            }
            (config_dir / "rembg_gray.json").write_text(json.dumps(parent), encoding="utf-8")
            black = {
                "preprocessing_id": "rembg_black",
                "backend": "aligned_mask_recompose",
                "parent_preprocessing_id": "rembg_gray",
                "parent_config_sha256": canonical_json_hash(parent),
                "background_value": 0,
            }
            resolved = resolve_live_config(black, config_dir)
            self.assertEqual(resolved["backend"], "rembg")
            self.assertEqual(resolved["model_name"], "u2net")
            self.assertEqual(resolved["background_value"], 0)

    def test_external_derivative_requires_matching_provenance_and_hashes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            settings = settings_for(root)
            settings.ensure_writable_directories()
            source = root / "source.bmp"
            Image.new("RGB", (10, 12), "white").save(source)
            source_hash = sha256_file(source)
            config = {
                "preprocessing_id": "raw_letterbox_v1",
                "backend": "raw_letterbox",
                "output_size": [10, 10],
                "background_value": 128,
            }
            config_path = root / "raw.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            derivative = root / "derivative"
            output = derivative / "val" / "normal" / "sample.png"
            output.parent.mkdir(parents=True)
            Image.new("RGB", (10, 10), "gray").save(output)
            manifest_hash = "frozen-manifest"
            fields = {
                "status": "ok",
                "parent_sample_id": "sample",
                "split": "val",
                "label": "normal",
                "source_sha256": source_hash,
                "parent_manifest_sha256": manifest_hash,
                "preprocessing_id": "raw_letterbox_v1",
                "config_sha256": canonical_json_hash(config),
                "model_sha256": "",
                "output_relpath": "val/normal/sample.png",
                "output_sha256": sha256_file(output),
                "mask_relpath": "",
                "mask_sha256": "",
                "processing_ms": "4.2",
                "quality_flags": "",
            }
            with (derivative / "derivative_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(fields))
                writer.writeheader(); writer.writerow(fields)
            model = {
                "preprocessing_config_path": str(config_path),
                "preprocessing_config_sha256": canonical_json_hash(config),
                "preprocessing_id": "raw_letterbox_v1",
                "preprocessing_model_sha256": None,
                "derivative_root": str(derivative),
            }
            result, mask, provenance = PreprocessingResolver(model, settings).resolve(
                source=source,
                sample={"sample_id": "sample", "split": "val", "label": "normal", "source_sha256": source_hash},
                manifest_hash=manifest_hash,
                force_live=False,
            )
            self.assertEqual(result, output)
            self.assertIsNone(mask)
            self.assertEqual(provenance["source"], "verified_derivative")


class ComparisonPersistenceTests(unittest.TestCase):
    def test_results_survive_store_recreation_and_truncated_tail(self):
        with TemporaryDirectory() as directory:
            settings = settings_for(Path(directory))
            store = ComparisonStore(settings)
            comparison_id = store.create({"name": "test"}, [{"sample_id": "one"}])
            store.append_result(comparison_id, {"model_id": "m", "sample_id": "one", "status": "completed"})
            with (store.path(comparison_id) / "sample_results.jsonl").open("a", encoding="utf-8") as stream:
                stream.write('{"partial"')
            restored = ComparisonStore(settings).results(comparison_id)
            self.assertEqual(len(restored), 1)
            self.assertEqual(restored[0]["sample_id"], "one")

    def test_locked_test_requires_exact_confirmation_before_job_creation(self):
        with TemporaryDirectory() as directory:
            settings = settings_for(Path(directory))
            store = ComparisonStore(settings)

            class Registry:
                def get(self, model_id):
                    raise AssertionError("Registry must not be touched before test confirmation")

            engine = BenchmarkEngine(settings, Registry(), store)
            with self.assertRaisesRegex(ValueError, "RUN LOCKED TEST"):
                engine.prepare(
                    {
                        "model_ids": ["model"],
                        "dataset_mode": "official_manifest",
                        "split": "test",
                        "manifest": "unused.csv",
                        "source_root": "unused",
                        "image_count": 1,
                    }
                )

    def test_comparison_rejects_more_than_four_models(self):
        with TemporaryDirectory() as directory:
            settings = settings_for(Path(directory))
            engine = BenchmarkEngine(settings, object(), ComparisonStore(settings))
            with self.assertRaisesRegex(ValueError, "one and four"):
                engine.prepare(
                    {
                        "model_ids": ["a", "b", "c", "d", "e"],
                        "split": "val",
                        "manifest": "unused.csv",
                        "source_root": "unused",
                        "image_count": 1,
                    }
                )
            with self.assertRaisesRegex(ValueError, "one and four"):
                engine.prepare({"model_ids": [], "split": "val", "manifest": "unused.csv", "source_root": "unused", "image_count": 1})

    def test_snapshot_verification_detects_checkpoint_mutation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.ckpt"; checkpoint.write_bytes(b"one")
            config = {"preprocessing_id": "raw", "backend": "raw_letterbox"}
            config_path = root / "config.json"; config_path.write_text(json.dumps(config), encoding="utf-8")
            contract = {"id": "m", "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "preprocessing_config_path": str(config_path), "preprocessing_config_sha256": canonical_json_hash(config)}
            contract["contract_sha256"] = canonical_json_hash(contract)
            BenchmarkEngine._verify_contract(contract)
            checkpoint.write_bytes(b"two")
            with self.assertRaisesRegex(ValueError, "Checkpoint changed"):
                BenchmarkEngine._verify_contract(contract)

    def test_comparison_id_and_manifest_source_paths_reject_traversal(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = ComparisonStore(settings_for(root))
            for invalid in (".", "..", "a/b", "a\\b", "not-an-id"):
                with self.assertRaises(ValueError):
                    store.path(invalid)
            with self.assertRaisesRegex(ValueError, "escapes"):
                safe_source_path(root / "raw", "../secret.bmp")
            with self.assertRaisesRegex(ValueError, "relative"):
                safe_source_path(root / "raw", str((root / "secret.bmp").resolve()))

    def test_mixed_camera_angle_selection_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"; source.mkdir()
            Image.new("L", (8, 10), 1).save(source / "G01-260203-120000-001.bmp")
            Image.new("L", (8, 10), 2).save(source / "G02-260203-120001-002.bmp")
            rows = build_rows(source, {"2026-02-03": "val"}, label="normal", include_hash=True)
            manifest = root / "split.csv"; write_manifest(manifest, rows, overwrite=False)
            manifest_hash = sha256_manifest(manifest)

            class Registry:
                def get(self, _model_id):
                    return {"id": "m", "family": "patchcore", "status": "ready_uncalibrated", "issues": [], "manifest_sha256": manifest_hash, "angle": "G01"}

            engine = BenchmarkEngine(settings_for(root), Registry(), ComparisonStore(settings_for(root)))
            with self.assertRaisesRegex(ValueError, "exactly one camera angle"):
                engine.prepare({"model_ids": ["m"], "dataset_mode": "official_manifest", "split": "val", "manifest": str(manifest), "source_root": str(source), "image_count": 2, "seed": 42})

    def test_prepare_snapshots_contract_and_locked_test_is_full_and_one_shot(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); settings = settings_for(root)
            source = root / "source"; source.mkdir()
            Image.new("L", (8, 10), 1).save(source / "G01-260204-120000-001.bmp")
            rows = build_rows(source, {"2026-02-04": "test"}, label="normal", include_hash=True)
            manifest = root / "split.csv"; write_manifest(manifest, rows, overwrite=False)
            manifest_hash = sha256_manifest(manifest)
            model = {"id": "m", "family": "patchcore", "status": "ready_uncalibrated", "issues": [], "manifest_sha256": manifest_hash, "angle": "G01", "checkpoint_sha256": "checkpoint", "preprocessing_config_sha256": "config", "preprocessing_model_sha256": ""}

            class Registry:
                def get(self, _model_id): return dict(model)

            store = ComparisonStore(settings); engine = BenchmarkEngine(settings, Registry(), store)
            request = {"model_ids": ["m"], "dataset_mode": "official_manifest", "split": "test", "manifest": str(manifest), "source_root": str(source), "image_count": 1, "seed": 42, "locked_test_confirmation": "RUN LOCKED TEST"}
            comparison_id = engine.prepare(request)
            saved = store.read_json(comparison_id, "comparison.json")
            self.assertEqual(saved["model_contracts"][0]["checkpoint_sha256"], "checkpoint")
            self.assertIn("contract_sha256", saved["model_contracts"][0])
            self.assertTrue((settings.workspace / "locked_test_record.json").is_file())
            with self.assertRaisesRegex(ValueError, "already evaluated"):
                engine.prepare(request)

    def test_exploratory_evaluation_allows_different_training_manifests(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); source = root / "images"; source.mkdir()
            Image.new("RGB", (8, 10), "white").save(source / "one.png")
            models = {
                "a": {"id": "a", "family": "patchcore", "status": "ready_uncalibrated", "issues": [], "manifest_sha256": "training-a", "angle": "G01"},
                "b": {"id": "b", "family": "patchcore", "status": "ready_uncalibrated", "issues": [], "manifest_sha256": "training-b", "angle": "G01"},
            }

            class Registry:
                def get(self, model_id): return dict(models[model_id])

            store = ComparisonStore(settings_for(root))
            comparison_id = BenchmarkEngine(settings_for(root), Registry(), store).prepare(
                {"model_ids": ["a", "b"], "source_root": str(source), "image_count": None}
            )
            config = store.read_json(comparison_id, "comparison.json")
            self.assertEqual(config["dataset_mode"], "exploratory_folder")
            self.assertEqual(config["image_count"], 1)
            self.assertIn("different manifests", config["warnings"][0])
            self.assertTrue((store.path(comparison_id) / "evaluation_snapshot.json").is_file())

    def test_paired_summaries_exclude_sample_failed_by_any_model(self):
        def row(model, sample, status="completed"):
            return {"model_id": model, "sample_id": sample, "status": status, "label": "normal", "raw_image_score": 0.1, "prediction": None, "preprocessing_ms": 1, "inference_ms": 2, "total_ms": 3, "quality_flags": []}
        summaries, paired_count = paired_model_summaries(
            [row("a", "one"), row("a", "two"), row("b", "one"), row("b", "two", "error")],
            ["a", "b"],
            2,
        )
        self.assertEqual(paired_count, 1)
        self.assertEqual(summaries["a"]["successful_count"], 1)
        self.assertEqual(summaries["b"]["excluded_unpaired_count"], 1)


class ModelRegistryTests(unittest.TestCase):
    def _artifact(self, root: Path, name: str, *, angle: str = "", manifest_hash: str = ""):
        folder = root / name; folder.mkdir()
        checkpoint = folder / "G01.ckpt"; checkpoint.write_bytes(name.encode())
        metadata = {
            "model": {"class": "Patchcore", "angle": angle, "image_size": 256},
            "dataset": {"preprocessing_id": "raw_letterbox_v1", "manifest_sha256": manifest_hash},
        }
        metadata_path = folder / "G01.metadata.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        return checkpoint, metadata_path

    def test_guided_contract_update_persists_angle_manifest_and_threshold(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); settings = settings_for(root)
            settings.preprocessing_configs_dir.mkdir(parents=True)
            config = {"preprocessing_id": "raw_letterbox_v1", "backend": "raw_letterbox", "output_size": [10, 10], "background_value": 128}
            config_path = settings.preprocessing_configs_dir / "raw_letterbox_v1.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            checkpoint, metadata = self._artifact(root, "model")
            registry = ModelRegistry(settings)
            imported = registry.import_checkpoint(checkpoint, metadata_path=metadata)
            self.assertEqual(imported["status"], "incomplete")
            updated = registry.update_contract(imported["id"], {"angle": "G01", "manifest_sha256": "abc123", "image_threshold": 0.42})
            self.assertEqual(updated["angle"], "G01")
            self.assertEqual(updated["manifest_sha256"], "abc123")
            self.assertEqual(updated["image_threshold"], 0.42)
            self.assertEqual(updated["calibration"]["method"], "manual_unverified")
            self.assertEqual(updated["status"], "ready")

    def test_same_sanitized_name_does_not_overwrite_another_checkpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); settings = settings_for(root)
            settings.preprocessing_configs_dir.mkdir(parents=True)
            config = {"preprocessing_id": "raw_letterbox_v1", "backend": "raw_letterbox", "output_size": [10, 10], "background_value": 128}
            (settings.preprocessing_configs_dir / "raw_letterbox_v1.json").write_text(json.dumps(config), encoding="utf-8")
            first, first_meta = self._artifact(root, "first", angle="G01", manifest_hash="hash")
            second, second_meta = self._artifact(root, "second", angle="G01", manifest_hash="hash")
            registry = ModelRegistry(settings)
            one = registry.import_checkpoint(first, display_name="same name", metadata_path=first_meta)
            two = registry.import_checkpoint(second, display_name="same name", metadata_path=second_meta)
            self.assertNotEqual(one["id"], two["id"])
            self.assertEqual(len(registry.list()), 2)


if __name__ == "__main__":
    unittest.main()
