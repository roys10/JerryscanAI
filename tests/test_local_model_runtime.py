from __future__ import annotations

import base64
import io
import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

import numpy as np
from PIL import Image

from backend.inference.local_model import (
    LocalModelManifest,
    LocalPatchCoreRuntime,
    ModelFolderError,
    RawPatchCoreEngine,
    _defect_localization_mask,
    _display_artifacts,
)
from backend.inference.manager import JerryScanModelManager, ModelNotReadyError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_BYTES = b"checkpoint"
WEIGHT_BYTES = b"u2net-weight"


def _document(model_id: str, *, threshold=60.0, rembg: bool = False) -> dict:
    preprocessing_id = "rembg_u2net_gray_v1" if rembg else "raw_letterbox_v1"
    backend = "rembg" if rembg else "raw_letterbox"
    weight = None
    if rembg:
        weight = {
            "file": "u2net.onnx",
            "shared_fallback": "../preprocessing/rembg/u2net.onnx",
            "sha256": hashlib.sha256(WEIGHT_BYTES).hexdigest(),
            "size_bytes": len(WEIGHT_BYTES),
        }
    return {
        "schema_version": "1.0",
        "model": {
            "id": model_id,
            "display_name": "Example production model",
            "family": "patchcore",
            "angle": "G01",
            "image_size": 256,
        },
        "input": {"width": 32, "height": 48},
        "artifacts": {
            "checkpoint": {
                "file": "G01.ckpt",
                "sha256": hashlib.sha256(CHECKPOINT_BYTES).hexdigest(),
                "size_bytes": len(CHECKPOINT_BYTES),
            },
            "metadata": "G01.metadata.json",
        },
        "preprocessing": {
            "id": preprocessing_id,
            "config": {
                "schema_version": "1.0",
                "preprocessing_id": preprocessing_id,
                "backend": backend,
                "output_size": [1024, 1024],
                "background_value": 128,
                **(
                    {
                        "model_name": "u2net",
                        "model_filename": "u2net.onnx",
                        "mask_threshold": 128,
                    }
                    if rembg
                    else {}
                ),
            },
            "weight": weight,
        },
        "decision_threshold": {
            "score": "raw_patchcore_image_score",
            "value": threshold,
            "rule": "fail_if_score_greater_than_or_equal",
            "provenance": "test",
        },
        "max_image_bytes": 1024 * 1024,
    }


def _write_folder(
    root: Path,
    *,
    with_checkpoint: bool = True,
    threshold=60.0,
    rembg: bool = False,
) -> Path:
    folder = root / (
        "Patchcore_rembg_example_256_c10_seed42"
        if rembg
        else "Patchcore_example_256_c10_seed42"
    )
    folder.mkdir()
    (folder / "model.json").write_text(
        json.dumps(_document(folder.name, threshold=threshold, rembg=rembg)),
        encoding="utf-8",
    )
    if with_checkpoint:
        (folder / "G01.ckpt").write_bytes(CHECKPOINT_BYTES)
    if rembg:
        (folder / "u2net.onnx").write_bytes(WEIGHT_BYTES)
    metadata = {
        "model": {
            "class": "Patchcore",
            "angle": "G01",
            "model_set": folder.name,
            "image_size": 256,
        },
        "dataset": {
            "preprocessing_id": (
                "rembg_u2net_gray_v1" if rembg else "raw_letterbox_v1"
            )
        },
    }
    (folder / "G01.metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return folder


def _upgrade_to_multi_angle(folder: Path, count: int = 4) -> None:
    """Expand a temporary schema-1.0 fixture into a schema-1.1 bundle."""
    document = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    checkpoint = document.pop("artifacts")["checkpoint"]
    threshold = document.pop("decision_threshold")
    document["schema_version"] = "1.1"
    document["model"].pop("angle")
    document["angles"] = {}
    source_metadata = json.loads(
        (folder / "G01.metadata.json").read_text(encoding="utf-8")
    )
    for number in range(1, count + 1):
        angle = f"G{number:02d}"
        document["angles"][angle] = {
            "checkpoint": {**checkpoint, "file": f"{angle}.ckpt"},
            "metadata": f"{angle}.metadata.json",
            "decision_threshold": {**threshold, "value": 60.0 + number},
        }
        (folder / f"{angle}.ckpt").write_bytes(CHECKPOINT_BYTES)
        metadata = json.loads(json.dumps(source_metadata))
        metadata["model"]["angle"] = angle
        (folder / f"{angle}.metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
    (folder / "model.json").write_text(json.dumps(document), encoding="utf-8")


class ManifestTests(unittest.TestCase):
    def test_all_four_tracked_preprocessing_contracts(self):
        expected = {
            "Patchcore_raw_letterbox_v1_256_c10_seed42": ("raw_letterbox", 128, 35),
            "Patchcore_fixed_crop_v1_256_c10_seed42": ("fixed_crop", 128, 36),
            "Patchcore_rembg_u2net_gray_v1_256_c10_seed42": ("rembg", 128, 34),
            "Patchcore_rembg_u2net_black_v1_256_c10_seed42": ("rembg", 0, 34),
        }
        for model_id, contract in expected.items():
            with self.subTest(model_id=model_id):
                manifest = LocalModelManifest.load(
                    PROJECT_ROOT / "models" / model_id, require_artifacts=False
                )
                self.assertEqual(
                    (
                        manifest.preprocessing_config["backend"],
                        manifest.preprocessing_config["background_value"],
                    ),
                    contract[:2],
                )
                self.assertEqual(manifest.decision_threshold, contract[2])
                self.assertEqual(
                    (manifest.original_width, manifest.original_height), (1025, 1281)
                )
                self.assertEqual(len(manifest.checkpoint_sha256), 64)

    def test_artifact_and_metadata_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary))
            manifest = LocalModelManifest.load(folder)
            self.assertEqual(manifest.model_id, folder.name)
            self.assertEqual(manifest.display_name, "Example production model")
            metadata = json.loads(manifest.metadata.read_text(encoding="utf-8"))
            metadata["dataset"]["preprocessing_id"] = "wrong"
            manifest.metadata.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ModelFolderError, "metadata mismatch"):
                LocalModelManifest.load(folder)

    def test_paths_cannot_escape_model_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary))
            document = json.loads((folder / "model.json").read_text(encoding="utf-8"))
            document["artifacts"]["checkpoint"]["file"] = "../outside.ckpt"
            (folder / "model.json").write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ModelFolderError, "directly inside"):
                LocalModelManifest.load(folder)

    def test_decision_threshold_must_be_positive_for_quality_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary), threshold=0)
            with self.assertRaisesRegex(ModelFolderError, "positive finite"):
                LocalModelManifest.load(folder)

    def test_checkpoint_tampering_is_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary))
            (folder / "G01.ckpt").write_bytes(b"checkpoinu")
            with self.assertRaisesRegex(ModelFolderError, "SHA-256 mismatch"):
                LocalModelManifest.load(folder)

    def test_rembg_weight_corruption_is_rejected_before_onnx_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary), rembg=True)
            (folder / "u2net.onnx").write_bytes(b"u2net-weighu")
            with self.assertRaisesRegex(ModelFolderError, "SHA-256 mismatch"):
                LocalModelManifest.load(folder)

    def test_optional_metadata_binding_must_match_authoritative_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary))
            metadata_path = folder / "G01.metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["checkpoint_artifact"] = {
                "sha256": "0" * 64,
                "size_bytes": len(CHECKPOINT_BYTES),
            }
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ModelFolderError, "checkpoint_artifact"):
                LocalModelManifest.load(folder)

    def test_missing_checkpoint_leaves_manager_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary), with_checkpoint=False)
            manager = JerryScanModelManager(folder, runtime_factory=lambda _: object())
            with self.assertRaises(ModelFolderError):
                manager.load_selected()
            with self.assertRaises(ModelNotReadyError):
                manager.inspect("G01", b"not used")

    def test_schema_1_1_declares_and_validates_each_angle_independently(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary))
            document = json.loads((folder / "model.json").read_text(encoding="utf-8"))
            checkpoint = document.pop("artifacts")["checkpoint"]
            metadata_name = "G01.metadata.json"
            threshold = document.pop("decision_threshold")
            document["schema_version"] = "1.1"
            document["model"].pop("angle")
            document["angles"] = {
                "G01": {
                    "checkpoint": checkpoint,
                    "metadata": metadata_name,
                    "decision_threshold": threshold,
                },
                "G02": {
                    "checkpoint": {**checkpoint, "file": "G02.ckpt"},
                    "metadata": "G02.metadata.json",
                    "decision_threshold": {**threshold, "value": 9.5},
                },
            }
            (folder / "model.json").write_text(json.dumps(document), encoding="utf-8")
            (folder / "G02.ckpt").write_bytes(CHECKPOINT_BYTES)
            metadata = json.loads((folder / metadata_name).read_text(encoding="utf-8"))
            metadata["model"]["angle"] = "G02"
            (folder / "G02.metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )

            manifest = LocalModelManifest.load(folder)

            self.assertEqual(manifest.required_angles, ("G01", "G02"))
            self.assertEqual(manifest.angles["G01"].decision_threshold, 60.0)
            self.assertEqual(manifest.angles["G02"].decision_threshold, 9.5)

            engines = iter([_FakeEngine(score=10), _FakeEngine(score=10)])
            runtime = LocalPatchCoreRuntime(
                manifest,
                preprocessor_factory=lambda _: _FakePreprocessor(),
                engine_factory=lambda _: next(engines),
            )
            stream = io.BytesIO()
            Image.new("RGB", (32, 48), "white").save(stream, format="PNG")
            image_bytes = stream.getvalue()

            self.assertEqual(runtime.predict(image_bytes, "G01")["status"], "PASS")
            g02_result = runtime.predict(image_bytes, "G02")
            self.assertEqual(g02_result["status"], "FAIL")
            self.assertEqual(g02_result["image_threshold"], 9.5)
            self.assertEqual(g02_result["angle"], "G02")

    def test_schema_1_1_keeps_one_two_three_or_four_valid_artifacts(self):
        for available_count in range(1, 5):
            with self.subTest(available_count=available_count):
                with tempfile.TemporaryDirectory() as temporary:
                    folder = _write_folder(Path(temporary))
                    _upgrade_to_multi_angle(folder)
                    for number in range(available_count + 1, 5):
                        (folder / f"G{number:02d}.ckpt").unlink()

                    manifest = LocalModelManifest.load(folder)

                    expected = tuple(f"G{number:02d}" for number in range(1, available_count + 1))
                    self.assertEqual(manifest.configured_angles, ("G01", "G02", "G03", "G04"))
                    self.assertEqual(manifest.required_angles, expected)
                    self.assertEqual(
                        set(manifest.unavailable_angles),
                        {f"G{number:02d}" for number in range(available_count + 1, 5)},
                    )

    def test_schema_1_1_corrupt_artifact_does_not_hide_valid_sibling(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary))
            _upgrade_to_multi_angle(folder, count=2)
            (folder / "G02.ckpt").write_bytes(b"checkpoinu")

            manifest = LocalModelManifest.load(folder)

            self.assertEqual(manifest.required_angles, ("G01",))
            self.assertEqual(
                manifest.unavailable_angles["G02"]["stage"], "artifact_validation"
            )
            self.assertIn("SHA-256 mismatch", manifest.unavailable_angles["G02"]["detail"])

    def test_schema_1_1_all_failed_is_structured_not_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary))
            _upgrade_to_multi_angle(folder, count=2)
            (folder / "G01.ckpt").unlink()
            (folder / "G02.ckpt").unlink()
            manifest = LocalModelManifest.load(folder)

            with self.assertRaises(ModelFolderError) as caught:
                LocalPatchCoreRuntime(
                    manifest,
                    preprocessor_factory=lambda _: _FakePreprocessor(),
                    engine_factory=lambda _: _FakeEngine(),
                )

            self.assertEqual(caught.exception.configured_angles, ("G01", "G02"))
            self.assertEqual(set(caught.exception.unavailable_angles), {"G01", "G02"})

    def test_schema_1_1_shared_preprocessing_failure_blocks_every_angle(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary), rembg=True)
            _upgrade_to_multi_angle(folder, count=2)
            (folder / "u2net.onnx").unlink()

            with self.assertRaises(ModelFolderError) as caught:
                LocalModelManifest.load(folder)

            self.assertEqual(caught.exception.configured_angles, ("G01", "G02"))
            self.assertEqual(set(caught.exception.unavailable_angles), {"G01", "G02"})
            self.assertTrue(
                all(
                    reason["stage"] == "shared_preprocessing_artifact"
                    for reason in caught.exception.unavailable_angles.values()
                )
            )


class _FakePreprocessor:
    name = "raw_letterbox"
    version = "test"
    model_name = ""
    model_sha256 = ""

    def __init__(self):
        self.received_size = None

    def process(self, image):
        self.received_size = image.size
        output = Image.new("RGB", (20, 12), "gray")
        mask = Image.new("L", output.size, 255)
        return output, mask, {"quality_flags": [], "marker": "preprocessed"}


class _FakeEngine:
    def __init__(self, score=7.25, pixel_display_bounds=(0.0, 100.0), device="cpu"):
        self.device = device
        self.device_fallback_reason = None
        self.received_size = None
        self.closed = False
        self.score = score
        self.pixel_display_bounds = pixel_display_bounds

    def predict(self, image):
        self.received_size = image.size
        return self.score, np.arange(24, dtype=np.float32).reshape(4, 6), 3.5

    def close(self):
        self.closed = True


class RuntimeTests(unittest.TestCase):
    def test_engine_initialization_failure_degrades_only_that_angle(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary))
            _upgrade_to_multi_angle(folder, count=2)
            manifest = LocalModelManifest.load(folder)

            def engine_factory(checkpoint):
                if checkpoint.name == "G02.ckpt":
                    raise RuntimeError("simulated angle load failure")
                return _FakeEngine()

            runtime = LocalPatchCoreRuntime(
                manifest,
                preprocessor_factory=lambda _: _FakePreprocessor(),
                engine_factory=engine_factory,
            )

            self.assertEqual(runtime.available_angles, ("G01",))
            self.assertEqual(
                runtime.unavailable_angles["G02"]["stage"], "engine_initialization"
            )

    def test_manager_health_reports_partial_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary))
            _upgrade_to_multi_angle(folder, count=2)

            def runtime_factory(manifest):
                return LocalPatchCoreRuntime(
                    manifest,
                    preprocessor_factory=lambda _: _FakePreprocessor(),
                    engine_factory=lambda checkpoint: (
                        _FakeEngine()
                        if checkpoint.name == "G01.ckpt"
                        else (_ for _ in ()).throw(RuntimeError("cannot load G02"))
                    ),
                )

            manager = JerryScanModelManager(folder, runtime_factory=runtime_factory)
            manager.load_selected()

            health = manager.health()
            self.assertEqual(health["status"], "degraded")
            self.assertTrue(health["ready_for_inference"])
            self.assertFalse(health["ready_for_decisions"])
            self.assertEqual(health["coverage"], "partial")
            self.assertEqual(health["configured_angles"], ["G01", "G02"])
            self.assertEqual(health["available_angles"], ["G01"])
            self.assertEqual(health["required_angles"], ["G01"])
            self.assertIn("G02", health["unavailable_angles"])

    def test_manager_health_reports_all_failed_angles(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary))
            _upgrade_to_multi_angle(folder, count=2)
            (folder / "G01.ckpt").unlink()
            (folder / "G02.ckpt").unlink()
            manager = JerryScanModelManager(
                folder,
                runtime_factory=lambda manifest: LocalPatchCoreRuntime(
                    manifest,
                    preprocessor_factory=lambda _: _FakePreprocessor(),
                    engine_factory=lambda _: _FakeEngine(),
                ),
            )

            with self.assertRaises(ModelFolderError):
                manager.load_selected()

            health = manager.health()
            self.assertEqual(health["status"], "not_ready")
            self.assertFalse(health["ready_for_inference"])
            self.assertEqual(health["coverage"], "none")
            self.assertEqual(health["configured_angles"], ["G01", "G02"])
            self.assertEqual(set(health["unavailable_angles"]), {"G01", "G02"})

    def test_string_interpolation_is_normalized_for_checkpoint_compatibility(self):
        from enum import Enum

        class _InterpolationMode(Enum):
            NEAREST = "nearest"
            BILINEAR = "bilinear"

        resize = SimpleNamespace(interpolation="bilinear")
        unchanged = SimpleNamespace(interpolation=_InterpolationMode.NEAREST)
        transform = SimpleNamespace(transforms=[resize, unchanged])

        converted = RawPatchCoreEngine._normalize_transform_interpolation(
            transform,
            _InterpolationMode,
        )

        self.assertEqual(converted, 1)
        self.assertIs(resize.interpolation, _InterpolationMode.BILINEAR)
        self.assertIs(unchanged.interpolation, _InterpolationMode.NEAREST)

    def test_distinct_gpu_angle_engines_can_use_configured_concurrency(self):
        manifest = LocalModelManifest.load(
            PROJECT_ROOT
            / "models"
            / "Patchcore_rembg_u2net_black_v1_256_c10_seed42",
            require_artifacts=False,
        )
        barrier = threading.Barrier(len(manifest.required_angles))

        class _ConcurrentEngine(_FakeEngine):
            def predict(self, image):
                barrier.wait(timeout=2)
                return super().predict(image)

        runtime = LocalPatchCoreRuntime(
            manifest,
            preprocessor_factory=lambda _: _FakePreprocessor(),
            engine_factory=lambda _, **__: _ConcurrentEngine(device="cuda"),
            gpu_model_count=len(manifest.required_angles),
            gpu_inference_concurrency=len(manifest.required_angles),
        )
        stream = io.BytesIO()
        Image.new(
            "RGB",
            (manifest.original_width, manifest.original_height),
            "white",
        ).save(stream, format="PNG")
        image_bytes = stream.getvalue()
        results = {}
        errors = []

        def inspect(angle):
            try:
                results[angle] = runtime.predict(image_bytes, angle)
            except Exception as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        threads = [
            threading.Thread(target=inspect, args=(angle,))
            for angle in manifest.required_angles
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)

        self.assertEqual(errors, [])
        self.assertEqual(set(results), set(manifest.required_angles))
        self.assertTrue(all(result["status"] == "PASS" for result in results.values()))

    def test_patchcore_prefers_cuda_and_falls_back_to_cpu_when_initialization_fails(self):
        engine = RawPatchCoreEngine.__new__(RawPatchCoreEngine)
        cuda = SimpleNamespace(type="cuda")
        cpu = SimpleNamespace(type="cpu")
        empty_cache = Mock()
        engine.torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, empty_cache=empty_cache),
            device=lambda name: cuda if name == "cuda" else cpu,
        )
        engine.model = object()
        engine.device_fallback_reason = None
        engine._initialize_on_device = Mock(
            side_effect=[RuntimeError("GPU out of memory"), None]
        )

        engine._load_on_preferred_device(Path("model.ckpt"), object(), "cuda")

        self.assertEqual(
            [call.args[2].type for call in engine._initialize_on_device.call_args_list],
            ["cuda", "cpu"],
        )
        empty_cache.assert_called_once_with()
        self.assertEqual(
            engine.device_fallback_reason,
            "RuntimeError: GPU out of memory",
        )

    def test_defect_localization_focuses_high_anomaly_core_for_display_only(self):
        y, x = np.ogrid[-1:1:101j, -1:1:101j]
        normalized = np.exp(-4 * (x * x + y * y)).astype(np.float32)
        broad_mask = normalized > 0.5

        focused_mask, focused_threshold = _defect_localization_mask(normalized)

        self.assertIsNotNone(focused_threshold)
        self.assertGreater(focused_threshold, 0.5)
        self.assertGreater(int(focused_mask.sum()), 0)
        self.assertLess(int(focused_mask.sum()), int(broad_mask.sum()))
        self.assertEqual(focused_mask[50, 50], 1)

    def test_defect_localization_uses_second_adaptive_core_separation(self):
        normalized = np.concatenate(
            [
                np.full(100, 0.55, dtype=np.float32),
                np.full(60, 0.72, dtype=np.float32),
                np.full(20, 0.88, dtype=np.float32),
                np.array([1.0], dtype=np.float32),
            ]
        ).reshape(1, -1)

        focused_mask, focused_threshold = _defect_localization_mask(normalized)

        self.assertIsNotNone(focused_threshold)
        self.assertGreater(focused_threshold, 0.72)
        self.assertEqual(int(focused_mask.sum()), 21)
        self.assertEqual(focused_mask[0, -1], 1)

    def test_defect_localization_preserves_peak_when_scores_quantize_together(self):
        normalized = np.full((4, 4), 0.75, dtype=np.float32)
        normalized[2, 3] = 1.0

        focused_mask, _ = _defect_localization_mask(normalized)

        self.assertEqual(focused_mask[2, 3], 1)

    def test_pass_display_suppresses_defect_contour(self):
        y, x = np.ogrid[-1:1:101j, -1:1:101j]
        anomaly_map = np.exp(-4 * (x * x + y * y)).astype(np.float32)
        model_input = Image.new("RGB", (101, 101), (128, 128, 128))

        _, pass_overlay, _ = _display_artifacts(
            anomaly_map,
            model_input,
            (0.0, 1.0),
            show_defect_contours=False,
        )
        _, fail_overlay, _ = _display_artifacts(
            anomaly_map,
            model_input,
            (0.0, 1.0),
            show_defect_contours=True,
        )
        decode = lambda value: np.asarray(
            Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[1])))
        ).astype(np.int16)
        pass_rgb = decode(pass_overlay)
        fail_rgb = decode(fail_overlay)

        self.assertLess(int(np.max(pass_rgb[..., 0] - pass_rgb[..., 1])), 20)
        self.assertGreater(int(np.max(fail_rgb[..., 0] - fail_rgb[..., 1])), 80)

    def test_defect_localization_is_empty_below_checkpoint_pixel_cutoff(self):
        normalized = np.full((8, 8), 0.49, dtype=np.float32)

        focused_mask, focused_threshold = _defect_localization_mask(normalized)

        self.assertIsNone(focused_threshold)
        self.assertEqual(int(focused_mask.sum()), 0)

    def test_original_to_preprocessor_to_patchcore_flow(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary), threshold=7.0)
            manifest = LocalModelManifest.load(folder)
            preprocessor = _FakePreprocessor()
            engine = _FakeEngine()
            runtime = LocalPatchCoreRuntime(
                manifest,
                preprocessor_factory=lambda _: preprocessor,
                engine_factory=lambda _: engine,
            )
            source = Image.new("RGB", (32, 48), "white")
            stream = io.BytesIO()
            source.save(stream, format="PNG")

            result = runtime.predict(stream.getvalue(), "G01")

            self.assertEqual(preprocessor.received_size, (32, 48))
            self.assertEqual(engine.received_size, (20, 12))
            self.assertEqual(result["raw_image_score"], 7.25)
            self.assertEqual(result["model_display_name"], "Example production model")
            self.assertEqual(result["anomaly_map_shape"], [4, 6])
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["decision"], "FAIL")
            self.assertEqual(result["image_threshold"], 7.0)
            self.assertEqual(
                result["threshold_rule"], "fail_if_score_greater_than_or_equal"
            )
            self.assertTrue(
                result["display_contract"].startswith(
                    "checkpoint_pixel_minmax_display_only_never_used_for_decision"
                )
            )
            self.assertIn(
                "defect_localization=display_only_nested_otsu",
                result["display_contract"],
            )
            self.assertAlmostEqual(
                result["quality_score_percentage"],
                100 - 30 * 7.25 / 7.0,
            )
            self.assertEqual(result["quality_failure_boundary_percentage"], 70.0)
            self.assertEqual(
                result["quality_score_contract"],
                "relative_quality_zero_raw_is_100_threshold_is_70",
            )
            self.assertTrue(result["heatmap_image"].startswith("data:image/jpeg;base64,"))
            self.assertTrue(
                result["defect_overlay_image"].startswith("data:image/jpeg;base64,")
            )
            self.assertTrue(result["segmentation_image"].startswith("data:image/png;base64,"))

            runtime.close()
            self.assertTrue(engine.closed)

    def test_score_equal_to_threshold_is_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary), threshold=60)
            manifest = LocalModelManifest.load(folder)
            runtime = LocalPatchCoreRuntime(
                manifest,
                preprocessor_factory=lambda _: _FakePreprocessor(),
                engine_factory=lambda _: _FakeEngine(score=60),
            )
            stream = io.BytesIO()
            Image.new("RGB", (32, 48), "white").save(stream, format="PNG")
            result = runtime.predict(stream.getvalue(), "G01")
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["quality_score_percentage"], 70.0)

    def test_score_below_threshold_is_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary), threshold=60)
            manifest = LocalModelManifest.load(folder)
            runtime = LocalPatchCoreRuntime(
                manifest,
                preprocessor_factory=lambda _: _FakePreprocessor(),
                engine_factory=lambda _: _FakeEngine(score=59.999),
            )
            stream = io.BytesIO()
            Image.new("RGB", (32, 48), "white").save(stream, format="PNG")
            self.assertEqual(runtime.predict(stream.getvalue(), "G01")["status"], "PASS")

    def test_quality_percentage_matches_but_does_not_make_raw_score_decision(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary), threshold=60)
            manifest = LocalModelManifest.load(folder)
            runtime = LocalPatchCoreRuntime(
                manifest,
                preprocessor_factory=lambda _: _FakePreprocessor(),
                engine_factory=lambda _: _FakeEngine(
                    score=55,
                    pixel_display_bounds=(0, 100),
                ),
            )
            stream = io.BytesIO()
            Image.new("RGB", (32, 48), "white").save(stream, format="PNG")

            result = runtime.predict(stream.getvalue(), "G01")

            self.assertAlmostEqual(result["quality_score_percentage"], 72.5)
            self.assertEqual(result["quality_failure_boundary_percentage"], 70.0)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["raw_image_score"], 55.0)
            self.assertEqual(result["image_threshold"], 60.0)
            self.assertEqual(
                result["decision_contract"], "configured_raw_patchcore_image_score"
            )

    def test_already_preprocessed_image_size_is_wrong_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary))
            preprocessor = _FakePreprocessor()
            manager = JerryScanModelManager(
                folder,
                runtime_factory=lambda manifest: LocalPatchCoreRuntime(
                    manifest,
                    preprocessor_factory=lambda _: preprocessor,
                    engine_factory=lambda _: _FakeEngine(),
                ),
            )
            manager.load_selected()
            stream = io.BytesIO()
            Image.new("RGB", (1024, 1024), "white").save(stream, format="PNG")
            result = manager.inspect("G01", stream.getvalue())
            self.assertEqual(result["status"], "WRONG_INPUT")
            self.assertEqual(result["error"]["code"], "image_size_mismatch")
            self.assertIsNone(preprocessor.received_size)

    def test_health_reports_selected_inference_device(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary))
            manager = JerryScanModelManager(
                folder,
                runtime_factory=lambda manifest: LocalPatchCoreRuntime(
                    manifest,
                    preprocessor_factory=lambda _: _FakePreprocessor(),
                    engine_factory=lambda _: _FakeEngine(),
                ),
            )
            manager.load_selected()

            health = manager.health()

            self.assertEqual(health["inference_device"], "cpu")
            self.assertIsNone(health["device_fallback_reason"])
            self.assertEqual(health["display_name"], "Example production model")

    def test_default_device_policy_assigns_every_angle_to_cpu(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {}, clear=True
        ):
            folder = _write_folder(Path(temporary))
            _upgrade_to_multi_angle(folder, count=3)
            manifest = LocalModelManifest.load(folder)
            requested = []

            def factory(checkpoint, *, preferred_device):
                requested.append((checkpoint.stem, preferred_device))
                return _FakeEngine(device=preferred_device)

            runtime = LocalPatchCoreRuntime(
                manifest,
                preprocessor_factory=lambda _: _FakePreprocessor(),
                engine_factory=factory,
            )

            self.assertEqual(
                requested, [("G01", "cpu"), ("G02", "cpu"), ("G03", "cpu")]
            )
            self.assertEqual(runtime.gpu_model_count, 0)
            self.assertEqual(runtime.cpu_inference_concurrency, 1)

    def test_gpu_assignment_follows_manifest_order_and_skips_unavailable_angle(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary))
            _upgrade_to_multi_angle(folder, count=4)
            (folder / "G02.ckpt").unlink()
            manifest = LocalModelManifest.load(folder)
            requested = []

            def factory(checkpoint, *, preferred_device):
                requested.append((checkpoint.stem, preferred_device))
                return _FakeEngine(device=preferred_device)

            runtime = LocalPatchCoreRuntime(
                manifest,
                preprocessor_factory=lambda _: _FakePreprocessor(),
                engine_factory=factory,
                gpu_model_count=2,
            )

            self.assertEqual(
                requested, [("G01", "cuda"), ("G03", "cuda"), ("G04", "cpu")]
            )
            self.assertEqual(runtime.available_angles, ("G01", "G03", "G04"))
            self.assertIn("G02", runtime.unavailable_angles)

    def test_invalid_device_environment_values_fail_startup(self):
        manifest = LocalModelManifest.load(
            PROJECT_ROOT
            / "models"
            / "Patchcore_rembg_u2net_black_v1_256_c10_seed42",
            require_artifacts=False,
        )
        cases = {
            "JERRYSCAN_GPU_MODEL_COUNT": "-1",
            "JERRYSCAN_CPU_INFERENCE_CONCURRENCY": "0",
            "JERRYSCAN_GPU_INFERENCE_CONCURRENCY": "many",
        }
        for name, value in cases.items():
            with self.subTest(name=name), patch.dict(
                os.environ, {name: value}, clear=True
            ):
                with self.assertRaisesRegex(ModelFolderError, name):
                    LocalPatchCoreRuntime(
                        manifest,
                        preprocessor_factory=lambda _: _FakePreprocessor(),
                        engine_factory=lambda _: _FakeEngine(),
                    )

    def test_cuda_unavailable_falls_back_selected_engine_to_cpu(self):
        engine = RawPatchCoreEngine.__new__(RawPatchCoreEngine)
        cpu = SimpleNamespace(type="cpu")
        engine.torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            device=lambda name: cpu,
        )
        engine.model = None
        engine.device_fallback_reason = None
        engine._initialize_on_device = Mock()

        engine._load_on_preferred_device(Path("model.ckpt"), object(), "cuda")

        engine._initialize_on_device.assert_called_once_with(
            Path("model.ckpt"), ANY, cpu
        )
        self.assertEqual(
            engine.device_fallback_reason,
            "CUDA was requested but is not available",
        )

    def test_health_reports_requested_actual_devices_and_per_angle_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary))
            _upgrade_to_multi_angle(folder, count=2)

            def runtime_factory(manifest):
                def engine_factory(_, *, preferred_device):
                    engine = _FakeEngine(device="cpu")
                    if preferred_device == "cuda":
                        engine.device_fallback_reason = "simulated CUDA OOM"
                    return engine

                return LocalPatchCoreRuntime(
                    manifest,
                    preprocessor_factory=lambda _: _FakePreprocessor(),
                    engine_factory=engine_factory,
                    gpu_model_count=1,
                )

            manager = JerryScanModelManager(folder, runtime_factory=runtime_factory)
            manager.load_selected()
            health = manager.health()

            self.assertEqual(
                health["requested_inference_devices"],
                {"G01": "cuda", "G02": "cpu"},
            )
            self.assertEqual(
                health["inference_devices"], {"G01": "cpu", "G02": "cpu"}
            )
            self.assertEqual(
                health["device_fallback_reasons"]["G01"], "simulated CUDA OOM"
            )
            self.assertEqual(health["device_policy"]["gpu_model_count"], 1)

    def test_cpu_patchcore_inference_is_serialized_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary))
            _upgrade_to_multi_angle(folder, count=2)
            manifest = LocalModelManifest.load(folder)
            state = {"active": 0, "maximum": 0}
            state_lock = threading.Lock()

            class _MeasuredCpuEngine(_FakeEngine):
                def predict(self, image):
                    with state_lock:
                        state["active"] += 1
                        state["maximum"] = max(state["maximum"], state["active"])
                    time.sleep(0.05)
                    try:
                        return super().predict(image)
                    finally:
                        with state_lock:
                            state["active"] -= 1

            runtime = LocalPatchCoreRuntime(
                manifest,
                preprocessor_factory=lambda _: _FakePreprocessor(),
                engine_factory=lambda _, **__: _MeasuredCpuEngine(device="cpu"),
                cpu_inference_concurrency=1,
            )
            stream = io.BytesIO()
            Image.new("RGB", (32, 48), "white").save(stream, format="PNG")
            image_bytes = stream.getvalue()
            threads = [
                threading.Thread(target=runtime.predict, args=(image_bytes, angle))
                for angle in runtime.available_angles
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2)

            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(state["maximum"], 1)


class _BlockingRuntime:
    def __init__(self, manifest, *, blocked: bool):
        self.manifest = manifest
        self.model_id = manifest.model_id
        self.preprocessing_id = manifest.preprocessing_id
        self.angle = manifest.angle
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = False
        self.blocked = blocked

    def predict(self, image_bytes, angle):
        self.started.set()
        if self.blocked:
            self.release.wait(timeout=5)
        return {"status": "PASS", "model_id": self.model_id}

    def close(self):
        self.closed = True


class ManagerSynchronizationTests(unittest.TestCase):
    def test_reload_waits_for_inflight_inference_before_closing_old_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary))
            runtimes = []

            def factory(manifest):
                runtime = _BlockingRuntime(manifest, blocked=not runtimes)
                runtimes.append(runtime)
                return runtime

            manager = JerryScanModelManager(folder, runtime_factory=factory)
            manager.load_selected()
            old = runtimes[0]
            inspection = threading.Thread(
                target=lambda: manager.inspect("G01", b"image"), daemon=True
            )
            inspection.start()
            self.assertTrue(old.started.wait(timeout=1))

            reload_thread = threading.Thread(target=manager.load_selected, daemon=True)
            reload_thread.start()
            deadline = time.monotonic() + 1
            while len(runtimes) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(len(runtimes), 2)
            time.sleep(0.05)
            self.assertTrue(reload_thread.is_alive())
            self.assertFalse(old.closed)

            old.release.set()
            inspection.join(timeout=1)
            reload_thread.join(timeout=1)
            self.assertFalse(inspection.is_alive())
            self.assertFalse(reload_thread.is_alive())
            self.assertTrue(old.closed)


if __name__ == "__main__":
    unittest.main()
