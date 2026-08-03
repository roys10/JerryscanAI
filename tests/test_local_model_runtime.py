from __future__ import annotations

import io
import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from backend.inference.local_model import (
    LocalModelManifest,
    LocalPatchCoreRuntime,
    ModelFolderError,
)
from backend.inference.manager import JerryScanModelManager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_BYTES = b"checkpoint"
WEIGHT_BYTES = b"u2net-weight"


def _document(model_id: str, *, threshold=None, rembg: bool = False) -> dict:
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
        "exploratory_threshold": threshold,
        "max_image_bytes": 1024 * 1024,
    }


def _write_folder(
    root: Path,
    *,
    with_checkpoint: bool = True,
    threshold=None,
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


class ManifestTests(unittest.TestCase):
    def test_all_four_tracked_preprocessing_contracts(self):
        expected = {
            "Patchcore_raw_letterbox_v1_256_c10_seed42": ("raw_letterbox", 128),
            "Patchcore_fixed_crop_v1_256_c10_seed42": ("fixed_crop", 128),
            "Patchcore_rembg_u2net_gray_v1_256_c10_seed42": ("rembg", 128),
            "Patchcore_rembg_u2net_black_v1_256_c10_seed42": ("rembg", 0),
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
                    contract,
                )
                self.assertIsNone(manifest.exploratory_threshold)
                self.assertEqual(
                    (manifest.original_width, manifest.original_height), (1025, 1281)
                )
                self.assertEqual(len(manifest.checkpoint_sha256), 64)

    def test_artifact_and_metadata_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = _write_folder(Path(temporary))
            manifest = LocalModelManifest.load(folder)
            self.assertEqual(manifest.model_id, folder.name)
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
            result = manager.inspect("G01", b"not used")
            self.assertEqual(result["status"], "REVIEW")
            self.assertEqual(result["decision"], "UNDECIDED")
            self.assertEqual(result["error"]["code"], "model_folder_not_ready")


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
    def __init__(self):
        self.received_size = None
        self.closed = False

    def predict(self, image):
        self.received_size = image.size
        return 7.25, np.arange(24, dtype=np.float32).reshape(4, 6), 3.5

    def close(self):
        self.closed = True


class RuntimeTests(unittest.TestCase):
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
            self.assertEqual(result["anomaly_map_shape"], [4, 6])
            self.assertEqual(result["status"], "SHADOW")
            self.assertEqual(result["decision"], "UNDECIDED")
            self.assertEqual(result["exploratory_decision"], "EXPLORATORY_FAULT")
            self.assertTrue(result["heatmap_image"].startswith("data:image/jpeg;base64,"))
            self.assertTrue(result["segmentation_image"].startswith("data:image/png;base64,"))

            runtime.close()
            self.assertTrue(engine.closed)

    def test_already_preprocessed_image_size_is_rejected_for_review(self):
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
            self.assertEqual(result["status"], "REVIEW")
            self.assertEqual(result["error"]["code"], "image_size_mismatch")
            self.assertIsNone(preprocessor.received_size)


class _BlockingRuntime:
    def __init__(self, manifest, *, blocked: bool):
        self.manifest = manifest
        self.model_id = manifest.model_id
        self.preprocessing_id = manifest.preprocessing_id
        self.angle = manifest.angle
        self.mode = "shadow"
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = False
        self.blocked = blocked

    def predict(self, image_bytes, angle):
        self.started.set()
        if self.blocked:
            self.release.wait(timeout=5)
        return {"status": "SHADOW", "model_id": self.model_id}

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
