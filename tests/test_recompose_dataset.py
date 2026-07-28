import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from training.datasets.create_dataset_manifest import (
    build_rows,
    sha256_file,
    sha256_manifest,
    write_manifest,
)
from training.preprocessing.preprocess_dataset import canonical_json_hash, write_derivative_manifest
from training.preprocessing.recompose_dataset import main, recompose, verify_parent


class RecomposeDatasetTests(unittest.TestCase):
    def _parent(self, root: Path):
        source = root / "source"; source.mkdir()
        raw = source / "G01-260201-120000-001.bmp"
        Image.new("L", (4, 3), 200).save(raw)
        manifest = root / "split.csv"
        frozen = build_rows(source, {"2026-02-01": "train"}, label="normal", include_hash=True)
        write_manifest(manifest, frozen, overwrite=False)
        parent = root / "gray"; (parent / "train" / "normal").mkdir(parents=True); (parent / "masks" / "train" / "normal").mkdir(parents=True)
        image_path = parent / "train" / "normal" / "G01-260201-120000-001.png"
        mask_path = parent / "masks" / "train" / "normal" / "G01-260201-120000-001.png"
        Image.new("RGB", (4, 3), (100, 120, 140)).save(image_path)
        mask = Image.new("L", (4, 3), 0); mask.putpixel((1, 1), 255); mask.save(mask_path)
        parent_config = {"preprocessing_id": "rembg_u2net_gray_v1", "backend": "rembg"}; parent_hash = canonical_json_hash(parent_config)
        (parent / "preprocessing_config.json").write_text(json.dumps(parent_config)); (parent / "summary.json").write_text(json.dumps({"preprocessing_id": "rembg_u2net_gray_v1", "config_sha256": parent_hash}))
        row = {"schema_version": "1.0", "parent_sample_id": frozen[0].sample_id, "split": "train", "label": "normal", "source_relpath": frozen[0].source_relpath, "source_sha256": frozen[0].source_sha256, "parent_manifest_sha256": sha256_manifest(manifest), "preprocessing_id": "rembg_u2net_gray_v1", "config_sha256": parent_hash, "backend": "rembg", "backend_version": "x", "model_name": "u2net", "model_sha256": "x", "output_relpath": "train/normal/G01-260201-120000-001.png", "output_sha256": sha256_file(image_path), "mask_relpath": "masks/train/normal/G01-260201-120000-001.png", "mask_sha256": sha256_file(mask_path), "status": "ok", "width": "4", "height": "3", "channels": "3", "processing_ms": "1", "mask_area_ratio": "", "component_count": "", "model_score": "", "bbox_xyxy": "", "quality_flags": "", "error": ""}
        write_derivative_manifest(parent / "derivative_manifest.csv", [row])
        return manifest, parent, parent_hash

    def test_recompose_replaces_only_masked_background(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest, parent, parent_hash = self._parent(root)
            output = recompose(parent / "train/normal/G01-260201-120000-001.png", parent / "masks/train/normal/G01-260201-120000-001.png", 0)
            self.assertEqual(output.getpixel((1, 1)), (100, 120, 140)); self.assertEqual(output.getpixel((0, 0)), (0, 0, 0))
            self.assertEqual(len(verify_parent(parent, manifest, {"parent_preprocessing_id": "rembg_u2net_gray_v1", "parent_config_sha256": parent_hash})), 1)

    def test_cli_creates_contract_and_copies_mask(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest, parent, parent_hash = self._parent(root)
            config = root / "config.json"; config.write_text(json.dumps({"preprocessing_id": "black", "parent_preprocessing_id": "rembg_u2net_gray_v1", "parent_config_sha256": parent_hash, "background_value": 0, "mask_materialization": "copy"}))
            with patch.object(sys, "argv", ["recompose", "--parent-root", str(parent), "--manifest", str(manifest), "--config", str(config), "--output-root", str(root / "out")]): self.assertEqual(main(), 0)
            result = Image.open(root / "out/black/train/normal/G01-260201-120000-001.png")
            self.assertEqual(result.getpixel((0, 0)), (0, 0, 0)); self.assertTrue((root / "out/black/masks/train/normal/G01-260201-120000-001.png").is_file())
