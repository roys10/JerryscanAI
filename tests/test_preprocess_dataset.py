import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from training.datasets.create_dataset_manifest import build_rows, write_manifest
from training.preprocessing.preprocess_dataset import (
    FixedCropBackend,
    RawLetterboxBackend,
    canonical_json_hash,
    clean_binary_mask,
    resolve_dataset_backend,
)


class PreprocessDatasetTests(unittest.TestCase):
    def test_config_hash_is_key_order_independent(self):
        self.assertEqual(canonical_json_hash({"a": 1, "b": 2}), canonical_json_hash({"b": 2, "a": 1}))

    def test_raw_letterbox_preserves_aspect_ratio(self):
        backend = RawLetterboxBackend(
            {"output_size": [100, 100], "background_value": 128}
        )
        image = Image.new("RGB", (50, 100), (255, 255, 255))

        output, mask, _ = backend.process(image)

        self.assertEqual(output.size, (100, 100))
        self.assertIsNone(mask)
        self.assertEqual(output.getpixel((0, 50)), (128, 128, 128))
        self.assertEqual(output.getpixel((50, 50)), (255, 255, 255))

    def test_mask_cleanup_keeps_largest_component(self):
        import numpy as np

        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:4, 2:4] = 255
        mask[8:18, 8:18] = 255

        cleaned, components = clean_binary_mask(mask, threshold=128, dilation_px=0)

        self.assertEqual(components, 2)
        self.assertEqual(int((cleaned > 0).sum()), 100)

    def test_fixed_crop_rejects_coordinates_outside_image(self):
        backend = FixedCropBackend(
            {
                "crop_xyxy": [0, 0, 60, 100],
                "output_size": [100, 100],
                "background_value": 128,
            }
        )
        with self.assertRaisesRegex(ValueError, "outside image"):
            backend.process(Image.new("RGB", (50, 100)))

    def test_dataset_backend_resolves_black_live_config(self):
        with TemporaryDirectory() as directory:
            config_dir = Path(directory)
            parent = {
                "preprocessing_id": "gray",
                "backend": "rembg",
                "background_value": 128,
            }
            (config_dir / "gray.json").write_text(json.dumps(parent))
            black = {
                "preprocessing_id": "black",
                "backend": "aligned_mask_recompose",
                "parent_preprocessing_id": "gray",
                "parent_config_sha256": canonical_json_hash(parent),
                "background_value": 0,
            }
            black_path = config_dir / "black.json"

            with patch(
                "training.preprocessing.preprocess_dataset.create_backend"
            ) as create_backend:
                resolved, backend = resolve_dataset_backend(black, black_path)

            self.assertIs(backend, create_backend.return_value)
            self.assertEqual(resolved["backend"], "rembg")
            self.assertEqual(resolved["preprocessing_id"], "black")
            self.assertEqual(resolved["background_value"], 0)
            create_backend.assert_called_once_with(resolved)


if __name__ == "__main__":
    unittest.main()
