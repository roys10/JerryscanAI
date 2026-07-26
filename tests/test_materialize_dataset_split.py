from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from training.datasets.create_dataset_manifest import build_rows, write_manifest
from training.datasets.materialize_dataset_split import (
    load_manifest_rows,
    verify_sources,
)


class MaterializeExistingManifestTests(unittest.TestCase):
    def test_existing_manifest_is_loaded_and_source_hashes_are_verified(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            Image.new("L", (8, 10), color=128).save(
                source / "G01-260201-120000-001.bmp"
            )
            rows = build_rows(
                source,
                {"2026-02-01": "train"},
                label="normal",
                include_hash=True,
            )
            manifest = root / "split.csv"
            write_manifest(manifest, rows, overwrite=False)

            loaded = load_manifest_rows(manifest)
            verify_sources(source, loaded)

            self.assertEqual(loaded, rows)

    def test_source_hash_mismatch_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            image_path = source / "G01-260201-120000-001.bmp"
            Image.new("L", (8, 10), color=128).save(image_path)
            rows = build_rows(
                source,
                {"2026-02-01": "train"},
                label="normal",
                include_hash=True,
            )
            manifest = root / "split.csv"
            write_manifest(manifest, rows, overwrite=False)
            Image.new("L", (8, 10), color=64).save(image_path)

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_sources(source, load_manifest_rows(manifest))


if __name__ == "__main__":
    unittest.main()
