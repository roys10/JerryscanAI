import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from training.models import train_patchcore
from training.models.train_patchcore import validate_materialized_dataset


class TrainPatchcoreDatasetTests(unittest.TestCase):
    def test_cloud_training_notebook_code_cells_compile(self):
        project_root = Path(train_patchcore.__file__).resolve().parents[2]
        notebook_paths = [
            project_root / "training" / "models" / "train-patchcore_notebook.ipynb"
        ]
        for notebook_path in notebook_paths:
            with self.subTest(notebook=notebook_path.name):
                notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
                code_cells = [
                    cell
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "code"
                ]

                self.assertGreaterEqual(len(code_cells), 5)
                for index, cell in enumerate(code_cells):
                    compile(
                        "".join(cell["source"]),
                        f"{notebook_path.name}:cell{index}",
                        "exec",
                    )

    def test_training_module_resolves_repository_root(self):
        project_root = Path(train_patchcore.__file__).resolve().parents[2]
        self.assertTrue((project_root / "pyproject.toml").is_file())
        self.assertTrue((project_root / "training").is_dir())

    def create_dataset(self, root: Path) -> Path:
        manifest = root / "split.csv"
        rows = []
        for split in ("train", "val", "test"):
            folder = root / "dataset" / split / "normal"
            folder.mkdir(parents=True)
            sample_id = f"G01-{split}-001"
            (folder / f"{sample_id}.png").write_bytes(b"not-decoded-in-dry-run")
            rows.append({"sample_id": sample_id, "split": split, "label": "normal"})
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("sample_id", "split", "label"))
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    def test_materialized_folders_must_match_manifest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.create_dataset(root)

            folders, counts, manifest_hash = validate_materialized_dataset(
                root / "dataset", manifest
            )

            self.assertEqual(counts, {"train": 1, "val": 1, "test": 1})
            self.assertTrue(all(folder.is_dir() for folder in folders.values()))
            self.assertEqual(len(manifest_hash), 64)

    def test_missing_materialized_sample_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.create_dataset(root)
            (root / "dataset" / "val" / "normal" / "G01-val-001.png").unlink()

            with self.assertRaisesRegex(ValueError, "val does not match manifest"):
                validate_materialized_dataset(root / "dataset", manifest)

    def test_unverified_label_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.create_dataset(root)
            text = manifest.read_text(encoding="utf-8").replace("normal", "unverified")
            manifest.write_text(text, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "label=normal"):
                validate_materialized_dataset(root / "dataset", manifest)


if __name__ == "__main__":
    unittest.main()
