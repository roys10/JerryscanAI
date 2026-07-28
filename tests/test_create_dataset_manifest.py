from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from training.datasets.create_dataset_manifest import (
    build_rows,
    main,
    materialize,
    parse_capture_filename,
    sha256_manifest,
)


class DatasetManifestTests(unittest.TestCase):
    def test_manifest_hash_is_stable_across_line_endings(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            lf = root / "lf.csv"
            crlf = root / "crlf.csv"
            lf.write_bytes(b"sample_id,split\nG01-001,train\n")
            crlf.write_bytes(b"sample_id,split\r\nG01-001,train\r\n")

            self.assertEqual(sha256_manifest(lf), sha256_manifest(crlf))

    def test_parse_capture_filename(self):
        angle, captured_at, sequence = parse_capture_filename(
            Path("G01-260203-091625-674.bmp")
        )
        self.assertEqual(angle, "G01")
        self.assertEqual(captured_at.isoformat(), "2026-02-03T09:16:25")
        self.assertEqual(sequence, 674)

    def test_build_rows_preserves_day_groups(self):
        with TemporaryDirectory() as directory:
            source = Path(directory)
            names = [
                "G01-260201-120000-001.bmp",
                "G01-260202-120000-002.bmp",
                "G01-260203-120000-003.bmp",
            ]
            for name in names:
                Image.new("L", (8, 10), color=128).save(source / name)

            rows = build_rows(
                source,
                {
                    "2026-02-01": "train",
                    "2026-02-02": "val",
                    "2026-02-03": "test",
                },
                label="unverified",
                include_hash=False,
                configured_date_flags={"2026-02-03": ["isolated_capture"]},
            )

            self.assertEqual([row.split for row in rows], ["train", "val", "test"])
            self.assertEqual(rows[0].quality_flags, "label_unverified")
            self.assertEqual(rows[2].quality_flags, "isolated_capture;label_unverified")
            self.assertTrue(all((row.width, row.height, row.channels) == (8, 10, 1) for row in rows))

    def test_duplicate_hashes_are_rejected(self):
        with TemporaryDirectory() as directory:
            source = Path(directory)
            for name in (
                "G01-260201-120000-001.bmp",
                "G01-260201-120001-002.bmp",
            ):
                Image.new("L", (8, 10), color=128).save(source / name)

            with self.assertRaisesRegex(ValueError, "Byte-exact duplicate"):
                build_rows(
                    source,
                    {"2026-02-01": "train"},
                    label="normal",
                    include_hash=True,
                )

    def test_cli_rejects_quality_flag_for_absent_date(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            for day in ("01", "02", "03"):
                Image.new("L", (8, 10), color=128).save(
                    source / f"G01-2602{day}-120000-001.bmp"
                )
            manifest = Path(directory) / "split.csv"
            argv = [
                "create_dataset_manifest.py",
                "--source",
                str(source),
                "--manifest",
                str(manifest),
                "--train-dates",
                "2026-02-01",
                "--val-dates",
                "2026-02-02",
                "--test-dates",
                "2026-02-03",
                "--date-flag",
                "2026-02-04=isolated_capture",
            ]
            with patch("sys.argv", argv):
                with self.assertRaisesRegex(SystemExit, "Quality-flag dates"):
                    main()
            self.assertFalse(manifest.exists())

    def test_materialization_is_promoted_from_temporary_directory(self):
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
                include_hash=False,
            )
            output = root / "materialized"

            materialize(source, output, rows, mode="hardlink")

            self.assertTrue(
                (output / "train" / "normal" / "G01-260201-120000-001.bmp").is_file()
            )
            self.assertFalse((root / ".materialized.partial").exists())


if __name__ == "__main__":
    unittest.main()
