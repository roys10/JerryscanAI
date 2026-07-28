"""Materialize an existing frozen manifest without rewriting it."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from training.datasets.create_dataset_manifest import (
    FIELDNAMES,
    ManifestRow,
    materialize,
    sha256_file,
    sha256_manifest,
)


def load_manifest_rows(manifest: Path) -> list[ManifestRow]:
    if not manifest.is_file():
        raise ValueError(f"Manifest does not exist: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = set(FIELDNAMES) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        rows = []
        for raw in reader:
            values = {field: raw[field] for field in FIELDNAMES}
            for field in ("sequence_no", "width", "height", "channels"):
                values[field] = int(values[field])
            rows.append(ManifestRow(**values))
    if not rows:
        raise ValueError(f"Manifest has no samples: {manifest}")
    if len({row.sample_id for row in rows}) != len(rows):
        raise ValueError("Manifest contains duplicate sample IDs")
    return rows


def verify_sources(source: Path, rows: list[ManifestRow]) -> None:
    for index, row in enumerate(rows, start=1):
        path = source / Path(row.source_relpath)
        if not path.is_file():
            raise ValueError(f"Source image does not exist: {path}")
        if path.stem != row.sample_id:
            raise ValueError(
                f"Source sample ID mismatch: {path.stem} != {row.sample_id}"
            )
        if not row.source_sha256:
            raise ValueError(f"Manifest has no source hash for {row.sample_id}")
        actual_hash = sha256_file(path)
        if actual_hash != row.source_sha256:
            raise ValueError(f"Source hash mismatch for {row.sample_id}")
        if index % 500 == 0:
            print(f"Verified {index}/{len(rows)} source files")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify source hashes and materialize an existing split manifest."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("hardlink", "copy"), default="hardlink")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = args.source.resolve()
    manifest = args.manifest.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        print(f"Source directory does not exist: {source}")
        return 1
    try:
        rows = load_manifest_rows(manifest)
        verify_sources(source, rows)
        materialize(source, output, rows, args.mode)
    except (OSError, ValueError) as exc:
        print(f"Materialization failed: {exc}")
        return 1

    counts = Counter(row.split for row in rows)
    print(f"Manifest SHA-256: {sha256_manifest(manifest)}")
    print(f"Output: {output}")
    print(f"Train: {counts['train']}; val: {counts['val']}; test: {counts['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
