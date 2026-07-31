from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from training.datasets.create_dataset_manifest import sha256_manifest
from training.datasets.materialize_dataset_split import load_manifest_rows


def safe_source_path(source_root: Path, source_relpath: str) -> Path:
    root = source_root.expanduser().resolve()
    relative = Path(source_relpath)
    if relative.is_absolute():
        raise ValueError("Manifest source path must be relative")
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError("Manifest source path escapes the original-image root")
    return path


def inspect_dataset(source_root: Path, manifest: Path) -> dict[str, Any]:
    source_root, manifest = source_root.resolve(), manifest.resolve()
    if not source_root.is_dir():
        raise ValueError(f"Original-image root does not exist: {source_root}")
    rows = load_manifest_rows(manifest)
    splits: dict[str, int] = {}
    labels: dict[str, int] = {}
    missing = 0
    for row in rows:
        splits[row.split] = splits.get(row.split, 0) + 1
        labels[row.label] = labels.get(row.label, 0) + 1
        missing += not safe_source_path(source_root, row.source_relpath).is_file()
    return {
        "source_root": str(source_root),
        "manifest": str(manifest),
        "manifest_sha256": sha256_manifest(manifest),
        "sample_count": len(rows),
        "split_counts": splits,
        "label_counts": labels,
        "missing_source_count": missing,
    }


def select_samples(
    manifest: Path, *, split: str, count: int, seed: int
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("Image count must be positive")
    rows = sorted(
        (row for row in load_manifest_rows(manifest) if row.split == split),
        key=lambda row: row.sample_id,
    )
    if not rows:
        raise ValueError(f"Manifest has no samples in split: {split}")
    if count > len(rows):
        raise ValueError(f"Requested {count} images but split {split} contains {len(rows)}")
    selected = random.Random(seed).sample(rows, count)
    return [asdict(row) for row in sorted(selected, key=lambda row: row.sample_id)]
