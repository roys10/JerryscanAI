from __future__ import annotations

import hashlib
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from training.datasets.create_dataset_manifest import sha256_file, sha256_manifest
from training.datasets.materialize_dataset_split import load_manifest_rows
from training.preprocessing.runtime import canonical_json_hash


SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


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


def scan_exploratory_folder(
    source_root: Path, *, camera_angle: str, label_mode: str
) -> tuple[list[dict[str, Any]], str]:
    root = source_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Original-image folder does not exist: {root}")
    if label_mode not in {"unlabeled", "verified_normal"}:
        raise ValueError("Label mode must be unlabeled or verified_normal")
    candidates = sorted(
        (
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    if not candidates:
        raise ValueError("Original-image folder contains no supported images")
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_resolved: set[Path] = set()
    seen_ids: set[str] = set()
    for index, candidate in enumerate(candidates, start=1):
        relative = candidate.relative_to(root).as_posix()
        resolved = safe_source_path(root, relative)
        path_key = relative.casefold()
        if path_key in seen_paths or resolved in seen_resolved:
            raise ValueError(f"Duplicate image entry in folder: {relative}")
        seen_paths.add(path_key)
        seen_resolved.add(resolved)
        sample_id = "folder-" + hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20]
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate stable sample ID for: {relative}")
        seen_ids.add(sample_id)
        rows.append(
            {
                "schema_version": "1.0",
                "sample_id": sample_id,
                "source_relpath": relative,
                "source_sha256": sha256_file(resolved),
                "camera_angle": camera_angle,
                "split": "exploratory",
                "label": "normal" if label_mode == "verified_normal" else "unlabeled",
                "sequence_no": index,
                "quality_flags": "",
                "notes": "exploratory folder snapshot; folder names were not used as labels",
            }
        )
    snapshot_hash = canonical_json_hash(
        {"schema_version": "1.0", "mode": "exploratory_folder", "samples": rows}
    )
    return rows, snapshot_hash


def select_exploratory_samples(
    rows: list[dict[str, Any]], *, count: int | None, seed: int
) -> list[dict[str, Any]]:
    if count is None:
        return list(rows)
    if count <= 0:
        raise ValueError("Image count must be positive or All")
    if count > len(rows):
        raise ValueError(f"Requested {count} images but folder contains {len(rows)}")
    selected = random.Random(seed).sample(rows, count)
    return sorted(selected, key=lambda row: row["sample_id"])
