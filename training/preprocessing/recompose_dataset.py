"""Create a background-fill variant from an audited segmentation derivative.

This deliberately never runs a segmentation model: it verifies the completed
parent derivative, then recomposites its aligned RGB images using its aligned
binary masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from training.datasets.create_dataset_manifest import sha256_file, sha256_manifest
from training.datasets.materialize_dataset_split import load_manifest_rows
from training.preprocessing.preprocess_dataset import (
    DERIVATIVE_FIELDS,
    canonical_json_hash,
    read_derivative_manifest,
    write_derivative_manifest,
)


def _require(value: str, description: str) -> str:
    if not value:
        raise ValueError(f"Missing {description}")
    return value


def verify_parent(parent_root: Path, manifest: Path, config: dict[str, Any]) -> list[dict[str, str]]:
    """Verify parent metadata, split identities, and every parent image/mask hash."""
    parent_config_path = parent_root / "preprocessing_config.json"
    parent_summary_path = parent_root / "summary.json"
    parent_manifest_path = parent_root / "derivative_manifest.csv"
    if not all(path.is_file() for path in (parent_config_path, parent_summary_path, parent_manifest_path)):
        raise ValueError("Parent derivative is missing required contract files")
    parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
    parent_summary = json.loads(parent_summary_path.read_text(encoding="utf-8"))
    expected_parent_id = str(config["parent_preprocessing_id"])
    expected_parent_config_hash = str(config["parent_config_sha256"])
    if parent_config.get("preprocessing_id") != expected_parent_id:
        raise ValueError("Parent preprocessing ID does not match config")
    if canonical_json_hash(parent_config) != expected_parent_config_hash:
        raise ValueError("Parent preprocessing config hash does not match config")
    if parent_summary.get("preprocessing_id") != expected_parent_id or parent_summary.get("config_sha256") != expected_parent_config_hash:
        raise ValueError("Parent summary metadata does not match parent config")
    frozen_rows = {row.sample_id: row for row in load_manifest_rows(manifest)}
    parent_manifest_hash = sha256_manifest(manifest)
    rows = read_derivative_manifest(parent_manifest_path)
    if len(rows) != len(frozen_rows) or len({row.get("parent_sample_id") for row in rows}) != len(rows):
        raise ValueError("Parent derivative does not contain exactly the frozen sample identities")
    for index, row in enumerate(rows, start=1):
        sample_id = _require(row.get("parent_sample_id", ""), "parent sample ID")
        frozen = frozen_rows.get(sample_id)
        if frozen is None or row.get("split") != frozen.split or row.get("label") != frozen.label:
            raise ValueError(f"Parent split identity mismatch for {sample_id}")
        if row.get("source_sha256") != frozen.source_sha256 or row.get("parent_manifest_sha256") != parent_manifest_hash:
            raise ValueError(f"Parent frozen manifest hash mismatch for {sample_id}")
        if row.get("preprocessing_id") != expected_parent_id or row.get("config_sha256") != expected_parent_config_hash or row.get("status") != "ok":
            raise ValueError(f"Parent derivative metadata/status mismatch for {sample_id}")
        for rel_key, hash_key in (("output_relpath", "output_sha256"), ("mask_relpath", "mask_sha256")):
            relpath, expected_hash = _require(row.get(rel_key, ""), rel_key), _require(row.get(hash_key, ""), hash_key)
            path = parent_root / relpath
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise ValueError(f"Parent {rel_key} hash mismatch for {sample_id}")
        if index % 500 == 0:
            print(f"Verified parent {index}/{len(rows)}", flush=True)
    return rows


def atomic_copy_or_link(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if mode == "hardlink":
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
    else:
        shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def recompose(parent_image: Path, parent_mask: Path, background_value: int) -> Image.Image:
    with Image.open(parent_image) as source, Image.open(parent_mask) as mask_source:
        image = source.convert("RGB")
        mask = mask_source.convert("L")
        if image.size != mask.size:
            raise ValueError(f"Parent image/mask geometry mismatch: {image.size} != {mask.size}")
        pixels = np.asarray(image).copy()
        binary = np.asarray(mask) >= 128
        pixels[~binary] = background_value
        return Image.fromarray(pixels, mode="RGB")


def save_png_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="PNG", optimize=False)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompose an audited segmentation derivative using its masks.")
    parser.add_argument("--parent-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    parent_root, manifest, config_path, output_root = (args.parent_root.resolve(), args.manifest.resolve(), args.config.resolve(), args.output_root.resolve())
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        preprocessing_id = str(config["preprocessing_id"])
        if not preprocessing_id or Path(preprocessing_id).name != preprocessing_id:
            raise ValueError("preprocessing_id must be one safe folder name")
        parent_rows = verify_parent(parent_root, manifest, config)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Parent verification failed: {exc}")
        return 1
    final_output, partial_output = output_root / preprocessing_id, output_root / f".{preprocessing_id}.partial"
    if final_output.exists():
        print(f"Final output already exists: {final_output}")
        return 1
    config_hash = canonical_json_hash(config)
    if partial_output.exists() and not args.resume:
        print(f"Partial output exists; inspect it or pass --resume: {partial_output}")
        return 1
    partial_output.mkdir(parents=True, exist_ok=True)
    saved_config = partial_output / "preprocessing_config.json"
    if saved_config.exists() and canonical_json_hash(json.loads(saved_config.read_text(encoding="utf-8"))) != config_hash:
        print("Cannot resume: preprocessing config does not match partial dataset")
        return 1
    saved_config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    checkpoint = partial_output / "derivative_manifest.partial.csv"
    complete = {row["parent_sample_id"]: row for row in read_derivative_manifest(checkpoint) if row.get("status") == "ok"}
    rows: list[dict[str, Any]] = []
    for index, parent in enumerate(parent_rows, start=1):
        sample_id, output_relpath, mask_relpath = parent["parent_sample_id"], parent["output_relpath"], parent["mask_relpath"]
        output_path, mask_path = partial_output / output_relpath, partial_output / mask_relpath
        old = complete.get(sample_id)
        if old and output_path.is_file() and mask_path.is_file() and sha256_file(output_path) == old.get("output_sha256") and sha256_file(mask_path) == old.get("mask_sha256"):
            rows.append(old); continue
        started = time.perf_counter()
        row = dict(parent)
        row.update({"schema_version": "1.0", "preprocessing_id": preprocessing_id, "config_sha256": config_hash, "backend": "aligned_mask_recompose", "backend_version": "1", "output_relpath": output_relpath, "output_sha256": "", "mask_relpath": mask_relpath, "mask_sha256": "", "status": "error", "error": ""})
        try:
            output = recompose(parent_root / output_relpath, parent_root / mask_relpath, int(config["background_value"]))
            save_png_atomic(output, output_path)
            atomic_copy_or_link(parent_root / mask_relpath, mask_path, str(config.get("mask_materialization", "hardlink")))
            row.update({"status": "ok", "output_sha256": sha256_file(output_path), "mask_sha256": sha256_file(mask_path), "width": output.width, "height": output.height, "channels": 3, "processing_ms": round((time.perf_counter() - started) * 1000, 3), "quality_flags": ""})
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if index % 100 == 0:
            write_derivative_manifest(checkpoint, rows); print(f"Recomposited {index}/{len(parent_rows)}", flush=True)
    write_derivative_manifest(checkpoint, rows); write_derivative_manifest(partial_output / "derivative_manifest.csv", rows)
    statuses = Counter(row["status"] for row in rows)
    summary = {"preprocessing_id": preprocessing_id, "parent_preprocessing_id": config["parent_preprocessing_id"], "parent_manifest_sha256": sha256_manifest(manifest), "parent_config_sha256": config["parent_config_sha256"], "config_sha256": config_hash, "sample_count": len(rows), "split_counts": dict(Counter(row["split"] for row in rows)), "status_counts": dict(statuses), "quality_flag_counts": {}}
    (partial_output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if statuses.get("error", 0):
        print(f"Recomposition failed for {statuses['error']} samples: {partial_output}"); return 1
    checkpoint.unlink(missing_ok=True); partial_output.replace(final_output)
    print(f"Created dataset: {final_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
