"""Create a split-preserving preprocessing dataset from the shared runtime."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from training.datasets.create_dataset_manifest import sha256_file, sha256_manifest
from training.datasets.materialize_dataset_split import load_manifest_rows
from training.preprocessing.runtime import (
    FixedCropBackend,
    RawLetterboxBackend,
    RembgBackend,
    Sam2Backend,
    align_foreground,
    canonical_json_hash,
    clean_binary_mask,
    contain_on_canvas,
    create_backend,
    process_single_image,
    resolve_live_config,
    save_png_atomic,
)


DERIVATIVE_FIELDS = [
    "schema_version", "parent_sample_id", "split", "label", "source_relpath",
    "source_sha256", "parent_manifest_sha256", "preprocessing_id",
    "config_sha256", "backend", "backend_version", "model_name",
    "model_sha256", "output_relpath", "output_sha256", "mask_relpath",
    "mask_sha256", "status", "width", "height", "channels", "processing_ms",
    "mask_area_ratio", "component_count", "model_score", "bbox_xyxy",
    "quality_flags", "error",
]


def write_derivative_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=DERIVATIVE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_derivative_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a versioned preprocessing dataset from a frozen split manifest."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source, manifest = args.source.resolve(), args.manifest.resolve()
    config_path, output_root = args.config.resolve(), args.output_root.resolve()
    if not source.is_dir() or not manifest.is_file() or not config_path.is_file():
        print("Source, manifest, or config path does not exist")
        return 1
    config = json.loads(config_path.read_text(encoding="utf-8"))
    preprocessing_id = str(config["preprocessing_id"])
    if not preprocessing_id or Path(preprocessing_id).name != preprocessing_id:
        print("preprocessing_id must be one safe folder name")
        return 1
    final_output = output_root / preprocessing_id
    partial_output = output_root / f".{preprocessing_id}.partial"
    if final_output.exists():
        print(f"Final output already exists: {final_output}")
        return 1

    parent_rows = load_manifest_rows(manifest)
    parent_manifest_hash = sha256_manifest(manifest)
    config_hash = canonical_json_hash(config)
    checkpoint_manifest = partial_output / "derivative_manifest.partial.csv"
    created_partial = not partial_output.exists()
    if partial_output.exists() and not args.resume:
        print(f"Partial output exists; inspect it or pass --resume: {partial_output}")
        return 1
    if created_partial:
        partial_output.mkdir(parents=True)
        (partial_output / "preprocessing_config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
    else:
        saved_config = json.loads(
            (partial_output / "preprocessing_config.json").read_text(encoding="utf-8")
        )
        if canonical_json_hash(saved_config) != config_hash:
            print("Cannot resume: preprocessing config does not match partial dataset")
            return 1
    try:
        backend = create_backend(config)
    except (RuntimeError, ValueError) as exc:
        if created_partial:
            shutil.rmtree(partial_output, ignore_errors=True)
        print(f"Cannot initialize backend: {exc}")
        return 1

    derivative_rows: list[dict[str, Any]] = []
    completed_rows = {
        row["parent_sample_id"]: row
        for row in read_derivative_manifest(checkpoint_manifest)
        if row.get("status") == "ok"
        and (partial_output / row.get("output_relpath", "missing")).is_file()
    }
    for index, parent in enumerate(parent_rows, start=1):
        if parent.sample_id in completed_rows:
            derivative_rows.append(completed_rows[parent.sample_id])
            continue
        started = time.perf_counter()
        source_path = source / Path(parent.source_relpath)
        output_relpath = Path(parent.split) / "normal" / f"{parent.sample_id}.png"
        mask_relpath = Path("masks") / parent.split / "normal" / f"{parent.sample_id}.png"
        row: dict[str, Any] = {
            "schema_version": "1.0", "parent_sample_id": parent.sample_id,
            "split": parent.split, "label": parent.label,
            "source_relpath": parent.source_relpath,
            "source_sha256": parent.source_sha256,
            "parent_manifest_sha256": parent_manifest_hash,
            "preprocessing_id": preprocessing_id, "config_sha256": config_hash,
            "backend": backend.name, "backend_version": backend.version,
            "model_name": backend.model_name, "model_sha256": backend.model_sha256,
            "output_relpath": output_relpath.as_posix(), "output_sha256": "",
            "mask_relpath": "", "mask_sha256": "", "status": "error",
            "width": "", "height": "", "channels": "", "processing_ms": "",
            "mask_area_ratio": "", "component_count": "", "model_score": "",
            "bbox_xyxy": "", "quality_flags": "", "error": "",
        }
        try:
            data = source_path.read_bytes()
            if hashlib.sha256(data).hexdigest() != parent.source_sha256:
                raise ValueError("Source hash does not match parent manifest")
            image = Image.open(io.BytesIO(data)).convert("RGB")
            output, output_mask, metrics = backend.process(image)
            output_path = partial_output / output_relpath
            save_png_atomic(output, output_path)
            row.update({
                "status": "ok", "output_sha256": sha256_file(output_path),
                "width": output.width, "height": output.height,
                "channels": len(output.getbands()),
                "mask_area_ratio": metrics.get("mask_area_ratio", ""),
                "component_count": metrics.get("component_count", ""),
                "model_score": metrics.get("sam_predicted_score", ""),
                "bbox_xyxy": metrics.get("bbox_xyxy", ""),
                "quality_flags": ";".join(metrics.get("quality_flags", [])),
            })
            if output_mask is not None:
                mask_path = partial_output / mask_relpath
                save_png_atomic(output_mask, mask_path)
                row["mask_relpath"] = mask_relpath.as_posix()
                row["mask_sha256"] = sha256_file(mask_path)
            fail_flags = set(config.get("fail_on_quality_flags", []))
            row_flags = set(str(row["quality_flags"]).split(";")) - {""}
            if fail_flags & row_flags:
                row["status"] = "qa_failed"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["processing_ms"] = round((time.perf_counter() - started) * 1000, 3)
        derivative_rows.append(row)
        if index % 100 == 0:
            write_derivative_manifest(checkpoint_manifest, derivative_rows)
            print(f"Processed {index}/{len(parent_rows)}", flush=True)

    write_derivative_manifest(checkpoint_manifest, derivative_rows)
    write_derivative_manifest(partial_output / "derivative_manifest.csv", derivative_rows)
    status_counts = Counter(row["status"] for row in derivative_rows)
    flag_counts = Counter(
        flag for row in derivative_rows
        for flag in str(row["quality_flags"]).split(";") if flag
    )
    summary = {
        "preprocessing_id": preprocessing_id,
        "parent_manifest_sha256": parent_manifest_hash,
        "config_sha256": config_hash, "sample_count": len(derivative_rows),
        "status_counts": dict(status_counts), "quality_flag_counts": dict(flag_counts),
    }
    (partial_output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    failed_count = status_counts.get("error", 0) + status_counts.get("qa_failed", 0)
    if failed_count:
        print(f"Preprocessing failed QA for {failed_count} samples: {partial_output}")
        return 1
    checkpoint_manifest.unlink()
    partial_output.replace(final_output)
    print(f"Created dataset: {final_output}")
    print(f"Samples: {len(derivative_rows)}; flags: {dict(flag_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
