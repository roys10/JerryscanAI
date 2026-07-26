"""Select a diverse, non-test annotation set from a derivative mask dataset."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np


SELECTION_FIELDS = [
    "sample_id",
    "split",
    "source_image",
    "seed_mask",
    "mask_area_ratio",
    "bbox_xyxy",
    "review_status",
    "notes",
]


def feature_vector(row: dict[str, str]) -> list[float]:
    x1, y1, x2, y2 = (float(value) for value in row["bbox_xyxy"].split(","))
    width, height = x2 - x1, y2 - y1
    return [
        float(row["mask_area_ratio"]),
        (x1 + x2) / 2,
        (y1 + y2) / 2,
        width,
        height,
    ]


def select_diverse(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count <= 0:
        return []
    if count > len(rows):
        raise ValueError(f"Requested {count} samples from only {len(rows)} candidates")
    features = np.asarray([feature_vector(row) for row in rows], dtype=np.float64)
    scale = np.maximum(features.std(axis=0), 1e-9)
    features = (features - features.mean(axis=0)) / scale
    first = int(np.argmax(np.linalg.norm(features, axis=1)))
    selected = [first]
    minimum_distance = np.linalg.norm(features - features[first], axis=1)
    minimum_distance[first] = -1
    while len(selected) < count:
        index = int(np.argmax(minimum_distance))
        selected.append(index)
        distance = np.linalg.norm(features - features[index], axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected] = -1
    return [rows[index] for index in selected]


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select train/validation images and seed masks for manual correction."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--derivative-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-count", default=100, type=int)
    parser.add_argument("--val-count", default=25, type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = args.source.resolve()
    derivative_root = args.derivative_root.resolve()
    output = args.output.resolve()
    manifest = derivative_root / "derivative_manifest.csv"
    if output.exists():
        print(f"Output already exists: {output}")
        return 1
    if not source.is_dir() or not manifest.is_file():
        print("Source directory or derivative manifest does not exist")
        return 1
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    candidates = {
        split: [
            row
            for row in rows
            if row["split"] == split
            and row["status"] == "ok"
            and row["mask_relpath"]
            and row["bbox_xyxy"]
        ]
        for split in ("train", "val")
    }
    try:
        selected = select_diverse(candidates["train"], args.train_count)
        selected += select_diverse(candidates["val"], args.val_count)
    except ValueError as exc:
        print(exc)
        return 1

    temporary = output.with_name(f".{output.name}.partial")
    if temporary.exists():
        print(f"Partial output already exists: {temporary}")
        return 1
    selection_rows = []
    try:
        for row in selected:
            split = row["split"]
            sample_id = row["parent_sample_id"]
            source_path = source / Path(row["source_relpath"])
            mask_path = derivative_root / Path(row["mask_relpath"])
            image_destination = temporary / "images" / split / source_path.name
            mask_destination = temporary / "seed_masks" / split / f"{sample_id}.png"
            link_or_copy(source_path, image_destination)
            link_or_copy(mask_path, mask_destination)
            selection_rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "source_image": image_destination.relative_to(temporary).as_posix(),
                    "seed_mask": mask_destination.relative_to(temporary).as_posix(),
                    "mask_area_ratio": row["mask_area_ratio"],
                    "bbox_xyxy": row["bbox_xyxy"],
                    "review_status": "pending",
                    "notes": "",
                }
            )
        with (temporary / "selection_manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=SELECTION_FIELDS)
            writer.writeheader()
            writer.writerows(selection_rows)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        f"Created annotation set: {output} "
        f"({args.train_count} train, {args.val_count} validation, 0 test)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
