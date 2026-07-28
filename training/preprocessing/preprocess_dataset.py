"""Create a split-preserving, versioned preprocessing dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from training.datasets.create_dataset_manifest import sha256_file, sha256_manifest
from training.datasets.materialize_dataset_split import load_manifest_rows


DERIVATIVE_FIELDS = [
    "schema_version",
    "parent_sample_id",
    "split",
    "label",
    "source_relpath",
    "source_sha256",
    "parent_manifest_sha256",
    "preprocessing_id",
    "config_sha256",
    "backend",
    "backend_version",
    "model_name",
    "model_sha256",
    "output_relpath",
    "output_sha256",
    "mask_relpath",
    "mask_sha256",
    "status",
    "width",
    "height",
    "channels",
    "processing_ms",
    "mask_area_ratio",
    "component_count",
    "model_score",
    "bbox_xyxy",
    "quality_flags",
    "error",
]


def canonical_json_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def save_png_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="PNG", optimize=False)
    os.replace(temporary, path)


def contain_on_canvas(
    image: Image.Image, size: tuple[int, int], background: int
) -> Image.Image:
    canvas_w, canvas_h = size
    scale = min(canvas_w / image.width, canvas_h / image.height)
    resized_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (background,) * 3)
    offset = ((canvas_w - resized.width) // 2, (canvas_h - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas


def clean_binary_mask(
    mask: np.ndarray, threshold: int, dilation_px: int
) -> tuple[np.ndarray, int]:
    binary = (mask >= threshold).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    foreground_components = component_count - 1
    if foreground_components <= 0:
        raise ValueError("Segmentation produced an empty mask")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    binary = (labels == largest).astype(np.uint8)
    if dilation_px > 0:
        kernel_size = dilation_px * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        binary = cv2.dilate(binary, kernel, iterations=1)
    return binary * 255, foreground_components


def align_foreground(
    image: Image.Image, mask: np.ndarray, config: dict[str, Any]
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Cannot align an empty mask")
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    crop_w, crop_h = x2 - x1, y2 - y1
    padding = round(max(crop_w, crop_h) * float(config.get("bbox_padding_ratio", 0)))
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(image.width, x2 + padding), min(image.height, y2 + padding)

    image_crop = image.crop((x1, y1, x2, y2))
    mask_crop = Image.fromarray(mask).crop((x1, y1, x2, y2))
    canvas_w, canvas_h = map(int, config["output_size"])
    target_fraction = float(config.get("object_scale", 0.9))
    scale = min(
        canvas_w * target_fraction / image_crop.width,
        canvas_h * target_fraction / image_crop.height,
    )
    resized_size = (
        max(1, round(image_crop.width * scale)),
        max(1, round(image_crop.height * scale)),
    )
    image_crop = image_crop.resize(resized_size, Image.Resampling.LANCZOS)
    mask_crop = mask_crop.resize(resized_size, Image.Resampling.NEAREST)

    center_x, center_y = config.get("object_center", [0.5, 0.5])
    left = round(canvas_w * float(center_x) - image_crop.width / 2)
    top = round(canvas_h * float(center_y) - image_crop.height / 2)
    left = min(max(left, 0), canvas_w - image_crop.width)
    top = min(max(top, 0), canvas_h - image_crop.height)
    background = int(config["background_value"])
    image_canvas = Image.new("RGB", (canvas_w, canvas_h), (background,) * 3)
    mask_canvas = Image.new("L", (canvas_w, canvas_h), 0)
    image_canvas.paste(image_crop, (left, top), mask_crop)
    mask_canvas.paste(mask_crop, (left, top))
    return image_canvas, mask_canvas, {
        "bbox_xyxy": f"{x1},{y1},{x2},{y2}",
        "aligned_bbox_xyxy": f"{left},{top},{left + image_crop.width},{top + image_crop.height}",
    }


class RawLetterboxBackend:
    name = "raw_letterbox"
    version = "1"
    model_name = ""
    model_sha256 = ""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def process(
        self, image: Image.Image
    ) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
        output_size = tuple(map(int, self.config["output_size"]))
        result = contain_on_canvas(
            image, output_size, int(self.config["background_value"])
        )
        return result, None, {}


class FixedCropBackend(RawLetterboxBackend):
    name = "fixed_crop"

    def process(
        self, image: Image.Image
    ) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
        crop = tuple(int(value) for value in self.config["crop_xyxy"])
        x1, y1, x2, y2 = crop
        if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            raise ValueError(f"Fixed crop {crop} is outside image size {image.size}")
        cropped = image.crop(crop)
        output_size = tuple(map(int, self.config["output_size"]))
        result = contain_on_canvas(
            cropped, output_size, int(self.config["background_value"])
        )
        return result, None, {"bbox_xyxy": ",".join(str(value) for value in crop)}


class RembgBackend:
    name = "rembg"

    def __init__(self, config: dict[str, Any]) -> None:
        model_dir = Path(str(config["model_dir"]))
        if not model_dir.is_absolute():
            model_dir = (Path.cwd() / model_dir).resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        os.environ["U2NET_HOME"] = str(model_dir)
        try:
            from rembg import new_session, remove
        except ImportError as exc:
            raise RuntimeError(
                'rembg is not installed. Install the "preprocess-rembg" optional dependency.'
            ) from exc
        self.config = config
        self.remove = remove
        self.model_name = str(config["model_name"])
        self.session = new_session(self.model_name)
        self.version = importlib.metadata.version("rembg")
        model_file = model_dir / str(config["model_filename"])
        self.model_sha256 = sha256_file(model_file) if model_file.is_file() else ""
        expected_hash = str(config.get("expected_model_sha256", "")).lower()
        if expected_hash and self.model_sha256.lower() != expected_hash:
            raise RuntimeError(
                f"rembg model hash mismatch: {self.model_sha256} != {expected_hash}"
            )

    def process(
        self, image: Image.Image
    ) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
        predicted = self.remove(
            image,
            session=self.session,
            only_mask=True,
            post_process_mask=bool(self.config.get("rembg_post_process_mask", False)),
        )
        mask_array = np.asarray(predicted.convert("L"))
        mask, component_count = clean_binary_mask(
            mask_array,
            int(self.config["mask_threshold"]),
            int(self.config.get("mask_dilation_px", 0)),
        )
        area_ratio = float(np.count_nonzero(mask) / mask.size)
        flags = []
        min_area, max_area = self.config.get("valid_mask_area_ratio", [0, 1])
        if not float(min_area) <= area_ratio <= float(max_area):
            flags.append("mask_area_out_of_range")
        if component_count > 1:
            flags.append("multiple_components_removed")
        output, output_mask, geometry = align_foreground(image, mask, self.config)
        return output, output_mask, {
            **geometry,
            "mask_area_ratio": area_ratio,
            "component_count": component_count,
            "quality_flags": flags,
        }


class Sam2Backend:
    name = "sam2"

    def __init__(self, config: dict[str, Any]) -> None:
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError(
                "SAM 2 is not installed. Use the official SAM 2 environment in WSL/Linux."
            ) from exc
        self.config = config
        self.torch = torch
        requested_device = str(config.get("device", "auto"))
        self.device = (
            "cuda"
            if requested_device == "auto" and torch.cuda.is_available()
            else "cpu"
            if requested_device == "auto"
            else requested_device
        )
        checkpoint = Path(str(config["checkpoint"]))
        if not checkpoint.is_absolute():
            checkpoint = (Path.cwd() / checkpoint).resolve()
        if not checkpoint.is_file():
            raise RuntimeError(f"SAM 2 checkpoint does not exist: {checkpoint}")
        model = build_sam2(
            str(config["model_config"]), str(checkpoint), device=self.device
        )
        self.predictor = SAM2ImagePredictor(model)
        self.model_name = str(config["model_name"])
        self.model_sha256 = sha256_file(checkpoint)
        expected_hash = str(config.get("expected_model_sha256", "")).lower()
        if expected_hash and self.model_sha256.lower() != expected_hash:
            raise RuntimeError(
                f"SAM 2 model hash mismatch: {self.model_sha256} != {expected_hash}"
            )
        self.version = importlib.metadata.version("sam-2")

    def process(
        self, image: Image.Image
    ) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
        box = np.asarray(self.config["box_prompt_xyxy"], dtype=np.float32)
        with self.torch.inference_mode():
            self.predictor.set_image(np.asarray(image))
            masks, scores, _ = self.predictor.predict(
                box=box,
                multimask_output=bool(self.config.get("multimask_output", True)),
            )
        min_area, max_area = self.config.get("valid_mask_area_ratio", [0, 1])
        areas = np.asarray([np.count_nonzero(mask) / mask.size for mask in masks])
        valid = np.flatnonzero((areas >= float(min_area)) & (areas <= float(max_area)))
        flags = []
        if len(valid):
            selected = int(valid[np.argmax(np.asarray(scores)[valid])])
        else:
            selected = int(np.argmax(scores))
            flags.append("no_candidate_in_area_range")
        raw_mask = masks[selected].astype(np.uint8) * 255
        mask, component_count = clean_binary_mask(
            raw_mask,
            int(self.config.get("mask_threshold", 128)),
            int(self.config.get("mask_dilation_px", 0)),
        )
        area_ratio = float(np.count_nonzero(mask) / mask.size)
        if not float(min_area) <= area_ratio <= float(max_area):
            flags.append("mask_area_out_of_range")
        if component_count > 1:
            flags.append("multiple_components_removed")
        output, output_mask, geometry = align_foreground(image, mask, self.config)
        return output, output_mask, {
            **geometry,
            "mask_area_ratio": area_ratio,
            "component_count": component_count,
            "quality_flags": flags,
            "sam_predicted_score": float(np.asarray(scores)[selected]),
        }


def create_backend(
    config: dict[str, Any],
) -> RawLetterboxBackend | FixedCropBackend | RembgBackend | Sam2Backend:
    backend = config.get("backend")
    if backend == "raw_letterbox":
        return RawLetterboxBackend(config)
    if backend == "fixed_crop":
        return FixedCropBackend(config)
    if backend == "rembg":
        return RembgBackend(config)
    if backend == "sam2":
        return Sam2Backend(config)
    raise ValueError(f"Unsupported preprocessing backend: {backend}")


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a matching .partial dataset after interruption.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = args.source.resolve()
    manifest = args.manifest.resolve()
    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
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
            "schema_version": "1.0",
            "parent_sample_id": parent.sample_id,
            "split": parent.split,
            "label": parent.label,
            "source_relpath": parent.source_relpath,
            "source_sha256": parent.source_sha256,
            "parent_manifest_sha256": parent_manifest_hash,
            "preprocessing_id": preprocessing_id,
            "config_sha256": config_hash,
            "backend": backend.name,
            "backend_version": backend.version,
            "model_name": backend.model_name,
            "model_sha256": backend.model_sha256,
            "output_relpath": output_relpath.as_posix(),
            "output_sha256": "",
            "mask_relpath": "",
            "mask_sha256": "",
            "status": "error",
            "width": "",
            "height": "",
            "channels": "",
            "processing_ms": "",
            "mask_area_ratio": "",
            "component_count": "",
            "model_score": "",
            "bbox_xyxy": "",
            "quality_flags": "",
            "error": "",
        }
        try:
            data = source_path.read_bytes()
            if hashlib.sha256(data).hexdigest() != parent.source_sha256:
                raise ValueError("Source hash does not match parent manifest")
            image = Image.open(io.BytesIO(data)).convert("RGB")
            output, output_mask, metrics = backend.process(image)
            output_path = partial_output / output_relpath
            save_png_atomic(output, output_path)
            row.update(
                {
                    "status": "ok",
                    "output_sha256": sha256_file(output_path),
                    "width": output.width,
                    "height": output.height,
                    "channels": len(output.getbands()),
                    "mask_area_ratio": metrics.get("mask_area_ratio", ""),
                    "component_count": metrics.get("component_count", ""),
                    "model_score": metrics.get("sam_predicted_score", ""),
                    "bbox_xyxy": metrics.get("bbox_xyxy", ""),
                    "quality_flags": ";".join(metrics.get("quality_flags", [])),
                }
            )
            if output_mask is not None:
                mask_path = partial_output / mask_relpath
                save_png_atomic(output_mask, mask_path)
                row["mask_relpath"] = mask_relpath.as_posix()
                row["mask_sha256"] = sha256_file(mask_path)
            fail_flags = set(config.get("fail_on_quality_flags", []))
            row_flags = set(str(row["quality_flags"]).split(";")) - {""}
            if fail_flags & row_flags:
                row["status"] = "qa_failed"
        except Exception as exc:  # Keep a reviewable failure manifest.
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
        flag
        for row in derivative_rows
        for flag in str(row["quality_flags"]).split(";")
        if flag
    )
    summary = {
        "preprocessing_id": preprocessing_id,
        "parent_manifest_sha256": parent_manifest_hash,
        "config_sha256": config_hash,
        "sample_count": len(derivative_rows),
        "status_counts": dict(status_counts),
        "quality_flag_counts": dict(flag_counts),
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
