"""Shared single-image preprocessing runtime for training, lab, and production."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from training.datasets.create_dataset_manifest import sha256_file


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

    def process(self, image: Image.Image) -> tuple[Image.Image, None, dict[str, Any]]:
        output_size = tuple(map(int, self.config["output_size"]))
        result = contain_on_canvas(
            image, output_size, int(self.config["background_value"])
        )
        return result, None, {}


class FixedCropBackend(RawLetterboxBackend):
    name = "fixed_crop"

    def process(self, image: Image.Image) -> tuple[Image.Image, None, dict[str, Any]]:
        crop = tuple(int(value) for value in self.config["crop_xyxy"])
        x1, y1, x2, y2 = crop
        if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            raise ValueError(f"Fixed crop {crop} is outside image size {image.size}")
        result = contain_on_canvas(
            image.crop(crop),
            tuple(map(int, self.config["output_size"])),
            int(self.config["background_value"]),
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
        expected = str(config.get("expected_model_sha256", "")).lower()
        if expected and self.model_sha256.lower() != expected:
            raise RuntimeError("rembg model hash mismatch")

    def process(
        self, image: Image.Image
    ) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
        predicted = self.remove(
            image,
            session=self.session,
            only_mask=True,
            post_process_mask=bool(self.config.get("rembg_post_process_mask", False)),
        )
        mask, component_count = clean_binary_mask(
            np.asarray(predicted.convert("L")),
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
            raise RuntimeError("SAM 2 is not installed") from exc
        self.config, self.torch = config, torch
        requested = str(config.get("device", "auto"))
        self.device = "cuda" if requested == "auto" and torch.cuda.is_available() else (
            "cpu" if requested == "auto" else requested
        )
        checkpoint = Path(str(config["checkpoint"]))
        if not checkpoint.is_absolute():
            checkpoint = (Path.cwd() / checkpoint).resolve()
        if not checkpoint.is_file():
            raise RuntimeError(f"SAM 2 checkpoint does not exist: {checkpoint}")
        model = build_sam2(str(config["model_config"]), str(checkpoint), device=self.device)
        self.predictor = SAM2ImagePredictor(model)
        self.model_name = str(config["model_name"])
        self.model_sha256 = sha256_file(checkpoint)
        expected = str(config.get("expected_model_sha256", "")).lower()
        if expected and self.model_sha256.lower() != expected:
            raise RuntimeError("SAM 2 model hash mismatch")
        self.version = importlib.metadata.version("sam-2")

    def process(
        self, image: Image.Image
    ) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
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
        selected = int(valid[np.argmax(np.asarray(scores)[valid])]) if len(valid) else int(np.argmax(scores))
        if not len(valid):
            flags.append("no_candidate_in_area_range")
        mask, component_count = clean_binary_mask(
            masks[selected].astype(np.uint8) * 255,
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


Backend = RawLetterboxBackend | FixedCropBackend | RembgBackend | Sam2Backend


def create_backend(config: dict[str, Any]) -> Backend:
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


def resolve_live_config(
    config: dict[str, Any], config_dir: Path | None = None
) -> dict[str, Any]:
    if config.get("backend") != "aligned_mask_recompose":
        return dict(config)
    if config_dir is None:
        raise ValueError("config_dir is required for a recomposition live pipeline")
    parent_id = str(config.get("parent_preprocessing_id", ""))
    parent_path = config_dir / f"{parent_id}.json"
    if not parent_id or not parent_path.is_file():
        raise ValueError(f"Parent preprocessing config does not exist: {parent_path}")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    actual_hash = canonical_json_hash(parent)
    expected_hash = str(config.get("parent_config_sha256", ""))
    if expected_hash and actual_hash != expected_hash:
        raise ValueError("Parent preprocessing config hash does not match live contract")
    resolved = resolve_live_config(parent, config_dir)
    resolved.update(
        {
            "schema_version": config.get("schema_version", resolved.get("schema_version")),
            "preprocessing_id": config["preprocessing_id"],
            "background_value": int(config["background_value"]),
            "composite_mode": config.get("composite_mode", "hard_background"),
            "live_parent_preprocessing_id": parent_id,
            "live_parent_config_sha256": actual_hash,
        }
    )
    return resolved


def process_single_image(
    image: Image.Image,
    config: dict[str, Any],
    *,
    config_dir: Path | None = None,
    backend_instance: Backend | None = None,
) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
    resolved = resolve_live_config(config, config_dir)
    backend = backend_instance or create_backend(resolved)
    started = time.perf_counter()
    output, mask, metrics = backend.process(image.convert("RGB"))
    flags = list(metrics.get("quality_flags", []))
    fail_flags = set(resolved.get("fail_on_quality_flags", []))
    return output, mask, {
        **metrics,
        "status": "qa_failed" if fail_flags.intersection(flags) else "ok",
        "preprocessing_id": str(config["preprocessing_id"]),
        "config_sha256": canonical_json_hash(config),
        "resolved_config_sha256": canonical_json_hash(resolved),
        "backend": backend.name,
        "backend_version": backend.version,
        "model_name": backend.model_name,
        "model_sha256": backend.model_sha256,
        "processing_ms": round((time.perf_counter() - started) * 1000, 3),
        "quality_flags": flags,
    }
