from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image

from .benchmark import BenchmarkEngine, BenchmarkRunner
from .datasets import inspect_dataset, safe_source_path
from .patchcore_adapter import PatchCoreAdapter, save_anomaly_visualization
from .preprocessing import PreprocessingResolver
from .registry import ModelRegistry
from .settings import LabSettings
from .storage import ComparisonStore


settings = LabSettings.from_environment()
registry = ModelRegistry(settings)
store = ComparisonStore(settings)
engine = BenchmarkEngine(settings, registry, store)
runner = BenchmarkRunner(engine)

app = FastAPI(title="JerryscanAI Model Lab", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://127.0.0.1:5174"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CalibrationContract(BaseModel):
    method: str
    source_manifest_sha256: str
    source_split: str
    sample_ids_sha256: str
    target_fpr: float | None = None
    recorded_at_utc: str


class ImportModelRequest(BaseModel):
    checkpoint_path: str
    display_name: str | None = None
    metadata_path: str | None = None
    preprocessing_config_path: str | None = None
    derivative_root: str | None = None
    image_threshold: float | None = None
    calibration: CalibrationContract | None = None


class DiscoverRequest(BaseModel):
    root: str


class ContractUpdate(BaseModel):
    display_name: str | None = None
    angle: str | None = None
    manifest_sha256: str | None = None
    preprocessing_config_path: str | None = None
    derivative_root: str | None = None
    image_threshold: float | None = None
    calibration: CalibrationContract | None = None


class DatasetInspectRequest(BaseModel):
    source_root: str
    manifest: str


class ComparisonRequest(BaseModel):
    name: str | None = None
    model_ids: list[str] = Field(min_length=1, max_length=4)
    source_root: str
    dataset_mode: str = "exploratory_folder"
    label_mode: str = "unlabeled"
    manifest: str | None = None
    split: str = "val"
    image_count: int | None = Field(default=None, gt=0)
    seed: int = 42
    force_live_preprocessing: bool = False
    locked_test_confirmation: str | None = None


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_family": "patchcore", "workspace": str(settings.workspace)}


@app.get("/api/models")
def list_models() -> list[dict[str, Any]]:
    return registry.list()


@app.post("/api/models/import")
def import_model(request: ImportModelRequest) -> dict[str, Any]:
    try:
        return registry.import_checkpoint(
            Path(request.checkpoint_path),
            display_name=request.display_name,
            metadata_path=Path(request.metadata_path) if request.metadata_path else None,
            preprocessing_config_path=(
                Path(request.preprocessing_config_path)
                if request.preprocessing_config_path else None
            ),
            derivative_root=Path(request.derivative_root) if request.derivative_root else None,
            image_threshold=request.image_threshold,
            calibration=request.calibration.model_dump() if request.calibration else None,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/models/discover")
def discover_models(request: DiscoverRequest) -> list[dict[str, Any]]:
    try:
        return registry.discover(Path(request.root))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch("/api/models/{model_id}")
def update_model(model_id: str, request: ContractUpdate) -> dict[str, Any]:
    try:
        return registry.update_contract(model_id, request.model_dump(exclude_none=True))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _save_upload(upload: UploadFile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        while chunk := await upload.read(1024 * 1024):
            stream.write(chunk)


@app.post("/api/models/upload")
async def upload_model(
    checkpoint: UploadFile = File(...),
    display_name: str = Form(...),
    metadata: UploadFile | None = File(None),
    preprocessing_config: UploadFile | None = File(None),
    image_threshold: float | None = Form(None),
) -> dict[str, Any]:
    if not checkpoint.filename or not checkpoint.filename.lower().endswith(".ckpt"):
        raise HTTPException(status_code=400, detail="Checkpoint must be a .ckpt file")
    folder = settings.imports_dir / uuid.uuid4().hex
    checkpoint_path = folder / Path(checkpoint.filename).name
    await _save_upload(checkpoint, checkpoint_path)
    metadata_path = None
    config_path = None
    if metadata and metadata.filename:
        metadata_path = folder / Path(metadata.filename).name
        await _save_upload(metadata, metadata_path)
    if preprocessing_config and preprocessing_config.filename:
        config_path = folder / Path(preprocessing_config.filename).name
        await _save_upload(preprocessing_config, config_path)
    try:
        return registry.import_checkpoint(
            checkpoint_path,
            display_name=display_name,
            metadata_path=metadata_path,
            preprocessing_config_path=config_path,
            image_threshold=image_threshold,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/datasets/inspect")
def inspect(request: DatasetInspectRequest) -> dict[str, Any]:
    try:
        return inspect_dataset(Path(request.source_root), Path(request.manifest))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/comparisons")
def list_comparisons() -> list[dict[str, Any]]:
    return store.list()


@app.post("/api/comparisons", status_code=202)
def create_comparison(request: ComparisonRequest) -> dict[str, Any]:
    try:
        comparison_id = engine.prepare(request.model_dump())
        runner.start(comparison_id)
        return {"comparison_id": comparison_id, "state": "queued"}
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/comparisons/{comparison_id}/resume", status_code=202)
def resume_comparison(comparison_id: str) -> dict[str, Any]:
    try:
        store.read_json(comparison_id, "comparison.json")
        runner.start(comparison_id)
        return {"comparison_id": comparison_id, "state": "queued"}
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/comparisons/{comparison_id}")
def get_comparison(comparison_id: str) -> dict[str, Any]:
    try:
        result = {
            "config": store.read_json(comparison_id, "comparison.json"),
            "status": store.read_json(comparison_id, "status.json"),
            "results": store.latest_results(comparison_id),
        }
        summary = store.path(comparison_id) / "summary.json"
        result["summary"] = json.loads(summary.read_text(encoding="utf-8")) if summary.is_file() else None
        return result
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _safe_comparison_asset(comparison_id: str, asset_path: str) -> Path:
    root = store.path(comparison_id).resolve()
    path = (root / asset_path).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return path


@app.get("/api/comparisons/{comparison_id}/assets/{asset_path:path}")
def comparison_asset(comparison_id: str, asset_path: str) -> FileResponse:
    path = _safe_comparison_asset(comparison_id, asset_path)
    if path.suffix == ".npy":
        raise HTTPException(status_code=400, detail="Raw maps are export artifacts, not browser images")
    return FileResponse(path)


@app.get("/api/comparisons/{comparison_id}/originals/{sample_id}")
def original_image(comparison_id: str, sample_id: str) -> FileResponse:
    config = store.read_json(comparison_id, "comparison.json")
    sample = next(
        (item for item in store.read_json(comparison_id, "samples.json") if item["sample_id"] == sample_id),
        None,
    )
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample not found")
    try:
        path = safe_source_path(Path(config["source_root"]), sample["source_relpath"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Original image not found")
    return FileResponse(path)


@app.post("/api/single-image/{model_id}")
async def single_image(model_id: str, image: UploadFile = File(...)) -> dict[str, Any]:
    try:
        model = registry.get(model_id)
        if model["status"] == "incomplete":
            raise ValueError(f"Model is incomplete: {model['issues']}")
        folder = settings.workspace / "single_image" / uuid.uuid4().hex
        original = folder / "original.png"
        folder.mkdir(parents=True)
        temporary = folder / "upload"
        await _save_upload(image, temporary)
        with Image.open(temporary) as source:
            source.convert("RGB").save(original, format="PNG")
        temporary.unlink(missing_ok=True)
        resolver = PreprocessingResolver(model, settings)
        input_path, mask_path, provenance = resolver.process_untracked(
            Image.open(original).convert("RGB"), folder
        )
        output = PatchCoreAdapter(model).predict(input_path)
        raw_map = folder / "raw_anomaly_map.npy"
        np.save(raw_map, output.raw_anomaly_map, allow_pickle=False)
        heatmap = folder / "heatmap.png"
        save_anomaly_visualization(output.raw_anomaly_map, heatmap)
        return {
            "id": folder.name,
            "raw_image_score": output.raw_image_score,
            "image_threshold": output.image_threshold,
            "prediction": output.prediction,
            "preprocessing_ms": provenance["processing_ms"],
            "inference_ms": output.inference_ms,
            "total_ms": provenance["processing_ms"] + output.inference_ms,
            "quality_flags": provenance.get("quality_flags", []),
            "model_input_transform": output.transform_contract,
            "assets": {
                "original": "original.png",
                "input": "model_input.png",
                "mask": "mask.png" if mask_path else None,
                "heatmap": "heatmap.png",
            },
        }
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/single-image/{result_id}/assets/{filename}")
def single_image_asset(result_id: str, filename: str) -> FileResponse:
    if not re.fullmatch(r"[0-9a-f]{32}", result_id):
        raise HTTPException(status_code=404, detail="Asset not found")
    single_root = (settings.workspace / "single_image").resolve()
    root = (single_root / result_id).resolve()
    if single_root not in root.parents:
        raise HTTPException(status_code=404, detail="Asset not found")
    path = (root / Path(filename).name).resolve()
    if root not in path.parents or not path.is_file() or path.suffix == ".npy":
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(path)
