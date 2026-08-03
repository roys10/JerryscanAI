"""Local G01 manufacturing API backed by one selected model folder."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from backend.inference.alerts import AlertManager
from backend.inference.config import ConfigManager
from backend.inference.history import HistoryManager
from backend.inference.manager import JerryScanModelManager


def _origins() -> list[str]:
    configured = os.getenv(
        "JERRYSCAN_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


app = FastAPI(title="JerryscanAI local G01 backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = JerryScanModelManager()
history_manager = HistoryManager()
config_manager = ConfigManager()
alert_manager = AlertManager(config_manager, history_manager)


@app.on_event("startup")
async def load_selected_model() -> None:
    try:
        await run_in_threadpool(model_manager.load_selected)
    except Exception as exc:
        # The API still starts so /health and inspection responses can explain
        # exactly which local artifact is missing.  It never returns PASS.
        print(f"Model folder is not ready: {type(exc).__name__}: {exc}")


async def _read_bounded(upload: UploadFile) -> bytes:
    maximum = (
        model_manager.runtime.manifest.max_image_bytes
        if model_manager.runtime is not None
        else 25 * 1024 * 1024
    )
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(min(1024 * 1024, maximum + 1 - total))
        if not chunk:
            break
        total += len(chunk)
        if total > maximum:
            raise HTTPException(status_code=413, detail="Camera image is too large")
        chunks.append(chunk)
    return b"".join(chunks)


def _overall_status(results: dict[str, dict[str, Any]]) -> str:
    statuses = {result.get("status", "SYSTEM_ERROR") for result in results.values()}
    for status in ("SYSTEM_ERROR", "REVIEW", "FAIL", "SHADOW"):
        if status in statuses:
            return status
    return "PASS" if statuses == {"PASS"} else "SYSTEM_ERROR"


async def _record(results: dict[str, dict[str, Any]]) -> tuple[str, str]:
    overall = _overall_status(results)
    model_id = next(
        (result.get("model_id") for result in results.values() if result.get("model_id")),
        None,
    )
    session_id = await run_in_threadpool(
        history_manager.save_session, results, overall, model_id
    )
    await run_in_threadpool(alert_manager.evaluate_session, overall, session_id)
    return session_id, overall


async def _inspect_g01(upload: UploadFile, model_name: str | None) -> dict[str, Any]:
    contents = await _read_bounded(upload)
    return await run_in_threadpool(
        model_manager.inspect,
        "G01",
        contents,
        requested_model=model_name,
    )


@app.get("/models")
async def get_models() -> list[str]:
    return model_manager.get_model_names()


@app.post("/inspect/G01")
async def inspect_image(
    file: UploadFile = File(...), model_name: Optional[str] = None
) -> dict[str, Any]:
    result = await _inspect_g01(file, model_name)
    session_id, overall = await _record({"G01": result})
    return {**result, "session_id": session_id, "overall_status": overall}


@app.post("/inspect-batch")
async def inspect_batch(
    model_name: Optional[str] = None,
    G01: Optional[UploadFile] = File(None),
) -> dict[str, Any]:
    if G01 is None:
        raise HTTPException(status_code=400, detail="The required G01 image is missing")
    result = await _inspect_g01(G01, model_name)
    results = {"G01": result}
    session_id, overall = await _record(results)
    return {
        "session_id": session_id,
        "overall_status": overall,
        "angles": results,
        "mode": result.get("mode", "shadow"),
        "required_angles": ["G01"],
    }


@app.get("/settings")
async def get_settings() -> dict[str, Any]:
    return config_manager.get_all()


@app.post("/settings")
async def update_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {"status": "success", "settings": config_manager.update(settings)}


@app.post("/reload-model")
async def reload_model() -> dict[str, Any]:
    try:
        await run_in_threadpool(model_manager.load_selected)
    except Exception as exc:
        raise HTTPException(status_code=409, detail=model_manager.health()) from exc
    return model_manager.health()


@app.get("/history")
async def get_history(status: Optional[str] = None) -> list[dict[str, Any]]:
    return history_manager.get_history(status=status)


@app.get("/stats")
async def get_stats() -> dict[str, Any]:
    return history_manager.get_stats()


@app.post("/simulate-trigger")
async def simulate_trigger(model_name: Optional[str] = None) -> dict[str, Any]:
    test_dir = Path(__file__).resolve().parents[1] / "test_images"
    image_path = next(
        (
            path
            for suffix in (".bmp", ".png", ".jpg", ".jpeg")
            if (path := test_dir / f"G01{suffix}").is_file()
        ),
        None,
    )
    if image_path is None:
        raise HTTPException(status_code=404, detail="Required test_images/G01 image not found")
    result = await run_in_threadpool(
        model_manager.inspect,
        "G01",
        image_path.read_bytes(),
        requested_model=model_name,
    )
    session_id, overall = await _record({"G01": result})
    return {
        "session_id": session_id,
        "overall_status": overall,
        "angles": {"G01": result},
    }


@app.get("/health")
def health_check() -> dict[str, Any]:
    return {**model_manager.health(), "supported_angles": ["G01"]}


@app.get("/ready")
def readiness_check() -> JSONResponse:
    health = health_check()
    return JSONResponse(
        status_code=200 if health["ready_for_inference"] else 503,
        content=health,
    )


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
