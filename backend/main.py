"""Local multi-angle manufacturing API backed by one selected model folder."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles

from backend.inference.alerts import AlertManager
from backend.inference.config import ConfigManager
from backend.inference.history import HistoryManager
from backend.inference.local_model import InferenceRuntimeError
from backend.inference.manager import (
    JerryScanModelManager,
    ModelNotReadyError,
    ModelSelectionError,
)


# Load local backend settings before CORS and runtime managers read the
# environment. Existing process-level variables keep precedence by default.
load_dotenv(Path(__file__).resolve().parent / ".env")


def _origins() -> list[str]:
    configured = os.getenv(
        "JERRYSCAN_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


app = FastAPI(title="JerryscanAI local multi-angle backend")
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
            # Keep memory bounded but let the runtime return the normal
            # WRONG_INPUT response shape for an oversized image.
            chunks.append(chunk)
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _overall_status(results: dict[str, dict[str, Any]]) -> str:
    statuses = {result.get("status") for result in results.values()}
    if not statuses.issubset({"PASS", "FAIL", "WRONG_INPUT"}):
        raise RuntimeError(f"Unexpected inspection status: {sorted(map(str, statuses))}")
    if "WRONG_INPUT" in statuses:
        return "WRONG_INPUT"
    return "FAIL" if "FAIL" in statuses else "PASS"


async def _record(results: dict[str, dict[str, Any]]) -> tuple[str, str]:
    overall = _overall_status(results)
    model_id = next(
        (
            result.get("model_display_name") or result.get("model_id")
            for result in results.values()
            if result.get("model_display_name") or result.get("model_id")
        ),
        None,
    )
    session_id = await run_in_threadpool(
        history_manager.save_session, results, overall, model_id
    )
    await run_in_threadpool(alert_manager.evaluate_session, overall, session_id)
    return session_id, overall


async def _inspect_angle(
    angle: str, upload: UploadFile, model_name: str | None
) -> dict[str, Any]:
    contents = await _read_bounded(upload)
    try:
        return await run_in_threadpool(
            model_manager.inspect,
            angle,
            contents,
            requested_model=model_name,
        )
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelSelectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except InferenceRuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/models")
async def get_models() -> list[str]:
    return model_manager.get_model_names()


def _angle_availability() -> tuple[list[str], list[str], dict[str, dict[str, str]]]:
    """Return available, configured, and unavailable angle contracts."""
    available = model_manager.get_required_angles()
    configured_getter = getattr(model_manager, "get_configured_angles", None)
    unavailable_getter = getattr(model_manager, "get_unavailable_angles", None)
    configured = configured_getter() if configured_getter else list(available)
    unavailable = unavailable_getter() if unavailable_getter else {}
    return list(available), list(configured), dict(unavailable)


@app.post("/inspect/{angle_id}")
async def inspect_image(
    angle_id: str, file: UploadFile = File(...), model_name: Optional[str] = None
) -> dict[str, Any]:
    try:
        required_angles, configured_angles, unavailable_angles = _angle_availability()
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if angle_id not in required_angles:
        if angle_id in unavailable_angles:
            reason = unavailable_angles[angle_id]
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Angle {angle_id!r} is configured but unavailable at "
                    f"{reason['stage']}: {reason['detail']}"
                ),
            )
        raise HTTPException(
            status_code=404,
            detail=f"Angle {angle_id!r} is not configured; expected {configured_angles}",
        )
    result = await _inspect_angle(angle_id, file, model_name)
    session_id, overall = await _record({angle_id: result})
    return {**result, "session_id": session_id, "overall_status": overall}


@app.post("/inspect-batch")
async def inspect_batch(
    request: Request,
    model_name: Optional[str] = None,
) -> dict[str, Any]:
    try:
        required_angles, configured_angles, unavailable_angles = _angle_availability()
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    form = await request.form()
    submitted_fields = list(form.keys())
    if not submitted_fields:
        raise HTTPException(
            status_code=400,
            detail="No images provided",
        )

    unknown = [angle for angle in submitted_fields if angle not in configured_angles]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown camera angle fields: {', '.join(unknown)}; "
                f"configured angles are {configured_angles}"
            ),
        )

    unavailable = [angle for angle in submitted_fields if angle not in required_angles]
    if unavailable:
        reasons = []
        for angle in unavailable:
            reason = unavailable_angles.get(angle, {})
            detail = reason.get("detail", "model is not available")
            reasons.append(f"{angle}: {detail}")
        raise HTTPException(
            status_code=503,
            detail="Uploaded camera models are unavailable: " + "; ".join(reasons),
        )

    # Preserve the model contract's camera order while inspecting only the
    # non-empty subset supplied by this request.
    inspected_angles = [angle for angle in required_angles if angle in form]
    uploads = {angle: form.get(angle) for angle in inspected_angles}
    inspections = []
    for angle in inspected_angles:
        upload = uploads[angle]
        if not hasattr(upload, "read"):
            raise HTTPException(status_code=400, detail=f"{angle} must be an image upload")
        inspections.append(_inspect_angle(angle, upload, model_name))
    angle_results = await asyncio.gather(*inspections)
    results = dict(zip(inspected_angles, angle_results, strict=True))
    session_id, overall = await _record(results)
    return {
        "session_id": session_id,
        "overall_status": overall,
        "angles": results,
        "inspected_angles": inspected_angles,
        "required_angles": required_angles,
        "available_angles": required_angles,
        "configured_angles": configured_angles,
        "unavailable_angles": unavailable_angles,
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
        raise HTTPException(status_code=503, detail=model_manager.health()) from exc
    return model_manager.health()


@app.get("/history")
async def get_history(status: Optional[str] = None) -> list[dict[str, Any]]:
    try:
        return history_manager.get_history(status=status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/history/{session_id}")
async def get_history_session(session_id: str) -> dict[str, Any]:
    session = history_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Inspection session not found")
    return session


@app.get("/stats")
async def get_stats() -> dict[str, Any]:
    return history_manager.get_stats()


@app.get("/health")
def health_check() -> dict[str, Any]:
    return model_manager.health()


@app.get("/ready")
def readiness_check() -> JSONResponse:
    health = health_check()
    return JSONResponse(
        status_code=200 if health["ready_for_inference"] else 503,
        content=health,
    )


# Register the frontend last so every API route, including FastAPI's /docs and
# /openapi.json, keeps priority. Local backend development remains API-only
# until `npm run build` creates frontend/dist.
frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
