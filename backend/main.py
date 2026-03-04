
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference.manager import JerryScanModelManager
from inference.history import HistoryManager
from inference.config import ConfigManager
from inference.alerts import AlertManager
import os
import uvicorn
from typing import List, Optional, Dict, Any

app = FastAPI(title="JerryscanAI Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Managers
model_manager = JerryScanModelManager()
history_manager = HistoryManager()
config_manager = ConfigManager()
alert_manager = AlertManager(config_manager, history_manager)

@app.on_event("startup")
async def load_models():
    # Base dir is backend/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    models_dir = os.path.join(project_root, "models")
    
    # Hierarchical loading from 'models/' directory
    if os.path.exists(models_dir):
        print(f"Scanning models directory: {models_dir}")
        model_manager.load_all_models(models_dir)

    if not model_manager.models:
        print("\n" + "="*60)
        print("CRITICAL WARNING: MANUAL ACTION REQUIRED")
        print("="*60)
        print(f"No model folders found in {models_dir}.")
        print("Please CREATE a subfolder in 'models/' and COPY .ckpt files there.")
        print("Example: models/Standard/front.ckpt")
        print("="*60 + "\n")

@app.get("/models")
async def get_models():
    """Returns list of available model sets (folder names)."""
    return model_manager.get_model_names()

@app.post("/inspect/{angle_id}")
async def inspect_image(angle_id: str, file: UploadFile = File(...), model_name: Optional[str] = None):
    print(f"Inspecting angle: {angle_id} using model set: {model_name or 'default'}")
    try:
        if not model_manager.models:
             raise HTTPException(status_code=503, detail="System not ready. No models loaded.")

        contents = await file.read()
        try:
            model = model_manager.get_model(angle_id, model_name=model_name)
            return model.predict(contents)
        except KeyError as e:
            return {
                "status": "UNAVAILABLE",
                "message": str(e),
                "score": 0.0,
                "score_percentage": 0.0,
                "heatmap_image": None,
                "segmentation_image": None,
                "original_image": None
            }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Inspection Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inspect-batch")
async def inspect_batch(
    model_name: Optional[str] = None,
    front: Optional[UploadFile] = File(None),
    back: Optional[UploadFile] = File(None),
    side_l: Optional[UploadFile] = File(None),
    side_r: Optional[UploadFile] = File(None)
):
    """
    Processes multiple angles and saves a single history session.
    """
    files = {"front": front, "back": back, "side_l": side_l, "side_r": side_r}
    results = {}
    overall_status = "PASS"
    
    for angle_id, file in files.items():
        if file:
            contents = await file.read()
            try:
                model = model_manager.get_model(angle_id, model_name=model_name)
                res = model.predict(contents)
                results[angle_id] = res
                if res.get("status") == "FAIL":
                    overall_status = "FAIL"
            except KeyError:
                results[angle_id] = {"status": "UNAVAILABLE", "score": 0.0}
    
    if not results:
        raise HTTPException(status_code=400, detail="No images provided")

    session_id = history_manager.save_session(results, overall_status, model_name=model_name)
    alert_manager.evaluate_session(overall_status, session_id)
    return {"session_id": session_id, "overall_status": overall_status, "angles": results}

@app.get("/settings")
async def get_settings():
    return config_manager.get_all()

@app.post("/settings")
async def update_settings(settings: Dict[str, Any]):
    updated = config_manager.update(settings)
    return {"status": "success", "settings": updated}

@app.get("/history")
async def get_history(status: Optional[str] = None):
    return history_manager.get_history(status=status)

@app.get("/stats")
async def get_stats():
    return history_manager.get_stats()

@app.post("/simulate-trigger")
async def simulate_trigger(model_name: Optional[str] = None):
    # Logic to process ALL subfolders in test_images/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(base_dir, "test_images")
    
    if not os.path.exists(test_dir):
        raise HTTPException(status_code=404, detail=f"test_images directory not found at {test_dir}")
        
    subdirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    if not subdirs:
        raise HTTPException(status_code=404, detail="No test folders found in test_images")
        
    all_sessions_results = []
    global_batch_status = "PASS"

    for chosen_folder in subdirs:
        folder_path = os.path.join(test_dir, chosen_folder)
        results = {}
        overall_status = "PASS"
        
        for angle_id in ["front", "back", "side_l", "side_r"]:
            img_path = None
            for ext in [".jpg", ".png", ".jpeg"]:
                p = os.path.join(folder_path, f"{angle_id}{ext}")
                if os.path.exists(p):
                    img_path = p
                    break
            
            if img_path:
                with open(img_path, "rb") as f:
                    contents = f.read()
                    try:
                        model = model_manager.get_model(angle_id, model_name=model_name)
                        res = model.predict(contents)
                        results[angle_id] = res
                        if res.get("status") == "FAIL":
                            overall_status = "FAIL"
                    except KeyError:
                        results[angle_id] = {"status": "UNAVAILABLE", "score": 0}
            else:
                 results[angle_id] = {"status": "MISSING", "score": 0}

        session_id = history_manager.save_session(results, overall_status, model_name=model_name)
        alert_manager.evaluate_session(overall_status, session_id)
        
        if overall_status == "FAIL":
            global_batch_status = "FAIL"

        all_sessions_results.append({
            "folder": chosen_folder, 
            "session_id": session_id, 
            "overall_status": overall_status, 
            "angles": results, 
            "model_name": model_name
        })

    # the frontend expects an object with overall_status and angles (results dict)
    # We will map the very last result as the "live console" result, but all will be in history
    final_session = all_sessions_results[-1]
    return {
        "overall_status": global_batch_status,
        "angles": final_session["angles"],
        "batch_processed": len(all_sessions_results)
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok", 
        "model_sets": model_manager.get_model_names(),
        "details": {name: list(angles.keys()) for name, angles in model_manager.models.items()}
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

