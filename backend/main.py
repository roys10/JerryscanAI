
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference.manager import JerryScanModelManager
from inference.history import HistoryManager
import os
import uvicorn
from typing import List, Optional

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

@app.on_event("startup")
async def load_models():
    # Base dir is backend/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    models_dir = os.path.join(project_root, "models")
    
    # 1. Try loading from 'models/' directory (Preferred)
    if os.path.exists(models_dir):
        print(f"Scanning models directory: {models_dir}")
        model_manager.load_all_models(models_dir)

    # 2. Fallback: Try 'model.ckpt' in root/backend (Legacy/Single File)
    if not model_manager.models:
        legacy_paths = [
            os.path.join(base_dir, "model.ckpt"),
            os.path.join(project_root, "model.ckpt")
        ]
        for path in legacy_paths:
            if os.path.exists(path):
                try:
                    model_manager.load_model("default", path)
                    print(f"Loaded legacy default model from {path}")
                    break
                except Exception as e:
                    print(f"Failed to load legacy model: {e}")
        
        if not model_manager.models:
            print("\n" + "="*60)
            print("CRITICAL WARNING: MANUAL ACTION REQUIRED")
            print("="*60)
            print(f"No models found in {models_dir} or root.")
            print("Please COPY .ckpt files to 'models/' folder.")
            print("="*60 + "\n")

@app.post("/inspect/{angle_id}")
async def inspect_image(angle_id: str, file: UploadFile = File(...)):
    print(f"Inspecting angle: {angle_id}")
    try:
        if not model_manager.models:
             raise HTTPException(status_code=503, detail="System not ready. No models loaded.")

        contents = await file.read()
        try:
            model = model_manager.get_model(angle_id)
            return model.predict(contents)
        except KeyError:
            return {
                "status": "UNAVAILABLE",
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
                model = model_manager.get_model(angle_id)
                res = model.predict(contents)
                results[angle_id] = res
                if res.get("status") == "FAIL":
                    overall_status = "FAIL"
            except KeyError:
                results[angle_id] = {"status": "UNAVAILABLE", "score": 0.0}
    
    if not results:
        raise HTTPException(status_code=400, detail="No images provided")

    session_id = history_manager.save_session(results, overall_status)
    return {"session_id": session_id, "overall_status": overall_status, "angles": results}

@app.get("/history")
async def get_history(status: Optional[str] = None):
    return history_manager.get_history(status=status)

@app.get("/stats")
async def get_stats():
    return history_manager.get_stats()

@app.post("/simulate-trigger")
async def simulate_trigger():
    import random
    # Logic to pick a random subfolder in test_images/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(base_dir, "test_images")
    
    if not os.path.exists(test_dir):
        raise HTTPException(status_code=404, detail=f"test_images directory not found at {test_dir}")
        
    subdirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    if not subdirs:
        raise HTTPException(status_code=404, detail="No test folders found in test_images")
        
    chosen_folder = random.choice(subdirs)
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
                    model = model_manager.get_model(angle_id)
                    res = model.predict(contents)
                    results[angle_id] = res
                    if res.get("status") == "FAIL":
                        overall_status = "FAIL"
                except KeyError:
                    results[angle_id] = {"status": "UNAVAILABLE", "score": 0}
        else:
             results[angle_id] = {"status": "MISSING", "score": 0}

    session_id = history_manager.save_session(results, overall_status)
    return {"folder": chosen_folder, "session_id": session_id, "overall_status": overall_status, "angles": results}

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": list(model_manager.models.keys())}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

