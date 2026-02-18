
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference.manager import JerryScanModelManager
import os
import uvicorn

app = FastAPI(title="JerryscanAI Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Manager
model_manager = JerryScanModelManager()

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
    # Only if no models loaded yet? Or in addition? 
    # Let's say if we have specific models, we use them.
    # If we have NO models, we warn.
    
    if not model_manager.models:
        # Check specific locations for legacy model.ckpt
        legacy_paths = [
            os.path.join(base_dir, "model.ckpt"),
            os.path.join(project_root, "model.ckpt")
        ]
        loaded_legacy = False
        for path in legacy_paths:
            if os.path.exists(path):
                try:
                    model_manager.load_model("default", path)
                    print(f"Loaded legacy default model from {path}")
                    loaded_legacy = True
                    break
                except Exception as e:
                    print(f"Failed to load legacy model: {e}")
        
        if not loaded_legacy:
            print("\n" + "="*60)
            print("CRITICAL WARNING: MANUAL ACTION REQUIRED")
            print("="*60)
            print(f"No models found in {models_dir}")
            print(f"And no legacy 'model.ckpt' found in {legacy_paths}")
            print("Please COPY .ckpt files to 'models/' or 'model.ckpt' to root.")
            print("="*60 + "\n")

@app.post("/inspect/{angle_id}")
async def inspect_image(angle_id: str, file: UploadFile = File(...)):
    print(f"Inspecting angle: {angle_id}")
    try:
        # Check if ANY logic is possible (fail fast if 0 models)
        if not model_manager.models:
             raise HTTPException(
                 status_code=503, 
                 detail="System not ready. No models loaded."
             )

        contents = await file.read()
        print(angle_id, "here")
        # Get appropriate model (default or specific angle)
        try:
            model = model_manager.get_model(angle_id)
        except KeyError:
             # Graceful Fallback for Batch Mode
             # If specific angle model is missing, return UNAVAILABLE status
             return {
                 "status": "UNAVAILABLE",
                 "score": 0.0,
                 "score_percentage": 0.0,
                 "threshold_percentage": 0.0,
                 "heatmap_image": None,
                 "segmentation_image": None,
                 "original_image": None # Or return the uploaded image encoded?
             }
        
        # Run prediction
        result = model.predict(contents)
 
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Inspection Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": list(model_manager.models.keys())}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
