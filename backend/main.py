
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
    # Identify checkpoint path relative to this file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_path = os.path.join(base_dir, "model.ckpt")
    
    if os.path.exists(ckpt_path):
        try:
            model_manager.load_model("default", ckpt_path)
            print(f"Loaded default model from {ckpt_path}")
        except Exception as e:
            print(f"Failed to load default model: {e}")
    else:
        print("\n" + "="*60)
        print("CRITICAL WARNING: MANUAL ACTION REQUIRED")
        print("="*60)
        print(f"Model checkpoint NOT found at: {ckpt_path}")
        print("Please COPY 'model.ckpt' to the project root or backend folder.")
        print("The system cannot perform inspections without this file.")
        print("="*60 + "\n")

@app.post("/inspect")
async def inspect_image(file: UploadFile = File(...), angle_id: str = None):
    try:
        # Check if models are loaded BEFORE reading file (fail fast)
        if not model_manager.models:
             raise HTTPException(
                 status_code=503, 
                 detail="Model not loaded. Server is running but 'model.ckpt' is missing."
             )

        contents = await file.read()
        
        # Get appropriate model (default or specific angle)
        try:
            model = model_manager.get_model(angle_id)
        except KeyError:
             # If specific angle fails, try default. If that fails, it's a 503.
             # But the check above handles the empty case.
             # This handles "angle_id not found" specifically if we had multiple models.
             raise HTTPException(status_code=404, detail=f"Model for angle '{angle_id}' not found.")
        
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
