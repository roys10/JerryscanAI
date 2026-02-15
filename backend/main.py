
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.inference.manager import JerryScanModelManager
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
        print(f"Warning: Model checkpoint not found at {ckpt_path}")

@app.post("/inspect")
async def inspect_image(file: UploadFile = File(...), angle_id: str = None):
    try:
        contents = await file.read()
        
        # Get appropriate model (default or specific angle)
        model = model_manager.get_model(angle_id)
        
        # Run prediction
        result = model.predict(contents)
 
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": list(model_manager.models.keys())}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
