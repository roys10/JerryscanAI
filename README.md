# JerryscanAI

An AI-powered surface defect detection system using Anomalib (Padim).

## 🚀 Deployment Instructions

### 1. Backend Setup
1.  **Install `uv`** (if not already installed):
    ```bash
    # Windows
    irm https://astral.sh/uv/install.ps1 | iex
    
    # macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Sync dependencies & Start Server** (from the project root):
    ```bash
    uv sync
    uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
    ```
3.  **IMPORTANT: Model Deployment**
    The model file (`model.ckpt`) is too large for Git and is ignored.
    You must **manually copy** `model.ckpt` to the project root:
    ```
    Project/JerryscanAI/
    ├── backend/
    ├── frontend/
    └── model.ckpt  <-- PLACE HERE
    ```
    Server runs at `http://localhost:8000`.
    Server runs at `http://localhost:8000`.

### 2. Frontend Setup
1.  Navigate to `frontend/`:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Run the development server:
    ```bash
    npm run dev
    ```
    App runs at `http://localhost:5173`.

## 🛠 Features
-   **Multi-Angle Inspection**: Supports Front/Back/Side views.
-   **Real-time Anomaly Detection**: Upload an image to inspect.
-   **Dual Visualization**: Toggle between **Heatmap** (Anomaly Map) and **Segmentation Mask** (Red outline).
-   **Precise Scoring**: Matches Anomalib CLI output (Float32 precision).

## 📦 Project Structure
-   `backend/`: FastAPI server + `inference/` (Model logic).
-   `frontend/`: React + Vite application.
-   `cli_predictions/`: Sample images for testing.
-   `model_inference_standalone_script.py`: CLI tool for testing logical parity.
