# JerryscanAI 🧠

A professional AI-powered surface defect detection system built with **FastAPI** and **React**, utilizing **Anomalib (Padim)** for high-precision anomaly detection on production lines.

## 🚀 Deployment Instructions

### 1. Backend Setup
1.  **Install `uv`** (if not already installed):
    ```bash
    # Windows
    irm https://astral.sh/uv/install.ps1 | iex
    
    # macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Sync dependencies**:
    ```bash
    uv sync
    ```
3.  **Model Deployment (Hierarchical)**
    The system supports multiple model sets. Create a subfolder for each set inside `models/`:
    ```
    Project/JerryscanAI/
    ├── models/
    │   ├── Standard/
    │   │   ├── front.ckpt
    │   │   ├── back.ckpt
    │   │   └── ...
    │   └── Optimized_V2/
    │       └── ...
    ```
4.  **Start Server**:
    ```bash
    cd backend
    python .\main.py
    ```

### 2. Frontend Setup
1.  **Install Dependencies**:
    ```bash
    cd frontend
    npm install
    ```
2.  **Environment Setup**:
    Create a `.env` file in `frontend/` (copy from `.env_templates`)

3.  **Run Development Server**:
    ```bash
    npm run dev
    ```

## 🛠 Features

-   **Multi-Angle Batch Inspection**: Supports simultaneous inspection of Front, Back, Side L, and Side R views.
-   **Professional Alerting System**:
    -   **Custom Rules**: Define alerts based on "Failure Streaks" or "Pass Rate Drops".
    -   **Multi-Channel**: Supports multiple Email recipients and Generic Webhooks per rule.
    -   **Immediate Persistence**: Dynamic rule management with instant backend synchronization.
-   **History & Analytics**: Comprehensive dashboard for tracking pass rates, total scans, and historical inspection reports with heatmaps.
-   **Multi-Model Management**: Switch between different trained model sets on the fly from the UI.
-   **Simulation Suite**: Built-in "Simulation Trigger" to process `test_images/` batches for system verification and alert testing.

## 📂 Project Structure

-   `backend/`: FastAPI core, configuration management, and alert dispatching.
-   `backend/inference/`: Model loading, Padim logic, and history persistence.
-   `frontend/`: React + Vite + Lucide Icons + CSS (Inspection & History).
-   `models/`: Hierarchical storage for `.ckpt` weight files.
-   `test_images/`: Directory for batch simulation samples.
-   `standalone_scripts/`: Utilities for standalone model testing and parity verification.
