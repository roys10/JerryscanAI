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
4.  **Start Server Locally**:
    ```bash
    cd backend
    python .\main.py
    ```

### 1B. Production Deployment (CI/CD via Docker)
The backend is configured for automated deployment via GitHub Actions and Docker by building a private image in the GitHub Container Registry (GHCR).

1. **Clone to Remote Server**: Ensure your remote Linux machine has the repository cloned to `~/jerryscanai` and that Docker is installed.
2. **GitHub Secrets**: First, generate a **Personal Access Token (classic)** with `read:packages` permissions on GitHub. Then, in your GitHub repository settings, add the following secrets:
   - `HOST_IP`: IP or domain of your Linux machine.
   - `HOST_USERNAME`: Your login username (e.g., `ubuntu`).
   - `HOST_PASSWORD`: Your login password.
   - `GHCR_PAT`: The Personal Access Token you generated (used to pull the image onto your server).
3. **Automated Deploy**: Pushing to the `main` branch will automatically trigger the `.github/workflows/deploy.yml` action. It will build the image, push it to GHCR, connect to your server, authenticate Docker, pull the latest image, and restart the container.

Alternatively, manually deploy on your server:
```bash
docker compose pull
docker compose up -d
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

---

The **Model Lab** is a dedicated research environment for evaluating new models (Padim/Patchcore) and comparing performance metrics side-by-side.

### Running the Lab
```bash
uv run --extra lab streamlit run model_lab/app.py
```

### Dataset Setup (`test_dataset/`)
> [!IMPORTANT]
> To use the Model Lab, download the `test_dataset` from the **JerryscanAI Google Drive** and place it in the project root.

Organize your test images by camera angle and category:
```
test_dataset/
├── front/
│   ├── normal/       # Good samples
│   ├── fault/        # Defective samples
│   └── ground_truth/ # (Optional) Semantic masks
├── back/
│   └── ...
├── side_l/
│   └── ...
└── side_r/
    └── ...
```

---

## 🛠 Features

-   **Multi-Angle Batch Inspection**: Simultaneous inspection of Front, Back, Side L, and Side R views.
-   **Professional Alerting System**:
    -   **Custom Rules**: "Failure Streaks" or "Pass Rate Drops" triggers.
    -   **Multi-Channel**: Multiple Email recipients and Webhooks per rule.
-   **History & Analytics**: Real-time stats, pass rates, and interactive historical logs with heatmaps.
-   **Multi-Model Management**: Hot-swappable model sets during runtime.
-   **Simulation Suite**: Batch process `test_images/` to verify alert rules and system logic.

## 📂 Project Structure

-   `backend/`: FastAPI core and alerting engine.
-   `backend/inference/`: AI logic, history persistence, and model management.
-   `frontend/`: React application (Live Dashboard & History).
-   `model_lab/`: Streamlit-based benchmarking and evaluation suite.
-   `models/`: Storage for versioned `.ckpt` weight files.
-   `test_dataset/`: Angle-aware directory for model evaluation.
-   `test_images/`: Samples for end-to-end system simulation.
