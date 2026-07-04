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
    The backend scans the top-level `models/` directory for subfolders. Each subfolder name becomes the model set name, and every `.ckpt` file inside that folder is loaded as an angle model.
    ```
    Project/JerryscanAI/
    ├── models/
    │   ├── model1/
    │   │   ├── G01.ckpt
    │   │   ├── G02.ckpt
    │   │   └── ...
    │   └── model2/
    │       ├── G01.ckpt
    │       └── ...
    ```
    Example: the repository already contains model sets named `Padim_v0_old` and `RembgAlignedPatchcore`; place your checkpoint files inside those folders so the backend can discover them automatically.
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
   - `SMTP_PASSWORD`: Your Gmail App Password (or email service SMTP password). The workflow will use this to populate `backend/.env` on the remote host at deploy time.
2.  **Automated Deploy**: Pushing to the `backend-CD` branch will automatically trigger the `.github/workflows/deploy.yml` action. It will build the image, push it to GHCR, create `backend/.env` on the remote server using the `SMTP_PASSWORD` secret, connect to your server, authenticate Docker, pull the latest image, and restart the container.

> The deployment uses `docker-compose.yml` and expects a `backend/.env` file on the remote host. The workflow automatically creates this file at deploy time by using the `SMTP_PASSWORD` GitHub secret. The compose file mounts `backend/.env` into the container and loads it via `env_file`, so SMTP and PORT values are available at runtime.


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
    Create a `.env` file in `frontend/` by copying `frontend/.env_template` and updating `VITE_BACKEND_URL` as needed.

3.  **Run Development Server**:
    ```bash
    npm run dev
    ```

### 1C. Backend Environment Setup
1.  Copy `backend/.env.example` to `backend/.env` and fill in your SMTP credentials and port.
2.  When running local Docker or remote deployment, ensure the backend environment file is available to the container so SMTP settings are resolved correctly.

> Note: The GitHub Actions deployment workflow currently pushes the Docker image and deploys `docker-compose.yml` to the remote host, but it does not automatically create `backend/.env` on the server. You must provision the remote `backend/.env` file manually or extend the workflow to populate it from secrets.


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
