# JerryScan AI

JerryScan AI is a deep learning-based visual inspection system designed to automatically detect defects in plastic jerrycans on a production line. The system utilizes unsupervised anomaly detection to identify manufacturing flaws such as pinholes, deformations, and contamination, ensuring high-quality output with minimal false negatives.

## Project Overview

Traditional rule-based inspection systems often struggle with the subtle variances in production environments (lighting, rotation, tolerances). JerryScan AI addresses this by learning the "normal" appearance of a jerrycan from defect-free samples and flagging any deviations as anomalies.

**Goal:** Reduce false negatives and provide a robust, production-ready AI inspection system that adapts to new formats without extensive labeled datasets.

## Key Features

-   **Deep Learning Anomaly Detection:** Uses unsupervised learning (PatchCore/PaDiM) trained only on "OK" samples.
-   **Real-Time Inspection:** optimized for high-speed production lines.
-   **Visual Feedback:** Genreates heatmaps explicitly identifying defect regions.
-   **Operator Dashboard:** Web-based UI for live monitoring, defect review, and system configuration.
-   **Traceability:** Logs every inspection with timestamps, images, and confidence scores.

## Architecture

The system is modular, consisting of four main components:

1.  **Image Acquisition:** Captures high-resolution images via industrial cameras or manual upload.
2.  **Inference Engine:** Runs the anomaly detection model (PyTorch) to compare images against the learned baseline.
3.  **Defect Classification:** Evaluates anomaly scores against configurable thresholds to decide Pass/Fail.
4.  **Operator Interface:** A React-based dashboard for visualization and control.

### Data Flow
`Image Capture -> Preprocessing -> Inference Engine -> Anomaly Scoring -> Classification -> Dashboard & Logging`

## Technology Stack

-   **Backend:** Python (FastAPI/Flask)
-   **AI/ML:** PyTorch (PatchCore/PaDiM)
-   **Frontend:** React
-   **Database:** SQLite / PostgreSQL

## Team

**Group 512**

-   Roy Segev
-   Ron Sheva
-   Matanel Lazerovich
-   Neta Weinreb
-   Ariel Gluzman

**Supervisor:** Moshe Butman

## License

[MIT](LICENSE)
