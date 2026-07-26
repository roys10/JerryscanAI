# JerryscanAI contributor and agent guide

## Mission

JerryscanAI detects surface defects on jerrycans captured by fixed production-line cameras. The application is largely complete; current work focuses on building a reproducible model-development and evaluation process.

Read [docs/model-development.md](docs/model-development.md) before changing datasets, preprocessing, training, inference calibration, or Model Lab metrics.
Follow [docs/preprocessing.md](docs/preprocessing.md) when importing segmentation weights or generating derivative datasets.

## Repository map

- `backend/`: FastAPI service and model inference.
- `frontend/`: React user interface.
- `model_lab/`: Streamlit comparison UI. Treat its current metrics as exploratory until the calibration issues in the model-development guide are fixed.
- `training/`: reproducible dataset, preprocessing, and anomaly-model training workflows.
- `standalone_scripts/`: legacy deployment and offline inference utilities.
- `models/`: local model artifacts. Checkpoints are ignored by Git.
- `data_manifests/`: versioned sample identities and frozen split assignments. Large images do not belong in Git.
- `test_dataset/`: small local/manual evaluation data, not the source of truth for an experiment split.

## Non-negotiable ML rules

1. Raw captures are immutable. Never rename, overwrite, preprocess in place, or move them to create a split.
2. Split raw sample identities before preprocessing. Every raw/rembg/static-ROI/SAM/YOLO derivative must inherit the same split from the same manifest.
3. Finish and audit a preprocessing dataset before training its PatchCore model. A preprocessing model/config and its generated dataset are separate versioned artifacts.
4. Group correlated captures by production session or capture day. Never use a random frame-level split for production-line sequences.
5. Train on `train`, select preprocessing/model settings and thresholds on `val`, and evaluate `test` once after decisions are frozen.
6. Never report defect-detection AUROC, F1, or recall from an all-normal set. Normal-only holdouts measure false positives and score drift only.
7. Keep real faults out of normal training data. Record how normal labels were verified.
8. Compare one controlled variable at a time. Use the same manifest, model seed, resolution, and metrics when comparing preprocessors.
9. Store raw image anomaly scores for metrics. Do not compute AUROC from clipped percentages or per-image-normalized scores.
10. Record enough metadata to reproduce every model: Git commit, manifest hash, preprocessing config/checkpoint, model config, seed, package lock, thresholds, hardware, latency, and metrics.
11. Do not tune on the final test set, including by visually choosing masks, morphology, thresholds, or model checkpoints.

## G01 split policy

The verified baseline split is defined by `data_manifests/G01/split_v2.csv`:

- Train: captures from 2026-02-01 and 2026-02-02.
- Validation: captures from 2026-02-03.
- Test: captures from 2026-02-04 and the isolated 2026-02-05 capture.

This intentionally uses future production sessions as validation/test data. Do not replace it with an 80/20 random split. The February 5 image remains flagged as an isolated capture, but the owner confirmed that it used the same camera/setup.

## Agent work boundaries

Use these roles when parallel work is explicitly requested:

- Dataset steward: audits labels, duplicates, capture groups, manifests, and leakage. Owns `data_manifests/` and dataset utilities.
- Preprocessing researcher: evaluates static ROI, rembg, SAM, and YOLO segmentation. Stores masks separately and records exact configs.
- Model researcher: evaluates PatchCore and challengers using the frozen manifest and model-independent raw outputs.
- Integration owner: changes shared inference, Model Lab, dependency files, and experiment registry after research outputs are agreed.

Agents should not edit the same files concurrently. Research agents return evidence and recommendations; the integration owner makes shared changes. A handoff must state files inspected/changed, commands run, assumptions, unresolved questions, and metric evidence.

## Development commands

```powershell
uv sync
uv run python -m unittest discover -s tests
uv run --extra lab streamlit run model_lab/app.py
uv run uvicorn backend.main:app --reload
```

For the frontend:

```powershell
cd frontend
npm install
npm run dev
```

Generate or verify a dataset manifest with
`python -m training.datasets.create_dataset_manifest`; see
[data_manifests/README.md](data_manifests/README.md). Prefer the locked `uv`
environment over notebook-level `pip install` commands.

## Definition of done for an ML experiment

- The split manifest and its SHA-256 are recorded.
- Inputs and generated masks pass data QA; failures are counted rather than silently dropped.
- Configuration, seed, dependencies, and model artifact are versioned or hashed.
- Validation selected the threshold; test did not influence any decision.
- Image-level raw scores and latency are saved per sample.
- The report includes false-positive rate on normal data and recall at the chosen operating point on real, labeled faults.
- Results can be reproduced from a documented command without editing a notebook cell.
