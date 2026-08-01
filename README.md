# JerryscanAI

JerryscanAI is a production-line surface-defect inspection system for jerrycans captured by fixed cameras. The repository contains the manufacturing application, reproducible dataset and preprocessing workflows, PatchCore training, and a separate Model Lab for controlled model comparison.

PatchCore is the only model family supported by the current training and Model Lab workflow. The design leaves room for future model families, but they are not implemented or advertised as supported.

## Repository areas

- `backend/`: FastAPI manufacturing-line API, inspection history, alerts, and current production inference.
- `frontend/`: React/Vite manufacturing-line interface.
- `training/`: frozen dataset manifests, derivative generation, preprocessing runtime, and PatchCore training.
- `data_manifests/`: versioned raw-sample identities and split assignments. Images are stored outside Git.
- `model_lab/`: independent FastAPI and React/Vite PatchCore comparison application.
- `models/`: versioned model-folder documentation and local model/preprocessing artifacts. See [models/README.md](models/README.md); large weights are ignored by Git.
- `tests/`: dataset, preprocessing, training, and Model Lab safeguards.

Read [AGENTS.md](AGENTS.md) before contributing. Model experiments must follow [docs/model-development.md](docs/model-development.md), preprocessing must follow [docs/preprocessing.md](docs/preprocessing.md), and training commands are documented in [training/README.md](training/README.md).

## Environment

Install [`uv`](https://docs.astral.sh/uv/) and synchronize the Python environment from the repository root:

```powershell
uv sync
```

Frontend dependencies are installed separately in their respective application directories with `npm install`.

## Manufacturing application

Start the production API from the repository root:

```powershell
uv run uvicorn backend.main:app --reload
```

Start the production frontend in another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Model checkpoints are currently discovered from subdirectories under `models/`, with one checkpoint per camera angle:

```text
models/
└── <model-set>/
    ├── G01.ckpt
    ├── G02.ckpt
    ├── G03.ckpt
    └── G04.ckpt
```

> [!WARNING]
> The production inference path has not yet been migrated to the new immutable preprocessing and calibration contracts used by Model Lab. A model registered successfully in Model Lab is not automatically production-ready. Complete an end-to-end GPU validation and production inference migration before deployment, especially for fixed-crop or segmentation-based preprocessing.

For container deployment, use the repository's Docker configuration:

```bash
docker compose pull
docker compose up -d
```

## Model Lab

Model Lab is a separate research application. It does not start with or expose the manufacturing application. The current implementation supports comparisons of one to four PatchCore models.

Start the Model Lab API:

```powershell
uv run python -m model_lab
```

Start its independent frontend in a second terminal:

```powershell
cd model_lab/frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5174`. The API listens on `http://127.0.0.1:8010`. Set `JERRYSCAN_LAB_WORKSPACE` to place imported checkpoints, caches, and comparison results outside the default ignored `.model_lab/` directory.

### Comparison workflow

1. Register an existing PatchCore checkpoint, discover a model directory, or upload a checkpoint and its metadata.
2. Complete any missing angle, training-manifest, preprocessing, derivative-root, or calibration fields shown by the model contract editor.
3. Select one to four models, one folder of original camera images, and an image amount (or **All**). Optionally name the comparison.
4. Keep the default `Unlabeled` choice for score/latency/visual exploration, or explicitly confirm that every selected image is verified normal to enable normal-only false-positive metrics.
5. Run the comparison. Model Lab hashes the folder into a canonical internal snapshot and selects the same raw sample identities once for every model.
6. Each model receives its declared preprocessing. A verified derivative or cache may be reused; otherwise Model Lab preprocesses the original image and records the provenance.
7. Review paired raw scores, false alarms when labels and calibration permit them, preprocessing quality flags, latency, actual model inputs, masks, and heatmaps.

The normal folder workflow is exploratory and does not ask for manifests or
splits. Expand **Advanced / Official benchmark** only when running a controlled
evaluation from an existing frozen manifest. Selection seed, force-live
preprocessing, official split, and locked-test confirmation are also there.

Use **Force live preprocessing** when measuring end-to-end behavior. Normal comparisons may reuse only derivatives whose source, manifest, configuration, preprocessing-model, and output hashes match the registered contract.

The Single Image screen always starts from an original camera image and executes the selected model's live preprocessing pipeline.

### Evaluation safeguards

- Exploratory folder comparison is the default; validation is the default split only inside Official benchmark mode.
- Exploratory folders are never labeled from their directory names. Unlabeled comparisons report score distributions, latency, QA, and visual results only.
- A model's stored manifest hash describes its training data; it is not required to match a new evaluation folder or official evaluation manifest. Different training manifests are recorded and warned about.
- A test comparison must use the complete frozen test split and creates a one-shot lock tied to immutable model contracts.
- Runs snapshot checkpoint, preprocessing, calibration, and manifest identities and verify them again when resumed.
- Metrics use raw image anomaly scores, never clipped display percentages.
- Models are summarized only on sample identities completed successfully by every selected model.
- With an all-normal dataset, AUROC, F1, precision, and recall are unavailable rather than reported as zero.
- Models without an image threshold remain uncalibrated: raw-score and latency analysis is available, but decisions and false-positive rate are not.
- Per-image heatmap scaling is display-only and is never used as an evaluation score.
- Uncalibrated models show Decision unavailable. The legacy production `50%` threshold and percentages such as `PASS 38.5%` came from flawed normalization and must not be reused as Model Lab calibration.

The old `model_lab/app.py` Streamlit interface remains temporarily as a legacy exploratory reference and must not be used for benchmark claims. Full Model Lab details are in [model_lab/README.md](model_lab/README.md).

## Training and preprocessing

Raw images are immutable and remain outside Git. Frozen manifests select raw sample identities before preprocessing, and every derivative inherits the same split.

The shared preprocessing runtime is used by batch derivative generation and Model Lab live preprocessing. Current configurations include raw letterboxing, fixed crop, U2Net/rembg background replacement, and prepared SAM support. PatchCore training consumes audited derivative datasets and records checkpoint metadata next to each model.

See [training/README.md](training/README.md) for exact commands and [docs/training-experiment-log.md](docs/training-experiment-log.md) for the recorded experiment history.

## Verification

Run the Python test suite with:

```powershell
uv run python -m unittest discover -s tests
```

Verify the Model Lab frontend with:

```powershell
cd model_lab/frontend
npm run lint
npm run build
```
