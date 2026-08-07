# JerryscanAI

JerryscanAI is a production-line surface-defect inspection system for jerrycans captured by fixed cameras. The repository contains the manufacturing application, reproducible dataset and preprocessing workflows, PatchCore training, and a separate Model Lab for controlled model comparison.

PatchCore is the only model family supported by the current training and Model Lab workflow. The design leaves room for future model families, but they are not implemented or advertised as supported.

## Repository areas

- `backend/`: FastAPI manufacturing-line API, inspection history, alerts, and local inference.
- `frontend/`: React/Vite manufacturing-line interface.
- `training/`: frozen dataset manifests, derivative generation, preprocessing runtime, and PatchCore training.
- `data_manifests/`: versioned raw-sample identities and split assignments. Images are stored outside Git.
- `model_lab/`: independent FastAPI and React/Vite PatchCore comparison application.
- `models/`: model-folder contracts and local model/preprocessing artifacts. See [models/README.md](models/README.md); learned artifacts are ignored by Git.
- `tests/`: dataset, preprocessing, training, runtime, and Model Lab safeguards.

Read [AGENTS.md](AGENTS.md) before contributing. Model experiments must follow [docs/model-development.md](docs/model-development.md), preprocessing must follow [docs/preprocessing.md](docs/preprocessing.md), and training commands are documented in [training/README.md](training/README.md).

## Environment

Install [`uv`](https://docs.astral.sh/uv/) and synchronize the Python environment from the repository root:

```powershell
uv sync
```

Frontend dependencies are installed separately in their respective application directories with `npm install`.

## Manufacturing application

The current manufacturing runtime is intentionally simple and G01-only. You
choose one model by pointing the backend at its folder. For every original
camera image, the backend runs that folder's preprocessing and PatchCore
checkpoint, then returns the raw anomaly score, a fixed-scale heatmap, a red
defect-location contour, and the preprocessing mask as separate UI views.
The original G01 contract is exactly 1025 x 1281 pixels; resized or already
preprocessed images are rejected before preprocessing.

Each usable folder has this shape:

```text
models/Patchcore_<preprocessing>_256_c10_seed42/
|-- model.json             # tracked runtime and preprocessing contract
|-- README.md              # tracked setup note
|-- G01.metadata.json      # tracked training/reproducibility record
|-- G01.ckpt               # supplied locally; ignored by Git
```

The stable `model.id` remains tied to the folder and training metadata, while
the optional `model.display_name` in `model.json` controls the name shown to
operators. Changing the display name does not require renaming the folder.

Each current folder already includes its matching metadata; normally you add
only that run's `G01.ckpt`. The rembg folders also need `u2net.onnx`. They first look beside `model.json`,
which makes the folder portable, then fall back to the shared
`models/preprocessing/rembg/u2net.onnx` copy.

In PowerShell, select a folder and start the API:

```powershell
$env:JERRYSCAN_MODEL_FOLDER = (Resolve-Path "models/Patchcore_rembg_u2net_black_v1_256_c10_seed42")
uv sync --extra preprocess-rembg
uv run uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/health`. A complete folder reports
`ready`; a missing or mismatched model artifact is reported by `/health` and
inspection requests return HTTP 503 rather than an inspection result.
PatchCore inference prefers CUDA automatically. If CUDA is unavailable or the
checkpoint cannot load and warm up on the GPU, startup retries on CPU. The
selected `inference_device` and any `device_fallback_reason` are reported by
`/health`.
On startup, the backend verifies the checkpoint and any U2Net weight against
the SHA-256 and byte size recorded in `model.json` before either model is loaded.

Start the frontend in another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Each model's `model.json` contains its own provisional raw-score threshold:
raw letterbox 35, fixed crop 36, and both U2Net variants 34. Each value is the
rounded ceiling above that model's maximum score on the 994 normal `split_v2`
validation images, producing zero observed validation false positives. These
are not percentages, and real labeled faults are still required to validate
defect recall. Invalid images or failed input preprocessing return
`WRONG_INPUT`; model/runtime failures are HTTP errors and are not recorded as
defective cans.

The operator UI converts the raw anomaly score to a relative quality index:
100% means no measured anomaly and 70% is the configured failure boundary.
Values at or below 70% are `FAIL`. This index is a monotonic presentation of
the same raw-score rule, not confidence, probability, accuracy, or a second
decision threshold.

For Docker, create `backend/.env` from `backend/.env.example` and run
`docker compose up --build`. Compose loads it as the container environment,
mounts the same file read-only for the backend, mounts
`models/` read-only, and keeps
runtime files under `runtime-data/`. Inspection history is stored in the active
`runtime-data/inspections_history.db` SQLite database. This project keeps that
same database file in Git as requested; `settings.json` and SQLite's temporary
WAL/SHM files remain ignored.

For a normal local run, non-secret alert settings live in
`backend/config.json`. You may edit the SMTP server/user, recipients, webhook,
and alert rules there. Keep the SMTP password out of JSON and provide it only
through the `SMTP_PASSWORD` environment variable. Docker overrides the config
path so container settings persist under `runtime-data/`.

The original `.github/workflows/deploy.yml` workflow is retained. A push to
`backend-CD` builds `ghcr.io/roys10/jerryscanai:latest`, copies Compose to the
configured remote server, creates `backend/.env` from the `SMTP_USER` and
`SMTP_PASSWORD` GitHub secrets plus the selected container model path, and
restarts the backend. It does not run the local
test suite. Because checkpoints and ONNX weights are ignored by Git, the
selected model folder and its large artifacts must already exist in the remote
server's `~/jerryscanai/models` directory.

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

The shared preprocessing runtime is used by batch derivative generation, Model Lab live preprocessing, and the local manufacturing runtime. Current configurations include raw letterboxing, fixed crop, U2Net/rembg background replacement, and prepared SAM support. PatchCore training consumes audited derivative datasets and records checkpoint metadata next to each model.

See [training/README.md](training/README.md) for exact commands and [docs/training-experiment-log.md](docs/training-experiment-log.md) for the recorded experiment history.

## Verification

Run the Python test suite with:

```powershell
uv run python -m unittest discover -s tests
```

Verify both frontends with:

```powershell
cd frontend
npm run lint
npm run build

cd ../model_lab/frontend
npm run lint
npm run build
```
