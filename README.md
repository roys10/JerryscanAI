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

The manufacturing runtime uses one explicitly selected model-set folder. A
folder can declare one camera (schema 1.0) or multiple cameras (schema 1.1).
For every original camera image, the backend runs the folder's shared
preprocessing and the checkpoint declared for that angle, then returns the raw
anomaly score, a fixed-scale heatmap, a red defect-location contour, and the
preprocessing mask, when the selected pipeline produces one, as separate UI
views. The current camera contract is exactly 1025 x 1281 pixels; resized or
already preprocessed images are rejected before preprocessing.

Each usable folder has this shape:

```text
models/Patchcore_<preprocessing>_256_c10_seed42/
|-- model.json             # tracked runtime and preprocessing contract
|-- README.md              # tracked setup note
|-- G01.metadata.json ... G04.metadata.json  # tracked training records
|-- G01.ckpt ... G04.ckpt                    # local; ignored by Git
```

The stable `model.id` remains tied to the folder and training metadata, while
the optional `model.display_name` in `model.json` controls the name shown to
operators. Changing the display name does not require renaming the folder.

Each current folder already includes its matching metadata; normally you add
the checkpoints declared in `model.json`. Multi-angle files sit beside one
another in the same model-set folder. The rembg folders also need one shared
`u2net.onnx`. They first look beside `model.json`,
which makes the folder portable, then fall back to the shared
`models/preprocessing/rembg/u2net.onnx` copy.

Copy `backend/.env.example` to `backend/.env`, then select the model folder with
the `JERRYSCAN_MODEL_FOLDER` environment variable:

```dotenv
JERRYSCAN_MODEL_FOLDER=models/Patchcore_rembg_u2net_black_v1_256_c10_seed42
```

From the repository root, install the rembg dependency used by this example and
start the API:

```console
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
For multi-angle batches, preprocessing remains single-flight because the
preprocessing backend is shared, while the independent angle-specific
PatchCore engines may infer concurrently. This reduces end-to-end latency on
adequately provisioned hardware without changing model scores or decisions.
On startup, the backend verifies every declared checkpoint and any U2Net weight against
the SHA-256 and byte size recorded in `model.json` before either model is loaded.

Start the frontend in another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Each angle in `model.json` contains its own raw-score threshold. G01's current
thresholds came from its normal validation set. The multi-angle U2Net-black
contract explicitly marks G02-G04's current value of 34 as a temporary,
uncalibrated operator default; those values require angle-specific validation.
Raw thresholds are not percentages, and real labeled faults are still required
to validate defect recall. Invalid images or failed input preprocessing return
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

The container keeps Uvicorn on `0.0.0.0:8000` internally and Compose publishes
it on VM port 80 on all network interfaces. With the VM firewall allowing port
80, the API is therefore available at `http://<VM-IP>/health`; port 8000 does
not need to be opened externally.

For a normal local run, non-secret alert settings live in
`backend/config.json`. You may edit the SMTP server/user, recipients, webhook,
and alert rules there. Keep the SMTP password out of JSON and provide it only
through the `SMTP_PASSWORD` environment variable. Docker overrides the config
path so container settings persist under `runtime-data/`.

A push to `backend-CD` triggers `.github/workflows/deploy.yml`. The workflow
builds `ghcr.io/roys10/jerryscanai:latest`, copies Compose to the configured
remote server, creates `backend/.env` from the `SMTP_USER` and `SMTP_PASSWORD`
GitHub secrets, sets `JERRYSCAN_MODEL_FOLDER` to the deployed U2Net-black
model-set path, and restarts the backend. It does not run the local test suite.
Because checkpoints and ONNX weights are ignored by Git, the selected model
folder and its large artifacts must already exist in the remote server's
`~/jerryscanai/models` directory.

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
