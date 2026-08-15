# Local model folders

The manufacturing backend uses exactly one folder at a time. It does not scan
this directory and does not choose a model alphabetically. Set
`JERRYSCAN_MODEL_FOLDER` to the folder you want to run.

## What a folder contains

```text
Patchcore_rembg_u2net_black_v1_256_c10_seed42/
|-- model.json
|-- README.md
|-- .gitignore
|-- G01.ckpt ... G04.ckpt
|-- G01.metadata.json ... G04.metadata.json
`-- u2net.onnx             # rembg only; optional if shared fallback exists
```

Commit `model.json`, `README.md`, `.gitignore`, and the small reviewed angle
metadata JSON files. A schema-1.1 multi-angle contract contains one
checkpoint/metadata/threshold entry per required camera under `angles`. The
complete live preprocessing configuration and U2Net weight are shared because
every angle was trained with that same preprocessing contract. Schema-1.0
single-angle folders remain supported.

The folder name and `model.id` are the stable technical identity used to bind
the folder to its training metadata. Change `model.display_name` independently
to control the human-readable name shown in the manufacturing UI.

Keep `G*.ckpt` and learned preprocessing weights local. They are large
supplied artifacts and are ignored in each current folder. The loader requires
the tracked training metadata at runtime and checks that its
model set, angle, image size, and preprocessing ID match `model.json`. This
catches a checkpoint/metadata pair copied into the wrong folder.

The production Docker image includes the Git-tracked portions of `models/`.
Compose mounts only the selected bundle's four `.ckpt` files and
`u2net.onnx` from the host at their expected paths. Do not mount the entire
host `models/` directory over `/app/models`: a directory bind mount would hide
the current contracts and metadata baked into the image.

## Current choices

- `Patchcore_raw_letterbox_v1_256_c10_seed42`
- `Patchcore_fixed_crop_v1_256_c10_seed42`
- `Patchcore_rembg_u2net_gray_v1_256_c10_seed42`
- `Patchcore_rembg_u2net_black_v1_256_c10_seed42`

Raw letterbox and fixed crop need no preprocessing weight. Each rembg folder
looks for its own `u2net.onnx` first and then for
`models/preprocessing/rembg/u2net.onnx`. To move a rembg model folder to another
machine by itself, copy the ONNX file into that model folder too.

## Select and run one model

Copy `backend/.env.example` to `backend/.env` and set the
`JERRYSCAN_MODEL_FOLDER` environment variable to the selected directory. For
example:

```dotenv
JERRYSCAN_MODEL_FOLDER=models/Patchcore_rembg_u2net_black_v1_256_c10_seed42
JERRYSCAN_GPU_MODEL_COUNT=0
JERRYSCAN_CPU_INFERENCE_CONCURRENCY=1
JERRYSCAN_GPU_INFERENCE_CONCURRENCY=1
```

Then run these commands from the repository root:

```console
uv sync --extra preprocess-rembg
uv run uvicorn backend.main:app --reload
```

Upload original camera images for any non-empty subset of currently available
angles. The backend reports both configured and available angles through
`/health`; a missing, corrupt, incompatible, or failed angle is listed
separately and an upload for it is rejected explicitly. Angles omitted from a
batch are not inspected, do not affect its overall decision, and are not stored
in that history session. Do not manually preprocess images; the
selected folder's shared pipeline runs automatically before the matching angle checkpoint. The current contracts
expect original 1025 x 1281 camera dimensions. Any other size, including a
1024 x 1024 derivative, correctly returns `WRONG_INPUT` before preprocessing.
Device placement is deliberately opt-in. `JERRYSCAN_GPU_MODEL_COUNT=0` keeps
all loaded angles on CPU and is the Docker/VM default. A positive count assigns
that many usable angle models to CUDA in `model.json` order, skipping angles
whose artifacts are unavailable; all remaining models stay on CPU. Use `1` on
a 6 GB GTX 1660 Super. A requested CUDA model falls back individually to CPU if
CUDA is unavailable or initialization fails. Requested devices, actual devices,
and fallback reasons are visible at `/health`.

U2Net preprocessing stays on CPU and is single-flight. CPU and GPU PatchCore
inference each default to one operation at a time. The corresponding
`JERRYSCAN_CPU_INFERENCE_CONCURRENCY` and
`JERRYSCAN_GPU_INFERENCE_CONCURRENCY` values should only be raised after
measuring memory use and latency on the target machine.

The backend checks each local checkpoint and rembg weight against the expected
SHA-256 and byte size in `model.json` before Torch or ONNX loads it. Schema-1.1
angle artifacts are fail-closed independently: usable siblings remain
available, and the failed angle receives a structured reason without borrowing
another angle's checkpoint. The service is not ready only when no angle loads
or the shared preprocessing artifact/runtime fails. Partial coverage is marked
`degraded` and is not reported as ready for full production decisions.

Each angle declares its own threshold on `raw_patchcore_image_score`; thresholds
are never inferred from another checkpoint. G01's U2Net-black value of 34 is a
provisional ceiling from its normal validation set. The current G02-G04 values
of 34 are explicitly marked temporary, uncalibrated operator defaults in
`model.json` and must be replaced after angle-specific validation. Raw-score
thresholds are not percentages. A defensible final threshold requires labeled
fault validation followed by one locked-test evaluation.
