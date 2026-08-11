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
```

Then run these commands from the repository root:

```console
uv sync --extra preprocess-rembg
uv run uvicorn backend.main:app --reload
```

Upload one original camera image for every angle declared by the selected
folder. Do not manually preprocess them; the selected folder's shared pipeline
runs automatically before the matching angle checkpoint. The current contracts
expect original 1025 x 1281 camera dimensions. Any other size, including a
1024 x 1024 derivative, correctly returns `WRONG_INPUT` before preprocessing.
Multi-angle batch requests run distinct PatchCore engines concurrently after a
single-flight shared preprocessing stage. This favors inspection latency on a
machine with sufficient CPU/GPU and memory; requests for the same angle remain
serialized through that angle's engine.

The backend checks each local checkpoint and rembg weight against the expected
SHA-256 and byte size in `model.json` before Torch or ONNX loads it. A partial,
wrong, or corrupted copy leaves the model not ready instead of running it.

Each angle declares its own threshold on `raw_patchcore_image_score`; thresholds
are never inferred from another checkpoint. G01's U2Net-black value of 34 is a
provisional ceiling from its normal validation set. The current G02-G04 values
of 34 are explicitly marked temporary, uncalibrated operator defaults in
`model.json` and must be replaced after angle-specific validation. Raw-score
thresholds are not percentages. A defensible final threshold requires labeled
fault validation followed by one locked-test evaluation.
