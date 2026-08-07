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
|-- G01.ckpt
|-- G01.metadata.json
`-- u2net.onnx             # rembg only; optional if shared fallback exists
```

Commit `model.json`, `README.md`, `.gitignore`, and the small reviewed
`G01.metadata.json`. The model contract contains
the PatchCore identity, angle, checkpoint names, complete live preprocessing
configuration, expected original image size, artifact SHA-256/byte sizes, and
the raw PatchCore image-score decision threshold.

The folder name and `model.id` are the stable technical identity used to bind
the folder to its training metadata. Change `model.display_name` independently
to control the human-readable name shown in the manufacturing UI.

Keep `G01.ckpt` and learned preprocessing weights local. They are large
supplied artifacts and are ignored in each current folder. The loader requires
the tracked training metadata at runtime and checks that its
model set, angle, image size, and preprocessing ID match `model.json`. This
catches a checkpoint/metadata pair copied into the wrong folder.

## Current G01 choices

- `Patchcore_raw_letterbox_v1_256_c10_seed42`
- `Patchcore_fixed_crop_v1_256_c10_seed42`
- `Patchcore_rembg_u2net_gray_v1_256_c10_seed42`
- `Patchcore_rembg_u2net_black_v1_256_c10_seed42`

Raw letterbox and fixed crop need no preprocessing weight. Each rembg folder
looks for its own `u2net.onnx` first and then for
`models/preprocessing/rembg/u2net.onnx`. To move a rembg model folder to another
machine by itself, copy the ONNX file into that model folder too.

## Select and run one model

From the repository root in PowerShell:

```powershell
$env:JERRYSCAN_MODEL_FOLDER = (Resolve-Path "models/Patchcore_rembg_u2net_black_v1_256_c10_seed42")
uv sync --extra preprocess-rembg
uv run uvicorn backend.main:app --reload
```

Upload the original G01 camera image. Do not manually preprocess it; the
selected folder's pipeline runs automatically. The fixed-crop contract also
expects the original 1025 x 1281 camera dimensions. Any other size, including a
1024 x 1024 derivative, correctly returns `WRONG_INPUT` before preprocessing.

The backend checks each local checkpoint and rembg weight against the expected
SHA-256 and byte size in `model.json` before Torch or ONNX loads it. A partial,
wrong, or corrupted copy leaves the model not ready instead of running it.

Each contract has a provisional threshold on `raw_patchcore_image_score`: raw
letterbox 35, fixed crop 36, and both U2Net variants 34. The values are rounded
ceilings above each model's maximum score on 994 normal `split_v2` validation
images, producing zero observed validation false positives. They are not
percentages. A defensible final threshold still requires labeled fault
validation followed by one locked-test evaluation.
