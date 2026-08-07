# JerryscanAI Model Lab

Model Lab is a separate PatchCore research application in this repository. It
does not start with, modify, or expose the manufacturing-line application. It
uses original camera images plus a frozen manifest, selects the same sample IDs
once, and supplies every model with the preprocessing contract registered with
that model.

## Start

From the repository root, start the API:

```powershell
uv sync
uv run python -m model_lab
```

In a second terminal, start the independent React interface:

```powershell
cd model_lab/frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5174`. The API listens only on
`http://127.0.0.1:8010` by default. Set `JERRYSCAN_LAB_WORKSPACE` to move the
registry, uploads, preprocessing cache, and persistent comparisons outside the
repository.

## Workflow

1. Register a checkpoint already on disk, discover a models folder, or upload a
   checkpoint with its metadata and preprocessing configuration.
2. Complete any explicitly listed contract gaps. A checkpoint is never assigned
   a model family, camera angle, preprocessor, manifest, or threshold by guessing.
3. Select one to four ready PatchCore models, one folder containing original
   camera images, and an image count or **All**. Optionally name the comparison.
   Keep images unlabeled for exploration, or explicitly confirm that all are
   verified normal.
4. Run the comparison. Models execute sequentially and results append after each
   sample, so an interrupted job can resume without repeating completed work.
5. Review raw-score distributions, false alarms when an image threshold is
   declared, preprocessing provenance, quality flags, and separate latency.

For repository-backed models, keep `G01.ckpt` and `G01.metadata.json` beside
each other in the appropriate model-set folder. The metadata is small and
should normally be committed after review; checkpoints and preprocessing
weights remain outside Git. See [the model artifact guide](../models/README.md)
for the required layout and artifact policy.

Each comparison snapshots the complete registered model contracts and verifies
checkpoint, preprocessing-config, and preprocessing-model hashes before a run
or resume. Results are summarized only over sample IDs successfully evaluated
by every selected model. An incomplete paired set is labeled `incomplete` and
can be resumed to retry failed rows.

Validation is the default. A locked-test evaluation must use the entire test
split and creates `.model_lab/locked_test_record.json`, tied to the manifest,
sample identities, and immutable model contracts. Model Lab refuses another
test evaluation while this record exists. Resetting it is an explicit
administrative filesystem action after the experiment owner documents why the
previous final evaluation is invalid; no reset is exposed in the normal UI.

Thresholds entered in the model editor are recorded as `manual_unverified`.
A scientifically calibrated threshold should later include its calibration
method, source manifest/split, sample-identity hash, target FPR, and timestamp.
With all-normal data, AUROC, F1, precision, and recall are shown as unavailable
instead of zero.

Current imported PatchCore models are uncalibrated unless their registry record
contains a documented image-score calibration. In that state Model Lab reports
the raw image score but shows **Decision unavailable**. Do not import the legacy
production `50%` cutoff: the previously shown values such as `PASS 38.5%` came
from flawed pixel/per-image percentage normalization and are neither raw
PatchCore scores nor valid Model Lab thresholds.

PatchCore evaluation calls the checkpoint's inner torch model after applying
the checkpoint training transform exactly once. Raw anomaly maps are persisted
unchanged as `.npy`. Heatmaps and overlays use per-map scaling only for display,
are labeled display-only, and never feed scores, thresholds, or metrics.

## Exploratory folders and official benchmarks

The default screen is intentionally simple: choose original camera images and
each model applies its own registered preprocessing. Model Lab recursively
scans supported images, hashes every file, creates stable sample IDs, and saves
the canonical internal evaluation snapshot with the comparison. A subset uses
the hidden deterministic seed 42 unless changed under Advanced; all models
always receive the same selected IDs. Folder names never create labels.

Expand **Advanced / Official benchmark** to use an existing frozen manifest and
split. This is the only mode that exposes locked-test semantics. A registered
model's manifest hash identifies its training data, not the evaluation data;
models trained on different manifests may be compared on a new folder or
official manifest, with that difference retained as a warning.

## Manufacturing-runtime boundary

Model Lab and the manufacturing application share preprocessing code, but they
remain separate applications. A Model Lab registration does not select or
modify the model used by the manufacturing backend.

The manufacturing backend loads exactly one local model folder selected with
`JERRYSCAN_MODEL_FOLDER`. That folder has its own tracked `model.json`, which
binds the checkpoint, training metadata, preprocessing configuration, original
G01 dimensions, and any preprocessing weight by identity and hash. See
[the local model-folder guide](../models/README.md). Model Lab registrations
continue to use the lab registry and workspace so experiments cannot silently
change the line runtime.

The manufacturing folders currently use the owner's provisional raw-score
threshold 60 (`score >= 60` means `FAIL`). Model Lab calibration remains
independent, and its results must not be treated as production PASS/FAIL
qualification until labeled real faults are used for validation and the locked
test is evaluated after all decisions are frozen.

The previous Streamlit files remain temporarily as a legacy exploratory UI and
must not be used for benchmark claims.
