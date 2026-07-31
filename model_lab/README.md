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
   checkpoint bundle.
2. Complete any explicitly listed contract gaps. A checkpoint is never assigned
   a model family, camera angle, preprocessor, manifest, or threshold by guessing.
3. Select one to four ready PatchCore models, the original image root, frozen
   manifest, split, image count, and sample seed.
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

## Production integration boundary

The shared preprocessing runtime is ready for production use, but the current
manufacturing backend has not yet been migrated to resolve and enforce these
registered model contracts. A Model Lab registration is therefore **not a
deployable production bundle yet**. Production integration must make
`backend/inference` load the same immutable preprocessing, calibration, and
PatchCore contracts before any selected lab model is deployed on the line.

The previous Streamlit files remain temporarily as a legacy exploratory UI and
must not be used for benchmark claims.
