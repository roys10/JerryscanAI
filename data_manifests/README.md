# Dataset manifests

Versioned manifests define sample identity and split membership without committing multi-gigabyte image data. Preprocessing pipelines and models must join on `sample_id`; they must not create new random splits.

## G01 baseline

Generate the current verified-normal whole-day split from the external raw data:

```powershell
uv run python -m training.datasets.create_dataset_manifest `
  --source E:\LearningProjects\AI\JerryscanAI\training_data\G01 `
  --manifest data_manifests\G01\split_v2.csv `
  --train-dates 2026-02-01 2026-02-02 `
  --val-dates 2026-02-03 `
  --test-dates 2026-02-04 2026-02-05 `
  --label normal `
  --date-flag 2026-02-05=isolated_capture `
  --date-flag 2026-02-05=setup_confirmed `
  --expected-count 3991 `
  --hash
```

`split_v1.csv` is retained as the pre-confirmation artifact with `label=unverified`; use `split_v2.csv` for experiments. Do not overwrite a manifest after an experiment has recorded its hash. Use `--materialize-output <path>` only when a framework requires physical `{train,val,test}/<label>` folders. Materialization never changes the source, requires a new output path, and cleans up its temporary directory if it fails.

Materialize the verified split on the same drive using hardlinks, then validate the training command without loading Anomalib:

```powershell
python -m training.datasets.materialize_dataset_split `
  --source E:\LearningProjects\AI\JerryscanAI\training_data\G01 `
  --manifest data_manifests\G01\split_v2.csv `
  --output E:\LearningProjects\AI\JerryscanAI\training_data\G01_split_v2

python -m training.models.train_patchcore `
  --dataset-root E:\LearningProjects\AI\JerryscanAI\training_data\G01_split_v2 `
  --manifest data_manifests\G01\split_v2.csv `
  --dry-run
```

The training script verifies exact sample IDs for all three folders. During fitting it passes only `train` and `val` to Anomalib; `test` remains locked.

## Schema

`split_v2.csv` contains:

- stable identity: `schema_version`, `sample_id`, `source_relpath`;
- capture metadata: `camera_angle`, `captured_at_local`, `capture_date`, `session_id`, `sequence_no`;
- experiment assignment: `split`, `label`, `defect_type`;
- file metadata: `width`, `height`, `channels`, `source_sha256`;
- review metadata: `quality_flags`, `notes`.

`captured_at_local` intentionally has no UTC offset until the camera timezone and clock are confirmed. The summary JSON records split/date counts and the manifest SHA-256.

Derivative datasets need a separate manifest that retains `parent_sample_id` and adds the preprocessor/version, checkpoint and config hashes, output path/hash, and status. Never edit the raw split assignment in a derivative manifest.
