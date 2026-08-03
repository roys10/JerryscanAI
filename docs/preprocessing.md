# Preprocessing dataset workflow

## Execution order

PatchCore models are trained only after their preprocessing dataset is complete and audited:

1. Freeze raw sample IDs and splits in `data_manifests/G01/split_v2.csv`.
2. Import or train the preprocessing model and record its exact artifact hash.
3. Generate images and masks for all train/validation/test IDs without changing split membership.
4. Review automatic QA, mask stability, and representative images.
5. Dry-run `train_patchcore.py` against the derivative dataset.
6. Train the same PatchCore configurations across approved variants.

The preprocessing model may use train and validation data for configuration. The locked anomaly-test images must not be used to choose prompts, thresholds, morphology, checkpoints, or supervised segmentation labels.

## Current variants

| ID | Status | Purpose |
|---|---|---|
| `raw_letterbox_v1` | Complete | Model-free aspect-ratio-preserving control |
| `rembg_u2net_gray_v1` | Complete | Existing-method equivalent with pinned U²-Net and reproducible masks |
| `rembg_u2net_black_v1` | Complete | Black-background recomposition of the completed aligned U²-Net derivative; no U²-Net rerun |
| `rembg_isnet_gray_v1` | Full run blocked by smoke-test QA | Newer rembg model retained as a documented rejected/pending variant |
| `fixed_crop_v1` | Complete | Model-free conservative crop based on training-mask bounds |
| `sam2_tiny_box_gray_v1` | Checkpoint/config prepared; Linux runtime unavailable locally | SAM 2.1 tiny with one frozen box prompt for G01 |
| `yolo_jerrycan_seg_v1` | Pending reviewed labels/license | Supervised single-class production segmentation candidate |

The official rembg library supports reusable inference sessions and automatically downloaded model weights. This project pins rembg 2.0.76 and stores weights in `models/preprocessing/rembg/`; each derivative manifest records the model SHA-256. See the [official rembg repository](https://github.com/danielgatis/rembg).

The ISNet weight was imported and pinned, but its 10-image train/validation
smoke test produced component-cleanup warnings on 4/10 samples. Manual review
found intermittent white-background regions retained inside the handle opening.
Do not generate the full ISNet derivative until a revised, versioned mask rule
passes a new smoke test; rejecting the variant is also a valid outcome.

The completed G01 variants have the following audited properties:

| Variant | Images | Masks | Failures/flags | Median preprocessing |
|---|---:|---:|---:|---:|
| `raw_letterbox_v1` | 3,991 | n/a | 0 | Recorded in its derivative manifest |
| `fixed_crop_v1` | 3,991 | n/a | 0 | 133.7 ms/image |
| `rembg_u2net_gray_v1` | 3,991 | 3,991 | 0 | 593.1 ms/image |
| `rembg_u2net_black_v1` | 3,991 | 3,991 | 0 | Recorded in its derivative manifest; U²-Net was not rerun |

All four complete variants contain exactly 1,997 train, 994 validation, and 1,000
locked-test identities from `split_v2.csv`; all pass the PatchCore training dry
run. U²-Net mask area ranges from 0.4837 to 0.4988 with one retained connected
object per image.

## Output contract

Each variant is stored outside Git:

```text
<external>/training_data/derived/G01/<preprocessing_id>/
├── train/normal/*.png
├── val/normal/*.png
├── test/normal/*.png
├── masks/train/normal/*.png       # segmentation variants only
├── masks/val/normal/*.png
├── masks/test/normal/*.png
├── preprocessing_config.json
├── derivative_manifest.csv
└── summary.json
```

The derivative manifest records the parent sample and manifest hash, preprocessing/config/model versions and hashes, output/mask hashes, geometry, timing, status, and quality flags. Dataset generation verifies every raw source hash. Failed or out-of-range masks keep the variant under `.partial`; completed variants are promoted atomically.

Imported and planned preprocessing weights are tracked in
`training/preprocessing/configs/model_registry.json`. Large artifacts remain
ignored by Git; their hashes and source/license review are versioned.

## Commands

Install the pinned rembg environment:

```powershell
uv sync --extra preprocess-rembg
```

Generate a variant:

```powershell
.\.venv\Scripts\python.exe -m training.preprocessing.preprocess_dataset `
  --source E:\LearningProjects\AI\JerryscanAI\training_data\G01 `
  --manifest data_manifests\G01\split_v2.csv `
  --config training\preprocessing\configs\rembg_u2net_gray_v1.json `
  --output-root E:\LearningProjects\AI\JerryscanAI\training_data\derived\G01
```

If a long run is interrupted, repeat the command with `--resume`. Never delete the source dataset or reuse a preprocessing ID with changed settings.

Create the controlled black-background counterpart without re-running U²-Net:

```powershell
python -m training.preprocessing.recompose_dataset `
  --parent-root E:\LearningProjects\AI\JerryscanAI\training_data\derived\G01\rembg_u2net_gray_v1 `
  --manifest data_manifests\G01\split_v2.csv `
  --config training\preprocessing\configs\rembg_u2net_black_v1.json `
  --output-root E:\LearningProjects\AI\JerryscanAI\training_data\derived\G01
```

The recomposer verifies the completed parent contract, parent metadata, frozen
split identities, and every parent output/mask hash before recomposition. It
retains foreground RGB values where the aligned binary mask is at least 128
and writes RGB zero elsewhere. Masks are hardlinked when possible (otherwise
copied), and the new dataset is written to `.partial` then promoted atomically
only when every sample succeeds. Use `--resume` only with the matching partial
directory; a changed configuration or parent needs a new preprocessing ID.

## SAM 2.1 workflow

Use the official SAM 2.1 tiny checkpoint first; it is the lowest-cost useful SAM comparison. Determine one G01 box prompt from training images only, freeze it in configuration, and use a deterministic candidate-selection rule based on predicted quality plus plausible area, centroid, and component count. Do not treat filename order as video unless capture continuity is independently confirmed. SAM 2 code and checkpoints are Apache-2.0; follow the [official repository](https://github.com/facebookresearch/sam2).

The checkpoint is already imported and hash-pinned. The official installation
instructions require Linux and strongly recommend WSL on Windows. WSL is not
accessible on the current workstation, so run the SAM smoke test in a
Linux/WSL or Kaggle environment rather than forcing an unsupported native
Windows installation.

Before full generation:

1. Use U²-Net training masks to estimate conservative object bounds.
2. Review at least 50 diverse training masks and 25 validation masks.
3. Freeze the box prompt and mask-selection rules.
4. Smoke-test boundary defects, handle holes, cap edges, and the bottom textured region.
5. Generate all splits with the shared derivative contract.

## YOLO segmentation workflow

YOLO requires reviewed polygon/mask labels; generated masks are not accepted directly as ground truth. Build an initial 100-image annotation set from train/validation only, sampled across normal appearance and segmentation failure modes. Correct SAM/U²-Net masks manually, freeze mask semantics, then run a 25/50/100-label learning curve before deciding whether more labels are justified.

After an approved segmentation variant is complete, create the seed annotation set with:

```powershell
python -m training.preprocessing.create_segmentation_annotation_set `
  --source E:\LearningProjects\AI\JerryscanAI\training_data\G01 `
  --derivative-root E:\LearningProjects\AI\JerryscanAI\training_data\derived\G01\rembg_u2net_gray_v1 `
  --output E:\LearningProjects\AI\JerryscanAI\training_data\annotations\G01_seg_v1 `
  --train-count 100 `
  --val-count 25
```

The selector uses mask area and geometry diversity and always selects zero locked-test samples. Its seed masks must be manually corrected before YOLO training.

Ultralytics documents AGPL-3.0 and Enterprise licensing. Do not install or train the Ultralytics implementation for production until the project owner chooses a compatible license. See the [official segmentation documentation](https://docs.ultralytics.com/tasks/segment) and [license summary](https://www.ultralytics.com/license).

## Acceptance checks

- Every variant has exactly 1,997 train, 994 validation, and 1,000 test images.
- Sample IDs match `split_v2`; no row changes split.
- No source, output, or mask hash is missing.
- Mask area, component count, bounding box, centroid, and failures are summarized.
- Representative boundaries are visually reviewed at original resolution.
- Segmentation latency and peak memory are recorded separately from PatchCore.
- PatchCore receives the same resolution and hyperparameters for controlled preprocessing comparisons.
