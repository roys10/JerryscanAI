# Model development plan

## Current baseline and immediate conclusion

Keep PatchCore as the first baseline, but make the comparison protocol trustworthy before training many architectures. The current G01 data are 3,991 grayscale BMP captures at 1025×1281 from one fixed angle. They span four main capture days and one isolated later image. Adjacent images are extremely similar, so random splitting would overestimate generalization.

The first controlled benchmark should compare background-handling approaches with the same PatchCore configuration. Model families should only expand after raw scores, thresholds, and metrics are correct in Model Lab.

## Frozen G01 split v2

| Split | Capture sessions | Images | Purpose |
|---|---:|---:|---|
| Train | 2026-02-01 and 2026-02-02 | 1,997 | Fit normal-only models |
| Validation | 2026-02-03 | 994 | Select preprocessing, hyperparameters, and provisional normal cutoff |
| Test | 2026-02-04 and 2026-02-05 | 1,000 | Locked future-session evaluation |

The split is intentionally about 50/25/25 because only four complete production days are available. Holding out whole days is more important than obtaining a conventional ratio. Train and validation may be combined for a final fit only if a separate calibration set remains; refitting changes the score distribution and invalidates a threshold calibrated on the old validation model. The test set must stay untouched.

Confirmed on 2026-07-06:

- The manufacturer states that all 3,991 captures are defect-free.
- Adjacent captures are expected to represent different jerrycans.
- The isolated 2026-02-05 image used the same camera/setup.

The current `split_v2.csv` therefore uses `label=normal`. It contains 3,991 unique source hashes and has canonical manifest SHA-256 `0fab56bc5aa7034763430617af10d3e8d9aea2aa7e137fb865458fcc13168512`; the manifest hash normalizes CSV line endings so Windows and Linux record the same experiment identity. A deterministic 750-image audit found closely overlapping intensity histograms across splits and no missing, resized, or sampled hash-mismatched files. Validation/test retain moderate spatial variation between capture days, which is desirable for measuring future-session robustness rather than near-duplicate memorization.

Open metadata questions remain: whether capture days correspond to different lots/shifts or camera adjustments, and whether filename timestamps are reliable local camera time.

### Why retain validation when every image is normal?

Validation is still required to select preprocessing, resolution, PatchCore coreset ratio, model checkpoint, and a provisional normal-score threshold without looking at test. It measures false-positive behavior and day-to-day score drift. It cannot measure defect recall, F1, or useful two-class AUROC by itself. Add real labeled faults in a new manifest version (or a separate versioned fault manifest) without allowing the same physical fault/can/run into both validation and test.

For the current normal-only stage, use validation to choose a threshold tied to an acceptable false-reject rate—for example, a high normal-score quantile—and use test once to estimate the resulting false-positive rate on a later session. Do not select the model or threshold on test. Keep the 50/25/25 day split for now: 1,997 diverse normal training images are adequate for the first PatchCore learning-curve experiment, while whole-day isolation is more valuable than moving correlated frames into training.

## Dataset and derivative layout

Raw data stay outside Git. The manifest is the source of truth:

```text
data_manifests/G01/
├── split_v1.csv              # historical, labels unverified
├── split_v1.summary.json
├── split_v2.csv              # current verified-normal baseline
└── split_v2.summary.json

<external data root>/
├── raw/G01/<sample>.bmp
└── derived/G01/
    ├── masks/
    │   ├── static_roi_v1/<sample>.png
    │   ├── rembg_u2net_v1/<sample>.png
    │   ├── sam2_box_v1/<sample>.png
    │   └── yolo_seg_v1/<sample>.png
    └── images/
        ├── rembg_u2net_gray_v1/<sample>.png
        └── sam2_box_feathered_v1/<sample>.png
```

Store masks separately from composited RGB/grayscale images. A derivative record should include `parent_sample_id`, preprocessor name/version, checkpoint hash, config hash, output hash, and status. This permits testing gray fill, feathered alpha, mask dilation, and anomaly-map ROI suppression without rerunning segmentation.

## Preprocessing experiment order

| Priority | Variant | Rationale |
|---:|---|---|
| 1 | Raw image | Required control; proves whether preprocessing helps |
| 1 | Fixed crop/static ROI mask | Deterministic and well matched to a fixed camera; no mask jitter |
| 1 | Existing rembg pipeline | Reproduces the current baseline |
| 2 | rembg `u2net` plus `isnet-general-use` or `birefnet-general` | `rembg` is a wrapper around materially different models, not one model |
| 2 | SAM 2.1 tiny/small with one fixed box prompt per angle | Low annotation burden and more constrained than automatic mask generation |
| 3 | Single-object-class supervised YOLO instance segmentation | Likely stable after domain training, but needs reviewed masks and licensing review |

For each learned mask, compare separately versioned hard gray replacement, feathered boundary, and small-dilation configurations. Keep anomaly-map-only ROI suppression as a distinct arm: it preserves input pixels and does not prevent PatchCore from storing background patches. Apply the ROI before aggregating the image-level score, then calibrate a separate threshold for that scoring rule. Hard mask edges and mask jitter can become false anomalies. Preserve all can material and a narrow context margin so preprocessing does not erase boundary defects.

For rembg, record the package version, exact model/checkpoint hash, alpha threshold, and postprocessing. For SAM 2.1, freeze a deterministic choice among candidate masks using its predicted-quality score plus ROI, area, and component plausibility. Only test SAM video propagation if the captures are confirmed to be consecutive video frames; filename order alone is insufficient. If used, define reset behavior and measure drift.

Create a representative gold-mask audit set across lighting, position, can appearance, and known faults. Report IoU/Dice, boundary quality, catastrophic failure rate, area/centroid variability, preprocessing latency and peak memory, and downstream anomaly metrics. If captures are confirmed consecutive, also report adjacent-frame mask IoU or boundary displacement. Add gates for plausible area, centroid, bounding box, component count, and segmentation confidence; flagged masks must not silently enter training.

G01 is grayscale, while these pretrained segmenters conventionally consume RGB tensors. Freeze and record channel replication, normalization, resize/letterbox policy, image interpolation, and mask interpolation. Custom YOLO labels, SAM mask choices, mask thresholds, and morphology may use train/validation or a separate segmentation dataset only—not the locked anomaly test set.

Primary references: [rembg](https://github.com/danielgatis/rembg), [U²-Net](https://arxiv.org/abs/2005.09007), [SAM](https://arxiv.org/abs/2304.02643), [SAM 2](https://arxiv.org/abs/2408.00714), [official SAM 2 repository](https://github.com/facebookresearch/sam2), and [Ultralytics segmentation documentation](https://docs.ultralytics.com/tasks/segment/). rembg code is MIT, the official U²-Net repository is Apache-2.0, and official SAM/SAM 2 code and checkpoints are Apache-2.0; still verify each exact third-party rembg weight. Ultralytics documents AGPL-3.0 and Enterprise licensing options; resolve the deployment license before adopting its training code or models in production ([license summary](https://www.ultralytics.com/license)).

## Model shortlist

1. PatchCore remains the reference normal-only model. Benchmark resolution and coreset ratio; its memory and latency grow with the memory bank ([paper](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html)).
2. PaDiM is a cheap aligned-image sanity baseline ([paper](https://arxiv.org/abs/2011.08785)).
3. EfficientAD-S is the first production-oriented challenger because it targets low-latency unsupervised anomaly detection ([paper](https://openaccess.thecvf.com/content/WACV2024/papers/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.pdf)).
4. Reverse Distillation provides a trainable reconstruction/distillation alternative ([paper](https://openaccess.thecvf.com/content/CVPR2022/html/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.html)).
5. Dinomaly is a later, heavier modern candidate, especially if one model will eventually cover several views ([paper](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Dinomaly_The_Less_Is_More_Philosophy_in_Multi-Class_Unsupervised_Anomaly_CVPR_2025_paper.html)).

Do not start with a full model × preprocessor Cartesian product. First run PatchCore on raw, static ROI, current rembg, and SAM box-prompt inputs. For each challenger, retain raw as the control and add the best PatchCore-selected preprocessing variant; a preprocessing choice can interact with the model architecture.

## PatchCore benchmark matrix

Start small and expand only when results justify it:

- Input: raw, static ROI, current rembg, SAM 2.1 box prompt.
- Resolution: 256 and one higher resolution that preserves the smallest business-relevant defect.
- Coreset ratio: 0.01, 0.05, and 0.10.
- Training-set learning curve: approximately 250, 500, 1,000, and all training images.
- Seed: at least three for trainable models. Run multiple PatchCore coreset seeds as well, or first demonstrate that its result is seed-stable.

Square-resizing 1025×1281 images to 256×256 distorts aspect ratio and may erase small defects. Treat letterboxing/cropping and resolution as explicit variables, but do not change them while comparing preprocessors.

## Evaluation protocol

The production decision metric should be defect recall at an agreed maximum false-positive rate on normal cans. Also record:

- raw-score AUROC and AUPR when both classes exist;
- precision, recall, specificity/FPR, balanced accuracy, confusion matrix, and the exact threshold;
- pixel AUPR and AUPRO/region overlap when defect masks exist;
- median and p95 latency, throughput, peak memory, artifact size, and preprocessing failure rate;
- per-sample raw score, prediction, timing, model ID, preprocessing ID, and manifest sample ID;
- paired block/cluster bootstrap confidence intervals by physical can, run, or time block when comparing models. An ordinary per-image bootstrap would treat correlated adjacent frames as independent, and one complete test day still provides weak evidence about session-to-session variation.

Threshold selection belongs to validation. If validation is normal-only, choose a provisional normal-score quantile tied to the allowed false-reject rate and label it provisional. Do not use the maximum normal score as an unexamined production threshold.

## Current training and Model Lab status

The original training and comparison-path problems have been addressed for the
current PatchCore workflow:

- training consumes the frozen manifest and explicit train/validation folders;
- the locked test folder is not passed to Anomalib during fitting;
- experiment metadata are written beside each checkpoint;
- Model Lab compares raw image scores and treats heatmap normalization as
  display-only;
- each registered model supplies its own preprocessing contract; and
- normal-only data do not produce defect AUROC, F1, precision, or recall.

Important limitations remain. PatchCore is still the only supported model
family in the current lab path, pixel/region metrics require reviewed defect
masks, and none of the four current G01 models has a threshold calibrated with
enough real labeled faults. Their provisional per-model raw-score thresholds
are ceilings above the maximum score on the 994 normal validation images, so
they establish an observed validation false-positive operating point only.
Current normal-only comparisons cannot establish the best defect detector or
qualify defect recall scientifically.

Anomalib already provides model-aware preprocessing, postprocessing, evaluators, and deployment inference. Prefer those interfaces over reconstructing calibration from checkpoint attributes ([deployment guide](https://anomalib.readthedocs.io/en/stable/markdown/guides/reference/deploy/), [metrics reference](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/metrics/index.html)).

## Execution milestones

1. **Complete:** materialize `split_v2` and dry-run `python -m training.models.train_patchcore`; it verifies the manifest and consumes explicit train/validation folders without a random internal split.
2. **Complete for PatchCore:** Model Lab saves raw model outputs, keeps display
   heatmaps separate, applies per-model preprocessing, and exposes calibration
   only when a threshold contract exists.
3. **Partially complete:** raw letterbox, fixed crop, and U²-Net variants are
   complete and audited. ISNet failed its smoke-test mask QA. SAM 2.1 is
   checkpoint/config ready but awaits a Linux runtime.
4. Add a reviewed real-fault validation/test set and define the acceptable production false-positive rate.
5. Run the controlled PatchCore matrix, select preprocessing, then add PaDiM and EfficientAD-S.
