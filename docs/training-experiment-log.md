# G01 PatchCore experiment log

## Purpose and current conclusion

This document is the presentation-ready record of the G01 PatchCore
experiments. It explains what the model learns, which variables are controlled,
what has actually been tested, what remains unmeasured, and how results should
be interpreted.

The current experiment asks one narrow question:

> With the same normal images, frozen split, PatchCore configuration, and seed,
> which of four background-handling pipelines gives the most robust anomaly
> scores on a later normal production session?

This stage can compare normal-data robustness and false-positive behavior. It
cannot establish which model detects defects best because every currently
verified G01 image is normal.

Current status as of 2026-07-31:

- All four preprocessing datasets are complete and identity-validated.
- The frozen stage-one configuration is implemented and reproducible.
- Standard all-GPU PatchCore completed feature extraction for the first
  variant, then exceeded the college GPU memory during memory-bank
  construction.
- CPU temporary embedding storage then completed the first full G01 training
  run for `raw_letterbox_v1` with all 1,997 training images and a 10% coreset.
- Canonical `G01.ckpt` and `G01.metadata.json` artifacts were produced. The
  checkpoint is approximately 1.3 GiB, but load integrity and artifact hashes
  have not yet been verified.
- No raw validation scores or anomaly-quality metrics have been reported, and
  the other three preprocessing variants remain untrained.

This log complements [model-development.md](model-development.md) and
[preprocessing.md](preprocessing.md). Add a new result row after every material
training attempt. Never overwrite a failed attempt with its replacement.

## How PatchCore training and scoring work

PatchCore is a normal-only anomaly detector. It does not learn a defect class
and does not update the pretrained backbone with gradient descent in this
experiment.

1. **Feature extraction.** Each normal training image is resized to the
   configured input size and passed through a pretrained convolutional
   backbone. The current experiment uses WideResNet50-2 and reads intermediate
   features from `layer2` and `layer3`.
2. **Patch embeddings.** Intermediate feature maps preserve spatial
   information. PatchCore combines local features into an embedding for each
   image region, producing many patch vectors per image rather than one global
   vector.
3. **Normal memory bank.** Patch embeddings from all normal training images are
   collected. This full pool describes the normal appearance of the jerrycan
   and any image content retained by the selected preprocessing pipeline.
4. **Coreset selection.** Storing every patch is expensive. Greedy
   K-center selection retains a representative subset. Stage one keeps 10% of
   the complete patch pool.
5. **Nearest-neighbor scoring.** At inference, query-image patches are compared
   with the normal memory bank. Large nearest-neighbor distances indicate
   regions unlike the training data. These distances feed the anomaly map and
   image-level anomaly score; Anomalib's `num_neighbors=9` controls the
   neighborhood used by its PatchCore scoring/reweighting implementation.

The memory bank is therefore the trained model's essential learned artifact.
Preprocessing matters because background pixels, hard mask edges, crop
boundaries, alignment differences, and foreground texture can all become
patches in that bank.

CPU embedding offload does not change this learning rule. It changes where
temporary pre-coreset tensors are stored while the same embeddings and
`KCenterGreedy` algorithm are used. Training and evaluation batch sizes are
also operational memory/throughput controls; they do not intentionally change
the training identities, feature definition, or final coreset rule.

## Frozen G01 data roles

The active manifest is `data_manifests/G01/split_v2.csv`.

| Split | Capture sessions | Images | Exact role |
|---|---|---:|---|
| Train | 2026-02-01 and 2026-02-02 | 1,997 | Supply verified-normal patch embeddings used to build the memory bank |
| Validation | 2026-02-03 | 994 | Compare preprocessing/model settings, examine normal-score drift, and select a provisional normal cutoff without looking at test |
| Test | 2026-02-04 plus one isolated 2026-02-05 capture | 1,000 | Locked later-session estimate after all choices are frozen |

Manifest facts:

- 3,991 unique grayscale BMP captures at 1025 x 1281.
- All labels are `normal`, based on manufacturer/project-owner confirmation.
- Neighboring captures are understood to show different jerrycans.
- The isolated 2026-02-05 image used the same camera/setup.
- Canonical manifest SHA-256:
  `0fab56bc5aa7034763430617af10d3e8d9aea2aa7e137fb865458fcc13168512`.

Capture date is not treated as a predictive feature. Whole capture days are
group boundaries because adjacent production-line frames share strongly
correlated imaging conditions. A random frame split would risk overly
optimistic evaluation. With only four complete days, preserving independent
future sessions is more valuable than moving more correlated frames into
training; this is why the split is approximately 50/25/25 rather than 80/10/10.

`split_v1.csv` has the same identities, hashes, and assignments, but its labels
were still `unverified`. `split_v2.csv` records the verified-normal state and is
the only manifest used by the current trainer.

### What normal-only validation can measure

It can measure:

- the distribution of raw anomaly scores on later normal production data;
- false-positive/false-reject rate at a frozen threshold;
- score drift and stability across normal sessions;
- latency, throughput, peak memory, and artifact size;
- preprocessing failures and mask stability.

It cannot measure:

- defect recall or missed-defect rate;
- precision or defect-class F1;
- meaningful two-class AUROC or AUPR;
- whether a pipeline removed or weakened real defect evidence;
- whether lower normal scores imply better defect separation.

Real, reviewed defect images are required before selecting the production model
or claiming defect-detection performance.

## Experiment variable dictionary

The classification below distinguishes variables that define the scientific
model/input comparison from operational settings that primarily affect
execution. Both kinds must be recorded for reproducibility.

| Variable | Meaning | Current value | Type and effect |
|---|---|---|---|
| `preprocessing_id` | Versioned input-generation pipeline, including geometry and background treatment | Four stage-one values listed below | **Scientific:** changes image content and therefore learned patch embeddings |
| `image_size` | Square tensor size passed to PatchCore preprocessing | 256 | **Scientific:** changes retained spatial detail and patch count |
| `backbone` | Pretrained feature extractor architecture | `wide_resnet50_2` | **Scientific:** changes representation and memory requirements |
| `layers` | Backbone stages used for local embeddings | `layer2`, `layer3` | **Scientific:** changes feature scales/dimensions |
| pretrained weights | Whether/how the backbone was initialized | ImageNet-pretrained Anomalib/timm weights (`pre_trained=True`) | **Scientific:** changes the feature representation; exact package/artifact provenance must be retained |
| train batch size | Images feature-extracted together | 8 | **Operational:** bounds transient extraction memory/throughput; frozen for reproducibility |
| evaluation batch size | Validation images scored together | 1 | **Operational:** bounds the patch-to-memory-bank distance-matrix peak; frozen for reproducibility |
| workers | Data-loader worker processes | 4 | **Operational:** affects input throughput and host resources |
| coreset sampling ratio | Fraction of complete patch pool retained | 0.10 | **Scientific:** changes memory-bank coverage, artifact size, inference cost, and potentially accuracy |
| `num_neighbors` | PatchCore scoring neighborhood | 9 | **Scientific:** changes anomaly scoring |
| seed | Random state for data/model components and coreset selection | 42 | **Scientific/reproducibility:** coreset choice can vary by seed |
| embedding storage | Temporary location of the full pre-coreset pool | `cpu` | **Operational:** reduces peak VRAM; does not intentionally change embeddings or coreset algorithm |
| train sample count | Normal images contributing embeddings | 1,997 | **Scientific:** changes normal coverage and memory-bank size |
| validation sample count | Normal images used for selection/calibration | 994 | **Scientific evaluation design:** fixed future session |
| threshold method | Rule converting raw image score to normal/anomalous | Pending | **Scientific evaluation:** must be selected on validation and frozen before test |
| target false-positive rate | Acceptable normal cans rejected by the system | Pending business decision | **Scientific/production objective:** required to choose a threshold and report recall at an operating point |
| score calibration | Mapping, if any, from raw score to comparable probability/percentage | None in stage one | **Scientific evaluation:** never compare clipped or per-image-normalized percentages as raw model evidence |
| accelerator | Hardware used for tensor operations | One CUDA GPU | **Operational:** record exact hardware/runtime because speed and numerical behavior can differ |

Changing any scientific variable creates a new experiment arm. Operational
changes must still be logged, but CPU storage or a smaller evaluation batch
must not be presented as a different learned model unless measurements show
that they alter the saved memory bank.

## Frozen stage-one configuration

| Configuration item | Frozen value |
|---|---|
| Angle | G01 |
| Manifest | `data_manifests/G01/split_v2.csv` |
| Model | Anomalib PatchCore 2.2 |
| Backbone/layers | `wide_resnet50_2`; `layer2,layer3` |
| Pretrained | Yes |
| Input | 256 x 256 |
| Train/validation/locked test | 1,997 / 994 / 1,000 |
| Train/evaluation batch | 8 / 1 |
| Workers | 4 |
| Coreset ratio | 0.10 |
| Neighbors | 9 |
| Seed | 42 |
| Embedding storage | CPU until coreset construction |
| Accelerator | GPU, one device |
| Test use during fit | None; identity validation only |

### Command flow

The portable notebook
`training/models/train-patchcore_notebook.ipynb` calls the canonical module
`training.models.train_patchcore`. For each preprocessing variant it first runs
the command with `--dry-run`, then removes that flag for actual training.

```bash
python -m training.models.train_patchcore \
  --dataset-root <DATA_ROOT>/<preprocessing_id> \
  --manifest data_manifests/G01/split_v2.csv \
  --angle G01 \
  --preprocessing-id <preprocessing_id> \
  --model-set Patchcore_<preprocessing_id>_256_c10_seed42 \
  --image-size 256 \
  --batch-size 8 \
  --eval-batch-size 1 \
  --num-workers 4 \
  --accelerator gpu \
  --coreset-sampling-ratio 0.10 \
  --num-neighbors 9 \
  --embedding-storage cpu \
  --seed 42 \
  --models-dir <OUTPUT_ROOT>/models \
  --results-dir <OUTPUT_ROOT>/results
```

The trainer verifies exact train, validation, and test identities before
training. Anomalib receives train as `normal_dir` and validation as
`normal_test_dir`, mirrored for fit-time validation using
`ValSplitMode.SAME_AS_TEST`. The real test directory is never passed to
Anomalib.

## Stage-one preprocessing comparison

Only `preprocessing_id` changes across these four arms.

| Experiment arm | Input presented to PatchCore | Background/boundary hypothesis | Dataset QA |
|---|---|---|---|
| `raw_letterbox_v1` | Full original frame, aspect-ratio-preserving resize, gray padding | Control: PatchCore may learn stable production background, but background changes may cause false anomalies | 3,991 images; 1,997/994/1,000 identities; dry run passed |
| `fixed_crop_v1` | Deterministic crop `[0,0,780,1200]`, then resize/letterbox | Removing known irrelevant frame area may reduce background variation without learned-mask jitter | 3,991 images; zero failures; median preprocessing 133.7 ms/image; dry run passed |
| `rembg_u2net_gray_v1` | U2-Net foreground aligned/centered; exterior filled gray 128 | Object isolation may reduce environmental variation; hard mask/alignment errors may themselves become anomalies | 3,991 images plus masks; zero failures/flags; mask area 0.4837-0.4988; median 593.1 ms/image; dry run passed |
| `rembg_u2net_black_v1` | Same aligned U2-Net foreground/mask; exterior filled RGB zero | Tests background fill value while holding segmentation and alignment fixed | 3,991 images plus masks; U2-Net not rerun; zero failures/flags; dry run passed |

rembg is the inference wrapper; U2-Net is the segmentation model. Fixed crop
and letterbox are deterministic transformations, not learned preprocessing
models.

The gray-versus-black comparison is especially controlled because both use the
same completed U2-Net masks and aligned foreground pixels. A difference is
therefore attributable to background recomposition value, subject to ordinary
runtime variation.

Other preprocessing research is outside stage one:

- `rembg_isnet_gray_v1` was blocked after 4/10 smoke-test samples triggered
  component-cleanup warnings and review found retained white background inside
  the handle opening.
- SAM 2.1 tiny has a prepared checkpoint/configuration but no accepted full
  derivative.
- YOLO segmentation is deferred pending reviewed labels and a deployment
  licensing decision.

## Actual attempts and results

No anomaly-quality metric should be entered until raw per-sample scores are
saved. `Pending` means unmeasured, not zero.

| ID/date | Scope and configuration | Status/evidence reached | Checkpoint | Raw validation scores | Quality metrics | Resource results |
|---|---|---|---|---|---|---|
| QA-1, before training | All four variants; exact manifest/folder dry run | **Passed.** Each arm matched 1,997 train, 994 validation, 1,000 locked-test identities and canonical manifest hash | Not applicable | Not applicable | Dataset identity only | Not measured |
| RUN-1, 2026-07-29 | `raw_letterbox_v1`; frozen stage-one model; standard device embedding storage; train/eval batch 8/8 in that attempt | **Failed after feature extraction.** All 250 train batches completed in about 11 s; OOM occurred while stacking the complete embedding pool before coreset selection | None | None | None | 20.07 GiB GPU; attempted additional 11.70 GiB allocation with only 7.15 GiB free |
| DEV-1, 2026-07-30 | CPU-offload implementation; synthetic 32 x 384 embedding pool | **Passed locally.** Standard `KCenterGreedy` returned a three-row coreset | Not a G01 model | Not applicable | Implementation smoke only | Not representative of G01 |
| QA-2, 2026-07-30 | Full G01 dry run with `--embedding-storage cpu` and evaluation batch wiring | **Passed.** Canonical hash and 1,997/994/1,000 identities verified; full repository suite passed 26 tests | Not applicable | Not applicable | Command/data contract only | No full embedding pool loaded |
| RUN-2, 2026-07-30 | `raw_letterbox_v1`; `CpuOffloadPatchcore`; all 1,997 train images; coreset 0.10; evaluation batch 1 | **Training and artifact production succeeded.** Metadata created at `2026-07-30T19:17:41.200539+00:00`; counts are 1,997/994/1,000 and `test_used_during_training=false` | `G01.ckpt` and `G01.metadata.json` both reported timestamp 2026-07-30 19:17. Checkpoint 1293.3936643600464 MiB (about 1.3 GiB). Hash/load integrity pending | Pending | Pending; no validation or defect claim | Fit 1329.343450333923 s (about 22 min 9 s). Peak GPU/RAM values not yet recorded in this log |
| RUN-3, pending | `fixed_crop_v1`; same configuration | **Ready but not yet verified as run** | Pending | Pending | Pending | Pending |
| RUN-4, pending | `rembg_u2net_gray_v1`; same configuration | **Ready but not yet verified as run** | Pending | Pending | Pending | Pending |
| RUN-5, pending | `rembg_u2net_black_v1`; same configuration | **Ready but not yet verified as run** | Pending | Pending | Pending | Pending |

During RUN-2, the notebook display remained at 17% after its connection became
stale/disconnected. This was not the final server-process state: training
continued independently and produced the timestamped checkpoint and metadata.
For long remote runs, verify the server process and canonical artifacts rather
than treating a frozen notebook progress display as proof of failure.

Earlier Linux path-capitalization, GUI OpenCV, and Pandas compatibility errors
occurred before useful training. They are preserved in the troubleshooting
appendix rather than treated as model results.

## Evaluation plan and metrics

### Stage-one normal validation

Save sample ID, preprocessing/model IDs, raw image score, inference time,
thresholded prediction, and a reproducible anomaly map when spatial review is
needed. Report score median/IQR/high quantiles, outliers by session, FPR at the
declared threshold, median/p95 latency, throughput, artifact size, peak
host/GPU memory, and preprocessing failure/latency separately.

Raw score scales may differ across independently fitted PatchCore models. A
lower median score in one arm does not by itself mean better anomaly detection.
Compare thresholded normal behavior under a predeclared calibration rule, score
stability, mask QA, resource costs, and eventually performance on the same real
defects.

If validation remains all-normal, use a predeclared high-score quantile tied to
an acceptable FPR; do not default uncritically to the maximum. The production
FPR requirement is not yet defined, so final calibration is pending.

### Locked normal test

After all choices are frozen, run the 1,000-image test once to estimate FPR,
validation-to-test score drift, and resource behavior. Do not tune after
inspection and then report the same test as unbiased.

### Required defect evaluation

With reviewed defects, make recall at the agreed maximum normal FPR the primary
metric. Also report precision, specificity/FPR, F1, balanced accuracy,
confusion matrix, raw-score AUROC/AUPR, and pixel AUPR/AUPRO when masks exist.
Break results down by defect type/size/position/session and use block/cluster
bootstrap uncertainty rather than treating adjacent frames as independent.

### Presentation-ready interpretation

For the mentor presentation:

- Say “best normal-session robustness” rather than “best defect detector” for
  stage-one results.
- Show the split by production session and explain leakage prevention before
  showing metrics.
- Present one controlled-variable table so the audience can see that only
  preprocessing changed.
- Show raw-score distributions and representative false positives; do not use
  clipped percentages or per-image-normalized heatmap maxima as comparable
  model scores.
- Label missing defect metrics as “not measurable with the current all-normal
  dataset,” not as zero or unavailable due to implementation.
- Separate preprocessing time, model fit time, inference time, and memory-bank
  size.
- Include failed runs only when they explain experimental constraints; do not
  present environment repairs as model findings.

## Planned controlled experiments

These are plans, not completed results. Do not run a full Cartesian product.

After stage one, save all raw validation scores; compare stability, false
positives under one calibration rule, segmentation QA, and cost; obtain real
defects before declaring a winner; then freeze preprocessing and vary one model
setting at a time.

### Recommended first training-size/coreset comparison

**Recommendation, not measured evidence:** prefer using all 1,997 training
images with a 1% coreset as the first resource-conscious arm. This exposes the
feature extractor and coreset selector to every available normal training image
while producing a smaller final memory bank than a 1,000-image/10% arm. The
expected benefit is broader observed normal-can diversity with less
nearest-neighbor work at inference, but validation must test that hypothesis.

At the current 256 px feature geometry, PatchCore produces approximately 1,024
candidate patch embeddings per image:

| Candidate configuration | Candidate patch pool | Retained coreset | Interpretation |
|---|---:|---:|---|
| All 1,997 train images, 1% coreset | Approximately 2,044,928 patches | Approximately 20,449 patches | Observes every training image, then compresses aggressively |
| Deterministic 1,000-image subset, 10% coreset | Approximately 1,024,000 patches | Approximately 102,400 patches | Observes fewer training images but retains many more patches from them |

Thus, the full-data/1% arm is expected to retain about one-fifth as many final
patches as the 1,000-image/10% arm. Exact counts should be read from the
resulting artifact because implementation rounding may differ.

The two controls answer different questions:

- **Training-image count** controls how many captured cans and normal
  variations the model can observe before compression. More images provide an
  opportunity for more diversity; correlated production frames mean they are
  not assumed to be 1,997 statistically independent conditions.
- **Coreset ratio** controls how aggressively the observed patch embeddings are
  compressed into the final memory bank. Lower ratios reduce bank size and are
  expected to reduce inference memory/distance work, but may discard useful
  normal coverage.

The full-data/1% and 1,000-image/10% candidates change two scientific variables
at once, so their direct comparison is useful for choosing a practical
configuration but cannot identify which variable caused a difference. After
that comparison, run the planned training-size and coreset-ratio arms while
holding the other variable fixed.

Do not assume the 1,997-image/1% model is superior, and do not claim defect
performance from normal-only validation. Compare its raw normal-score
stability, false-positive behavior, artifact size, and latency against the
deterministic 1,000-image configuration; evaluate defect recall only after
reviewed defects exist.

### PatchCore sensitivity studies

| Study | Planned values | Hold constant | Question |
|---|---|---|---|
| Resolution | 256 and at least one higher resolution chosen from the smallest business-relevant defect | Preprocessing, backbone/layers, train identities, seed, coreset ratio | Does extra spatial detail improve small-defect evidence enough to justify memory/latency? |
| Coreset ratio | 0.01, 0.05, 0.10 | Input, resolution, train identities, seed | How much memory-bank compression is possible without losing performance? |
| Training learning curve | Approximately 250, 500, 1,000, all 1,997 | Validation/test, preprocessing, model settings, deterministic subset rule | How many normal cans are needed for stable coverage? |
| Coreset seed stability | At least three seeds | All other variables | Is the selected memory bank/result stable to coreset sampling? |
| Neighbors | Values justified after baseline results | All other variables | Does scoring neighborhood improve defect/normal separation? |

The 1,000-image arm must use a deterministic, versioned subset of current train
identities. The same selected IDs must be used for all preprocessing arms,
while validation and test remain unchanged. It is a scientific training-size
experiment, not an invisible operational workaround.

Later challengers are PaDiM, EfficientAD-S, and, if justified, Reverse
Distillation. Use raw input as a control plus the best justified preprocessing
arm rather than repeating every model across every preprocessor.

## Artifacts and reproducibility contract

Each successful stage-one arm should produce:

```text
<OUTPUT_ROOT>/
├── training_environment.freeze.txt
├── models/
│   └── Patchcore_<preprocessing_id>_256_c10_seed42/
│       ├── G01.ckpt
│       └── G01.metadata.json
└── results/
    └── <model-independent per-sample raw outputs>
```

The checkpoint is insufficient by itself. Metadata must record:

- Git state, full command, manifest path/hash, preprocessing/config/model
  hashes, split counts, and `test_used_during_training=false`;
- model implementation, backbone/layers/pretraining, image size, coreset,
  neighbors, seed, train/eval batches, workers, accelerator, and embedding
  storage;
- software versions and platform, CPU/RAM, GPU/VRAM, and CUDA identity;
- fit time, peak process/CUDA memory, checkpoint path/hash/size, and load test;
- once available, calibration rule/threshold/target FPR and versioned
  per-sample scores, predictions, timings, and metrics.

Current code writes most training/hardware fields only after fitting succeeds.
For failures, retain the command, traceback, and external resource evidence.
Hash and load-test copied artifacts before marking a run complete.

## Appendix A: execution environment

### College server used for RUN-1

| Resource | Recorded value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU allocation | 20.07 GiB |
| System RAM | 1.5 TiB total; approximately 1.3 TiB available when measured |
| Swap | None |
| Python | 3.11.15 |
| PyTorch/torchvision | 2.12.1+cu130 / 0.27.1 |
| Driver/runtime report | NVIDIA driver 580.95.05; driver CUDA 13.0 |

CPU storage uses the large host-RAM allocation for the temporary full pool,
then runs standard `KCenterGreedy` on GPU. Estimated host peak is roughly
24 GiB during concatenation; actual G01 measurements are pending. Evaluation
batch 1 separately bounds validation distance-matrix VRAM. Neither is a
scientific model change. Higher-memory cloud GPU is only a contingency; no
Lightning.ai JerryscanAI result exists.

### Local workstation

The local Windows environment was used for manifest/dataset QA, preprocessing,
repository tests, and synthetic CPU-offload verification. No successful full
G01 PatchCore training result is recorded locally.

## Appendix B: condensed troubleshooting record

These were execution repairs, not experimental findings. None required dataset
regeneration.

| Issue | Point of failure | Resolution/evidence |
|---|---|---|
| Linux path capitalization | Before dataset validation | Corrected `/tf/Jerryscan-data/G01`; no images moved |
| GUI OpenCV required missing `libxcb.so.1` | Anomalib import before training | Replaced GUI wheel with `opencv-python-headless==4.13.0.92`; notebook verifies `GUI: NONE` |
| Pandas 3 incompatibility with Anomalib 2.2 Folder dataset | Datamodule setup | Pinned Pandas 2.3.3 / `<3` |
| Windows/Linux CSV line endings produced different byte hashes | Manifest reporting | Canonical line-ending-normalized hash now records `0fab56...` on both platforms; identities/splits never changed |
| All-GPU embedding concatenation OOM | After all 250 feature-extraction batches | Added operational CPU temporary storage; RUN-2 subsequently completed full G01 training/artifact production |

Relevant repository commits before the current uncommitted experiment work:

- `faedfdc`: normalize manifest hashing and pin headless OpenCV.
- `cce0518`: pin Pandas below 3 for Anomalib compatibility.

## Next evidence required

1. Hash and load-test the RUN-2 checkpoint and verify its metadata contents;
   record peak RAM/VRAM if recoverable.
2. Train the other three stage-one arms with no scientific-variable changes.
3. Export model-independent raw validation scores and anomaly maps.
4. Define the acceptable production false-positive rate with the project
   stakeholders.
5. Acquire/version reviewed real defect images and masks.
6. Select preprocessing/model/threshold on validation, then evaluate locked
   test once.
