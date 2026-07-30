# Training experiment log

## Purpose and update rule

This is the chronological source of truth for G01 anomaly-model training
attempts. It records what was actually run, the hardware and software involved,
the controlled variables, failures, artifacts, and decisions. It complements
the experiment policy in [model-development.md](model-development.md) and the
derivative-dataset contract in [preprocessing.md](preprocessing.md).

Update this file after every material training attempt, including failed
attempts. Do not silently replace an entry after fixing a problem. Add a new
entry and link it to the earlier failure. Keep observed facts separate from
recommendations.

Last updated: 2026-07-30.

## Current status

### Confirmed facts

- Four complete preprocessing variants are available for the controlled
  stage-one PatchCore comparison.
- All four variants inherit the same 3,991 identities and frozen split from
  `data_manifests/G01/split_v2.csv`.
- Folder and identity dry-run validation passes with 1,997 train, 994
  validation, and 1,000 locked-test images per variant.
- No PatchCore checkpoint has yet been produced by the recorded college-server
  attempts.
- The latest college-server attempt extracted all training embeddings for the
  first variant, then ran out of 20.07 GiB GPU memory while Anomalib combined
  the stored embeddings before coreset selection.
- The college server has enough system memory to try CPU offload: 1.5 TiB total
  RAM and approximately 1.3 TiB available at the time measured. It has no swap.
- CPU offload is the selected next experiment. A local implementation is
  present, the portable notebook selects it by default, and local unit/synthetic
  verification passes. A successful full college-platform GPU run is not yet
  recorded here.

### Current recommendation

Preserve the full 1,997-image training set, 256 x 256 input, WideResNet50-2
features, and 10% coreset while first testing CPU-offloaded embedding storage.
This changes memory placement rather than the scientific comparison variable.
If that does not work acceptably, create a deterministic, versioned
approximately 1,000-image training subset and treat it as an explicit
learning-curve experiment. Do not quietly drop samples.

## Frozen data identity and split

The active manifest is `data_manifests/G01/split_v2.csv`.

| Property | Recorded value |
|---|---|
| Camera angle | G01 |
| Source images | 3,991 grayscale BMP captures, 1025 x 1281 |
| Label state | All 3,991 confirmed normal by the manufacturer/project owner |
| Unique source hashes | 3,991 |
| Canonical manifest SHA-256 | `0fab56bc5aa7034763430617af10d3e8d9aea2aa7e137fb865458fcc13168512` |
| Train | 1,997 images from 2026-02-01 and 2026-02-02 |
| Validation | 994 images from 2026-02-03 |
| Test | 1,000 images from 2026-02-04 plus one isolated image from 2026-02-05 |

The capture date has no intended predictive meaning. Capture sessions/days are
used only as grouping boundaries because neighboring production-line frames
are highly correlated. A random frame split could place near-duplicate
conditions in train and evaluation and overstate generalization.

The approximately 50/25/25 ratio is intentional. With only four complete
production days, future-session isolation is more useful than moving more
correlated frames into training. The owner also confirmed that neighboring
frames show different jerrycans and that the isolated 2026-02-05 image used the
same camera/setup.

`split_v1.csv` is historical. It has the same 3,991 IDs, hashes, and split
assignments, but its labels were `unverified`. `split_v2.csv` records the later
normal-label and setup confirmation and is the only manifest used for current
training. The trainer rejects non-normal manifest rows, so v1 is not a valid
input to this normal-only workflow.

## Preprocessing datasets

Large derivatives are stored outside Git. On the college server they are under
`/tf/Jerryscan-data/G01/<preprocessing_id>`. Each accepted dataset contains
`train/normal`, `val/normal`, and `test/normal`; segmentation variants also
contain split-aligned masks.

| Preprocessing ID | Status | Images/masks | What it changes | Recorded outcome |
|---|---|---:|---|---|
| `raw_letterbox_v1` | Complete; accepted | 3,991 images | Preserves the complete original frame and aspect ratio, adding gray letterbox padding | Required raw-background control; identity dry run passed |
| `fixed_crop_v1` | Complete; accepted | 3,991 images | Applies deterministic crop `[0, 0, 780, 1200]`, then resize/letterbox; some original background remains | Zero failures; median preprocessing 133.7 ms/image; identity dry run passed |
| `rembg_u2net_gray_v1` | Complete; accepted | 3,991 images and 3,991 masks | Uses rembg 2.0.76 with U2-Net, aligns the can, and fills removed background with gray value 128 | Zero failures/flags; mask-area range 0.4837-0.4988; median preprocessing 593.1 ms/image; identity dry run passed |
| `rembg_u2net_black_v1` | Complete; accepted | 3,991 images and 3,991 masks | Reuses the audited aligned U2-Net images/masks and replaces mask-exterior pixels with RGB zero | U2-Net was not rerun; zero failures/flags; identity dry run passed |

Important terminology: rembg is the inference wrapper; U2-Net is the
segmentation model used by the two accepted rembg datasets. Fixed crop and
letterbox are deterministic geometric preprocessing methods, not learned
preprocessing models.

Other investigated preprocessing work is not part of the current four-way
training run:

- `rembg_isnet_gray_v1`: rejected/blocked after 4 of 10 smoke-test images
  triggered component-cleanup warnings and manual review found intermittent
  retained white background inside the handle opening.
- `sam2_tiny_box_gray_v1`: checkpoint and configuration prepared, but the
  local Windows runtime could not run the required Linux workflow. It is not a
  completed derivative.
- `yolo_jerrycan_seg_v1`: deferred pending reviewed segmentation labels and a
  deployment-license choice.

## Controlled stage-one PatchCore configuration

These values must stay identical across the four preprocessing variants. A
change creates a different experiment arm and must be named and logged.

| Variable | Stage-one value |
|---|---|
| Model | Anomalib PatchCore |
| Backbone | `wide_resnet50_2`, pretrained |
| Feature layers | `layer2`, `layer3` |
| Input size | 256 x 256 |
| Training images | All 1,997 manifest train identities |
| Validation images | All 994 manifest validation identities |
| Locked test | Filename/identity validation only; not passed to Anomalib |
| Training batch size | 8 |
| Evaluation batch size | 1 |
| Data workers | 4 on the portable notebook |
| Coreset sampling ratio | 0.10 |
| Nearest neighbors | 9 |
| Seed | 42 |
| Accelerator | One CUDA GPU |
| Pre-coreset embedding storage | CPU (`--embedding-storage cpu`) |
| Variant order | raw letterbox, fixed crop, U2-Net gray, U2-Net black |

PatchCore is a memory-bank method. In the recorded Anomalib 2.2 execution,
lowering the training batch size would reduce transient feature-extraction
memory but would not reduce the final complete embedding pool that caused the
OOM. The separately controlled evaluation batch size is 1 because validation
constructs a patch-to-memory-bank distance matrix whose temporary VRAM grows
with the evaluation batch and final coreset size. Evaluation batch size affects
memory, throughput, and numerical batching, not which images, features, or
saved memory bank are used. Likewise, changing only the coreset ratio does not
prevent the implementation from first materializing the full pool. Training
variants one at a time also does not fix the issue: the notebook already runs
them sequentially and the first variant failed by itself.

The validation folder is exposed to Anomalib as its normal test directory and
mirrored using `ValSplitMode.SAME_AS_TEST`, because Anomalib's `Folder`
datamodule has no dedicated validation-directory argument. The real test folder
is checked against the manifest by JerryscanAI code but is deliberately never
passed to Anomalib during fitting.

## Software and portability record

The portable runner is
`training/models/train-patchcore_notebook.ipynb`; the canonical command-line
implementation it calls is `training/models/train_patchcore.py`.
CPU-offload behavior is isolated in
`training/models/cpu_offload_patchcore.py`.

The college runtime reported:

| Component | Recorded value |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.12.1+cu130 |
| torchvision | 0.27.1 |
| CUDA available to PyTorch | Yes |
| NVIDIA driver | 580.95.05 |
| Driver-reported CUDA | 13.0 |
| Anomalib | 2.2.0 for the controlled notebook stack |
| Lightning | 2.6.1 for the controlled notebook stack |
| OpenCV | `opencv-python-headless==4.13.0.92` |
| Pandas | 2.3.3, constrained below 3 |

The notebook intentionally preserves the platform-provided, compatible
CUDA-enabled PyTorch and torchvision pair. It installs/pins the
hardware-independent stack and saves `training_environment.freeze.txt` under
the persistent output root.

Two dependency corrections were required:

1. Anomalib pulled a GUI OpenCV build that imported `libxcb.so.1`, which was
   absent on the headless server. The notebook now removes GUI OpenCV wheels,
   installs `opencv-python-headless==4.13.0.92`, and asserts `GUI: NONE`.
2. Anomalib 2.2's Folder dataset failed with Pandas 3. The project now constrains
   Pandas below 3 and the notebook installs/verifies Pandas 2.3.3.

Manifest hashing was also normalized across CSV line endings. Before the fix,
Linux printed `93ffd5020ed35f75e988a79d311cca012c118571a578b69d836854f8f4375daa`
for the transferred file while Windows printed the canonical hash. The sample
content and split had not changed. Current code records the canonical
`0fab56...` hash on both platforms.

Relevant repository commits:

- `faedfdc` - normalize manifest hashing across platforms and pin headless
  OpenCV.
- `cce0518` - pin Pandas below 3 for Anomalib compatibility.

### Repository verification as of 2026-07-30

```powershell
uv run python -m unittest discover -s tests
```

Result: 26 tests passed after the CPU-offload and separate evaluation-batch
changes. The suite includes dataset/manifest validation, portable-notebook
contract checks, CPU/evaluation command wiring, immediate CPU
detachment/offload checks, and CPU-pool consolidation through the standard
coreset sampler.

An end-to-end synthetic offload smoke test also succeeded: a 32 x 384 CPU
embedding pool was passed through Anomalib's standard `KCenterGreedy` path and
produced a three-row coreset. The actual G01 command dry run with
`--embedding-storage cpu` passed all 1,997 train, 994 validation, and 1,000
locked-test identities and printed the canonical manifest hash.

These checks validate the implementation, command wiring, identity contract,
and small synthetic coreset path. They do not establish that the full G01
coreset construction fits the 20.07 GiB college GPU; that remains the next
platform test.

## Platform and resource inventory

### Local Windows workstation

Confirmed use:

- Dataset inspection, manifest creation/audit, preprocessing generation, and
  repository tests.
- U2-Net preprocessing completed on CPU locally.
- SAM 2.1 full execution was blocked because its supported Linux/WSL runtime
  was unavailable.

No successful full PatchCore training run is recorded on the local workstation,
and its GPU/RAM specifications have not been recorded. Do not infer a local
training capacity from preprocessing success.

### College Jupyter server

| Resource | Recorded value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU memory visible to process | 20.07 GiB |
| System RAM | 1.5 TiB total |
| RAM at measurement | 117 GiB used, 49 GiB free, 1.3 TiB available including cache |
| Swap | None |
| Storage paths used | `/tf/JerryscanAI`, `/tf/Jerryscan-data/G01`, `/tf/jerryscanai-training-output` |

The GPU completed feature extraction at 256 x 256 for 1,997 images in 250
batches, but 20.07 GiB was insufficient for the original all-GPU embedding
concatenation. The very large available system RAM makes this the preferred
platform for the CPU-offload experiment.

### Lightning.ai options considered

No successful JerryscanAI training attempt on Lightning.ai is recorded yet.
The platform screen showed T4, L40S, A100, H100, and H200 options with
account-specific free-hour balances. Hardware availability and VRAM allocations
must be verified inside a running session with `nvidia-smi`.

Recommendations made before the CPU-offload decision:

- T4 (typically 16 GiB) is not suitable for the unchanged experiment because it
  offers less VRAM than the college allocation that failed.
- L40S (typically 48 GiB) is the preferred paid/free-hour GPU to try before an
  H100 if retaining the original all-GPU implementation.
- H100 (commonly 80 GiB on this platform, but verify the allocation) is the
  safest high-memory option and likely unnecessary if CPU offload succeeds.
- A100/H200 capacity depends on the exact platform allocation; do not assume
  fitness from the model name alone.

## Chronological training-attempt log

### Preparatory validation - complete

All four selected derivative folders passed the canonical dry run:

- exact train/validation/test folder identities match `split_v2.csv`;
- counts are 1,997/994/1,000;
- output paths are constructible;
- the locked test is not loaded;
- no model is trained during dry run.

This validates data organization and identity, not model quality.

### Attempt 0 - Linux path case mismatch

**Platform:** college Jupyter server.

**Status:** failed before dataset validation.

The notebook used `/tf/jerryscan-data/G01/raw_letterbox_v1`, while the uploaded
directory was `/tf/Jerryscan-data/G01/raw_letterbox_v1`. Linux paths are
case-sensitive. Correcting `DATA_ROOT` and rebuilding the `VARIANTS` mapping
resolved the assertion. No dataset was recreated or moved.

### Attempt 1 - GUI OpenCV import failure

**Platform:** college Jupyter server.

**Variant:** `raw_letterbox_v1`.

**Status:** failed before PatchCore started; no checkpoint.

Dataset identity validation passed. Importing Anomalib imported GUI OpenCV,
which failed with:

```text
ImportError: libxcb.so.1: cannot open shared object file: No such file or directory
```

Resolution: replace GUI OpenCV packages with the pinned headless wheel and
verify `GUI: NONE`. This was an environment failure, not a dataset or GPU
failure.

### Attempt 2 - Pandas 3 / Anomalib Folder incompatibility

**Platform:** college Jupyter server.

**Variant:** `raw_letterbox_v1`.

**Status:** failed while Anomalib constructed the Folder dataset; no
checkpoint.

The canonical manifest hash and counts were correct, CUDA was detected, and
PatchCore was instantiated. Anomalib then raised:

```text
ValueError: cannot set a frame with no defined index and a scalar
```

Resolution: pin Pandas 2.3.3 (`pandas>=2.3.3,<3`) and restart the kernel. This
was a dependency compatibility failure, not a dataset failure.

### Attempt 3 - GPU OOM during embedding-bank construction

**Platform:** college Jupyter server.

**Variant:** `raw_letterbox_v1`.

**Status:** failed after feature extraction; no checkpoint. The other three
variants were not attempted.

Confirmed progress:

- Correct manifest hash and 1,997/994/1,000 counts.
- GPU recognized as NVIDIA RTX PRO 6000 Blackwell Server Edition.
- All 250 training batches completed in approximately 11 seconds.
- Failure occurred at validation start, when
  `PatchcoreModel.subsample_embedding()` called
  `torch.vstack(self.embedding_store)`.

Recorded CUDA error:

```text
CUDA out of memory. Tried to allocate 11.70 GiB.
GPU capacity: 20.07 GiB; free: 7.15 GiB.
Process memory in use: 12.46 GiB.
PyTorch allocated: 11.80 GiB.
```

Interpretation: this is a genuine peak-memory limit in Anomalib's all-GPU
embedding-bank construction, not CUDA fragmentation, multi-model accumulation,
or an image-data error.

### Attempt 4 - CPU offload

**Status:** implemented and locally verified as of 2026-07-30; not yet run on
the college server with the full G01 embedding pool.

The new `--embedding-storage cpu` mode:

1. Runs the standard Anomalib PatchCore training step for each batch.
2. Immediately detaches that batch's embedding and moves it from CUDA to
   system RAM.
3. Concatenates the complete pre-coreset pool in system RAM.
4. Clears unused CUDA cache and transfers the single consolidated pool to the
   GPU.
5. Runs Anomalib's unchanged `KCenterGreedy` coreset selection on the GPU.

The standard mode remains available as `--embedding-storage device`. The CPU
mode is expected to require roughly 24 GiB available system RAM at the
concatenation peak because both the list of batch tensors and the new
consolidated tensor briefly coexist. The college server's approximately
1.3 TiB available RAM is comfortably above that estimate. Actual usage still
needs to be measured.

Validation now uses `--eval-batch-size 1` independently of the unchanged
training batch size 8. This bounds the later validation distance-matrix peak
after coreset selection. If the option is omitted, the CLI preserves its
general behavior by defaulting evaluation batch size to the training batch
size; the controlled notebook passes 1 explicitly.

Local verification completed:

- all 26 repository tests passed;
- a synthetic 32 x 384 CPU embedding pool completed standard
  `KCenterGreedy` selection and returned a three-row coreset;
- the full G01 dry run with the CPU flag validated the canonical manifest and
  1,997/994/1,000 identities;
- notebook/CLI verification confirmed the explicit evaluation batch size 1
  while retaining training batch size 8.

The dry run does not load images into PatchCore or exercise the full GPU
transfer/coreset peak, so the remote attempt remains pending.

Acceptance conditions:

- all 1,997 train identities are used;
- the controlled model variables remain unchanged;
- temporary/full embeddings are held in system RAM where needed;
- GPU peak memory remains below the 20.07 GiB allocation;
- coreset selection completes;
- `G01.ckpt` and `G01.metadata.json` are written;
- metadata records `implementation=CpuOffloadPatchcore` and
  `training.embedding_storage=cpu`;
- metadata records training/evaluation batch sizes, fit time, peak process RSS,
  peak CUDA allocated/reserved memory, and hardware identity;
- the locked test remains unused.

Do not mark this attempt successful until artifacts and metadata have been
inspected on persistent storage.

## Canonical college-server workflow

Pull the intended branch and inspect the commit:

```bash
cd /tf/JerryscanAI
git pull --ff-only
git log -1 --oneline
```

Restart the Jupyter kernel after dependency or notebook updates. Open
`training/models/train-patchcore_notebook.ipynb`, set only the repository,
data, and persistent output paths as necessary, then run from the top. On the
recorded college layout:

```python
DATA_ROOT = Path("/tf/Jerryscan-data/G01")
OUTPUT_ROOT = Path("/tf/jerryscanai-training-output")
```

Before spending GPU time, the dry-run cell must report 1,997 train, 994
validation, and 1,000 locked-test identities for every selected variant. The
environment cell must confirm CUDA, `GUI: NONE`, and Pandas 2.3.3.

The generated training command should contain:

```text
--embedding-storage cpu
--eval-batch-size 1
```

For a one-variant operational smoke test, select:

```python
VARIANTS_TO_TRAIN = ["raw_letterbox_v1"]
```

This is allowed as an operational check, but it is not a completed
preprocessing comparison. After the memory strategy succeeds, run all four
variants with identical scientific settings.

Monitor resources in a terminal:

```bash
nvidia-smi -l 2
free -h
```

Do not reinstall torch or torchvision in the notebook unless the platform's
pair is demonstrably incompatible. Do not regenerate datasets to repair a
Python dependency, path-capitalization, or GPU-memory failure.

## Expected artifacts

After a successful run, each variant should produce:

```text
<OUTPUT_ROOT>/
├── training_environment.freeze.txt
├── models/
│   └── Patchcore_<preprocessing_id>_256_c10_seed42/
│       ├── G01.ckpt
│       └── G01.metadata.json
└── results/
```

The metadata must include the manifest hash, preprocessing ID, sample counts,
model configuration, both training and evaluation batch sizes, worker
settings, seed, accelerator, package versions, Git commit/dirty state, command,
checkpoint path, and `test_used_during_training=false`.

Successful current runs also record:

- `training.embedding_storage`;
- `training.fit_seconds`;
- `training.peak_process_rss_bytes`;
- `training.peak_cuda_allocated_bytes`;
- `training.peak_cuda_reserved_bytes`;
- hardware platform and machine identity;
- CPU count and total system-memory bytes;
- GPU name and total GPU-memory bytes;
- PyTorch CUDA runtime version.

These values are written only after fitting succeeds. For failed runs, retain
the console traceback and external resource observations because final metadata
will not exist.

Copy or archive artifacts before an ephemeral platform session ends. An
existence check alone is insufficient: open the metadata, confirm its hashes
and variables, and verify the checkpoint can be loaded before declaring the
attempt complete.

## Evaluation and test-use policy

All current G01 images are normal. Therefore:

- training fits the PatchCore normal memory bank;
- validation may compare preprocessing, resolution, coreset settings, and a
  provisional normal-score cutoff;
- locked test may be evaluated once only after those choices are frozen;
- current normal-only validation/test can report false-positive rate and score
  drift, but not defect recall, defect F1, or meaningful two-class AUROC;
- real, reviewed defect images are still required to measure the production
  objective of defect recall at an agreed maximum normal false-positive rate.

The current trainer validates locked-test filenames but does not load their
pixels into Anomalib. Do not visually choose masks, morphology, model settings,
checkpoints, or thresholds using locked-test outputs.

## Fallback if CPU offload is unsuccessful

Create a deterministic, versioned approximately 1,000-image subset from the
existing 1,997 training identities. Preserve the same validation and locked
test identities. Record the selection algorithm, seed/grouping, selected
sample IDs, and subset-manifest hash. Use exactly the same subset for all four
preprocessing variants.

This fallback changes the training-data-size variable, so it must use a
different model-set name and must not be compared as though it were the
full-data stage-one experiment. Prefer also running a documented learning curve
(for example 250, 500, 1,000, and all images when hardware later permits).

## Open items

1. Complete and test CPU-offloaded PatchCore on the college server.
2. Record peak GPU memory, peak system RAM, wall time, and resulting artifact
   sizes for each successful variant.
3. If offload fails, define and version the deterministic 1,000-image fallback
   selection before training.
4. Train the four controlled preprocessing variants.
5. Generate model-independent raw validation scores and compare normal
   false-positive behavior/score drift without using locked test.
6. After selecting preprocessing, test resolution and coreset ratio as separate
   controlled variables.
7. Acquire and version reviewed real defects for threshold selection and
   recall/F1 evaluation.
8. Evaluate the locked test once the preprocessing/model/threshold decisions
   are frozen.
