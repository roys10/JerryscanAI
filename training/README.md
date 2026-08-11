# Training workspace

This directory contains the reproducible machine-learning workflow for
JerryscanAI. Run commands from the repository root with Python module syntax.

## Layout

- `datasets/`: frozen-manifest creation and split materialization.
- `preprocessing/`: derivative image and mask generation.
- `preprocessing/configs/`: versioned preprocessing configurations and model registry.
- `models/`: PatchCore training entry points.

Raw captures remain immutable and outside Git. Frozen sample identities remain
under `data_manifests/`, generated derivative datasets remain outside Git, and
trained checkpoints remain under the ignored `models/` artifact directory.

The normal execution order is:

1. Create or verify a frozen dataset manifest.
2. Generate and audit a complete preprocessing dataset.
3. Dry-run training against the derivative dataset.
4. Train the model and record its experiment metadata.

See [the preprocessing guide](../docs/preprocessing.md) and
[the model-development guide](../docs/model-development.md) for the controlled
experiment protocol.

## Portable GPU PatchCore training

Use `models/train-patchcore_notebook.ipynb` on any CUDA GPU platform after
uploading the approved derivative datasets to persistent storage. The platform
must provide a compatible CUDA-enabled PyTorch and torchvision pair; the
notebook preserves those hardware-specific packages, pins the rest of the
training stack, replaces Anomalib's GUI OpenCV dependency with the pinned
headless build, pins Pandas below 3 for Anomalib 2.2 compatibility, records
`pip freeze`, verifies CUDA, OpenCV, and Pandas, dry-runs every selected dataset
against `split_v2.csv`, and calls the canonical
`training.models.train_patchcore` module. Only data/output paths should normally
change between the college server, Lightning.ai, and another GPU provider.

The notebook enables `--embedding-storage cpu`. Standard Anomalib PatchCore
retains the complete pre-coreset embedding pool on the accelerator and briefly
duplicates it during concatenation. The CPU-offload implementation moves each
float32 batch embedding to system RAM immediately, consolidates the pool there,
clears cached CUDA allocations, and then runs the unchanged Anomalib
`KCenterGreedy` selection on the accelerator. This preserves the model inputs
and coreset algorithm while trading additional system RAM and transfer time for
lower peak VRAM. G01 at 256 px is expected to need roughly 24 GiB of available
system RAM during CPU concatenation; record actual peak RAM and VRAM for every
platform run.

The portable notebook also uses evaluation batch size 1 while retaining
training batch size 8. Evaluation constructs a patch-to-memory-bank distance
matrix whose temporary VRAM grows with evaluation batch size and the final
coreset. Batch size changes execution memory and throughput, not the samples,
features, or saved memory bank. Successful metadata records both batch sizes,
fit wall time, peak process RSS, peak CUDA allocated/reserved memory, and
hardware identity.
