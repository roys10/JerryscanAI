# Training workspace

This directory contains the reproducible machine-learning workflow for
JerryscanAI. Run commands from the repository root with Python module syntax.

## Layout

- `datasets/`: frozen-manifest creation and split materialization.
- `preprocessing/`: derivative image and mask generation.
- `preprocessing/configs/`: versioned preprocessing configurations and model registry.
- `models/`: PatchCore and future anomaly-model training entry points.

Raw captures remain immutable and outside Git. Frozen sample identities remain
under `data_manifests/`, generated derivative datasets remain outside Git, and
trained checkpoints remain under the ignored `models/` artifact directory.

The normal execution order is:

1. Create or verify a frozen dataset manifest.
2. Generate and audit a complete preprocessing dataset.
3. Dry-run training against the derivative dataset.
4. Train the model and record its experiment metadata.

See `docs/preprocessing.md` and `docs/model-development.md` for the controlled
experiment protocol.

## Portable GPU PatchCore training

Use `models/train-patchcore_notebook.ipynb` on any CUDA GPU platform after
uploading the approved derivative datasets to persistent storage. The platform
must provide a compatible CUDA-enabled PyTorch and torchvision pair; the
notebook preserves those hardware-specific packages, pins the rest of the
training stack, replaces Anomalib's GUI OpenCV dependency with the pinned
headless build, records `pip freeze`, verifies CUDA and OpenCV, dry-runs every
selected dataset against `split_v2.csv`, and calls the canonical
`training.models.train_patchcore` module. Only data/output paths should normally
change between the college server, Lightning.ai, and another GPU provider.
