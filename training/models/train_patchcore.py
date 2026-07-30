"""Train PatchCore from a materialized, manifest-verified dataset split."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from training.datasets.create_dataset_manifest import sha256_manifest


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
REQUIRED_SPLITS = ("train", "val", "test")


def parse_layers(value: str) -> tuple[str, ...]:
    layers = tuple(part.strip() for part in value.split(",") if part.strip())
    if not layers:
        raise argparse.ArgumentTypeError("At least one layer is required.")
    return layers


def manifest_split_ids(manifest: Path) -> dict[str, set[str]]:
    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", "split", "label"}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Manifest is missing required columns: {sorted(missing_columns)}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"Manifest has no samples: {manifest}")
    unsupported = sorted({row["split"] for row in rows} - set(REQUIRED_SPLITS))
    if unsupported:
        raise ValueError(f"Manifest contains unsupported splits: {unsupported}")
    non_normal = [row["sample_id"] for row in rows if row["label"] != "normal"]
    if non_normal:
        raise ValueError(
            "PatchCore training manifest must contain verified label=normal rows; "
            f"found {len(non_normal)} other rows"
        )

    result = {split: set() for split in REQUIRED_SPLITS}
    seen: set[str] = set()
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id in manifest: {sample_id}")
        seen.add(sample_id)
        result[row["split"]].add(sample_id)
    for split, sample_ids in result.items():
        if not sample_ids:
            raise ValueError(f"Manifest split is empty: {split}")
    return result


def folder_sample_ids(folder: Path) -> set[str]:
    if not folder.is_dir():
        raise ValueError(f"Dataset folder does not exist: {folder}")
    paths = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    sample_ids = {path.stem for path in paths}
    if len(sample_ids) != len(paths):
        duplicates = [
            sample_id
            for sample_id, count in Counter(path.stem for path in paths).items()
            if count > 1
        ]
        raise ValueError(f"Duplicate sample IDs in {folder}: {duplicates[:5]}")
    return sample_ids


def validate_materialized_dataset(
    dataset_root: Path, manifest: Path
) -> tuple[dict[str, Path], dict[str, int], str]:
    if not manifest.is_file():
        raise ValueError(f"Manifest does not exist: {manifest}")
    expected = manifest_split_ids(manifest)
    folders = {split: dataset_root / split / "normal" for split in REQUIRED_SPLITS}

    for split, folder in folders.items():
        actual = folder_sample_ids(folder)
        missing = sorted(expected[split] - actual)
        extra = sorted(actual - expected[split])
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing={missing[:5]} ({len(missing)} total)")
            if extra:
                details.append(f"extra={extra[:5]} ({len(extra)} total)")
            raise ValueError(f"{split} does not match manifest: {'; '.join(details)}")

    counts = {split: len(expected[split]) for split in REQUIRED_SPLITS}
    return folders, counts, sha256_manifest(manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train PatchCore using explicit manifest train/validation splits."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Materialized root containing train/normal, val/normal, and test/normal.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Frozen manifest whose sample IDs must exactly match the dataset folders.",
    )
    parser.add_argument("--angle", default="G01", help="Camera angle/checkpoint name.")
    parser.add_argument(
        "--preprocessing-id",
        default="raw_v1",
        help="Versioned preprocessing identifier recorded with the model.",
    )
    parser.add_argument(
        "--model-set",
        default="PatchcoreRaw256",
        help="Subfolder under models/ where the checkpoint is saved.",
    )
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument(
        "--eval-batch-size",
        default=None,
        type=int,
        help="Validation batch size; defaults to --batch-size.",
    )
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument(
        "--accelerator",
        default="auto",
        choices=("auto", "cpu", "gpu"),
        help="Execution device. PatchCore memory-bank construction may require CPU.",
    )
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--layers", default=("layer2", "layer3"), type=parse_layers)
    parser.add_argument("--coreset-sampling-ratio", default=0.1, type=float)
    parser.add_argument("--num-neighbors", default=9, type=int)
    parser.add_argument(
        "--embedding-storage",
        choices=("device", "cpu"),
        default="device",
        help=(
            "Temporary pre-coreset embedding storage. Use cpu when the complete "
            "float32 embedding pool does not fit in GPU memory."
        ),
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--results-dir", default=Path("results"), type=Path)
    parser.add_argument("--models-dir", default=Path("models"), type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing checkpoint and metadata file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset, manifest, and output paths without importing Anomalib.",
    )
    return parser


def git_state(project_root: Path) -> dict[str, str | bool | None]:
    try:
        commit = subprocess.run(
            ["git", "-c", f"safe.directory={project_root.as_posix()}", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-c", f"safe.directory={project_root.as_posix()}", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def total_system_memory_bytes() -> int | None:
    """Return total system RAM on platforms that expose POSIX sysconf."""
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def peak_process_rss_bytes() -> int | None:
    """Return peak process resident memory when the resource module is available."""
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if sys.platform == "darwin" else peak * 1024
    except (ImportError, OSError, ValueError):
        return None


def main() -> int:
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.resolve()
    manifest = args.manifest.resolve()
    try:
        folders, counts, manifest_hash = validate_materialized_dataset(
            dataset_root, manifest
        )
    except ValueError as exc:
        print(f"Dataset validation failed: {exc}")
        return 1

    if not 0 < args.coreset_sampling_ratio <= 1:
        print("--coreset-sampling-ratio must be in (0, 1]")
        return 1
    eval_batch_size = args.eval_batch_size or args.batch_size
    if (
        args.image_size <= 0
        or args.batch_size <= 0
        or eval_batch_size <= 0
        or args.num_neighbors <= 0
    ):
        print(
            "Image size, training/evaluation batch sizes, and number of "
            "neighbors must be positive"
        )
        return 1

    output_dir = (args.models_dir / args.model_set).resolve()
    output_ckpt = output_dir / f"{args.angle}.ckpt"
    output_metadata = output_dir / f"{args.angle}.metadata.json"
    existing = [path for path in (output_ckpt, output_metadata) if path.exists()]
    if existing and not args.overwrite:
        print(f"Output already exists; pass --overwrite: {existing[0]}")
        return 1

    print(f"Dataset root: {dataset_root}")
    print(f"Manifest: {manifest}")
    print(f"Manifest SHA-256: {manifest_hash}")
    print(
        f"Train: {counts['train']}; validation: {counts['val']}; "
        f"locked test (unused): {counts['test']}"
    )
    print(f"Checkpoint: {output_ckpt}")
    if args.dry_run:
        print("Dry run OK. No model was trained and the locked test split was not loaded.")
        return 0

    import torch
    from anomalib.data import Folder
    from anomalib.data.utils.split import TestSplitMode, ValSplitMode
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    from lightning import seed_everything

    if args.embedding_storage == "cpu":
        from training.models.cpu_offload_patchcore import CpuOffloadPatchcore

        patchcore_class = CpuOffloadPatchcore
    else:
        patchcore_class = Patchcore

    seed_everything(args.seed, workers=True)
    accelerator = args.accelerator
    if accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Accelerator: {accelerator}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Pre-coreset embedding storage: {args.embedding_storage}")
    print(f"Training batch size: {args.batch_size}; evaluation batch size: {eval_batch_size}")

    pre_processor = Patchcore.configure_pre_processor(
        image_size=(args.image_size, args.image_size)
    )
    model = patchcore_class(
        backbone=args.backbone,
        layers=args.layers,
        pre_trained=True,
        coreset_sampling_ratio=args.coreset_sampling_ratio,
        num_neighbors=args.num_neighbors,
        pre_processor=pre_processor,
    )

    # Anomalib Folder has no dedicated validation-directory parameter. Expose the
    # frozen validation directory as its test directory and mirror it for fit-time
    # validation. The real test directory is deliberately not passed to Anomalib.
    datamodule = Folder(
        name=args.angle,
        normal_dir=folders["train"],
        normal_test_dir=folders["val"],
        train_batch_size=args.batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=args.num_workers,
        test_split_mode=TestSplitMode.FROM_DIR,
        val_split_mode=ValSplitMode.SAME_AS_TEST,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    engine = Engine(
        accelerator=accelerator,
        devices=1,
        default_root_dir=args.results_dir,
        logger=False,
    )
    uses_cuda = accelerator == "gpu" and torch.cuda.is_available()
    if uses_cuda:
        torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    training_seconds = time.perf_counter() - training_started
    peak_cuda_allocated = torch.cuda.max_memory_allocated() if uses_cuda else None
    peak_cuda_reserved = torch.cuda.max_memory_reserved() if uses_cuda else None
    engine.trainer.save_checkpoint(output_ckpt)

    project_root = Path(__file__).resolve().parents[2]
    gpu_properties = torch.cuda.get_device_properties(0) if uses_cuda else None
    metadata = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "class": "Patchcore",
            "angle": args.angle,
            "model_set": args.model_set,
            "backbone": args.backbone,
            "layers": list(args.layers),
            "image_size": args.image_size,
            "coreset_sampling_ratio": args.coreset_sampling_ratio,
            "num_neighbors": args.num_neighbors,
            "implementation": patchcore_class.__name__,
        },
        "dataset": {
            "root": str(dataset_root),
            "manifest": str(manifest),
            "manifest_sha256": manifest_hash,
            "preprocessing_id": args.preprocessing_id,
            "counts": counts,
            "test_used_during_training": False,
        },
        "training": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "eval_batch_size": eval_batch_size,
            "num_workers": args.num_workers,
            "accelerator": accelerator,
            "embedding_storage": args.embedding_storage,
            "fit_seconds": training_seconds,
            "peak_process_rss_bytes": peak_process_rss_bytes(),
            "peak_cuda_allocated_bytes": peak_cuda_allocated,
            "peak_cuda_reserved_bytes": peak_cuda_reserved,
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "system_memory_bytes": total_system_memory_bytes(),
            "gpu_name": gpu_properties.name if gpu_properties else None,
            "gpu_total_memory_bytes": (
                gpu_properties.total_memory if gpu_properties else None
            ),
            "cuda_runtime": torch.version.cuda,
        },
        "software": {
            "python": sys.version,
            "anomalib": package_version("anomalib"),
            "lightning": package_version("lightning"),
            "torch": package_version("torch"),
        },
        "git": git_state(project_root),
        "command": sys.argv,
        "checkpoint": str(output_ckpt),
    }
    output_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved checkpoint: {output_ckpt}")
    print(f"Saved metadata: {output_metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
