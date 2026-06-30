import argparse
from pathlib import Path

import torch
from anomalib.data import Folder
from anomalib.data.utils.split import TestSplitMode, ValSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore

### command to run:
### .\.venv\Scripts\python.exe standalone_scripts\train_patchcore.py --normal-dir training_data\G01_aligned_last500\train\normal


def parse_layers(value: str) -> tuple[str, ...]:
    layers = tuple(part.strip() for part in value.split(",") if part.strip())
    if not layers:
        raise argparse.ArgumentTypeError("At least one layer is required.")
    return layers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Jerryscan PatchCore checkpoint from normal images."
    )
    parser.add_argument(
        "--normal-dir",
        required=True,
        type=Path,
        help="Folder containing good/normal training images.",
    )
    parser.add_argument("--angle", default="G01", help="Angle/checkpoint name, e.g. G01.")
    parser.add_argument(
        "--model-set",
        default="RembgAlignedPatchcore",
        help="Subfolder under models/ where the checkpoint will be saved.",
    )
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument(
        "--accelerator",
        default="auto",
        choices=("auto", "cpu", "gpu"),
        help="Use cpu to avoid GPU memory-bank OOM on large PatchCore datasets.",
    )
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--layers", default=("layer2", "layer3"), type=parse_layers)
    parser.add_argument("--coreset-sampling-ratio", default=0.1, type=float)
    parser.add_argument("--num-neighbors", default=9, type=int)
    parser.add_argument(
        "--normal-split-ratio",
        default=0.2,
        type=float,
        help="Fraction of normal images held out for validation.",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--results-dir", default=Path("results"), type=Path)
    parser.add_argument("--models-dir", default=Path("models"), type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths/imports without fitting the model.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    normal_dir = args.normal_dir.resolve()
    if not normal_dir.exists():
        print(f"Normal image folder does not exist: {normal_dir}")
        return 1

    image_count = sum(1 for path in normal_dir.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    if image_count == 0:
        print(f"No images found in normal folder: {normal_dir}")
        return 1

    accelerator = args.accelerator
    if accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Normal dir: {normal_dir}")
    print(f"Images: {image_count}")
    print(f"Accelerator: {accelerator}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pre_processor = Patchcore.configure_pre_processor(
        image_size=(args.image_size, args.image_size)
    )
    model = Patchcore(
        backbone=args.backbone,
        layers=args.layers,
        pre_trained=True,
        coreset_sampling_ratio=args.coreset_sampling_ratio,
        num_neighbors=args.num_neighbors,
        pre_processor=pre_processor,
    )

    datamodule = Folder(
        name=args.angle,
        normal_dir=normal_dir,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        normal_split_ratio=args.normal_split_ratio,
        test_split_mode=TestSplitMode.NONE,
        val_split_mode=ValSplitMode.FROM_TRAIN,
        seed=args.seed,
    )

    output_dir = args.models_dir / args.model_set
    output_dir.mkdir(parents=True, exist_ok=True)
    output_ckpt = output_dir / f"{args.angle}.ckpt"

    if args.dry_run:
        print(f"Dry run OK. Checkpoint would be saved to: {output_ckpt}")
        return 0

    engine = Engine(
        accelerator=accelerator,
        devices=1,
        default_root_dir=args.results_dir,
        logger=False,
    )
    engine.fit(model=model, datamodule=datamodule)
    engine.trainer.save_checkpoint(output_ckpt)
    print(f"Saved checkpoint: {output_ckpt.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
