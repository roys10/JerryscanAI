import argparse
import os
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def image_paths(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a reproducible train/holdout split for normal images."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-ratio", default=0.8, type=float)
    parser.add_argument(
        "--holdout-count",
        default=None,
        type=int,
        help="Exact number of images to hold out. Overrides --train-ratio.",
    )
    parser.add_argument(
        "--strategy",
        default="random",
        choices=("random", "last"),
        help="random uses a seeded shuffle; last holds out the final sorted filenames.",
    )
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    source_dir = args.source.resolve()
    output_dir = args.output.resolve()
    if not source_dir.exists():
        print(f"Source folder does not exist: {source_dir}")
        return 1
    if not 0 < args.train_ratio < 1:
        print("--train-ratio must be between 0 and 1")
        return 1

    paths = image_paths(source_dir)
    if not paths:
        print(f"No images found in: {source_dir}")
        return 1

    if args.holdout_count is not None:
        if not 0 < args.holdout_count < len(paths):
            print("--holdout-count must be greater than 0 and less than the image count")
            return 1
        train_count = len(paths) - args.holdout_count
    else:
        train_count = round(len(paths) * args.train_ratio)

    if args.strategy == "random":
        rng = random.Random(args.seed)
        shuffled = paths[:]
        rng.shuffle(shuffled)
        train_paths = set(shuffled[:train_count])
    else:
        train_paths = set(paths[:train_count])

    train_dir = output_dir / "train" / "normal"
    holdout_dir = output_dir / "holdout" / "normal"

    for path in paths:
        relative = path.relative_to(source_dir)
        target_root = train_dir if path in train_paths else holdout_dir
        link_or_copy(path, target_root / relative)

    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Train: {len(train_paths)} images -> {train_dir}")
    print(f"Holdout: {len(paths) - len(train_paths)} images -> {holdout_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
