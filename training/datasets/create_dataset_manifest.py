"""Create a validated, leakage-safe manifest and optional folder materialization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image


SCHEMA_VERSION = "1.0"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
FILENAME_PATTERN = re.compile(
    r"^(?P<angle>[^-]+)-(?P<date>\d{6})-(?P<time>\d{6})-(?P<sequence>\d+)"
    r"(?P<extension>\.[^.]+)$"
)
FIELDNAMES = [
    "schema_version",
    "sample_id",
    "source_relpath",
    "camera_angle",
    "captured_at_local",
    "capture_date",
    "session_id",
    "sequence_no",
    "split",
    "label",
    "defect_type",
    "width",
    "height",
    "channels",
    "source_sha256",
    "quality_flags",
    "notes",
]


@dataclass(frozen=True)
class ManifestRow:
    schema_version: str
    sample_id: str
    source_relpath: str
    camera_angle: str
    captured_at_local: str
    capture_date: str
    session_id: str
    sequence_no: int
    split: str
    label: str
    defect_type: str
    width: int
    height: int
    channels: int
    source_sha256: str
    quality_flags: str
    notes: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_manifest(path: Path) -> str:
    """Hash a CSV manifest canonically across Git line-ending conversions.

    Existing G01 derivative artifacts were created from the Windows CRLF form,
    so CRLF is the canonical byte representation retained for compatibility.
    """
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content.replace(b"\n", b"\r\n")).hexdigest()


def parse_capture_filename(path: Path) -> tuple[str, datetime, int]:
    match = FILENAME_PATTERN.fullmatch(path.name)
    if not match:
        raise ValueError(
            f"Filename does not match ANGLE-YYMMDD-HHMMSS-SEQUENCE.ext: {path.name}"
        )
    captured_at = datetime.strptime(
        match.group("date") + match.group("time"), "%y%m%d%H%M%S"
    )
    return match.group("angle"), captured_at, int(match.group("sequence"))


def date_set(values: list[str], option: str) -> set[str]:
    parsed: set[str] = set()
    for value in values:
        try:
            parsed.add(datetime.strptime(value, "%Y-%m-%d").date().isoformat())
        except ValueError as exc:
            raise ValueError(f"{option} expects YYYY-MM-DD, got {value!r}") from exc
    return parsed


def date_flags(values: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for value in values:
        try:
            capture_date, flag = value.split("=", maxsplit=1)
            parsed_date = datetime.strptime(capture_date, "%Y-%m-%d").date().isoformat()
        except ValueError as exc:
            raise ValueError(
                f"--date-flag expects YYYY-MM-DD=flag, got {value!r}"
            ) from exc
        if not flag or not re.fullmatch(r"[a-z0-9_]+", flag):
            raise ValueError(f"Invalid quality flag in --date-flag: {flag!r}")
        result.setdefault(parsed_date, []).append(flag)
    return result


def build_rows(
    source: Path,
    assignments: dict[str, str],
    label: str,
    include_hash: bool,
    configured_date_flags: dict[str, list[str]] | None = None,
) -> list[ManifestRow]:
    paths = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported images found in {source}")

    rows: list[ManifestRow] = []
    seen_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}
    for path in paths:
        angle, captured_at, sequence = parse_capture_filename(path)
        sample_id = path.stem
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)

        capture_date = captured_at.date().isoformat()
        if capture_date not in assignments:
            raise ValueError(
                f"Capture date {capture_date} for {path.name} has no split assignment"
            )

        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                channels = len(image.getbands())
        except Exception as exc:
            raise ValueError(f"Image validation failed for {path}: {exc}") from exc

        source_hash = sha256_file(path) if include_hash else ""
        quality_flags = list((configured_date_flags or {}).get(capture_date, []))
        if label == "unverified":
            quality_flags.append("label_unverified")
        quality_flags = sorted(set(quality_flags))
        if source_hash:
            previous = seen_hashes.get(source_hash)
            if previous is not None:
                raise ValueError(f"Byte-exact duplicate: {path.name} and {previous}")
            seen_hashes[source_hash] = path.name

        rows.append(
            ManifestRow(
                schema_version=SCHEMA_VERSION,
                sample_id=sample_id,
                source_relpath=path.relative_to(source).as_posix(),
                camera_angle=angle,
                captured_at_local=captured_at.isoformat(timespec="seconds"),
                capture_date=capture_date,
                session_id=f"{angle}_{capture_date}",
                sequence_no=sequence,
                split=assignments[capture_date],
                label=label,
                defect_type="",
                width=width,
                height=height,
                channels=channels,
                source_sha256=source_hash,
                quality_flags=";".join(quality_flags),
                notes="",
            )
        )
    return rows


def write_manifest(path: Path, rows: list[ManifestRow], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists: {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    os.replace(temporary, path)


def write_summary(path: Path, manifest: Path, rows: list[ManifestRow]) -> None:
    summary = {
        "schema_version": SCHEMA_VERSION,
        "manifest": manifest.name,
        "manifest_sha256": sha256_manifest(manifest),
        "sample_count": len(rows),
        "split_counts": dict(sorted(Counter(row.split for row in rows).items())),
        "date_counts": dict(sorted(Counter(row.capture_date for row in rows).items())),
        "label_counts": dict(sorted(Counter(row.label for row in rows).items())),
        "camera_angles": sorted({row.camera_angle for row in rows}),
        "source_hashes_included": all(bool(row.source_sha256) for row in rows),
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def materialize(
    source: Path,
    output: Path,
    rows: list[ManifestRow],
    mode: str,
) -> None:
    if output.exists():
        raise FileExistsError(f"Materialization output already exists: {output}")
    temporary = output.with_name(f".{output.name}.partial")
    if temporary.exists():
        raise FileExistsError(f"Partial materialization already exists: {temporary}")

    try:
        for row in rows:
            source_path = source / Path(row.source_relpath)
            destination = temporary / row.split / row.label / source_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if mode == "copy":
                shutil.copy2(source_path, destination)
            else:
                try:
                    os.link(source_path, destination)
                except OSError as exc:
                    raise OSError(
                        f"Cannot hardlink {source_path} to {destination}. "
                        "Use --materialize-mode copy for different filesystems."
                    ) from exc
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate timestamped captures and create an explicit split manifest."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--train-dates", required=True, nargs="+")
    parser.add_argument("--val-dates", required=True, nargs="+")
    parser.add_argument("--test-dates", required=True, nargs="+")
    parser.add_argument("--label", default="unverified")
    parser.add_argument(
        "--date-flag",
        action="append",
        default=[],
        help="Attach a quality flag to a date, e.g. 2026-02-05=isolated_capture.",
    )
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--hash", action="store_true", help="Hash files and reject exact duplicates.")
    parser.add_argument("--materialize-output", type=Path)
    parser.add_argument(
        "--materialize-mode", choices=("hardlink", "copy"), default="hardlink"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace manifest files only; materialized dataset paths must be new.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = args.source.resolve()
    if not source.is_dir():
        raise SystemExit(f"Source directory does not exist: {source}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.label):
        raise SystemExit("--label must be safe to use as one folder name")

    groups = {
        "train": date_set(args.train_dates, "--train-dates"),
        "val": date_set(args.val_dates, "--val-dates"),
        "test": date_set(args.test_dates, "--test-dates"),
    }
    all_dates = [date for dates in groups.values() for date in dates]
    if len(all_dates) != len(set(all_dates)):
        raise SystemExit("A capture date may belong to only one split")
    assignments = {date: split for split, dates in groups.items() for date in dates}

    try:
        configured_date_flags = date_flags(args.date_flag)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    rows = build_rows(
        source,
        assignments,
        args.label,
        args.hash,
        configured_date_flags,
    )
    if args.expected_count is not None and len(rows) != args.expected_count:
        raise SystemExit(
            f"Expected {args.expected_count} images but validated {len(rows)}"
        )
    present_dates = {row.capture_date for row in rows}
    unused_dates = set(assignments) - present_dates
    if unused_dates:
        raise SystemExit(f"Configured dates contain no images: {sorted(unused_dates)}")
    unused_flag_dates = set(configured_date_flags) - present_dates
    if unused_flag_dates:
        raise SystemExit(
            f"Quality-flag dates contain no images: {sorted(unused_flag_dates)}"
        )

    manifest = args.manifest.resolve()
    write_manifest(manifest, rows, args.overwrite)
    summary = manifest.with_suffix(".summary.json")
    write_summary(summary, manifest, rows)
    if args.materialize_output:
        materialize(
            source,
            args.materialize_output.resolve(),
            rows,
            args.materialize_mode,
        )

    counts = Counter(row.split for row in rows)
    print(f"Manifest: {manifest}")
    print(f"Summary: {summary}")
    print(f"Samples: {len(rows)}")
    print(f"Train: {counts['train']}; val: {counts['val']}; test: {counts['test']}")
    print(f"SHA-256: {sha256_manifest(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
