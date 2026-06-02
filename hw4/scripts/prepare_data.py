"""Extract and validate the released HW4 image restoration dataset."""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

from PIL import Image

from _bootstrap import PROJECT_ROOT  # noqa: F401
from hw4.data import find_pairs, list_test_images, stratified_split


DEFAULT_ARCHIVE = PROJECT_ROOT / "release_folder-20260526T091534Z-3-001.zip"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "hw4_realse_dataset"


def extract_dataset(archive_path: Path, output_parent: Path) -> None:
    output_parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as outer:
        nested = outer.read("release_folder/hw4_realse_dataset.zip")
    with zipfile.ZipFile(io.BytesIO(nested)) as dataset_archive:
        for info in dataset_archive.infolist():
            name = info.filename
            if (
                not name.endswith(".png")
                or "__MACOSX" in name
                or not name.startswith("hw4_realse_dataset/")
            ):
                continue
            target = output_parent / Path(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(dataset_archive.read(info))


def validate_data(data_root: Path) -> None:
    records = find_pairs(data_root)
    train, validation = stratified_split(records)
    test_images = list_test_images(data_root)
    all_paths = [record.degraded for record in records]
    all_paths += [record.clean for record in records]
    all_paths += test_images
    invalid: list[str] = []
    for path in all_paths:
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (256, 256):
                invalid.append(f"{path.name}: {image.mode} {image.size}")
    if invalid:
        raise ValueError(f"Unexpected image properties: {invalid[:5]}")
    print(f"Validated data root: {data_root}")
    print(f"Training pairs: {len(train)} (rain=1440, snow=1440)")
    print(f"Validation pairs: {len(validation)} (rain=160, snow=160)")
    print(f"Test images: {len(test_images)}")
    print("All images: RGB 256x256")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only:
        print(f"Extracting dataset from {args.archive}...")
        extract_dataset(args.archive, args.data_root.parent)
    validate_data(args.data_root)


if __name__ == "__main__":
    main()
