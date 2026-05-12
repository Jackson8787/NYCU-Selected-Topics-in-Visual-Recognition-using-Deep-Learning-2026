from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from hw3.data import (
    list_test_images,
    list_train_samples,
    load_test_id_map,
    masks_to_instances,
    read_image_rgb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate HW3 extracted data.")
    parser.add_argument("--data-root", type=Path, default=Path("data/hw3-data-release"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = list_train_samples(args.data_root)
    test_images = list_test_images(args.data_root)
    test_id_map = load_test_id_map(args.data_root)

    class_instances = Counter()
    class_files = Counter()
    image_sizes = []

    for sample in samples:
        image = read_image_rgb(sample.image_path)
        image_sizes.append((image.shape[1], image.shape[0]))
        masks, boxes, labels = masks_to_instances(sample.mask_paths)
        if masks.shape[0] != boxes.shape[0] or boxes.shape[0] != labels.shape[0]:
            raise ValueError(f"Target length mismatch: {sample.sample_id}")
        if (
            boxes.shape[0]
            and not ((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])).all()
        ):
            raise ValueError(f"Invalid box in sample: {sample.sample_id}")
        for class_id in sample.mask_paths:
            class_files[class_id] += 1
        for class_id in labels.tolist():
            class_instances[class_id] += 1

    missing = [path.name for path in test_images if path.name not in test_id_map]
    if missing:
        raise ValueError(f"Test images missing from id map: {missing[:5]}")

    print(f"train samples: {len(samples)}")
    print(f"test images: {len(test_images)}")
    print(f"unique train sizes: {len(set(image_sizes))}")
    print(
        f"width range: {min(w for w, _ in image_sizes)}-{max(w for w, _ in image_sizes)}"
    )
    print(
        f"height range: {min(h for _, h in image_sizes)}-{max(h for _, h in image_sizes)}"
    )
    print(f"mask files/class: {dict(sorted(class_files.items()))}")
    print(f"instances/class: {dict(sorted(class_instances.items()))}")
    print("data validation passed")


if __name__ == "__main__":
    main()
