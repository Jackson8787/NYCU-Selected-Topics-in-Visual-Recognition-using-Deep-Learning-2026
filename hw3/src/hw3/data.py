"""Dataset and data conversion utilities for HW3."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from hw3.config import CELL_CLASS_IDS, CLASS_NAMES


@dataclass(frozen=True)
class SampleInfo:
    """A train sample and its available class masks."""

    sample_id: str
    image_path: Path
    mask_paths: dict[int, Path]


def read_image_rgb(path: Path) -> np.ndarray:
    """Read a tif image as uint8 RGB, dropping alpha when present."""
    image = tifffile.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[-1] >= 4:
        image = image[..., :3]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.max() > 1:
            image = image / image.max()
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def read_mask(path: Path) -> np.ndarray:
    """Read a class mask tif as a 2D integer array."""
    mask = tifffile.imread(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return np.asarray(mask)


def list_train_samples(data_root: Path) -> list[SampleInfo]:
    """Return all train samples under an extracted data root."""
    train_root = data_root / "train"
    if not train_root.exists():
        raise FileNotFoundError(f"Missing train directory: {train_root}")

    samples: list[SampleInfo] = []
    for sample_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
        image_path = sample_dir / "image.tif"
        if not image_path.exists():
            continue
        mask_paths = {
            class_id: sample_dir / f"class{class_id}.tif"
            for class_id in CELL_CLASS_IDS
            if (sample_dir / f"class{class_id}.tif").exists()
        }
        samples.append(SampleInfo(sample_dir.name, image_path, mask_paths))
    return samples


def list_test_images(data_root: Path) -> list[Path]:
    """Return test tif images under the extracted test_release directory."""
    test_root = data_root / "test_release"
    if not test_root.exists():
        raise FileNotFoundError(f"Missing test_release directory: {test_root}")
    return sorted(test_root.glob("*.tif"))


def load_test_id_map(data_root: Path) -> dict[str, dict[str, int | str]]:
    """Load test file metadata keyed by file name."""
    mapping_path = data_root / "test_image_name_to_ids.json"
    with mapping_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return {row["file_name"]: row for row in rows}


def masks_to_instances(
    mask_paths: dict[int, Path],
    max_instances: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert class mask files into per-instance masks, boxes, and labels."""
    masks: list[np.ndarray] = []
    boxes: list[list[float]] = []
    labels: list[int] = []

    for class_id in CELL_CLASS_IDS:
        path = mask_paths.get(class_id)
        if path is None:
            continue
        class_mask = read_mask(path)
        for instance_id in np.unique(class_mask):
            if instance_id == 0:
                continue
            instance_mask = class_mask == instance_id
            ys, xs = np.where(instance_mask)
            if ys.size == 0 or xs.size == 0:
                continue
            x_min = float(xs.min())
            y_min = float(ys.min())
            x_max = float(xs.max() + 1)
            y_max = float(ys.max() + 1)
            if x_max <= x_min or y_max <= y_min:
                continue
            masks.append(instance_mask.astype(np.uint8))
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id)

    if not masks:
        return (
            np.zeros((0, 1, 1), dtype=np.uint8),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    boxes_array = np.asarray(boxes, dtype=np.float32)
    labels_array = np.asarray(labels, dtype=np.int64)
    if max_instances is not None and len(masks) > max_instances:
        areas = (boxes_array[:, 2] - boxes_array[:, 0]) * (
            boxes_array[:, 3] - boxes_array[:, 1]
        )
        keep = np.argsort(areas)[-max_instances:]
        keep.sort()
        masks = [masks[index] for index in keep]
        boxes_array = boxes_array[keep]
        labels_array = labels_array[keep]

    return (
        np.stack(masks, axis=0).astype(np.uint8),
        boxes_array,
        labels_array,
    )


class Hw3CellDataset(Dataset):
    """Torchvision-compatible instance segmentation dataset."""

    def __init__(
        self,
        data_root: Path | str,
        sample_ids: Iterable[str] | None = None,
        train: bool = False,
        flip_prob: float = 0.5,
        vflip_prob: float = 0.0,
        rotate90_prob: float = 0.0,
        max_instances: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        all_samples = list_train_samples(self.data_root)
        if sample_ids is not None:
            keep = set(sample_ids)
            all_samples = [sample for sample in all_samples if sample.sample_id in keep]
        self.samples = all_samples
        self.train = train
        self.flip_prob = flip_prob
        self.vflip_prob = vflip_prob
        self.rotate90_prob = rotate90_prob
        self.max_instances = max_instances

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = read_image_rgb(sample.image_path)
        masks, boxes, labels = masks_to_instances(sample.mask_paths, self.max_instances)
        if masks.shape[0] == 0:
            masks = np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8)

        if self.train and random.random() < self.flip_prob:
            image, masks, boxes = horizontal_flip(image, masks, boxes)
        if self.train and random.random() < self.vflip_prob:
            image, masks, boxes = vertical_flip(image, masks, boxes)
        if self.train and random.random() < self.rotate90_prob:
            k = random.randint(1, 3)
            image, masks, boxes = rotate_90(image, masks, boxes, k)

        image_tensor = (
            torch.as_tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        )
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        masks_tensor = torch.as_tensor(masks, dtype=torch.uint8)
        area_tensor = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (
            boxes_tensor[:, 2] - boxes_tensor[:, 0]
        )

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": torch.zeros((labels_tensor.shape[0],), dtype=torch.int64),
            "sample_id": sample.sample_id,
        }
        return image_tensor, target


def horizontal_flip(
    image: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flip image, masks, and boxes horizontally."""
    width = image.shape[1]
    image = np.ascontiguousarray(image[:, ::-1, :])
    if masks.size:
        masks = np.ascontiguousarray(masks[:, :, ::-1])
        flipped = boxes.copy()
        flipped[:, 0] = width - boxes[:, 2]
        flipped[:, 2] = width - boxes[:, 0]
        boxes = flipped
    return image, masks, boxes


def vertical_flip(
    image: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flip image, masks, and boxes vertically."""
    height = image.shape[0]
    image = np.ascontiguousarray(image[::-1, :, :])
    if masks.size:
        masks = np.ascontiguousarray(masks[:, ::-1, :])
        flipped = boxes.copy()
        flipped[:, 1] = height - boxes[:, 3]
        flipped[:, 3] = height - boxes[:, 1]
        boxes = flipped
    return image, masks, boxes


def rotate_90(
    image: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate image and masks by k * 90 degrees and recompute boxes."""
    image = np.ascontiguousarray(np.rot90(image, k=k, axes=(0, 1)))
    if masks.size:
        masks = np.ascontiguousarray(np.rot90(masks, k=k, axes=(1, 2)))
        boxes = boxes_from_masks(masks)
    return image, masks, boxes


def boxes_from_masks(masks: np.ndarray) -> np.ndarray:
    """Recompute xyxy boxes from binary masks."""
    boxes = []
    for mask in masks:
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            boxes.append([0.0, 0.0, 1.0, 1.0])
            continue
        boxes.append(
            [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        )
    return np.asarray(boxes, dtype=np.float32)


def collate_fn(batch):
    """Torchvision detection collate function."""
    return tuple(zip(*batch))


def sample_class_presence(samples: Iterable[SampleInfo]) -> dict[str, set[int]]:
    """Return class-presence sets by sample id."""
    return {sample.sample_id: set(sample.mask_paths.keys()) for sample in samples}


def make_stratified_folds(
    samples: list[SampleInfo],
    num_folds: int = 5,
    seed: int = 42,
) -> list[list[str]]:
    """Greedy multi-label stratification by available class masks."""
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda sample: len(sample.mask_paths), reverse=True)

    folds: list[list[SampleInfo]] = [[] for _ in range(num_folds)]
    class_counts = [Counter() for _ in range(num_folds)]

    for sample in shuffled:
        best_fold = min(
            range(num_folds),
            key=lambda fold_idx: (
                sum(class_counts[fold_idx][cid] for cid in sample.mask_paths),
                len(folds[fold_idx]),
            ),
        )
        folds[best_fold].append(sample)
        for class_id in sample.mask_paths:
            class_counts[best_fold][class_id] += 1

    return [[sample.sample_id for sample in fold] for fold in folds]


def describe_samples(samples: Iterable[SampleInfo]) -> dict[str, object]:
    """Return simple dataset statistics."""
    samples = list(samples)
    mask_files = Counter()
    for sample in samples:
        for class_id in sample.mask_paths:
            mask_files[CLASS_NAMES[class_id]] += 1
    return {
        "num_samples": len(samples),
        "mask_files": dict(mask_files),
    }


def split_train_val(
    all_ids: list[str],
    folds: list[list[str]],
    fold_index: int,
) -> tuple[list[str], list[str]]:
    """Return train/validation ids for one fold."""
    val_ids = set(folds[fold_index])
    train_ids = [sample_id for sample_id in all_ids if sample_id not in val_ids]
    return train_ids, sorted(val_ids)


def fold_summary(
    folds: list[list[str]], presence: dict[str, set[int]]
) -> list[dict[str, object]]:
    """Summarize class presence in each fold."""
    rows = []
    for index, fold in enumerate(folds):
        counts = defaultdict(int)
        for sample_id in fold:
            for class_id in presence[sample_id]:
                counts[class_id] += 1
        rows.append(
            {
                "fold": index,
                "samples": len(fold),
                "class_presence": {str(k): counts[k] for k in CELL_CLASS_IDS},
            }
        )
    return rows
