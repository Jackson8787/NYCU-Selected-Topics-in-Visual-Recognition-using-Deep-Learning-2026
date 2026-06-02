"""Dataset loading and deterministic splits for HW4 restoration."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PairRecord:
    """One degraded/clean training pair."""

    degraded: Path
    clean: Path
    degradation: str


def find_pairs(data_root: Path) -> list[PairRecord]:
    """Find all paired rain and snow restoration examples."""
    degraded_dir = data_root / "train" / "degraded"
    clean_dir = data_root / "train" / "clean"
    records: list[PairRecord] = []
    for degradation in ("rain", "snow"):
        for degraded in sorted(degraded_dir.glob(f"{degradation}-*.png")):
            clean = clean_dir / degraded.name.replace("-", "_clean-", 1)
            if not clean.exists():
                raise FileNotFoundError(
                    f"Missing clean target for {degraded.name}: {clean}"
                )
            records.append(PairRecord(degraded, clean, degradation))
    if len(records) != 3200:
        raise ValueError(f"Expected 3200 training pairs, found {len(records)}")
    return records


def stratified_split(
    records: list[PairRecord], val_per_type: int = 160, seed: int = 42
) -> tuple[list[PairRecord], list[PairRecord]]:
    """Split records while retaining equal rain and snow validation sets."""
    generator = random.Random(seed)
    train: list[PairRecord] = []
    val: list[PairRecord] = []
    for degradation in ("rain", "snow"):
        group = [record for record in records if record.degradation == degradation]
        generator.shuffle(group)
        val.extend(group[:val_per_type])
        train.extend(group[val_per_type:])
    return sorted(train, key=lambda item: item.degraded.name), sorted(
        val, key=lambda item: item.degraded.name
    )


def list_test_images(data_root: Path) -> list[Path]:
    """Return test images sorted numerically."""
    test_dir = data_root / "test" / "degraded"
    images = sorted(test_dir.glob("*.png"), key=lambda path: int(path.stem))
    expected = [f"{index}.png" for index in range(100)]
    if [path.name for path in images] != expected:
        raise ValueError("Test data must contain exactly 0.png through 99.png")
    return images


def load_rgb_tensor(path: Path) -> torch.Tensor:
    """Load an RGB image as a float tensor in [0, 1]."""
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class PairedRestorationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Paired restoration dataset with synchronized spatial augmentation."""

    def __init__(
        self, records: list[PairRecord], train: bool, patch_size: int = 128
    ) -> None:
        self.records = records
        self.train = train
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        degraded = load_rgb_tensor(record.degraded)
        clean = load_rgb_tensor(record.clean)
        if self.train:
            degraded, clean = self._augment(degraded, clean)
        return degraded, clean, record.degraded.name

    def _augment(
        self, degraded: torch.Tensor, clean: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, height, width = degraded.shape
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(f"Patch size {self.patch_size} exceeds image dimensions")
        top = torch.randint(0, height - self.patch_size + 1, (1,)).item()
        left = torch.randint(0, width - self.patch_size + 1, (1,)).item()
        degraded = degraded[
            :, top : top + self.patch_size, left : left + self.patch_size
        ]
        clean = clean[:, top : top + self.patch_size, left : left + self.patch_size]
        if torch.rand(()) < 0.5:
            degraded, clean = degraded.flip(-1), clean.flip(-1)
        if torch.rand(()) < 0.5:
            degraded, clean = degraded.flip(-2), clean.flip(-2)
        rotations = int(torch.randint(0, 4, (1,)).item())
        return torch.rot90(degraded, rotations, (-2, -1)), torch.rot90(
            clean, rotations, (-2, -1)
        )


class TestRestorationDataset(Dataset[tuple[torch.Tensor, str]]):
    """Test image dataset retaining competition filenames."""

    def __init__(self, images: list[Path]) -> None:
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.images[index]
        return load_rgb_tensor(path), path.name
