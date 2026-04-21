import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Dataset


class DigitDetectionDataset(Dataset):
    """COCO-style digit detection dataset for DETR fine-tuning."""

    def __init__(self, data_dir, split, processor=None, augment=False):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_dir = self.data_dir / split
        self.processor = processor
        self.augment = augment

        annotation_path = self.data_dir / f"{split}.json"
        with annotation_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations_by_image = {image["id"]: [] for image in self.images}
        for annotation in coco["annotations"]:
            item = dict(annotation)
            item["category_id"] = int(item["category_id"]) - 1
            self.annotations_by_image[item["image_id"]].append(item)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_info = self.images[index]
        image = Image.open(
            self.image_dir / image_info["file_name"]
        ).convert("RGB")
        if self.augment:
            image = apply_color_augmentation(image)
        annotations = {
            "image_id": image_info["id"],
            "annotations": self.annotations_by_image[image_info["id"]],
        }

        if self.processor is None:
            return image, annotations

        encoded = self.processor(
            images=image,
            annotations=annotations,
            return_tensors="pt",
        )
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "pixel_mask": encoded["pixel_mask"].squeeze(0),
            "labels": encoded["labels"][0],
        }


class DigitTestDataset(Dataset):
    """Unlabeled test images for CodaBench submission inference."""

    def __init__(self, data_dir):
        self.image_dir = Path(data_dir) / "test"
        self.images = sorted(
            self.image_dir.glob("*.png"),
            key=lambda path: int(path.stem),
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        image_id = int(image_path.stem)
        width, height = image.size
        return {
            "image": image,
            "image_id": image_id,
            "target_size": torch.tensor([height, width]),
        }


class DetrCollator:
    """Pads DETR inputs and keeps per-image labels as a list."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        pixel_values = [item["pixel_values"] for item in batch]
        encoded = self.processor.pad(pixel_values, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"],
            "pixel_mask": encoded["pixel_mask"],
            "labels": [item["labels"] for item in batch],
        }


def test_collate_fn(batch):
    return {
        "images": [item["image"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "target_sizes": torch.stack([item["target_size"] for item in batch]),
    }


def apply_color_augmentation(image):
    """Apply label-preserving image-only augmentation for digit detection."""

    if random.random() < 0.8:
        image = ImageEnhance.Brightness(image).enhance(
            random.uniform(0.75, 1.25)
        )
    if random.random() < 0.8:
        image = ImageEnhance.Contrast(image).enhance(
            random.uniform(0.75, 1.35)
        )
    if random.random() < 0.5:
        image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.2))
    if random.random() < 0.2:
        image = ImageOps.autocontrast(image)
    if random.random() < 0.1:
        image = image.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.6))
        )
    return image
