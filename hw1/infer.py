from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 100


class TestImageDataset(Dataset):
    def __init__(self, test_dir: Path, image_size: int, tta: int) -> None:
        self.paths = sorted(test_dir.glob("*.jpg"))
        if not self.paths:
            raise FileNotFoundError(f"No .jpg files found in {test_dir}")
        if tta == 1:
            resize_sizes = [int(image_size * 1.14)]
        elif tta == 2:
            resize_sizes = [int(image_size * 1.14)]
        elif tta == 4:
            resize_sizes = [int(image_size * 1.14), int(image_size * 1.28)]
        elif tta == 6:
            resize_sizes = [
                int(image_size * 1.10),
                int(image_size * 1.22),
                int(image_size * 1.34),
            ]
        else:
            raise ValueError(f"Unsupported TTA level: {tta}")

        self.transforms = []
        for resize_size in resize_sizes:
            base_transform = transforms.Compose(
                [
                    transforms.Resize(resize_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
            self.transforms.append(base_transform)
            if tta >= 2:
                flip_transform = transforms.Compose(
                    [
                        transforms.Resize(resize_size),
                        transforms.CenterCrop(image_size),
                        transforms.RandomHorizontalFlip(p=1.0),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                    ]
                )
                self.transforms.append(flip_transform)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image_path = self.paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            views = [transform(image) for transform in self.transforms]
        stacked_views = torch.stack(views, dim=0)
        return stacked_views, image_path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and create prediction.csv for CV HW1.")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to extracted data directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        required=True,
        help="One or more checkpoint paths.",
    )
    parser.add_argument("--output-csv", type=Path, default=Path("prediction.csv"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--tta",
        type=int,
        default=1,
        choices=[1, 2, 4, 6],
        help="TTA level: 1=single, 2=flip, 4=2 scales x flip, 6=3 scales x flip.",
    )
    return parser.parse_args()


def build_model(model_name: str) -> torch.nn.Module:
    model_fn = getattr(models, model_name)
    model = model_fn(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_list = []
    idx_to_label = None

    for checkpoint_path in args.checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_name = checkpoint.get("model_name", "resnet50")
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        class_to_idx = checkpoint.get("class_to_idx")
        if class_to_idx is None:
            raise KeyError(f"Checkpoint missing class_to_idx: {checkpoint_path}")
        current_idx_to_label = {
            idx: int(class_name) for class_name, idx in class_to_idx.items()
        }
        if idx_to_label is None:
            idx_to_label = current_idx_to_label
        elif idx_to_label != current_idx_to_label:
            raise ValueError(
                "Checkpoint class mappings do not match; cannot ensemble them safely."
            )

        model = build_model(model_name)
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to(device)
        models_list.append(model)

    assert idx_to_label is not None

    test_dataset = TestImageDataset(args.data_root / "test", args.image_size, args.tta)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    rows = []
    with torch.no_grad():
        for images, image_names in test_loader:
            images = images.to(device, non_blocking=True)
            batch_size, num_views, channels, height, width = images.shape
            combined_logits = None
            flat_images = images.view(batch_size * num_views, channels, height, width)
            for model in models_list:
                logits = model(flat_images)
                logits = logits.view(batch_size, num_views, -1).mean(dim=1)
                combined_logits = logits if combined_logits is None else combined_logits + logits

            combined_logits = combined_logits / len(models_list)
            predictions = combined_logits.argmax(dim=1).cpu().tolist()
            rows.extend(
                {"image_name": image_name, "pred_label": idx_to_label[pred]}
                for image_name, pred in zip(image_names, predictions)
            )

    submission = pd.DataFrame(rows)
    submission.to_csv(args.output_csv, index=False)
    print(f"Saved submission to {args.output_csv}")


if __name__ == "__main__":
    main()
