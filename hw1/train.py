from __future__ import annotations

import argparse
import csv
import copy
import json
import os
import random
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from tqdm.auto import tqdm
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, models, transforms

plt.switch_backend("Agg")


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet baseline for CV HW1.")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to extracted data directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "resnet50_baseline",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50", "resnet101"],
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--balanced-sampler",
        action="store_true",
        help="Use WeightedRandomSampler for class-balanced sampling.",
    )
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument(
        "--mix-prob",
        type=float,
        default=0.0,
        help="Probability of applying MixUp/CutMix on a training batch.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training.")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights.")
    parser.add_argument(
        "--torch-home",
        type=Path,
        default=None,
        help="Local cache directory for torchvision weights.",
    )
    parser.add_argument("--augmentation", type=str, default="trivial", choices=["trivial", "rand"])
    parser.add_argument("--randaugment-num-ops", type=int, default=2)
    parser.add_argument("--randaugment-magnitude", type=int, default=9)
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="EMA decay. Set > 0 to enable EMA evaluation.",
    )
    parser.add_argument(
        "--plot-top-k",
        type=int,
        default=15,
        help="Number of classes shown in the confusion matrix plot.",
    )
    parser.add_argument(
        "--make-submission",
        action="store_true",
        help="Run test inference with best.pt after training.",
    )
    parser.add_argument(
        "--submission-tta",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of TTA views for auto-submission.",
    )
    parser.add_argument(
        "--submission-prefix",
        type=str,
        default="submission",
        help="Prefix for the generated zip filename.",
    )
    parser.add_argument(
        "--submission-dir",
        type=Path,
        default=None,
        help="Directory to save timestamped submission files.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(
    image_size: int,
    augmentation: str,
    randaugment_num_ops: int,
    randaugment_magnitude: int,
) -> Tuple[transforms.Compose, transforms.Compose]:
    if augmentation == "rand":
        aug_transform = transforms.RandAugment(
            num_ops=randaugment_num_ops,
            magnitude=randaugment_magnitude,
        )
    else:
        aug_transform = transforms.TrivialAugmentWide()

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            aug_transform,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


class ModelEma:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.ema_model.state_dict()
        model_state = model.state_dict()
        for key, value in ema_state.items():
            model_value = model_state[key].detach()
            if not torch.is_floating_point(value):
                value.copy_(model_value)
            else:
                value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.ema_model.state_dict()


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    weight_map = {
        "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
        "resnet101": models.ResNet101_Weights.IMAGENET1K_V2,
    }
    model_fn = getattr(models, model_name)
    weights = weight_map[model_name] if pretrained else None
    try:
        model = model_fn(weights=weights)
    except Exception as error:
        print(f"Failed to load pretrained weights ({error}). Falling back to random initialization.")
        model = model_fn(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )
    return model


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def smooth_one_hot(labels: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    targets = torch.full((labels.size(0), num_classes), off_value, device=labels.device, dtype=torch.float32)
    targets.scatter_(1, labels.unsqueeze(1), on_value)
    return targets


def soft_target_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


def rand_bbox(size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
    _, _, height, width = size
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    center_x = np.random.randint(width)
    center_y = np.random.randint(height)

    x1 = np.clip(center_x - cut_w // 2, 0, width)
    y1 = np.clip(center_y - cut_h // 2, 0, height)
    x2 = np.clip(center_x + cut_w // 2, 0, width)
    y2 = np.clip(center_y + cut_h // 2, 0, height)
    return int(x1), int(y1), int(x2), int(y2)


def apply_mixup_or_cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    label_smoothing: float,
    mixup_alpha: float,
    cutmix_alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if images.size(0) < 2:
        return images, smooth_one_hot(labels, num_classes, label_smoothing)

    permutation = torch.randperm(images.size(0), device=images.device)
    shuffled_images = images[permutation]
    shuffled_labels = labels[permutation]

    use_cutmix = cutmix_alpha > 0.0 and (mixup_alpha <= 0.0 or random.random() < 0.5)
    if use_cutmix:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        x1, y1, x2, y2 = rand_bbox(images.size(), lam)
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = shuffled_images[:, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        mixed_images = lam * images + (1.0 - lam) * shuffled_images

    targets_a = smooth_one_hot(labels, num_classes, label_smoothing)
    targets_b = smooth_one_hot(shuffled_labels, num_classes, label_smoothing)
    mixed_targets = lam * targets_a + (1.0 - lam) * targets_b
    return mixed_images, mixed_targets


def ascii_curve(values: List[float], width: int = 40) -> str:
    if not values:
        return ""
    blocks = " .:-=+*#%@"
    min_v = min(values)
    max_v = max(values)
    if abs(max_v - min_v) < 1e-12:
        return blocks[-1] * min(len(values), width)

    sampled = values
    if len(values) > width:
        indices = np.linspace(0, len(values) - 1, width).astype(int)
        sampled = [values[i] for i in indices]

    chars = []
    for value in sampled:
        normalized = (value - min_v) / (max_v - min_v)
        idx = min(int(normalized * (len(blocks) - 1)), len(blocks) - 1)
        chars.append(blocks[idx])
    return "".join(chars)


def compute_confusion_matrix(labels: List[int], predictions: List[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(labels, predictions):
        matrix[true_label, pred_label] += 1
    return matrix


def summarize_confusions(confusion: np.ndarray, top_k: int = 5) -> List[str]:
    pairs = []
    for true_idx in range(confusion.shape[0]):
        for pred_idx in range(confusion.shape[1]):
            if true_idx == pred_idx:
                continue
            count = int(confusion[true_idx, pred_idx])
            if count > 0:
                pairs.append((count, true_idx, pred_idx))
    pairs.sort(reverse=True)
    return [f"{true_idx}->{pred_idx}: {count}" for count, true_idx, pred_idx in pairs[:top_k]]


def save_curves(history: List[Dict[str, float]], curve_dir: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    curve_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_acc, marker="o", label="train_acc")
    axes[0].plot(epochs, val_acc, marker="o", label="val_acc")
    axes[0].set_title("Accuracy Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_loss, marker="o", label="train_loss")
    axes[1].plot(epochs, val_loss, marker="o", label="val_loss")
    axes[1].set_title("Loss Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(curve_dir / "training_curves_latest.png", dpi=160)
    plt.close(fig)

    epoch_fig, epoch_ax = plt.subplots(figsize=(6, 4))
    epoch_ax.plot(epochs, val_acc, marker="o", color="tab:green")
    epoch_ax.set_title("Validation Accuracy by Epoch")
    epoch_ax.set_xlabel("Epoch")
    epoch_ax.set_ylabel("Validation Accuracy")
    epoch_ax.grid(True, alpha=0.3)
    epoch_fig.tight_layout()
    epoch_fig.savefig(curve_dir / "val_accuracy_by_epoch.png", dpi=160)
    epoch_fig.savefig(curve_dir / f"epoch_{epochs[-1]:03d}_val_accuracy.png", dpi=160)
    plt.close(epoch_fig)


def save_confusion_matrix_plot(
    confusion: np.ndarray,
    epoch: int,
    best_val_acc: float,
    save_path: Path,
    top_k: int,
) -> None:
    class_totals = confusion.sum(axis=1)
    top_indices = np.argsort(class_totals)[::-1][:top_k]
    trimmed = confusion[np.ix_(top_indices, top_indices)]

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(trimmed, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Confusion Matrix (epoch={epoch}, best_val_acc={best_val_acc:.4f})")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(top_indices)))
    ax.set_yticks(range(len(top_indices)))
    ax.set_xticklabels([str(idx) for idx in top_indices], rotation=90)
    ax.set_yticklabels([str(idx) for idx in top_indices])
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


class TestImageDataset(Dataset):
    def __init__(self, test_dir: Path, image_size: int) -> None:
        self.paths = sorted(test_dir.glob("*.jpg"))
        if not self.paths:
            raise FileNotFoundError(f"No .jpg files found in {test_dir}")
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.14)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image_path = self.paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        return self.transform(image), image_path.stem


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    ema: ModelEma | None,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    total_epochs: int,
    stage_name: str,
    num_classes: int,
    label_smoothing: float,
    mixup_alpha: float,
    cutmix_alpha: float,
    mix_prob: float,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    all_predictions: List[int] = []
    all_labels: List[int] = []

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch:02d}/{total_epochs} [{stage_name}]",
        leave=False,
        dynamic_ncols=True,
    )
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)
        batch_size = images.size(0)
        metric_labels = labels
        targets: torch.Tensor | None = None

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if mix_prob > 0.0 and random.random() < mix_prob and (mixup_alpha > 0.0 or cutmix_alpha > 0.0):
                images, targets = apply_mixup_or_cutmix(
                    images=images,
                    labels=labels,
                    num_classes=num_classes,
                    label_smoothing=label_smoothing,
                    mixup_alpha=mixup_alpha,
                    cutmix_alpha=cutmix_alpha,
                )

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                if targets is not None:
                    loss = soft_target_cross_entropy(logits, targets)
                else:
                    loss = criterion(logits, labels)

            if is_train:
                assert optimizer is not None
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if ema is not None:
                    ema.update(model)

        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * batch_size
        total_acc += (predictions == metric_labels).float().mean().item() * batch_size
        total_samples += batch_size
        all_predictions.extend(predictions.detach().cpu().tolist())
        all_labels.extend(metric_labels.detach().cpu().tolist())
        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_acc / total_samples:.4f}",
        )

    return {
        "loss": total_loss / total_samples,
        "acc": total_acc / total_samples,
        "predictions": all_predictions,
        "labels": all_labels,
    }


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def serialize_args(args: argparse.Namespace) -> Dict:
    payload = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def make_loader_kwargs(num_workers: int, prefetch_factor: int) -> Dict:
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def build_balanced_sampler(image_folder: datasets.ImageFolder) -> WeightedRandomSampler:
    targets = image_folder.targets
    class_counts = np.bincount(targets, minlength=len(image_folder.classes))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[target] for target in targets]
    sample_weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_submission_from_checkpoint(
    checkpoint_path: Path,
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    submission_dir: Path,
    submission_prefix: str,
    tta: int,
) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_name = checkpoint.get("model_name", "resnet50")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    class_to_idx = checkpoint.get("class_to_idx")
    model = build_model(model_name, NUM_CLASSES, pretrained=False)
    model.load_state_dict(state_dict)
    model.eval()

    if class_to_idx is None:
        raise KeyError("Checkpoint missing class_to_idx; cannot map model outputs back to competition labels.")
    idx_to_label = {idx: int(class_name) for class_name, idx in class_to_idx.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_dataset = TestImageDataset(data_root / "test", image_size=image_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **make_loader_kwargs(num_workers, prefetch_factor),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_dir.mkdir(parents=True, exist_ok=True)
    csv_path = submission_dir / f"{submission_prefix}_{timestamp}.csv"
    zip_path = submission_dir / f"{submission_prefix}_{timestamp}.zip"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_name", "pred_label"])

        with torch.no_grad():
            progress = tqdm(test_loader, desc="Inference [test]", leave=False, dynamic_ncols=True)
            for images, image_names in progress:
                images = images.to(device, non_blocking=True)
                if device.type == "cuda":
                    images = images.to(memory_format=torch.channels_last)
                logits = model(images)
                if tta >= 2:
                    flipped_images = torch.flip(images, dims=[3])
                    logits = (logits + model(flipped_images)) / 2.0
                predictions = logits.argmax(dim=1).cpu().tolist()
                for image_name, pred_label in zip(image_names, predictions):
                    writer.writerow([image_name, idx_to_label[pred_label]])

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(csv_path, arcname="prediction.csv")

    print(f"Saved timestamped submission CSV to {csv_path}")
    print(f"Saved timestamped submission ZIP to {zip_path}")
    return zip_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_dir = output_dir / "curves"
    confusion_dir = output_dir / "confusion_matrices"
    submission_dir = args.submission_dir or (output_dir / "submissions")
    torch_home = args.torch_home or (output_dir / "torch_cache")
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    train_dir = args.data_root / "train"
    val_dir = args.data_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Expected train/val under {args.data_root}")

    train_transform, eval_transform = build_transforms(
        image_size=args.image_size,
        augmentation=args.augmentation,
        randaugment_num_ops=args.randaugment_num_ops,
        randaugment_magnitude=args.randaugment_magnitude,
    )
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)

    if len(train_dataset.classes) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes, got {len(train_dataset.classes)}")

    train_sampler = build_balanced_sampler(train_dataset) if args.balanced_sampler else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **make_loader_kwargs(args.num_workers, args.prefetch_factor),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **make_loader_kwargs(args.num_workers, args.prefetch_factor),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    model = build_model(args.model_name, NUM_CLASSES, pretrained=not args.no_pretrained).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    ema = ModelEma(model, decay=args.ema_decay) if args.ema_decay > 0.0 else None

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.warmup_epochs > 0 and args.warmup_epochs < args.epochs:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    save_json(output_dir / "class_mapping.json", idx_to_class)
    serializable_args = serialize_args(args)

    best_val_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            ema,
            device,
            use_amp,
            epoch=epoch,
            total_epochs=args.epochs,
            stage_name="train",
            num_classes=NUM_CLASSES,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mix_prob=args.mix_prob,
        )
        eval_model = ema.ema_model if ema is not None else model
        val_metrics = run_epoch(
            eval_model,
            val_loader,
            criterion,
            None,
            None,
            None,
            device,
            use_amp=False,
            epoch=epoch,
            total_epochs=args.epochs,
            stage_name="val",
            num_classes=NUM_CLASSES,
            label_smoothing=args.label_smoothing,
            mixup_alpha=0.0,
            cutmix_alpha=0.0,
            mix_prob=0.0,
        )
        scheduler.step()

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "lr": current_lr,
            "elapsed_sec": elapsed,
        }
        history.append(epoch_log)

        confusion = compute_confusion_matrix(val_metrics["labels"], val_metrics["predictions"], NUM_CLASSES)
        confusion_summary = summarize_confusions(confusion, top_k=5)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} | "
            f"lr={current_lr:.6f} | time={elapsed:.1f}s"
        )
        print(f"  val_acc_curve:  {ascii_curve([row['val_acc'] for row in history])}")
        print(f"  train_acc_curve:{ascii_curve([row['train_acc'] for row in history])}")
        if confusion_summary:
            print("  top_confusions: " + ", ".join(confusion_summary))
        else:
            print("  top_confusions: none")

        checkpoint = {
            "epoch": epoch,
            "model_name": args.model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "args": serializable_args,
            "class_to_idx": class_to_idx,
        }
        if ema is not None:
            checkpoint["ema_state_dict"] = ema.state_dict()
        torch.save(checkpoint, output_dir / "last.pt")

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_state_dict = ema.state_dict() if ema is not None else model.state_dict()
            best_checkpoint = {
                "model_name": args.model_name,
                "model_state_dict": best_state_dict,
                "best_val_acc": best_val_acc,
                "epoch": epoch,
                "class_to_idx": class_to_idx,
                "args": serializable_args,
            }
            torch.save(best_checkpoint, output_dir / "best.pt")

        save_json(output_dir / "history.json", {"history": history, "best_val_acc": best_val_acc})
        save_curves(history, curve_dir)
        save_confusion_matrix_plot(
            confusion=confusion,
            epoch=epoch,
            best_val_acc=best_val_acc,
            save_path=confusion_dir / f"epoch_{epoch:03d}.png",
            top_k=args.plot_top_k,
        )

    print(f"Training finished. Best val acc: {best_val_acc:.4f}")

    if args.make_submission:
        build_submission_from_checkpoint(
            checkpoint_path=output_dir / "best.pt",
            data_root=args.data_root,
            image_size=args.image_size,
            batch_size=max(args.batch_size, 128),
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            submission_dir=submission_dir,
            submission_prefix=args.submission_prefix,
            tta=args.submission_tta,
        )


if __name__ == "__main__":
    main()
