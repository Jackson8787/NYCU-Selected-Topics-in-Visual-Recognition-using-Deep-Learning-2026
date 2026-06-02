"""Training, evaluation, and inference helpers."""

from __future__ import annotations

import csv
import math
import random
import zipfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torchvision.models import VGG16_Weights, VGG19_Weights, vgg16, vgg19

from .data import PairedRestorationDataset, TestRestorationDataset
from .model import PromptIR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CharbonnierEdgeLoss(nn.Module):
    """Charbonnier reconstruction loss with a small gradient consistency term."""

    def __init__(self, edge_weight: float = 0.05, eps: float = 1e-3) -> None:
        super().__init__()
        self.edge_weight = edge_weight
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        charbonnier = torch.sqrt((prediction - target).square() + self.eps**2).mean()
        if self.edge_weight <= 0:
            return charbonnier
        pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
        pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        edge = (pred_dx - target_dx).abs().mean() + (pred_dy - target_dy).abs().mean()
        return charbonnier + self.edge_weight * edge


def ssim_index(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Differentiable SSIM averaged over batch and channels."""
    return ssim_per_sample(prediction, target, window_size, data_range).mean()


def ssim_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Differentiable SSIM averaged per sample."""
    padding = window_size // 2
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = F.avg_pool2d(prediction, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    mu_x2 = mu_x.square()
    mu_y2 = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x = (
        F.avg_pool2d(prediction.square(), window_size, stride=1, padding=padding)
        - mu_x2
    )
    sigma_y = (
        F.avg_pool2d(target.square(), window_size, stride=1, padding=padding) - mu_y2
    )
    sigma_xy = (
        F.avg_pool2d(prediction * target, window_size, stride=1, padding=padding)
        - mu_xy
    )
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    ssim = numerator / denominator.clamp_min(1e-8)
    return ssim.mean(dim=(1, 2, 3))


class L1PSNRSSIMLoss(nn.Module):
    """L1 plus the PSNR/SSIM terms used by the referenced high-score Lab4 repo."""

    def __init__(self, psnr_weight: float = 0.1, ssim_weight: float = 0.1) -> None:
        super().__init__()
        self.psnr_weight = psnr_weight
        self.ssim_weight = ssim_weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction.clamp(0.0, 1.0)
        mse = F.mse_loss(prediction, target).clamp_min(1e-10)
        negative_psnr = 10.0 * torch.log10(mse)
        ssim_loss = 1.0 - ssim_index(prediction, target)
        return (
            F.l1_loss(prediction, target)
            + self.psnr_weight * negative_psnr
            + self.ssim_weight * ssim_loss
        )

    def per_sample_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        prediction = prediction.clamp(0.0, 1.0)
        per_sample_l1 = (prediction - target).abs().mean(dim=(1, 2, 3))
        per_sample_mse = (prediction - target).square().mean(dim=(1, 2, 3))
        negative_psnr = 10.0 * torch.log10(per_sample_mse.clamp_min(1e-10))
        ssim_loss = 1.0 - ssim_per_sample(prediction, target)
        return (
            per_sample_l1
            + self.psnr_weight * negative_psnr
            + self.ssim_weight * ssim_loss
        )


class VGGPerceptualLoss(nn.Module):
    """Composite reconstruction loss with a frozen VGG19 feature-space term."""

    def __init__(
        self,
        base_loss: str = "mse",
        perceptual_weight: float = 0.1,
        psnr_weight: float = 0.1,
        ssim_weight: float = 0.1,
        pretrained: bool = True,
        backbone: str = "vgg19",
        layer_preset: str = "extended",
    ) -> None:
        super().__init__()
        if base_loss not in {"mse", "l1_psnr_ssim"}:
            raise ValueError(f"Unsupported perceptual base loss: {base_loss}")
        if backbone not in {"vgg16", "vgg19"}:
            raise ValueError(f"Unsupported perceptual backbone: {backbone}")
        if layer_preset not in {"reference", "extended"}:
            raise ValueError(f"Unsupported perceptual layer preset: {layer_preset}")
        self.base_loss = base_loss
        self.perceptual_weight = perceptual_weight
        self.psnr_weight = psnr_weight
        self.ssim_weight = ssim_weight
        if backbone == "vgg16":
            weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            self.perceptual_extractor = vgg16(weights=weights).features[:16].eval()
            self.selected_layers = {3, 8, 15}
        else:
            weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
            self.perceptual_extractor = vgg19(weights=weights).features[:36].eval()
            self.selected_layers = {3, 8, 17, 26, 35}
        if layer_preset == "reference":
            self.selected_layers = {3, 8, 15}
        for parameter in self.perceptual_extractor.parameters():
            parameter.requires_grad = False
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        image = image.clamp(0.0, 1.0)
        return (image - self.imagenet_mean) / self.imagenet_std

    def perceptual_per_sample(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        prediction_features = self._normalize(prediction.float())
        target_features = self._normalize(target.float())
        per_sample = prediction.new_zeros(prediction.shape[0], dtype=torch.float32)
        for index, layer in enumerate(self.perceptual_extractor):
            prediction_features = layer(prediction_features)
            target_features = layer(target_features)
            if index in self.selected_layers:
                per_sample = per_sample + (
                    prediction_features - target_features
                ).abs().mean(dim=(1, 2, 3))
        return per_sample

    def per_sample_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        prediction = prediction.clamp(0.0, 1.0)
        if self.base_loss == "mse":
            base = (prediction - target).square().mean(dim=(1, 2, 3))
        else:
            base = L1PSNRSSIMLoss(
                psnr_weight=self.psnr_weight, ssim_weight=self.ssim_weight
            ).per_sample_loss(prediction, target)
        return base + self.perceptual_weight * self.perceptual_per_sample(
            prediction, target
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.per_sample_loss(prediction, target).mean()


def build_loss(
    name: str,
    edge_weight: float,
    psnr_weight: float = 0.1,
    ssim_weight: float = 0.1,
    perceptual_weight: float = 0.1,
    perceptual_pretrained: bool = True,
    perceptual_backbone: str = "vgg19",
    perceptual_layer_preset: str = "extended",
) -> nn.Module:
    if name == "l1":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    if name == "l1_mse":
        return L1MSELoss()
    if name == "l1_psnr_ssim":
        return L1PSNRSSIMLoss(psnr_weight=psnr_weight, ssim_weight=ssim_weight)
    if name == "mse_perceptual":
        return VGGPerceptualLoss(
            base_loss="mse",
            perceptual_weight=perceptual_weight,
            psnr_weight=psnr_weight,
            ssim_weight=ssim_weight,
            pretrained=perceptual_pretrained,
            backbone=perceptual_backbone,
            layer_preset=perceptual_layer_preset,
        )
    if name == "l1_psnr_ssim_perceptual":
        return VGGPerceptualLoss(
            base_loss="l1_psnr_ssim",
            perceptual_weight=perceptual_weight,
            psnr_weight=psnr_weight,
            ssim_weight=ssim_weight,
            pretrained=perceptual_pretrained,
            backbone=perceptual_backbone,
            layer_preset=perceptual_layer_preset,
        )
    if name == "charbonnier":
        return CharbonnierEdgeLoss(edge_weight=0.0)
    if name == "charbonnier_edge":
        return CharbonnierEdgeLoss(edge_weight=edge_weight)
    raise ValueError(f"Unsupported loss: {name}")


def degradation_weights(
    filenames: tuple[str, ...] | list[str],
    device: torch.device,
    rain_weight: float,
    snow_weight: float,
) -> torch.Tensor:
    values = [
        rain_weight if str(filename).startswith("rain-") else snow_weight
        for filename in filenames
    ]
    return torch.tensor(values, device=device, dtype=torch.float32)


def weighted_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    filenames: tuple[str, ...] | list[str],
    criterion: nn.Module,
    loss_name: str,
    device: torch.device,
    rain_weight: float,
    snow_weight: float,
    psnr_weight: float = 0.1,
    ssim_weight: float = 0.1,
) -> torch.Tensor:
    weights = degradation_weights(filenames, device, rain_weight, snow_weight)
    weights = weights / weights.mean().clamp_min(1e-8)
    if hasattr(criterion, "per_sample_loss"):
        per_sample = criterion.per_sample_loss(prediction, target)
        return (per_sample * weights).mean()
    view_shape = (weights.shape[0],) + (1,) * (prediction.ndim - 1)
    pixel_weights = weights.view(view_shape)
    if loss_name == "mse":
        per_pixel = (prediction - target).square()
        return (per_pixel * pixel_weights).mean()
    elif loss_name == "l1":
        per_pixel = (prediction - target).abs()
        return (per_pixel * pixel_weights).mean()
    elif loss_name == "l1_mse":
        per_pixel = (
            0.5 * (prediction - target).abs() + 0.5 * (prediction - target).square()
        )
        return (per_pixel * pixel_weights).mean()
    else:
        raise ValueError(
            "rain/snow weighted training currently supports mse, l1, l1_mse, "
            "l1_psnr_ssim, and perceptual variants"
        )


class L1MSELoss(nn.Module):
    """Blend L1 stability with the MSE objective used by PSNR."""

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5 * F.l1_loss(prediction, target) + 0.5 * F.mse_loss(
            prediction, target
        )


def build_model(profile: str = "baseline") -> PromptIR:
    """Build the official-size baseline, a wider variant, or a quick test model."""
    if profile == "tiny":
        return PromptIR(
            dim=16, num_blocks=(1, 1, 1, 1), refinement_blocks=1, heads=(1, 1, 2, 4)
        )
    if profile == "tiny_prompt7":
        return PromptIR(
            dim=16,
            num_blocks=(1, 1, 1, 1),
            refinement_blocks=1,
            heads=(1, 1, 2, 4),
            prompt_len=7,
        )
    if profile == "baseline":
        return PromptIR()
    if profile == "prompt7":
        return PromptIR(prompt_len=7)
    if profile == "prompt9":
        return PromptIR(prompt_len=9)
    if profile == "wide":
        return PromptIR(dim=56)
    if profile == "wide_deep":
        return PromptIR(dim=56, num_blocks=(4, 6, 8, 10), refinement_blocks=6)
    if profile != "baseline":
        raise ValueError(f"Unsupported model profile: {profile}")
    return PromptIR()


def create_loader(
    dataset: PairedRestorationDataset | TestRestorationDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def epoch_learning_rate(
    epoch: int, epochs: int, base_lr: float, warmup_epochs: int
) -> float:
    if epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def psnr_batch(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction.clamp(0.0, 1.0) - target).square().mean(dim=(1, 2, 3))
    return -10.0 * torch.log10(mse.clamp_min(1e-10))


def tta_forward(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Average the eight dihedral test-time augmentations."""
    outputs: list[torch.Tensor] = []
    for rotation in range(4):
        rotated = torch.rot90(image, rotation, dims=(-2, -1))
        restored = model(rotated)
        outputs.append(torch.rot90(restored, -rotation, dims=(-2, -1)))
        flipped = rotated.flip(-1)
        restored_flip = model(flipped).flip(-1)
        outputs.append(torch.rot90(restored_flip, -rotation, dims=(-2, -1)))
    return torch.stack(outputs, dim=0).mean(dim=0)


def predict_batch(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    amp: bool,
    tta: bool,
) -> torch.Tensor:
    with autocast_context(device, amp):
        return tta_forward(model, image) if tta else model(image)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    amp: bool,
    description: str = "Validation",
    tta: bool = False,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    progress = tqdm(loader, desc=description, leave=False, dynamic_ncols=True)
    for degraded, clean, _ in progress:
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        prediction = predict_batch(model, degraded, device, amp, tta)
        batch_scores = psnr_batch(prediction.float(), clean)
        total += float(batch_scores.sum().item())
        count += batch_scores.numel()
        progress.set_postfix(psnr=f"{total / count:.3f}")
    return total / count


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_psnr: float,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_psnr": best_psnr,
            "config": config,
        },
        path,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    device: torch.device,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    warmup_epochs: int,
    accumulation_steps: int,
    amp: bool,
    config: dict[str, Any],
    loss_name: str = "l1",
    edge_weight: float = 0.05,
    psnr_weight: float = 0.1,
    ssim_weight: float = 0.1,
    perceptual_weight: float = 0.1,
    perceptual_pretrained: bool = True,
    perceptual_backbone: str = "vgg19",
    perceptual_layer_preset: str = "extended",
    weight_decay: float = 0.01,
    scheduler_name: str = "cosine",
    onecycle_pct_start: float = 0.3,
    onecycle_div_factor: float = 25.0,
    onecycle_final_div_factor: float = 1_000.0,
    resume: Path | None = None,
    rain_weight: float = 1.0,
    snow_weight: float = 1.0,
) -> tuple[Path, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    writer = SummaryWriter(output_dir / "tensorboard")
    metrics_path = output_dir / "metrics.csv"
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler: torch.optim.lr_scheduler.OneCycleLR | None = None
    use_constant_lr = False
    if scheduler_name == "onecycle":
        updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=updates_per_epoch,
            pct_start=onecycle_pct_start,
            div_factor=onecycle_div_factor,
            final_div_factor=onecycle_final_div_factor,
        )
    elif scheduler_name == "constant":
        use_constant_lr = True
    elif scheduler_name != "cosine":
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    criterion = build_loss(
        loss_name,
        edge_weight,
        psnr_weight,
        ssim_weight,
        perceptual_weight,
        perceptual_pretrained,
        perceptual_backbone,
        perceptual_layer_preset,
    )
    best_psnr = float("-inf")
    best_path = checkpoint_dir / "best.pt"
    model.to(device)
    criterion.to(device)
    start_epoch = 1
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_psnr = float(checkpoint["best_psnr"])
        print(
            f"Resuming after epoch {start_epoch - 1} from {resume}; best_psnr={best_psnr:.3f}"
        )

    append_metrics = resume is not None and metrics_path.exists()
    with metrics_path.open(
        "a" if append_metrics else "w", newline="", encoding="utf-8"
    ) as metrics_file:
        metric_writer = csv.DictWriter(
            metrics_file,
            fieldnames=[
                "epoch",
                "learning_rate",
                "train_loss",
                "val_psnr",
                "best_psnr",
            ],
        )
        if not append_metrics:
            metric_writer.writeheader()
        for epoch in range(start_epoch, epochs + 1):
            if scheduler is None and not use_constant_lr:
                current_lr = epoch_learning_rate(
                    epoch, epochs, learning_rate, warmup_epochs
                )
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
            else:
                current_lr = optimizer.param_groups[0]["lr"]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            samples = 0
            progress = tqdm(
                train_loader, desc=f"Epoch {epoch:03d}/{epochs:03d}", dynamic_ncols=True
            )
            use_weighted_loss = rain_weight != 1.0 or snow_weight != 1.0
            for step, (degraded, clean, filenames) in enumerate(progress, start=1):
                degraded = degraded.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                with autocast_context(device, amp):
                    restored = model(degraded)
                    if use_weighted_loss:
                        loss = weighted_loss(
                            restored,
                            clean,
                            filenames,
                            criterion,
                            loss_name,
                            device,
                            rain_weight,
                            snow_weight,
                            psnr_weight,
                            ssim_weight,
                        )
                    else:
                        loss = criterion(restored, clean)
                    loss = loss / accumulation_steps
                loss.backward()
                if step % accumulation_steps == 0 or step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    current_lr = optimizer.param_groups[0]["lr"]
                batch_loss = float(loss.item()) * accumulation_steps
                running_loss += batch_loss * degraded.shape[0]
                samples += degraded.shape[0]
                progress.set_postfix(
                    loss=f"{running_loss / samples:.5f}", lr=f"{current_lr:.2e}"
                )

            train_loss = running_loss / samples
            val_psnr = evaluate(model, val_loader, device, amp)
            improved = val_psnr > best_psnr
            best_psnr = max(best_psnr, val_psnr)
            save_checkpoint(
                checkpoint_dir / "last.pt", model, optimizer, epoch, best_psnr, config
            )
            if improved:
                save_checkpoint(best_path, model, optimizer, epoch, best_psnr, config)
            metric_writer.writerow(
                {
                    "epoch": epoch,
                    "learning_rate": current_lr,
                    "train_loss": train_loss,
                    "val_psnr": val_psnr,
                    "best_psnr": best_psnr,
                }
            )
            metrics_file.flush()
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("psnr/validation", val_psnr, epoch)
            writer.add_scalar("learning_rate", current_lr, epoch)
            writer.flush()
            marker = "saved best.pt" if improved else "kept best.pt"
            tqdm.write(
                f"Epoch {epoch:03d}: train_loss={train_loss:.5f} "
                f"val_psnr={val_psnr:.3f} best_psnr={best_psnr:.3f} ({marker})"
            )
    writer.close()
    return best_path, best_psnr


@torch.inference_mode()
def create_submission(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    output_dir: Path,
    amp: bool,
    tta: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device).eval()
    predictions: dict[str, np.ndarray] = {}
    progress = tqdm(loader, desc="Test inference", dynamic_ncols=True)
    for degraded, filenames in progress:
        degraded = degraded.to(device, non_blocking=True)
        restored = predict_batch(model, degraded, device, amp, tta)
        arrays = (
            restored.float().clamp(0.0, 1.0).mul(255.0).round().byte().cpu().numpy()
        )
        predictions.update(
            {filename: array for filename, array in zip(filenames, arrays)}
        )
    npz_path = output_dir / "pred.npz"
    np.savez(npz_path, **predictions)
    zip_path = output_dir / "hw4_promptir_baseline_submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(npz_path, arcname="pred.npz")
    return zip_path
