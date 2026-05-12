"""Training loop helpers."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from tqdm import tqdm


def move_target_to_device(target: dict, device: torch.device) -> dict:
    """Move tensor target fields to device and keep metadata on CPU."""
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in target.items()
    }


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: torch.amp.GradScaler | None = None,
    print_freq: int = 20,
    skip_oom: bool = True,
) -> dict[str, float]:
    """Train one epoch and return average losses."""
    model.train()
    totals: dict[str, float] = {}
    steps = 0
    skipped_oom = 0

    progress = tqdm(data_loader, desc=f"train epoch {epoch}", leave=False)
    for images, targets in progress:
        try:
            images = [image.to(device) for image in images]
            targets = [move_target_to_device(target, device) for target in targets]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=scaler is not None
            ):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            if not math.isfinite(float(losses.detach().cpu())):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}: {loss_dict}"
                )

            if scaler is not None:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()
        except Exception as exc:
            if not skip_oom or "out of memory" not in str(exc).lower():
                raise
            skipped_oom += 1
            optimizer.zero_grad(set_to_none=True)
            images = []
            targets = []
            loss_dict = {}
            losses = None
            if device.type == "cuda":
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            progress.set_postfix(skipped_oom=skipped_oom)
            continue

        steps += 1
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        totals["loss"] = totals.get("loss", 0.0) + float(losses.detach().cpu())

        if steps % print_freq == 0:
            progress.set_postfix(loss=totals["loss"] / steps)

    averages = {key: value / max(steps, 1) for key, value in totals.items()}
    averages["skipped_oom"] = float(skipped_oom)
    return averages


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    args: dict,
) -> None:
    """Save a training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "args": args,
        },
        path,
    )
