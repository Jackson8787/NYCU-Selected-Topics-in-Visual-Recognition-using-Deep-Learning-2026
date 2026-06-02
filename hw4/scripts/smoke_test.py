"""GPU smoke checks for data, PromptIR training, checkpoints, and submissions."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from _bootstrap import PROJECT_ROOT  # noqa: F401
from hw4.data import (
    PairedRestorationDataset,
    find_pairs,
    list_test_images,
    stratified_split,
)
from hw4.engine import autocast_context, build_model, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "data" / "hw4_realse_dataset"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "smoke"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument(
        "--model-profile",
        choices=["baseline", "wide", "wide_deep", "tiny"],
        default="baseline",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    set_seed(42)
    device = torch.device("cuda")
    train_records, validation_records = stratified_split(find_pairs(args.data_root))
    assert len(train_records) == 2880 and len(validation_records) == 320
    assert len(list_test_images(args.data_root)) == 100
    dataset = PairedRestorationDataset(
        train_records[: args.batch_size], True, args.patch_size
    )
    degraded = torch.stack([dataset[index][0] for index in range(args.batch_size)]).to(
        device
    )
    clean = torch.stack([dataset[index][1] for index in range(args.batch_size)]).to(
        device
    )
    model = build_model(args.model_profile).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    progress = tqdm(range(1), desc="Smoke train batch", dynamic_ncols=True)
    for _ in progress:
        with autocast_context(device, True):
            restored = model(degraded)
            loss = nn.functional.l1_loss(restored, clean)
        loss.backward()
        optimizer.step()
        progress.set_postfix(loss=f"{loss.item():.5f}")
    assert restored.shape == degraded.shape
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "smoke.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    reloaded = build_model(args.model_profile).to(device)
    reloaded.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=False)["model"]
    )
    with torch.inference_mode(), autocast_context(device, True):
        one_prediction = (
            reloaded(degraded[:1]).float().clamp(0, 1).mul(255).byte().cpu().numpy()[0]
        )
    npz_path = args.output_dir / "pred.npz"
    np.savez(npz_path, **{"0.png": one_prediction})
    zip_path = args.output_dir / "smoke_submission.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(npz_path, arcname="pred.npz")
    assert one_prediction.shape == (3, args.patch_size, args.patch_size)
    print(
        f"Smoke test passed on {torch.cuda.get_device_name(0)}; output={restored.shape}"
    )
    print(f"Checkpoint reload passed: {checkpoint}")
    print(f"Submission container check passed: {zip_path}")


if __name__ == "__main__":
    main()
