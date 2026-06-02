"""Evaluate weighted two-checkpoint PromptIR ensembles on validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

from _bootstrap import PROJECT_ROOT  # noqa: F401
from hw4.data import PairedRestorationDataset, find_pairs, stratified_split
from hw4.engine import build_model, create_loader, predict_batch, psnr_batch


def parse_weights(text: str) -> list[float]:
    weights = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not weights:
        raise ValueError("At least one weight is required.")
    if any(weight < 0.0 or weight > 1.0 for weight in weights):
        raise ValueError("Weights must be in [0, 1].")
    return weights


def load_model(
    checkpoint_path: Path, device: torch.device, profile: str
) -> torch.nn.Module:
    model = build_model(profile).to(device).eval()
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    print(
        f"Loaded {checkpoint_path} "
        f"(epoch={state.get('epoch')}, best={state.get('best_psnr')})"
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    model_profiles = [
        "baseline",
        "prompt7",
        "prompt9",
        "wide",
        "wide_deep",
        "tiny",
        "tiny_prompt7",
    ]
    parser.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "data" / "hw4_realse_dataset"
    )
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument(
        "--weight-b",
        default="0,0.25,0.5,0.75,1",
        help="Comma-separated weights for checkpoint B; checkpoint A gets 1-w.",
    )
    parser.add_argument(
        "--model-profile",
        choices=model_profiles,
        default="baseline",
        help="Shared model profile for both checkpoints when A/B-specific profiles are not set.",
    )
    parser.add_argument(
        "--model-profile-a",
        choices=model_profiles,
        help="Model profile for checkpoint A. Defaults to --model-profile.",
    )
    parser.add_argument(
        "--model-profile-b",
        choices=model_profiles,
        help="Model profile for checkpoint B. Defaults to --model-profile.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    weights_b = parse_weights(args.weight_b)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profile_a = args.model_profile_a or args.model_profile
    profile_b = args.model_profile_b or args.model_profile
    model_a = load_model(args.checkpoint_a, device, profile_a)
    model_b = load_model(args.checkpoint_b, device, profile_b)

    _, val_records = stratified_split(find_pairs(args.data_root), seed=args.seed)
    dataset = PairedRestorationDataset(val_records, train=False)
    loader = create_loader(dataset, args.batch_size, False, args.num_workers)
    amp = not args.no_amp
    totals = {weight_b: 0.0 for weight_b in weights_b}
    count = 0
    with torch.inference_mode():
        for degraded, clean, _ in tqdm(
            loader, desc="Weighted ensemble validation", dynamic_ncols=True
        ):
            degraded = degraded.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            pred_a = predict_batch(
                model_a, degraded, device, amp=amp, tta=args.tta
            ).float()
            pred_b = predict_batch(
                model_b, degraded, device, amp=amp, tta=args.tta
            ).float()
            for weight_b in weights_b:
                restored = (1.0 - weight_b) * pred_a + weight_b * pred_b
                totals[weight_b] += float(psnr_batch(restored, clean).sum().item())
            count += clean.shape[0]

    best_weight = max(weights_b, key=lambda weight: totals[weight] / count)
    for weight_b in weights_b:
        print(
            f"weight_b={weight_b:.4f} "
            f"weight_a={1.0 - weight_b:.4f} "
            f"psnr={totals[weight_b] / count:.4f}"
        )
    print(f"Best weight_b: {best_weight:.4f}")


if __name__ == "__main__":
    main()
