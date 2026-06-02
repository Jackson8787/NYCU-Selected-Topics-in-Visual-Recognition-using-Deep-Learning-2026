"""Create a weighted two-checkpoint PromptIR ensemble submission."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from _bootstrap import PROJECT_ROOT  # noqa: F401
from hw4.data import TestRestorationDataset, list_test_images
from hw4.engine import build_model, create_loader, predict_batch


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
        type=float,
        required=True,
        help="Weight for checkpoint B; checkpoint A gets 1-weight_b.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.weight_b <= 1.0:
        raise ValueError("--weight-b must be in [0, 1]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profile_a = args.model_profile_a or args.model_profile
    profile_b = args.model_profile_b or args.model_profile
    model_a = load_model(args.checkpoint_a, device, profile_a)
    model_b = load_model(args.checkpoint_b, device, profile_b)
    dataset = TestRestorationDataset(list_test_images(args.data_root))
    loader = create_loader(dataset, 1, False, args.num_workers)
    amp = not args.no_amp
    predictions: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for degraded, filenames in tqdm(
            loader, desc="Weighted ensemble inference", dynamic_ncols=True
        ):
            degraded = degraded.to(device, non_blocking=True)
            pred_a = predict_batch(
                model_a, degraded, device, amp=amp, tta=args.tta
            ).float()
            pred_b = predict_batch(
                model_b, degraded, device, amp=amp, tta=args.tta
            ).float()
            restored = (1.0 - args.weight_b) * pred_a + args.weight_b * pred_b
            arrays = restored.clamp(0.0, 1.0).mul(255.0).round().byte().cpu().numpy()
            predictions.update(
                {filename: array for filename, array in zip(filenames, arrays)}
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / "pred.npz"
    np.savez(npz_path, **predictions)
    zip_path = args.output_dir / "hw4_promptir_weighted_ensemble_submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(npz_path, arcname="pred.npz")
    print(f"Submission ZIP: {zip_path}")


if __name__ == "__main__":
    main()
