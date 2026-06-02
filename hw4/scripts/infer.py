"""Generate a CodaBench submission using a trained PromptIR checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _bootstrap import PROJECT_ROOT  # noqa: F401
from hw4.data import TestRestorationDataset, list_test_images
from hw4.engine import build_model, create_loader, create_submission


MODEL_PROFILES = [
    "baseline",
    "prompt7",
    "prompt9",
    "wide",
    "wide_deep",
    "tiny",
    "tiny_prompt7",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "data" / "hw4_realse_dataset"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "submission"
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--model-profile",
        choices=MODEL_PROFILES,
        default="baseline",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_profile)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    dataset = TestRestorationDataset(list_test_images(args.data_root))
    loader = create_loader(dataset, 1, False, args.num_workers)
    zip_path = create_submission(
        model, loader, device, args.output_dir, not args.no_amp, args.tta
    )
    print(f"Submission ZIP: {zip_path}")


if __name__ == "__main__":
    main()
