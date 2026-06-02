"""Evaluate a checkpoint on the deterministic HW4 validation split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _bootstrap import PROJECT_ROOT  # noqa: F401
from hw4.data import PairedRestorationDataset, find_pairs, stratified_split
from hw4.engine import build_model, create_loader, evaluate


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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--degradation",
        choices=["all", "rain", "snow"],
        default="all",
        help="Evaluate all validation data or only one degradation type.",
    )
    parser.add_argument(
        "--model-profile",
        choices=MODEL_PROFILES,
        default="baseline",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_records = stratified_split(find_pairs(args.data_root), seed=args.seed)
    if args.degradation != "all":
        val_records = [
            record for record in val_records if record.degradation == args.degradation
        ]
        print(f"Evaluating {args.degradation} validation subset.")
    dataset = PairedRestorationDataset(val_records, train=False)
    loader = create_loader(dataset, args.batch_size, False, args.num_workers)
    model = build_model(args.model_profile).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    psnr = evaluate(
        model,
        loader,
        device,
        amp=not args.no_amp,
        description="Validation TTA" if args.tta else "Validation",
        tta=args.tta,
    )
    print(f"Validation PSNR: {psnr:.4f} dB")


if __name__ == "__main__":
    main()
