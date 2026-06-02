"""Train a PromptIR restoration model from scratch for HW4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _bootstrap import PROJECT_ROOT  # noqa: F401
from hw4.data import (
    PairedRestorationDataset,
    TestRestorationDataset,
    find_pairs,
    list_test_images,
    stratified_split,
)
from hw4.engine import (
    build_model,
    create_loader,
    create_submission,
    set_seed,
    train_model,
)


MODEL_PROFILES = [
    "baseline",
    "prompt7",
    "prompt9",
    "wide",
    "wide_deep",
    "tiny",
    "tiny_prompt7",
]


def json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def load_initial_weights(
    model: torch.nn.Module, checkpoint_path: Path, device: torch.device, partial: bool
) -> None:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_state = state["model"]
    if not partial:
        model.load_state_dict(checkpoint_state)
        print(f"Initialized model weights from {checkpoint_path}")
        return
    model_state = model.state_dict()
    matched = {
        name: tensor
        for name, tensor in checkpoint_state.items()
        if name in model_state and model_state[name].shape == tensor.shape
    }
    skipped = sorted(set(checkpoint_state) - set(matched))
    model_state.update(matched)
    model.load_state_dict(model_state)
    print(
        f"Partially initialized from {checkpoint_path}: "
        f"loaded {len(matched)} tensors, skipped {len(skipped)} shape/name mismatches."
    )
    if skipped:
        print(
            "Skipped tensors: "
            + ", ".join(skipped[:12])
            + (" ..." if len(skipped) > 12 else "")
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "data" / "hw4_realse_dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "baseline_promptir",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=15)
    parser.add_argument(
        "--scheduler", choices=["cosine", "onecycle", "constant"], default="cosine"
    )
    parser.add_argument("--onecycle-pct-start", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-all", action="store_true")
    parser.add_argument(
        "--degradation",
        choices=["all", "rain", "snow"],
        default="all",
        help="Train on all data or only one degradation type for expert models.",
    )
    parser.add_argument(
        "--model-profile",
        choices=MODEL_PROFILES,
        default="baseline",
    )
    parser.add_argument(
        "--loss",
        choices=[
            "l1",
            "mse",
            "l1_mse",
            "l1_psnr_ssim",
            "mse_perceptual",
            "l1_psnr_ssim_perceptual",
            "charbonnier",
            "charbonnier_edge",
        ],
        default="l1",
    )
    parser.add_argument("--edge-weight", type=float, default=0.05)
    parser.add_argument("--psnr-weight", type=float, default=0.1)
    parser.add_argument("--ssim-weight", type=float, default=0.1)
    parser.add_argument("--perceptual-weight", type=float, default=0.1)
    parser.add_argument(
        "--perceptual-pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet-pretrained VGG19 features for perceptual loss variants.",
    )
    parser.add_argument(
        "--perceptual-backbone",
        choices=["vgg16", "vgg19"],
        default="vgg19",
        help="Feature extractor used by perceptual loss variants.",
    )
    parser.add_argument(
        "--perceptual-layer-preset",
        choices=["reference", "extended"],
        default="extended",
        help="Layer preset for perceptual loss variants.",
    )
    parser.add_argument("--rain-weight", type=float, default=1.0)
    parser.add_argument("--snow-weight", type=float, default=1.0)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument(
        "--partial-init",
        action="store_true",
        help="Load only checkpoint tensors whose names and shapes match the selected model profile.",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--onecycle-div-factor", type=float, default=25.0)
    parser.add_argument("--onecycle-final-div-factor", type=float, default=1000.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the planned PromptIR baseline training."
        )
    set_seed(args.seed)
    device = torch.device("cuda")
    amp = not args.no_amp
    records = find_pairs(args.data_root)
    train_records, val_records = stratified_split(records, seed=args.seed)
    if args.degradation != "all":
        records = [
            record for record in records if record.degradation == args.degradation
        ]
        train_records = [
            record for record in train_records if record.degradation == args.degradation
        ]
        val_records = [
            record for record in val_records if record.degradation == args.degradation
        ]
        print(f"Training a {args.degradation} expert model.")
    if args.train_all:
        train_records = records
        print(
            "Training on all released training pairs; validation PSNR is leaky "
            "and should not be compared with held-out runs."
        )
    train_dataset = PairedRestorationDataset(
        train_records, train=True, patch_size=args.patch_size
    )
    val_dataset = PairedRestorationDataset(
        val_records, train=False, patch_size=args.patch_size
    )
    test_dataset = TestRestorationDataset(list_test_images(args.data_root))
    train_loader = create_loader(train_dataset, args.batch_size, True, args.num_workers)
    val_loader = create_loader(
        val_dataset, args.val_batch_size, False, args.num_workers
    )
    test_loader = create_loader(test_dataset, 1, False, args.num_workers)
    model = build_model(args.model_profile)
    if args.init_checkpoint is not None:
        load_initial_weights(model, args.init_checkpoint, device, args.partial_init)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config = vars(args).copy()
    config["data_root"] = str(args.data_root)
    config["output_dir"] = str(args.output_dir)
    config["parameters"] = parameter_count
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(json_ready(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Model profile: {args.model_profile}; parameters: {parameter_count:,}")
    print(
        f"Train/validation/test: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}; "
        f"batch={args.batch_size}; accumulation={args.accumulation_steps}; "
        f"loss={args.loss}; BF16 AMP={amp}"
    )
    best_path, best_psnr = train_model(
        model,
        train_loader,
        val_loader,
        device,
        args.output_dir,
        args.epochs,
        args.learning_rate,
        min(args.warmup_epochs, args.epochs),
        args.accumulation_steps,
        amp,
        config,
        args.loss,
        args.edge_weight,
        args.psnr_weight,
        args.ssim_weight,
        args.perceptual_weight,
        args.perceptual_pretrained,
        args.perceptual_backbone,
        args.perceptual_layer_preset,
        args.weight_decay,
        args.scheduler,
        args.onecycle_pct_start,
        args.onecycle_div_factor,
        args.onecycle_final_div_factor,
        args.resume,
        args.rain_weight,
        args.snow_weight,
    )
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    zip_path = create_submission(
        model, test_loader, device, args.output_dir / "submission", amp
    )
    (args.output_dir / "result.json").write_text(
        json.dumps(
            {
                "best_validation_psnr": best_psnr,
                "best_checkpoint": str(best_path),
                "submission_zip": str(zip_path),
                "model_profile": args.model_profile,
                "loss": args.loss,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "rain_weight": args.rain_weight,
                "snow_weight": args.snow_weight,
                "perceptual_weight": args.perceptual_weight,
                "perceptual_pretrained": args.perceptual_pretrained,
                "perceptual_backbone": args.perceptual_backbone,
                "perceptual_layer_preset": args.perceptual_layer_preset,
                "epochs": args.epochs,
                "train_all": args.train_all,
                "degradation": args.degradation,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Best validation PSNR: {best_psnr:.3f} dB")
    print(f"Best checkpoint: {best_path}")
    print(f"Submission ZIP: {zip_path}")


if __name__ == "__main__":
    main()
