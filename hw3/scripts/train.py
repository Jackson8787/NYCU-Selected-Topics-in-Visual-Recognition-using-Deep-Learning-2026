from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401

from hw3.coco import collect_predictions, dataset_to_coco_gt, evaluate_ap50
from hw3.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR
from hw3.data import Hw3CellDataset, collate_fn, list_train_samples, split_train_val
from hw3.engine import save_checkpoint, train_one_epoch
from hw3.model import build_model
from hw3.visualize import TensorboardLogger, append_metrics_csv, plot_metrics


def parse_min_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HW3 Mask R-CNN.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "run")
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--final", action="store_true", help="Train on all samples without validation."
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument(
        "--architecture",
        choices=["maskrcnn_r50_fpn", "maskrcnn_r50_fpn_v2"],
        default="maskrcnn_r50_fpn",
    )
    parser.add_argument(
        "--pretrained",
        choices=["imagenet", "coco", "none"],
        default="imagenet",
    )
    parser.add_argument(
        "--min-sizes", type=parse_min_sizes, default=(512, 640, 768, 896, 1024)
    )
    parser.add_argument("--max-size", type=int, default=1536)
    parser.add_argument("--small-cell-anchors", action="store_true")
    parser.add_argument("--detections-per-img", type=int, default=300)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--vflip-prob", type=float, default=0.0)
    parser.add_argument("--rotate90-prob", type=float, default=0.0)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument(
        "--no-pin-memory", action="store_true", help="Disable DataLoader pinned memory."
    )
    parser.add_argument(
        "--tensorboard", action="store_true", help="Write TensorBoard scalars."
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Disable PNG curve updates."
    )
    parser.add_argument(
        "--max-train-instances",
        type=int,
        default=None,
        help="Cap per-image training instances by keeping the largest masks.",
    )
    parser.add_argument(
        "--limit-train", type=int, default=None, help="Debug: limit train samples."
    )
    parser.add_argument(
        "--limit-val", type=int, default=None, help="Debug: limit validation samples."
    )
    parser.add_argument(
        "--resume", type=Path, default=None, help="Resume from a checkpoint."
    )
    return parser.parse_args()


def load_split_ids(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    samples = list_train_samples(args.data_root)
    all_ids = [sample.sample_id for sample in samples]
    if args.final:
        return all_ids, []
    if args.splits is None:
        raise ValueError("--splits is required unless --final is set")
    payload = json.loads(args.splits.read_text(encoding="utf-8"))
    folds = payload["folds"]
    if args.fold < 0 or args.fold >= len(folds):
        raise ValueError(f"--fold must be in [0, {len(folds) - 1}]")
    return split_train_val(all_ids, folds, args.fold)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids = load_split_ids(args)
    if args.limit_train is not None:
        train_ids = train_ids[: args.limit_train]
    if args.limit_val is not None:
        val_ids = val_ids[: args.limit_val]
    train_dataset = Hw3CellDataset(
        args.data_root,
        train_ids,
        train=True,
        flip_prob=args.hflip_prob,
        vflip_prob=args.vflip_prob,
        rotate90_prob=args.rotate90_prob,
        max_instances=args.max_train_instances,
    )
    val_dataset = (
        Hw3CellDataset(args.data_root, val_ids, train=False) if val_ids else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda" and not args.no_pin_memory,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=device.type == "cuda" and not args.no_pin_memory,
        )
        if val_dataset is not None
        else None
    )

    model = build_model(
        architecture=args.architecture,
        pretrained=args.pretrained,
        min_size=args.min_sizes,
        max_size=args.max_size,
        small_cell_anchors=args.small_cell_anchors,
        detections_per_img=args.detections_per_img,
    ).to(device)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(1, int(args.epochs * 0.6)), max(2, int(args.epochs * 0.85))],
        gamma=0.1,
    )
    scaler = (
        torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None
    )

    start_epoch = 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"resumed {args.resume} from epoch {start_epoch - 1}")

    best_ap50 = -1.0
    log_path = args.output_dir / "metrics.jsonl"
    csv_path = args.output_dir / "metrics.csv"
    plot_path = args.output_dir / "metrics.png"
    tensorboard = TensorboardLogger(args.output_dir / "tb", args.tensorboard)
    gt = dataset_to_coco_gt(val_dataset) if val_dataset is not None else None

    run_args = vars(args).copy()
    run_args["data_root"] = str(run_args["data_root"])
    run_args["output_dir"] = str(run_args["output_dir"])
    run_args["splits"] = str(run_args["splits"]) if run_args["splits"] else None
    run_args["min_sizes"] = list(run_args["min_sizes"])

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            losses = train_one_epoch(
                model, train_loader, optimizer, device, epoch, scaler=scaler
            )
            lr_scheduler.step()

            metrics = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **losses}
            if val_loader is not None and gt is not None:
                predictions = collect_predictions(
                    model,
                    val_loader,
                    device,
                    score_threshold=args.score_threshold,
                    mask_threshold=args.mask_threshold,
                )
                metrics.update(evaluate_ap50(gt, predictions))
                if metrics["ap50"] > best_ap50:
                    best_ap50 = metrics["ap50"]
                    save_checkpoint(
                        args.output_dir / "best.pth",
                        model,
                        optimizer,
                        epoch,
                        metrics,
                        run_args,
                    )
            else:
                save_checkpoint(
                    args.output_dir / "best.pth",
                    model,
                    optimizer,
                    epoch,
                    metrics,
                    run_args,
                )

            save_checkpoint(
                args.output_dir / "last.pth", model, optimizer, epoch, metrics, run_args
            )
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
            append_metrics_csv(csv_path, metrics)
            tensorboard.add_metrics(metrics, epoch)
            if not args.no_plot:
                plot_metrics(log_path, plot_path)
            print(json.dumps(metrics, indent=2))
    finally:
        tensorboard.close()

    print(f"finished training; checkpoints in {args.output_dir}")


if __name__ == "__main__":
    main()
