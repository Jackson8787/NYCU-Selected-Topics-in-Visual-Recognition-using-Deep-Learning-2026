from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401

from hw3.coco import collect_predictions, dataset_to_coco_gt, evaluate_ap50
from hw3.config import DEFAULT_DATA_ROOT
from hw3.data import Hw3CellDataset, collate_fn, list_train_samples, split_train_val
from hw3.model import build_model, load_checkpoint


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_min_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint and sweep thresholds."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
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
    parser.add_argument("--min-sizes", type=parse_min_sizes, default=(640,))
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--small-cell-anchors", action="store_true")
    parser.add_argument("--detections-per-img", type=int, default=300)
    parser.add_argument(
        "--score-thresholds", type=parse_float_list, default=[0.05, 0.1, 0.2, 0.3, 0.5]
    )
    parser.add_argument("--mask-thresholds", type=parse_float_list, default=[0.5])
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = list_train_samples(args.data_root)
    all_ids = [sample.sample_id for sample in samples]
    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    _, val_ids = split_train_val(all_ids, split_payload["folds"], args.fold)

    dataset = Hw3CellDataset(args.data_root, val_ids, train=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    gt = dataset_to_coco_gt(dataset)

    model = build_model(
        architecture=args.architecture,
        pretrained=args.pretrained,
        min_size=args.min_sizes,
        max_size=args.max_size,
        small_cell_anchors=args.small_cell_anchors,
        detections_per_img=args.detections_per_img,
    ).to(device)
    checkpoint = load_checkpoint(model, str(args.checkpoint), device)
    print(f"loaded checkpoint epoch={checkpoint.get('epoch', 'unknown')}")

    rows = []
    for score_threshold in args.score_thresholds:
        for mask_threshold in args.mask_thresholds:
            predictions = collect_predictions(
                model,
                loader,
                device,
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
            )
            metrics = evaluate_ap50(gt, predictions)
            row = {
                "score_threshold": score_threshold,
                "mask_threshold": mask_threshold,
                "predictions": len(predictions),
                **metrics,
            }
            rows.append(row)
            print(json.dumps(row, indent=2))

    best = max(rows, key=lambda row: row["ap50"])
    print("best:")
    print(json.dumps(best, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"rows": rows, "best": best}, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
