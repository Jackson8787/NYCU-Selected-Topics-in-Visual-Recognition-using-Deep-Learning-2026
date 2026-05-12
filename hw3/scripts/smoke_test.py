from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401

from hw3.coco import collect_predictions, dataset_to_coco_gt, evaluate_ap50
from hw3.data import Hw3CellDataset, collate_fn, list_train_samples
from hw3.engine import move_target_to_device
from hw3.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small HW3 pipeline smoke test.")
    parser.add_argument("--data-root", type=Path, default=Path("data/hw3-data-release"))
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--skip-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for module in [
        "torch",
        "torchvision",
        "tifffile",
        "imagecodecs",
        "pycocotools",
        "cv2",
        "albumentations",
    ]:
        __import__(module)
        print(f"import ok: {module}")

    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    samples = list_train_samples(args.data_root)
    ids = [sample.sample_id for sample in samples[: args.samples]]
    dataset = Hw3CellDataset(args.data_root, ids, train=False)
    image, target = dataset[0]
    print(f"sample image shape: {tuple(image.shape)}")
    print(f"sample instances: {target['labels'].numel()}")

    if args.skip_model:
        print("smoke test passed without model forward")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        architecture="maskrcnn_r50_fpn_v2",
        pretrained="coco",
        min_size=(256,),
        max_size=512,
        small_cell_anchors=True,
        detections_per_img=100,
    ).to(device)
    model.train()
    images = [image.to(device)]
    targets = [move_target_to_device(target, device)]
    losses = model(images, targets)
    print({key: float(value.detach().cpu()) for key, value in losses.items()})

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    gt = dataset_to_coco_gt(dataset)
    preds = collect_predictions(model, loader, device, score_threshold=0.99)
    print(evaluate_ap50(gt, preds))
    print("smoke test passed")


if __name__ == "__main__":
    main()
