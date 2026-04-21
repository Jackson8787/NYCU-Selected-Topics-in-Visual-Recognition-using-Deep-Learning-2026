import argparse
import contextlib
import io
import json
import math
import tempfile
import time
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (
    DetrCollator,
    DigitDetectionDataset,
    DigitTestDataset,
    test_collate_fn,
)
from utils import (
    build_model,
    build_processor,
    configure_torch,
    load_config,
    make_optimizer,
    make_scheduler,
    move_labels_to_device,
    resolve_path,
    save_checkpoint,
    save_json,
    set_seed,
)


class ValidImageDataset(DigitTestDataset):
    def __init__(self, data_dir):
        self.image_dir = Path(data_dir) / "valid"
        with (Path(data_dir) / "valid.json").open("r", encoding="utf-8") as f:
            coco = json.load(f)
        self.images = [
            self.image_dir / image["file_name"]
            for image in sorted(coco["images"], key=lambda item: item["id"])
        ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DETR baseline for NYCU HW2."
    )
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny sanity check.",
    )
    return parser.parse_args()


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    config,
    epoch,
):
    model.train()
    running_loss = 0.0
    accumulation_steps = config.get("gradient_accumulation_steps", 1)
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(progress, start=1):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        pixel_mask = batch["pixel_mask"].to(device, non_blocking=True)
        labels = move_labels_to_device(batch["labels"], device)

        with torch.amp.autocast(
            "cuda",
            enabled=config["amp"] and device.type == "cuda",
        ):
            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=labels,
            )
            loss = outputs.loss / accumulation_steps

        scaler.scale(loss).backward()
        if step % accumulation_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["max_grad_norm"],
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * accumulation_steps
        progress.set_postfix(loss=f"{running_loss / step:.4f}")

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def validate_loss(model, loader, device, config):
    model.eval()
    running_loss = 0.0
    progress = tqdm(loader, desc="valid loss", leave=False)

    for step, batch in enumerate(progress, start=1):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        pixel_mask = batch["pixel_mask"].to(device, non_blocking=True)
        labels = move_labels_to_device(batch["labels"], device)

        with torch.amp.autocast(
            "cuda",
            enabled=config["amp"] and device.type == "cuda",
        ):
            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=labels,
            )
            loss = outputs.loss

        running_loss += loss.item()
        progress.set_postfix(loss=f"{running_loss / step:.4f}")

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def validate_map(model, processor, loader, device, data_dir, config):
    model.eval()
    predictions = []
    threshold = config.get("map_score_threshold", 0.001)
    max_detections = config.get("map_max_detections_per_image", 50)

    for batch in tqdm(loader, desc="valid mAP", leave=False):
        inputs = processor(images=batch["images"], return_tensors="pt")
        inputs = {
            key: value.to(device, non_blocking=True)
            for key, value in inputs.items()
        }
        target_sizes = batch["target_sizes"].to(device, non_blocking=True)

        outputs = model(**inputs)
        results = processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
        )

        for image_id, result in zip(batch["image_ids"], results):
            boxes = result["boxes"].detach().cpu().tolist()
            scores = result["scores"].detach().cpu().tolist()
            labels = result["labels"].detach().cpu().tolist()
            for box, score, label in list(
                zip(boxes, scores, labels)
            )[:max_detections]:
                x_min, y_min, x_max, y_max = box
                predictions.append(
                    {
                        "image_id": int(image_id),
                        "bbox": [
                            float(x_min),
                            float(y_min),
                            float(x_max - x_min),
                            float(y_max - y_min),
                        ],
                        "score": float(score),
                        "category_id": int(label) + 1,
                    }
                )

    coco_gt = COCO(str(Path(data_dir) / "valid.json"))
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(predictions, f)
        pred_path = f.name

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = (
            coco_gt.loadRes(pred_path)
            if predictions
            else coco_gt.loadRes([])
        )
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    return {
        "AP": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
    }


def main():
    args = parse_args()
    configure_torch()
    project_dir = Path(__file__).resolve().parents[1]
    config = load_config(resolve_path(args.config, project_dir))

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.smoke:
        config["epochs"] = 1
        config["eval_every"] = 1

    set_seed(config["seed"])
    data_dir = resolve_path(config["data_dir"], project_dir)
    output_dir = resolve_path(config["output_dir"], project_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = build_processor(config)
    train_dataset = DigitDetectionDataset(
        data_dir,
        "train",
        processor,
        augment=config.get("augmentation", {}).get("color_jitter", False),
    )
    valid_dataset = DigitDetectionDataset(data_dir, "valid", processor)
    valid_image_dataset = ValidImageDataset(data_dir)

    if args.smoke:
        train_dataset = torch.utils.data.Subset(train_dataset, range(8))
        valid_dataset = torch.utils.data.Subset(valid_dataset, range(4))
        valid_image_dataset = torch.utils.data.Subset(
            valid_image_dataset,
            range(4),
        )

    collator = DetrCollator(processor)
    loader_kwargs = {
        "num_workers": config["num_workers"],
        "collate_fn": collator,
        "pin_memory": device.type == "cuda",
    }
    if config["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = config.get(
            "persistent_workers",
            True,
        )
        loader_kwargs["prefetch_factor"] = config.get("prefetch_factor", 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        **loader_kwargs,
    )
    valid_map_loader = DataLoader(
        valid_image_dataset,
        batch_size=config.get("map_batch_size", config["batch_size"]),
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=test_collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=(
            config.get("persistent_workers", True)
            if config["num_workers"] > 0
            else False
        ),
        prefetch_factor=(
            config.get("prefetch_factor", 2)
            if config["num_workers"] > 0
            else None
        ),
    )

    model = build_model(config).to(device)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(
        optimizer,
        config,
        math.ceil(
            len(train_loader)
            / config.get("gradient_accumulation_steps", 1)
        ),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config["amp"] and device.type == "cuda",
    )

    history = []
    best_valid_loss = math.inf
    best_valid_ap = -math.inf
    best_metric = config.get("best_metric", "loss")
    start = time.time()

    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            config,
            epoch,
        )
        valid_loss = None
        valid_metrics = None
        if epoch % config["eval_every"] == 0:
            valid_loss = validate_loss(model, valid_loader, device, config)
            if best_metric == "map" or config.get("track_map", False):
                valid_metrics = validate_map(
                    model,
                    processor,
                    valid_map_loader,
                    device,
                    data_dir,
                    config,
                )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_AP": valid_metrics["AP"] if valid_metrics else None,
            "valid_AP50": valid_metrics["AP50"] if valid_metrics else None,
            "valid_AP75": valid_metrics["AP75"] if valid_metrics else None,
            "lr": optimizer.param_groups[0]["lr"],
            "backbone_lr": optimizer.param_groups[1]["lr"],
            "elapsed_sec": round(time.time() - start, 2),
        }
        history.append(row)
        save_json(history, output_dir / "history.json")
        print(row)

        save_checkpoint(model, processor, output_dir, config, "last")
        if (
            best_metric == "map"
            and valid_metrics is not None
            and valid_metrics["AP"] > best_valid_ap
        ):
            best_valid_ap = valid_metrics["AP"]
            save_checkpoint(model, processor, output_dir, config, "best")
        elif (
            best_metric != "map"
            and valid_loss is not None
            and valid_loss < best_valid_loss
        ):
            best_valid_loss = valid_loss
            save_checkpoint(model, processor, output_dir, config, "best")

    if best_metric == "map":
        print(f"Finished. Best validation AP: {best_valid_ap:.4f}")
    else:
        print(f"Finished. Best validation loss: {best_valid_loss:.4f}")


if __name__ == "__main__":
    main()
