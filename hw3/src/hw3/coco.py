"""COCO conversion, AP50 evaluation, and submission helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from hw3.config import CELL_CLASS_IDS, CLASS_NAMES
from hw3.data import Hw3CellDataset, read_image_rgb


def encode_binary_mask(binary_mask: np.ndarray) -> dict:
    """Encode a 2D binary mask as compressed COCO RLE."""
    mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(mask)
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def dataset_to_coco_gt(dataset: Hw3CellDataset) -> dict:
    """Convert a dataset subset to COCO ground truth JSON."""
    images = []
    annotations = []
    ann_id = 1
    for image_id, sample in enumerate(dataset.samples):
        image = read_image_rgb(sample.image_path)
        height, width = image.shape[:2]
        _, target = dataset[image_id]
        images.append(
            {
                "id": image_id,
                "file_name": f"{sample.sample_id}.tif",
                "height": height,
                "width": width,
            }
        )
        boxes = target["boxes"].numpy()
        labels = target["labels"].numpy()
        masks = target["masks"].numpy()
        areas = target["area"].numpy()
        for box, label, binary_mask, area in zip(boxes, labels, masks, areas):
            x1, y1, x2, y2 = box.tolist()
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": float(area),
                    "segmentation": encode_binary_mask(binary_mask),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    return {
        "info": {"description": "HW3 validation ground truth"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": class_id, "name": CLASS_NAMES[class_id], "supercategory": "cell"}
            for class_id in CELL_CLASS_IDS
        ],
    }


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    score_threshold: float = 0.05,
    mask_threshold: float = 0.5,
) -> list[dict]:
    """Run model and collect COCO result dictionaries."""
    model.eval()
    results: list[dict] = []
    for images, targets in tqdm(data_loader, desc="eval", leave=False):
        images = [image.to(device) for image in images]
        outputs = model(images)
        for output, target in zip(outputs, targets):
            image_id = int(target["image_id"].item())
            scores = output["scores"].detach()
            keep = scores >= score_threshold
            boxes = output["boxes"].detach()[keep].cpu().numpy()
            labels = output["labels"].detach()[keep].cpu().numpy()
            masks = output["masks"].detach()[keep, 0].cpu().numpy()
            scores = scores[keep].cpu().numpy()
            for box, label, soft_mask, score in zip(boxes, labels, masks, scores):
                if int(label) not in CELL_CLASS_IDS:
                    continue
                binary_mask = soft_mask >= mask_threshold
                if not binary_mask.any():
                    continue
                x1, y1, x2, y2 = box.tolist()
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "segmentation": encode_binary_mask(binary_mask),
                        "score": float(score),
                    }
                )
    return results


def evaluate_ap50(coco_gt_dict: dict, predictions: list[dict]) -> dict[str, float]:
    """Evaluate segmentation AP50 with pycocotools."""
    if not predictions:
        return {"ap50": 0.0}
    with tempfile.TemporaryDirectory() as tmp_dir:
        gt_path = Path(tmp_dir) / "gt.json"
        pred_path = Path(tmp_dir) / "pred.json"
        gt_path.write_text(json.dumps(coco_gt_dict), encoding="utf-8")
        pred_path.write_text(json.dumps(predictions), encoding="utf-8")
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(str(pred_path))
        evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")
        evaluator.params.iouThrs = np.array([0.5])
        evaluator.evaluate()
        evaluator.accumulate()
        precision = evaluator.eval["precision"]
        valid = precision[precision > -1]
        ap50 = float(np.mean(valid)) if valid.size else 0.0
    return {"ap50": ap50}


@torch.no_grad()
def predict_test_images(
    model: torch.nn.Module,
    image_paths: Iterable[Path],
    image_id_map: dict[str, dict[str, int | str]],
    device: torch.device,
    score_threshold: float = 0.05,
    mask_threshold: float = 0.5,
) -> list[dict]:
    """Run test inference and produce COCO result dictionaries."""
    model.eval()
    results: list[dict] = []
    for image_path in tqdm(list(image_paths), desc="test"):
        image = read_image_rgb(image_path)
        tensor = torch.as_tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        output = model([tensor.to(device)])[0]
        meta = image_id_map[image_path.name]
        image_id = int(meta["id"])

        scores = output["scores"].detach()
        keep = scores >= score_threshold
        boxes = output["boxes"].detach()[keep].cpu().numpy()
        labels = output["labels"].detach()[keep].cpu().numpy()
        masks = output["masks"].detach()[keep, 0].cpu().numpy()
        scores = scores[keep].cpu().numpy()
        for box, label, soft_mask, score in zip(boxes, labels, masks, scores):
            if int(label) not in CELL_CLASS_IDS:
                continue
            binary_mask = soft_mask >= mask_threshold
            if not binary_mask.any():
                continue
            x1, y1, x2, y2 = box.tolist()
            results.append(
                {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "segmentation": encode_binary_mask(binary_mask),
                    "score": float(score),
                }
            )
    return results
