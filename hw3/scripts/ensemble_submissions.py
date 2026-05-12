from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge HW3 submission JSON files with mask NMS."
    )
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.10)
    parser.add_argument("--nms-threshold", type=float, default=0.50)
    parser.add_argument("--agreement-boost", type=float, default=0.0)
    parser.add_argument("--max-per-image-class", type=int, default=500)
    return parser.parse_args()


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / max(union, 1))


def xywh_to_xyxy(bbox: list[float]) -> np.ndarray:
    x, y, w, h = bbox
    return np.asarray([x, y, x + w, y + h], dtype=np.float32)


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    return intersection / max(area_a + area_b - intersection, 1.0)


def encode_mask(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def bbox_from_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    return [x1, y1, x2 - x1, y2 - y1]


def merge_group(
    items: list[dict], nms_threshold: float, agreement_boost: float, limit: int
) -> list[dict]:
    candidates = []
    for item in items:
        mask = mask_utils.decode(item["segmentation"]).astype(bool)
        if not mask.any():
            continue
        candidates.append(
            {
                "item": item,
                "mask": mask,
                "box": xywh_to_xyxy(item["bbox"]),
                "score": float(item["score"]),
                "votes": 1,
            }
        )

    candidates.sort(key=lambda row: row["score"], reverse=True)
    kept: list[dict] = []
    for candidate in candidates:
        duplicate_index = None
        for index, selected in enumerate(kept):
            if bbox_iou(candidate["box"], selected["box"]) < nms_threshold:
                continue
            if mask_iou(candidate["mask"], selected["mask"]) >= nms_threshold:
                duplicate_index = index
                break
        if duplicate_index is None:
            kept.append(candidate)
            if len(kept) >= limit:
                break
            continue

        selected = kept[duplicate_index]
        selected["votes"] += 1
        selected["score"] = max(selected["score"], candidate["score"])
        if candidate["score"] > float(selected["item"]["score"]):
            selected["item"] = candidate["item"]
            selected["mask"] = candidate["mask"]
            selected["box"] = candidate["box"]

    results = []
    for selected in kept:
        item = dict(selected["item"])
        score = selected["score"]
        if selected["votes"] > 1 and agreement_boost > 0:
            score = min(1.0, score * (1.0 + agreement_boost * (selected["votes"] - 1)))
        mask = selected["mask"]
        bbox = bbox_from_mask(mask)
        if bbox is None:
            continue
        item["bbox"] = bbox
        item["segmentation"] = encode_mask(mask)
        item["score"] = float(score)
        results.append(item)
    return results


def main() -> None:
    args = parse_args()
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for path in args.inputs:
        items = json.loads(path.read_text(encoding="utf-8"))
        for item in items:
            if float(item["score"]) < args.score_threshold:
                continue
            key = (int(item["image_id"]), int(item["category_id"]))
            groups[key].append(item)

    merged = []
    for items in tqdm(groups.values(), desc="merge"):
        merged.extend(
            merge_group(
                items,
                nms_threshold=args.nms_threshold,
                agreement_boost=args.agreement_boost,
                limit=args.max_per_image_class,
            )
        )
    merged.sort(
        key=lambda row: (
            int(row["image_id"]),
            int(row["category_id"]),
            -float(row["score"]),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {len(merged)} predictions to {args.output}")


if __name__ == "__main__":
    main()
