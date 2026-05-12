from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_utils
from tqdm import tqdm

import _bootstrap  # noqa: F401

from ensemble_submissions import bbox_from_mask, encode_mask, merge_group
from hw3.config import CELL_CLASS_IDS, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR
from hw3.data import list_test_images, load_test_id_map, read_image_rgb
from hw3.model import build_model, load_checkpoint


def parse_min_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tiled HW3 test inference.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "submission" / "test-results.json",
    )
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
    parser.add_argument("--detections-per-img", type=int, default=400)
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--model-score-threshold", type=float, default=0.05)
    parser.add_argument("--output-score-threshold", type=float, default=0.08)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--nms-threshold", type=float, default=0.50)
    parser.add_argument("--edge-margin", type=int, default=12)
    parser.add_argument("--min-mask-area", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def touches_internal_edge(
    bbox: list[float],
    tile_h: int,
    tile_w: int,
    y0: int,
    x0: int,
    image_h: int,
    image_w: int,
    margin: int,
) -> bool:
    if margin <= 0:
        return False
    x, y, w, h = bbox
    x2 = x + w
    y2 = y + h
    if x <= margin and x0 > 0:
        return True
    if y <= margin and y0 > 0:
        return True
    if x2 >= tile_w - margin and x0 + tile_w < image_w:
        return True
    if y2 >= tile_h - margin and y0 + tile_h < image_h:
        return True
    return False


@torch.no_grad()
def predict_tile(
    model: torch.nn.Module,
    tile: np.ndarray,
    image_id: int,
    y0: int,
    x0: int,
    image_h: int,
    image_w: int,
    device: torch.device,
    score_threshold: float,
    mask_threshold: float,
    edge_margin: int,
    min_mask_area: int,
) -> list[dict]:
    tile_h, tile_w = tile.shape[:2]
    tensor = torch.as_tensor(tile.transpose(2, 0, 1), dtype=torch.float32) / 255.0
    output = model([tensor.to(device)])[0]

    scores = output["scores"].detach()
    keep = scores >= score_threshold
    labels = output["labels"].detach()[keep].cpu().numpy()
    masks = output["masks"].detach()[keep, 0].cpu().numpy()
    scores = scores[keep].cpu().numpy()

    results = []
    for label, soft_mask, score in zip(labels, masks, scores):
        if int(label) not in CELL_CLASS_IDS:
            continue
        tile_mask = soft_mask >= mask_threshold
        if int(tile_mask.sum()) < min_mask_area:
            continue
        tile_bbox = bbox_from_mask(tile_mask)
        if tile_bbox is None:
            continue
        if touches_internal_edge(
            tile_bbox, tile_h, tile_w, y0, x0, image_h, image_w, edge_margin
        ):
            continue

        full_mask = np.zeros((image_h, image_w), dtype=np.uint8)
        full_mask[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile_mask.astype(np.uint8)
        bbox = bbox_from_mask(full_mask)
        if bbox is None:
            continue
        results.append(
            {
                "image_id": image_id,
                "category_id": int(label),
                "bbox": bbox,
                "segmentation": encode_mask(full_mask),
                "score": float(score),
            }
        )
    return results


def validate(
    results: list[dict], image_id_map: dict[str, dict[str, int | str]]
) -> None:
    known = {int(row["id"]): row for row in image_id_map.values()}
    for index, row in enumerate(results):
        image_id = int(row["image_id"])
        if image_id not in known:
            raise ValueError(f"Unknown image_id at {index}")
        if int(row["category_id"]) not in CELL_CLASS_IDS:
            raise ValueError(f"Bad category_id at {index}")
        decoded = mask_utils.decode(row["segmentation"])
        expected = known[image_id]
        if decoded.shape != (int(expected["height"]), int(expected["width"])):
            raise ValueError(f"Invalid RLE at {index}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = list_test_images(args.data_root)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    image_id_map = load_test_id_map(args.data_root)

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
    model.eval()

    all_results = []
    for image_path in tqdm(image_paths, desc="tiled test"):
        image = read_image_rgb(image_path)
        image_h, image_w = image.shape[:2]
        image_id = int(image_id_map[image_path.name]["id"])
        candidates = []
        for y0 in tile_starts(image_h, args.tile_size, args.stride):
            for x0 in tile_starts(image_w, args.tile_size, args.stride):
                tile = image[
                    y0 : min(y0 + args.tile_size, image_h),
                    x0 : min(x0 + args.tile_size, image_w),
                    :,
                ]
                candidates.extend(
                    predict_tile(
                        model,
                        tile,
                        image_id,
                        y0,
                        x0,
                        image_h,
                        image_w,
                        device,
                        score_threshold=args.model_score_threshold,
                        mask_threshold=args.mask_threshold,
                        edge_margin=args.edge_margin,
                        min_mask_area=args.min_mask_area,
                    )
                )

        by_class: dict[int, list[dict]] = {}
        for item in candidates:
            if float(item["score"]) < args.output_score_threshold:
                continue
            by_class.setdefault(int(item["category_id"]), []).append(item)
        for group in by_class.values():
            all_results.extend(
                merge_group(
                    group,
                    nms_threshold=args.nms_threshold,
                    agreement_boost=0.0,
                    limit=args.detections_per_img,
                )
            )

    all_results.sort(
        key=lambda row: (
            int(row["image_id"]),
            int(row["category_id"]),
            -float(row["score"]),
        )
    )
    validate(all_results, image_id_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(all_results, separators=(",", ":")), encoding="utf-8"
    )
    print(f"wrote {len(all_results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
