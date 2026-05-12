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


def parse_tta(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    valid = {"orig", "hflip", "vflip", "hvflip"}
    for item in items:
        if item not in valid:
            raise argparse.ArgumentTypeError(f"Unknown TTA transform: {item}")
    if "orig" not in items:
        items.insert(0, "orig")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HW3 test inference with flip TTA."
    )
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
    parser.add_argument("--min-sizes", type=parse_min_sizes, default=(512, 640))
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--small-cell-anchors", action="store_true")
    parser.add_argument("--detections-per-img", type=int, default=400)
    parser.add_argument("--model-score-threshold", type=float, default=0.05)
    parser.add_argument("--output-score-threshold", type=float, default=0.10)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--nms-threshold", type=float, default=0.50)
    parser.add_argument("--agreement-boost", type=float, default=0.03)
    parser.add_argument("--tta", type=parse_tta, default=["orig", "hflip"])
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def apply_transform(image: np.ndarray, transform: str) -> np.ndarray:
    if transform == "orig":
        return image
    if transform == "hflip":
        return np.ascontiguousarray(image[:, ::-1, :])
    if transform == "vflip":
        return np.ascontiguousarray(image[::-1, :, :])
    if transform == "hvflip":
        return np.ascontiguousarray(image[::-1, ::-1, :])
    raise ValueError(transform)


def invert_mask(mask: np.ndarray, transform: str) -> np.ndarray:
    if transform == "orig":
        return mask
    if transform == "hflip":
        return np.ascontiguousarray(mask[:, ::-1])
    if transform == "vflip":
        return np.ascontiguousarray(mask[::-1, :])
    if transform == "hvflip":
        return np.ascontiguousarray(mask[::-1, ::-1])
    raise ValueError(transform)


@torch.no_grad()
def predict_one(
    model: torch.nn.Module,
    image: np.ndarray,
    image_id: int,
    device: torch.device,
    transform: str,
    score_threshold: float,
    mask_threshold: float,
) -> list[dict]:
    aug_image = apply_transform(image, transform)
    tensor = torch.as_tensor(aug_image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
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
        binary_mask = invert_mask(soft_mask >= mask_threshold, transform)
        if not binary_mask.any():
            continue
        bbox = bbox_from_mask(binary_mask)
        if bbox is None:
            continue
        results.append(
            {
                "image_id": image_id,
                "category_id": int(label),
                "bbox": bbox,
                "segmentation": encode_mask(binary_mask),
                "score": float(score),
            }
        )
    return results


def validate(
    results: list[dict], image_id_map: dict[str, dict[str, int | str]]
) -> None:
    known = {int(row["id"]): row for row in image_id_map.values()}
    for index, row in enumerate(results):
        meta = known[int(row["image_id"])]
        decoded = mask_utils.decode(row["segmentation"])
        if decoded.shape != (int(meta["height"]), int(meta["width"])):
            raise ValueError(f"Invalid RLE shape at {index}")


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
    for image_path in tqdm(image_paths, desc="tta test"):
        image = read_image_rgb(image_path)
        image_id = int(image_id_map[image_path.name]["id"])
        candidates = []
        for transform in args.tta:
            candidates.extend(
                predict_one(
                    model,
                    image,
                    image_id,
                    device,
                    transform,
                    args.model_score_threshold,
                    args.mask_threshold,
                )
            )
        groups: dict[int, list[dict]] = {}
        for item in candidates:
            if float(item["score"]) < args.output_score_threshold:
                continue
            groups.setdefault(int(item["category_id"]), []).append(item)
        for group in groups.values():
            all_results.extend(
                merge_group(
                    group,
                    nms_threshold=args.nms_threshold,
                    agreement_boost=args.agreement_boost,
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
