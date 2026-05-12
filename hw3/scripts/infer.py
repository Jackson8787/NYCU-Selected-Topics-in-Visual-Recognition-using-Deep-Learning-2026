from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from pycocotools import mask as mask_utils

import _bootstrap  # noqa: F401

from hw3.coco import predict_test_images
from hw3.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR, SUBMISSION_FILENAME
from hw3.data import list_test_images, load_test_id_map
from hw3.model import build_model, load_checkpoint


def parse_min_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HW3 test inference.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "submission" / SUBMISSION_FILENAME,
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
    parser.add_argument("--min-sizes", type=parse_min_sizes, default=(1024,))
    parser.add_argument("--max-size", type=int, default=1536)
    parser.add_argument("--small-cell-anchors", action="store_true")
    parser.add_argument("--detections-per-img", type=int, default=300)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument(
        "--limit", type=int, default=None, help="Debug: limit test images."
    )
    return parser.parse_args()


def validate_submission(
    results: list[dict], image_id_map: dict[str, dict[str, int | str]]
) -> None:
    known = {int(row["id"]): row for row in image_id_map.values()}
    for index, row in enumerate(results):
        image_id = int(row["image_id"])
        if image_id not in known:
            raise ValueError(f"Unknown image_id at result {index}")
        if int(row["category_id"]) not in {1, 2, 3, 4}:
            raise ValueError(f"Invalid category_id at result {index}")
        if not (0.0 <= float(row["score"]) <= 1.0):
            raise ValueError(f"Invalid score at result {index}")
        if "bbox" in row:
            bbox = row["bbox"]
            if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                raise ValueError(f"Invalid bbox at result {index}")
        decoded = mask_utils.decode(row["segmentation"])
        expected = known[image_id]
        if decoded.shape != (int(expected["height"]), int(expected["width"])):
            raise ValueError(f"Invalid RLE at result {index}")


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

    results = predict_test_images(
        model,
        image_paths,
        image_id_map,
        device,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
    )
    validate_submission(results, image_id_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results), encoding="utf-8")
    print(f"wrote {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
