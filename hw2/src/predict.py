import argparse
from pathlib import Path

import torch
from PIL import ImageOps
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import DigitTestDataset, test_collate_fn
from utils import load_model_and_processor, save_json, zip_prediction


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CodaBench pred.json."
    )
    parser.add_argument("--checkpoint", default="outputs/baseline/best")
    parser.add_argument("--data-dir", default="nycu-hw2-data")
    parser.add_argument("--output", default="outputs/baseline/pred.json")
    parser.add_argument(
        "--zip-output",
        default="outputs/baseline/submission.zip",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--max-detections-per-image", type=int, default=50)
    parser.add_argument("--horizontal-flip-tta", action="store_true")
    parser.add_argument("--horizontal-flip-only", action="store_true")
    parser.add_argument("--tta-iou-threshold", type=float, default=0.74)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only predict N images.",
    )
    return parser.parse_args()


def xyxy_iou(box_a, box_b):
    x_min = max(box_a[0], box_b[0])
    y_min = max(box_a[1], box_b[1])
    x_max = min(box_a[2], box_b[2])
    y_max = min(box_a[3], box_b[3])
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def classwise_nms(detections, iou_threshold, max_detections):
    kept = []
    for label in sorted({detection["label"] for detection in detections}):
        items = [
            detection
            for detection in detections
            if detection["label"] == label
        ]
        items = sorted(
            items,
            key=lambda detection: detection["score"],
            reverse=True,
        )
        label_kept = []
        for item in items:
            if all(
                xyxy_iou(item["box"], kept_item["box"]) < iou_threshold
                for kept_item in label_kept
            ):
                label_kept.append(item)
        kept.extend(label_kept)
    return sorted(
        kept,
        key=lambda detection: detection["score"],
        reverse=True,
    )[:max_detections]


def detections_from_result(result):
    boxes = result["boxes"].detach().cpu().tolist()
    scores = result["scores"].detach().cpu().tolist()
    labels = result["labels"].detach().cpu().tolist()
    return [
        {
            "box": [float(value) for value in box],
            "score": float(score),
            "label": int(label),
        }
        for box, score, label in zip(boxes, scores, labels)
    ]


def flip_detections(detections, width):
    flipped = []
    for detection in detections:
        x_min, y_min, x_max, y_max = detection["box"]
        flipped.append(
            {
                "box": [width - x_max, y_min, width - x_min, y_max],
                "score": detection["score"],
                "label": detection["label"],
            }
        )
    return flipped


@torch.no_grad()
def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    checkpoint_dir = project_dir / args.checkpoint
    data_dir = project_dir / args.data_dir
    output_path = project_dir / args.output
    zip_path = project_dir / args.zip_output

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model_and_processor(checkpoint_dir, device)
    model.eval()

    dataset = DigitTestDataset(data_dir)
    if args.limit is not None:
        dataset = torch.utils.data.Subset(
            dataset,
            range(min(args.limit, len(dataset))),
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_collate_fn,
        pin_memory=True,
    )

    predictions = []
    for batch in tqdm(loader, desc="predict"):
        images = (
            [ImageOps.mirror(image) for image in batch["images"]]
            if args.horizontal_flip_only
            else batch["images"]
        )
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        target_sizes = batch["target_sizes"].to(device)

        outputs = model(**inputs)
        results = processor.post_process_object_detection(
            outputs,
            threshold=args.score_threshold,
            target_sizes=target_sizes,
        )

        detections_by_index = [
            detections_from_result(result)
            for result in results
        ]
        if args.horizontal_flip_only:
            for index, detections in enumerate(detections_by_index):
                width = int(batch["target_sizes"][index][1].item())
                detections_by_index[index] = flip_detections(detections, width)

        if args.horizontal_flip_tta:
            flipped_images = [
                ImageOps.mirror(image) for image in batch["images"]
            ]
            flipped_inputs = processor(
                images=flipped_images,
                return_tensors="pt",
            )
            flipped_inputs = {
                key: value.to(device) for key, value in flipped_inputs.items()
            }
            flipped_outputs = model(**flipped_inputs)
            flipped_results = processor.post_process_object_detection(
                flipped_outputs,
                threshold=args.score_threshold,
                target_sizes=target_sizes,
            )
            for index, result in enumerate(flipped_results):
                width = int(batch["target_sizes"][index][1].item())
                detections_by_index[index].extend(
                    flip_detections(detections_from_result(result), width)
                )

        for image_id, detections in zip(
            batch["image_ids"],
            detections_by_index,
        ):
            if args.horizontal_flip_tta:
                kept = classwise_nms(
                    detections,
                    args.tta_iou_threshold,
                    args.max_detections_per_image,
                )
            else:
                kept = sorted(
                    detections,
                    key=lambda detection: detection["score"],
                    reverse=True,
                )[: args.max_detections_per_image]
            for detection in kept:
                x_min, y_min, x_max, y_max = detection["box"]
                predictions.append(
                    {
                        "image_id": int(image_id),
                        "bbox": [
                            round(float(x_min), 3),
                            round(float(y_min), 3),
                            round(float(x_max - x_min), 3),
                            round(float(y_max - y_min), 3),
                        ],
                        "score": round(float(detection["score"]), 6),
                        "category_id": int(detection["label"]) + 1,
                    }
                )

    save_json(predictions, output_path)
    zip_prediction(output_path, zip_path)
    print(f"Wrote {len(predictions)} predictions to {output_path}")
    print(f"Wrote submission zip to {zip_path}")


if __name__ == "__main__":
    main()
