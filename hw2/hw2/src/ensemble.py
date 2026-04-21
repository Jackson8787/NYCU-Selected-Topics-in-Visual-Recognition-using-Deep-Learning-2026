import argparse
import contextlib
import io
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils import save_json, zip_prediction


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse detection prediction JSON files."
    )
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument(
        "--method",
        choices=["concat", "nms", "wbf"],
        default="wbf",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.55)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--max-detections-per-image", type=int, default=100)
    parser.add_argument("--output", required=True)
    parser.add_argument("--zip-output", default=None)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--metrics-output", default=None)
    return parser.parse_args()


def load_predictions(path, weight):
    with Path(path).open("r", encoding="utf-8") as f:
        predictions = json.load(f)

    grouped = defaultdict(list)
    for prediction in predictions:
        score = float(prediction["score"])
        if score <= 0:
            continue
        item = {
            "image_id": int(prediction["image_id"]),
            "category_id": int(prediction["category_id"]),
            "bbox": [float(value) for value in prediction["bbox"]],
            "score": score,
            "weight": float(weight),
        }
        grouped[(item["image_id"], item["category_id"])].append(item)
    return grouped


def xywh_to_xyxy(box):
    x, y, width, height = box
    return [x, y, x + width, y + height]


def xyxy_to_xywh(box):
    x_min, y_min, x_max, y_max = box
    return [x_min, y_min, max(0.0, x_max - x_min), max(0.0, y_max - y_min)]


def iou(box_a, box_b):
    a = xywh_to_xyxy(box_a)
    b = xywh_to_xyxy(box_b)
    x_min = max(a[0], b[0])
    y_min = max(a[1], b[1])
    x_max = min(a[2], b[2])
    y_max = min(a[3], b[3])
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def fuse_cluster(cluster):
    total = sum(item["score"] * item["weight"] for item in cluster)
    if total <= 0:
        best = max(cluster, key=lambda item: item["score"])
        return best["bbox"], best["score"]

    boxes = [xywh_to_xyxy(item["bbox"]) for item in cluster]
    fused = [
        sum(
            box[index] * item["score"] * item["weight"]
            for box, item in zip(boxes, cluster)
        )
        / total
        for index in range(4)
    ]
    score = sum(item["score"] for item in cluster) / len(cluster)
    score *= min(len(cluster), 2) / 2
    return xyxy_to_xywh(fused), score


def fuse_group(items, method, iou_threshold):
    items = sorted(
        items,
        key=lambda item: item["score"] * item["weight"],
        reverse=True,
    )
    if method == "concat":
        return items

    if method == "nms":
        kept = []
        for item in items:
            if all(
                iou(item["bbox"], kept_item["bbox"]) < iou_threshold
                for kept_item in kept
            ):
                kept.append(item)
        return kept

    clusters = []
    for item in items:
        best_index = None
        best_iou = iou_threshold
        for index, cluster in enumerate(clusters):
            cluster_box, _ = fuse_cluster(cluster)
            overlap = iou(item["bbox"], cluster_box)
            if overlap >= best_iou:
                best_index = index
                best_iou = overlap
        if best_index is None:
            clusters.append([item])
        else:
            clusters[best_index].append(item)

    fused = []
    for cluster in clusters:
        box, score = fuse_cluster(cluster)
        anchor = cluster[0]
        fused.append(
            {
                "image_id": anchor["image_id"],
                "category_id": anchor["category_id"],
                "bbox": box,
                "score": score,
                "weight": 1.0,
            }
        )
    return sorted(fused, key=lambda item: item["score"], reverse=True)


def fuse_predictions(
    prediction_paths,
    weights,
    method,
    iou_threshold,
    score_threshold,
    max_detections,
):
    grouped = defaultdict(list)
    for path, weight in zip(prediction_paths, weights):
        for key, items in load_predictions(path, weight).items():
            grouped[key].extend(items)

    by_image = defaultdict(list)
    for key, items in grouped.items():
        for item in fuse_group(items, method, iou_threshold):
            if item["score"] >= score_threshold:
                by_image[item["image_id"]].append(item)

    output = []
    for image_id, items in by_image.items():
        items = sorted(
            items,
            key=lambda item: item["score"],
            reverse=True,
        )[:max_detections]
        for item in items:
            output.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(item["category_id"]),
                    "bbox": [float(value) for value in item["bbox"]],
                    "score": float(item["score"]),
                }
            )
    return output


def evaluate(predictions, annotations):
    coco_gt = COCO(str(annotations))
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

    names = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_1",
        "AR_10",
        "AR_100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    return {
        name: float(value)
        for name, value in zip(names, evaluator.stats.tolist())
    }


def main():
    args = parse_args()
    weights = args.weights or [1.0] * len(args.predictions)
    if len(weights) != len(args.predictions):
        raise ValueError("--weights must match --predictions length")

    predictions = fuse_predictions(
        args.predictions,
        weights,
        args.method,
        args.iou_threshold,
        args.score_threshold,
        args.max_detections_per_image,
    )
    save_json(predictions, args.output)
    if args.zip_output:
        zip_prediction(args.output, args.zip_output)

    if args.annotations:
        metrics = evaluate(predictions, Path(args.annotations))
        print(json.dumps(metrics, indent=2))
        if args.metrics_output:
            save_json(metrics, args.metrics_output)


if __name__ == "__main__":
    main()
