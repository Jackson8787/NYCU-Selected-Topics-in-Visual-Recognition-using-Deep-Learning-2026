import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the full DETR baseline pipeline: train, evaluate, "
            "visualize, submit."
        )
    )
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--mode",
        choices=["all", "train", "eval", "predict"],
        default="all",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny end-to-end check.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--max-detections-per-image", type=int, default=50)
    parser.add_argument("--predict-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    return parser.parse_args()


def run_step(command, cwd):
    print("\n" + "=" * 80)
    print(" ".join(str(part) for part in command))
    print("=" * 80)
    sys.stdout.flush()
    subprocess.run(command, cwd=cwd, check=True)


def add_optional_train_args(command, args):
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.num_workers is not None:
        command.extend(["--num-workers", str(args.num_workers)])
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.smoke:
        command.append("--smoke")
        if args.batch_size is None:
            command.extend(["--batch-size", "2"])
        if args.num_workers is None:
            command.extend(["--num-workers", "0"])


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    python = sys.executable
    with (project_dir / args.config).open("r", encoding="utf-8") as f:
        config = json.load(f)
    output_dir = config.get("output_dir", "outputs/baseline")
    checkpoint = args.checkpoint or str(Path(output_dir) / "best")

    if args.mode in ["all", "train"]:
        command = [python, "src/train.py", "--config", args.config]
        add_optional_train_args(command, args)
        run_step(command, project_dir)

    if args.mode in ["all", "eval"]:
        command = [
            python,
            "src/evaluate.py",
            "--checkpoint",
            checkpoint,
            "--batch-size",
            str(args.eval_batch_size),
            "--save-predictions",
            str(Path(output_dir) / "valid_pred.json"),
            "--metrics-output",
            str(Path(output_dir) / "valid_metrics.json"),
            "--max-detections-per-image",
            str(args.max_detections_per_image),
        ]
        if args.smoke:
            command.extend(["--num-workers", "0", "--limit", "16"])
        run_step(command, project_dir)

    if args.mode in ["all", "train", "eval"]:
        run_step(
            [
                python,
                "src/visualize.py",
                "--history",
                str(Path(output_dir) / "history.json"),
                "--metrics",
                str(Path(output_dir) / "valid_metrics.json"),
                "--output-dir",
                str(Path(output_dir) / "figures"),
            ],
            project_dir,
        )

    if args.mode in ["all", "predict"]:
        command = [
            python,
            "src/predict.py",
            "--checkpoint",
            checkpoint,
            "--batch-size",
            str(args.predict_batch_size),
            "--score-threshold",
            str(args.score_threshold),
            "--max-detections-per-image",
            str(args.max_detections_per_image),
            "--output",
            str(Path(output_dir) / "pred.json"),
            "--zip-output",
            str(Path(output_dir) / "submission.zip"),
        ]
        if args.smoke:
            command.extend(["--num-workers", "0", "--limit", "3"])
        run_step(command, project_dir)


if __name__ == "__main__":
    main()
