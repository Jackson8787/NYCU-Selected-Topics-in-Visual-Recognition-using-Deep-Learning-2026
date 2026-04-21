import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot training and validation results."
    )
    parser.add_argument("--history", default="outputs/baseline/history.json")
    parser.add_argument(
        "--metrics",
        default="outputs/baseline/valid_metrics.json",
    )
    parser.add_argument("--output-dir", default="outputs/baseline/figures")
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_history(history_path, output_dir):
    history_path = Path(history_path)
    if not history_path.exists():
        print(f"Skip loss plot; missing {history_path}")
        return None

    history = load_json(history_path)
    if not history:
        print("Skip loss plot; empty history.")
        return None

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    valid_rows = [row for row in history if row.get("valid_loss") is not None]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="train loss")
    if valid_rows:
        plt.plot(
            [row["epoch"] for row in valid_rows],
            [row["valid_loss"] for row in valid_rows],
            marker="o",
            label="valid loss",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DETR Baseline Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = Path(output_dir) / "loss_curve.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"Wrote {output_path}")
    return output_path


def plot_metrics(metrics_path, output_dir):
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        print(f"Skip metrics plot; missing {metrics_path}")
        return None

    metrics = load_json(metrics_path)
    selected = {
        key: metrics[key]
        for key in ["AP", "AP50", "AP75", "AR_1", "AR_10", "AR_100"]
        if key in metrics
    }
    if not selected:
        print("Skip metrics plot; no supported metrics found.")
        return None

    plt.figure(figsize=(8, 5))
    plt.bar(selected.keys(), selected.values(), color="#4c78a8")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("COCO Validation Metrics")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = Path(output_dir) / "valid_metrics.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"Wrote {output_path}")
    return output_path


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    output_dir = project_dir / args.output_dir
    plot_history(project_dir / args.history, output_dir)
    plot_metrics(project_dir / args.metrics, output_dir)


if __name__ == "__main__":
    main()
