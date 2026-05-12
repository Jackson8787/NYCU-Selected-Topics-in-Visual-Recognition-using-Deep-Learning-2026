"""Experiment logging and plotting helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def append_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    """Append metrics to a CSV file, expanding columns as new keys appear."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []

    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)

    for key in metrics:
        if key not in fieldnames:
            fieldnames.append(key)

    rows.append({key: metrics.get(key, "") for key in fieldnames})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_metrics_jsonl(path: Path) -> list[dict[str, float]]:
    """Load numeric metrics from a JSONL log."""
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        rows.append(
            {
                key: float(value)
                for key, value in item.items()
                if isinstance(value, (int, float))
            }
        )
    return rows


def plot_metrics(log_path: Path, output_path: Path) -> None:
    """Write a compact PNG dashboard for the current experiment."""
    rows = load_metrics_jsonl(log_path)
    if not rows:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in rows if "epoch" in row]
    if not epochs:
        return

    loss_keys = [
        "loss",
        "loss_classifier",
        "loss_box_reg",
        "loss_mask",
        "loss_objectness",
        "loss_rpn_box_reg",
    ]
    eval_keys = ["ap50", "ap", "ar", "lr", "skipped_oom"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    for key in loss_keys:
        values = [row.get(key) for row in rows]
        if any(value is not None for value in values):
            axes[0].plot(
                epochs, values, marker="o", linewidth=1.5, markersize=3, label=key
            )
    axes[0].set_title("Training losses")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    for key in eval_keys:
        values = [row.get(key) for row in rows]
        if any(value is not None for value in values):
            axes[1].plot(
                epochs, values, marker="o", linewidth=1.5, markersize=3, label=key
            )
    axes[1].set_title("Validation and runtime metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


class TensorboardLogger:
    """Optional TensorBoard scalar writer with a no-op fallback."""

    def __init__(self, log_dir: Path, enabled: bool) -> None:
        self.writer = None
        if not enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:
            print(f"TensorBoard unavailable: {exc}")
            return
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def add_metrics(self, metrics: dict[str, Any], step: int) -> None:
        if self.writer is None:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, float(value), step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
