from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from hw3.visualize import append_metrics_csv, load_metrics_jsonl, plot_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot HW3 training metrics.")
    parser.add_argument(
        "run_dir", type=Path, help="Directory containing metrics.jsonl."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = args.run_dir / "metrics.jsonl"
    rows = load_metrics_jsonl(log_path)
    if not rows:
        raise FileNotFoundError(f"No metrics found in {log_path}")
    csv_path = args.run_dir / "metrics.csv"
    if csv_path.exists():
        csv_path.unlink()
    for row in rows:
        append_metrics_csv(csv_path, row)
    plot_metrics(log_path, args.run_dir / "metrics.png")
    print(f"wrote {csv_path}")
    print(f"wrote {args.run_dir / 'metrics.png'}")


if __name__ == "__main__":
    main()
