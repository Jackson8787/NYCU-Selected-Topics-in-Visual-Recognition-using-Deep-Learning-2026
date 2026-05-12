from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from hw3.data import (
    fold_summary,
    list_train_samples,
    make_stratified_folds,
    sample_class_presence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create greedy stratified K-fold splits."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/hw3-data-release"))
    parser.add_argument("--out", type=Path, default=Path("outputs/splits_5fold.json"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = list_train_samples(args.data_root)
    folds = make_stratified_folds(samples, num_folds=args.folds, seed=args.seed)
    presence = sample_class_presence(samples)
    payload = {
        "seed": args.seed,
        "num_folds": args.folds,
        "folds": folds,
        "summary": fold_summary(folds, presence),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
