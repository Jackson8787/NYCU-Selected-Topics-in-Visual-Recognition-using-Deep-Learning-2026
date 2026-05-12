from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import _bootstrap  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract HW3 released data tar.")
    parser.add_argument("--tar", type=Path, default=Path("hw3-data-release.tar"))
    parser.add_argument("--out", type=Path, default=Path("data/hw3-data-release"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.tar) as tar:
        tar.extractall(args.out)
    print(f"Extracted {args.tar} to {args.out}")


if __name__ == "__main__":
    main()
