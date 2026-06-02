"""Create a weighted ensemble submission from existing pred.npz files."""

from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

import numpy as np

from _bootstrap import PROJECT_ROOT  # noqa: F401
from evaluate_knn_route_ensemble import weighted_average


def parse_weights(raw: str, count: int) -> list[float]:
    weights = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(weights) != count:
        raise ValueError(f"got {len(weights)} weights for {count} inputs")
    return weights


def expected_keys() -> list[str]:
    return [f"{index}.png" for index in range(100)]


def load_prediction_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        keys = sorted(data.files, key=lambda name: int(Path(name).stem))
        if keys != expected_keys():
            raise ValueError(f"{path} does not contain keys 0.png through 99.png")
        predictions = {key: data[key] for key in keys}
    for key, array in predictions.items():
        if array.shape != (3, 256, 256) or array.dtype != np.uint8:
            raise ValueError(
                f"{path}:{key} has shape={array.shape} dtype={array.dtype}"
            )
    return predictions


def blend_uint8_arrays(arrays: list[np.ndarray], weights: list[float]) -> np.ndarray:
    blended = weighted_average(arrays, weights)
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip-name", default="hw4_npz_ensemble_submission.zip")
    args = parser.parse_args()

    weights = parse_weights(args.weights, len(args.inputs))
    loaded = [load_prediction_npz(path) for path in args.inputs]
    predictions = {}
    rows = []
    for key in expected_keys():
        arrays = [item[key] for item in loaded]
        predictions[key] = blend_uint8_arrays(arrays, weights)
        rows.append({"key": key, "sources": len(arrays)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / "pred.npz"
    np.savez(npz_path, **predictions)
    zip_path = args.output_dir / args.zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(npz_path, arcname="pred.npz")

    with (args.output_dir / "ensemble_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=["key", "sources"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Submission ZIP: {zip_path}")


if __name__ == "__main__":
    main()
