"""Shared constants for the HW3 instance segmentation project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "hw3-data-release"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

NUM_CLASSES = 5
CELL_CLASS_IDS = (1, 2, 3, 4)
CLASS_NAMES = {
    1: "class1",
    2: "class2",
    3: "class3",
    4: "class4",
}

SUBMISSION_FILENAME = "test-reults.json"
