import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Myutils.metrics import Metric


METRICS = ["miou", "dice", "precision", "recall", "acc", "hd95", "mae", "nsd"]
OUTPUT_FIELDS = ["gt_count", "pred_count", *METRICS]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the CPU-only seven-dataset smoke test."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "test",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare generated metrics with demo/expected_metrics.csv.",
    )
    return parser.parse_args()


def segment_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, prediction = cv2.threshold(
        gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(prediction, cv2.MORPH_OPEN, kernel)


def write_metrics(rows, output_path):
    fieldnames = ["dataset", "filename", *OUTPUT_FIELDS]
    numeric_rows = [{key: row[key] for key in OUTPUT_FIELDS} for row in rows]
    average = {
        key: float(np.mean([float(row[key]) for row in numeric_rows]))
        for key in OUTPUT_FIELDS
    }

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"dataset": "ALL", "filename": "AVERAGE", **average})


def verify_metrics(actual_path, expected_path, tolerance=1e-6):
    with actual_path.open(newline="", encoding="utf-8") as file:
        actual_rows = list(csv.DictReader(file))
    with expected_path.open(newline="", encoding="utf-8") as file:
        expected_rows = list(csv.DictReader(file))

    if len(actual_rows) != len(expected_rows):
        raise AssertionError("The number of metric rows does not match the reference.")

    for actual, expected in zip(actual_rows, expected_rows):
        if (
            actual["dataset"] != expected["dataset"]
            or actual["filename"] != expected["filename"]
        ):
            raise AssertionError("Metric row identifiers do not match the reference.")
        for key in OUTPUT_FIELDS:
            if abs(float(actual[key]) - float(expected[key])) > tolerance:
                raise AssertionError(
                    f"Metric mismatch for {actual['filename']} {key}: "
                    f"{actual[key]} != {expected[key]}"
                )


def main():
    args = parse_args()
    image_dir = args.data_root / "images"
    mask_dir = args.data_root / "masks"
    manifest_path = args.data_root / "samples.csv"
    prediction_dir = args.output_dir / "predicted_masks"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Sample manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as file:
        samples = list(csv.DictReader(file))
    if not samples:
        raise RuntimeError(f"Sample manifest is empty: {manifest_path}")

    start_time = time.perf_counter()
    rows = []
    for sample in samples:
        image_path = image_dir / sample["filename"]
        mask_path = mask_dir / sample["filename"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        ground_truth = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or ground_truth is None:
            raise RuntimeError(f"Failed to read {image_path.name} or its mask")
        if ground_truth.ndim == 3:
            ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)

        prediction = segment_image(image)
        scores = Metric.compute_all(
            ground_truth,
            prediction,
            metrics=METRICS,
            num_classes=int(sample["num_classes"]),
        )
        rows.append(
            {
                "dataset": sample["dataset"],
                "filename": image_path.name,
                **scores,
            }
        )
        cv2.imwrite(str(prediction_dir / image_path.name), prediction * 255)

    metrics_path = args.output_dir / "metrics.csv"
    write_metrics(rows, metrics_path)
    if args.verify:
        expected_path = Path(__file__).resolve().parent / "expected_metrics.csv"
        verify_metrics(metrics_path, expected_path)
        print("Metrics match demo/expected_metrics.csv")
    elapsed = time.perf_counter() - start_time
    print(f"Processed {len(rows)} images in {elapsed:.3f} seconds")
    print(f"Predicted masks: {prediction_dir}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
