import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the deterministic synthetic demo dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "test",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def make_sample(rng, sample_index, size=256):
    image = np.full((size, size), 36, dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.uint8)

    y_grid, x_grid = np.mgrid[:size, :size]
    background = 10 * np.sin(x_grid / 17.0) + 7 * np.cos(y_grid / 23.0)
    image += background

    object_count = 12 + sample_index * 3
    for _ in range(object_count):
        center = (
            int(rng.integers(18, size - 18)),
            int(rng.integers(18, size - 18)),
        )
        axes = (
            int(rng.integers(7, 20)),
            int(rng.integers(5, 15)),
        )
        angle = float(rng.uniform(0, 180))
        cv2.ellipse(mask, center, axes, angle, 0, 360, 1, thickness=-1)

    image[mask > 0] += rng.uniform(105, 145)
    image += rng.normal(0, 9, image.shape)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = np.clip(image, 0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image_bgr, mask


def main():
    args = parse_args()
    image_dir = args.output_dir / "images"
    mask_dir = args.output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    for sample_index in range(1, 4):
        image, mask = make_sample(rng, sample_index)
        filename = f"synthetic_{sample_index:02d}.png"
        cv2.imwrite(str(image_dir / filename), image)
        cv2.imwrite(str(mask_dir / filename), mask * 255)

    print(f"Generated 3 image-mask pairs in {args.output_dir}")


if __name__ == "__main__":
    main()
