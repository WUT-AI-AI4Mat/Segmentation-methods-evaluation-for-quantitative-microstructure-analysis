import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


SAMPLES = [
    {
        "dataset": "UHCS",
        "source_image": "UHCS/test/images/A_micrograph1006.tif",
        "source_mask": "UHCS/test/masks/A_micrograph1006.tif",
        "filename": "uhcs_A_micrograph1006.png",
        "num_classes": 7,
    },
    {
        "dataset": "Super",
        "source_image": "Super/test/images/45kx_SE_15kV-etched_0.tif",
        "source_mask": "Super/test/masks/45kx_SE_15kV-etched_0.tif",
        "filename": "super_45kx_SE_15kV-etched_0.png",
        "num_classes": 3,
    },
    {
        "dataset": "EBC",
        "source_image": "EBC/test/images/010417#4_S2480009.tif",
        "source_mask": "EBC/test/masks/010417#4_S2480009.tif",
        "filename": "ebc_010417_4_S2480009.png",
        "num_classes": 3,
    },
    {
        "dataset": "MetalDAM",
        "source_image": "MetalDAM/test/images/micrograph1.jpg",
        "source_mask": "MetalDAM/test/masks/micrograph1.png",
        "filename": "metaldam_micrograph1.png",
        "num_classes": 5,
    },
    {
        "dataset": "Aachen-Heerlen",
        "source_image": "Aachen-Heerlen/test/images/IMG_00004.png",
        "source_mask": "Aachen-Heerlen/test/masks/IMG_00004.png",
        "filename": "aachen_heerlen_IMG_00004.png",
        "num_classes": 2,
    },
    {
        "dataset": "EMPS",
        "source_image": "EMPS/test/images/01ac659240.png",
        "source_mask": "EMPS/test/masks/01ac659240.png",
        "filename": "emps_01ac659240.png",
        "num_classes": 2,
    },
    {
        "dataset": "Grain",
        "source_image": "Grain/test/images/A_01_04.png",
        "source_mask": "Grain/test/masks/A_01_04.png",
        "filename": "grain_A_01_04.png",
        "num_classes": 2,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare one test image-mask pair from each benchmark dataset."
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        required=True,
        help="Parent directory containing the seven dataset folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "test",
    )
    return parser.parse_args()


def read_image(path, flags):
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, flags)
    if image is None:
        raise RuntimeError(f"Failed to read {path}")
    return image


def write_png(path, image):
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"Failed to encode {path}")
    encoded.tofile(path)


def main():
    args = parse_args()
    image_dir = args.output_dir / "images"
    mask_dir = args.output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for sample in SAMPLES:
        source_image = args.datasets_root / sample["source_image"]
        source_mask = args.datasets_root / sample["source_mask"]
        if not source_image.is_file() or not source_mask.is_file():
            raise FileNotFoundError(
                f"Missing source pair: {source_image} / {source_mask}"
            )

        image = read_image(source_image, cv2.IMREAD_COLOR)
        mask = read_image(source_mask, cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Shape mismatch for {sample['dataset']}")

        write_png(image_dir / sample["filename"], image)
        write_png(mask_dir / sample["filename"], mask.astype(np.uint8))
        manifest_rows.append(
            {
                "dataset": sample["dataset"],
                "filename": sample["filename"],
                "num_classes": sample["num_classes"],
                "source_image": sample["source_image"],
                "source_mask": sample["source_mask"],
            }
        )

    manifest_path = args.output_dir / "samples.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Prepared {len(manifest_rows)} image-mask pairs in {args.output_dir}")


if __name__ == "__main__":
    main()
