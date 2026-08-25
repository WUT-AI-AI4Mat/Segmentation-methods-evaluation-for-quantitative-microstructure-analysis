import argparse
import os
import sys
import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from PIL import Image

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from peft import LoraConfig, get_peft_model

    from Myutils.visualizer import Visualizer
    from Myutils.metrics import Metric
    from Myutils.checkpoint import load_incremental_checkpoint
    print("Loaded SAM2 and evaluation dependencies.")
except ImportError as e:
    print(f"Failed to import a required dependency: {e}")
    sys.exit(1)


IMG_DIR = None
LBL_DIR = None

RESULT_ROOT = None

CHECKPOINT_PATH = None
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

FINETUNED_WEIGHT_PATH = None


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_MASK_AREA = 10000000
CROP_BOTTOM_PIXELS = 0

POINTS_PER_SIDE = 32
POINTS_PER_BATCH = 64
PRED_IOU_THRESH = 0.8
STABILITY_SCORE_THRESH = 0.8


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LoRA and mask-decoder fine-tuned SAM2.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--finetuned-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=MODEL_CFG)
    parser.add_argument("--points-per-side", type=int, default=POINTS_PER_SIDE)
    parser.add_argument("--points-per-batch", type=int, default=POINTS_PER_BATCH)
    parser.add_argument("--pred-iou-thresh", type=float, default=PRED_IOU_THRESH)
    parser.add_argument("--stability-score-thresh", type=float, default=STABILITY_SCORE_THRESH)
    parser.add_argument("--crop-bottom-pixels", type=int, default=CROP_BOTTOM_PIXELS)
    parser.add_argument("--max-mask-area", type=int, default=MAX_MASK_AREA)
    parser.add_argument("--device", default=DEVICE)
    return parser.parse_args()


def cv_imread(file_path, flags=cv2.IMREAD_COLOR):
    if not os.path.exists(file_path): return None
    img = None
    try:
        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, flags)
    except: pass
    if img is None: return None

    if flags == cv2.IMREAD_GRAYSCALE and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif flags == cv2.IMREAD_COLOR and img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def find_label_file(img_name, label_dir):
    base_name = os.path.splitext(img_name)[0]
    rules = [
        lambda x: x,
        lambda x: x.replace("RG", "RGMask"),
        lambda x: x + "_label",
        lambda x: x + "_mask",
        lambda x: x + "_seg"
    ]
    exts = ['.png', '.jpg', '.tif', '.bmp']

    for rule in rules:
        try: target = rule(base_name)
        except: continue
        for ext in exts:
            path = os.path.join(label_dir, target + ext)
            if os.path.exists(path): return path
    return None


def main():
    global IMG_DIR, LBL_DIR, RESULT_ROOT, CHECKPOINT_PATH
    global MODEL_CFG, FINETUNED_WEIGHT_PATH, DEVICE
    global MAX_MASK_AREA, CROP_BOTTOM_PIXELS, POINTS_PER_SIDE
    global POINTS_PER_BATCH, PRED_IOU_THRESH, STABILITY_SCORE_THRESH

    args = parse_args()
    IMG_DIR = os.path.join(args.dataset_root, "test", "images")
    LBL_DIR = os.path.join(args.dataset_root, "test", "masks")
    RESULT_ROOT = args.output_dir
    CHECKPOINT_PATH = args.checkpoint
    FINETUNED_WEIGHT_PATH = args.finetuned_checkpoint
    MODEL_CFG = args.config
    POINTS_PER_SIDE = args.points_per_side
    POINTS_PER_BATCH = args.points_per_batch
    PRED_IOU_THRESH = args.pred_iou_thresh
    STABILITY_SCORE_THRESH = args.stability_score_thresh
    CROP_BOTTOM_PIXELS = args.crop_bottom_pixels
    MAX_MASK_AREA = args.max_mask_area
    DEVICE = args.device

    print("Starting fine-tuned SAM2 evaluation.")

    save_plot_dir = os.path.join(RESULT_ROOT, "plots")
    excel_path = os.path.join(RESULT_ROOT, "metrics_summary.xlsx")
    os.makedirs(save_plot_dir, exist_ok=True)

    print(f"Loading base checkpoint: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(CHECKPOINT_PATH)

    try:
        sam2_model = build_sam2(MODEL_CFG, CHECKPOINT_PATH, device=DEVICE, apply_postprocessing=False)
    except Exception as e:
        raise RuntimeError(f"Failed to build SAM2: {e}") from e

    if os.path.exists(FINETUNED_WEIGHT_PATH):
        lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["qkv", "proj"],
            lora_dropout=0.05, bias="none"
        )
        sam2_model.image_encoder = get_peft_model(sam2_model.image_encoder, lora_config)

        loaded_count = load_incremental_checkpoint(
            sam2_model, FINETUNED_WEIGHT_PATH, map_location=DEVICE
        )
        sam2_model.image_encoder = sam2_model.image_encoder.merge_and_unload()

        print(f"Loaded {loaded_count} fine-tuned tensors.")
    else:
        raise FileNotFoundError(FINETUNED_WEIGHT_PATH)

    sam2_model.eval()

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=POINTS_PER_SIDE,
        points_per_batch=POINTS_PER_BATCH,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_SCORE_THRESH,
        crop_n_layers=0,
        min_mask_region_area=0,
        multimask_output=False
    )

    valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.bmp')
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(valid_exts)]
    print(f"Found {len(img_files)} test images.")

    all_metrics = []
    start_time = time.time()

    for img_file in tqdm(img_files, desc="Processing"):
        img_path = os.path.join(IMG_DIR, img_file)

        try:
            image = cv_imread(img_path, cv2.IMREAD_COLOR)
            if image is None: continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            h_raw, w_raw = image_rgb.shape[:2]
            if CROP_BOTTOM_PIXELS > 0 and h_raw > CROP_BOTTOM_PIXELS:
                image_rgb = image_rgb[:-CROP_BOTTOM_PIXELS, :]

            label_path = find_label_file(img_file, LBL_DIR)
            gt_label = None
            if label_path:
                gt_label = cv_imread(label_path, cv2.IMREAD_UNCHANGED)
                if gt_label is not None and gt_label.ndim == 3:
                    gt_label = cv2.cvtColor(gt_label, cv2.COLOR_BGR2GRAY)

            if gt_label is not None:
                h_lbl, w_lbl = gt_label.shape[:2]
                if CROP_BOTTOM_PIXELS > 0 and abs(h_lbl - h_raw) < 2:
                    gt_label = gt_label[:-CROP_BOTTOM_PIXELS, :]

                h_now, w_now = image_rgb.shape[:2]
                h_lbl, w_lbl = gt_label.shape[:2]
                h_min, w_min = min(h_now, h_lbl), min(w_now, w_lbl)

                if h_now > h_min: image_rgb = image_rgb[:h_min, :]
                if h_lbl > h_min: gt_label = gt_label[:h_min, :]
                if w_now > w_min: image_rgb = image_rgb[:, :w_min]
                if w_lbl > w_min: gt_label = gt_label[:, :w_min]

            raw_masks = mask_generator.generate(image_rgb)

            filtered_masks = []
            for mask in raw_masks:
                if mask['area'] <= MAX_MASK_AREA:
                    filtered_masks.append(mask)

            masks = filtered_masks

            current_metric = Metric.compute_all(gt_label, masks)
            current_metric['filename'] = img_file
            all_metrics.append(current_metric)

            base_name = os.path.splitext(img_file)[0]
            raw_save_path = os.path.join(save_plot_dir, f"{base_name}_raw_mask.png")
            Visualizer.save_raw_prediction(
                image_shape=image_rgb.shape,
                pred_result=masks,
                save_path=raw_save_path
            )

            plt.close('all')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Evaluation failed for {img_file}: {e}")
            continue

    end_time = time.time()
    total_processing_time = end_time - start_time
    avg_time_per_img = total_processing_time / len(img_files) if len(img_files) > 0 else 0

    print(f"Evaluation completed in {total_processing_time:.2f} s ({avg_time_per_img:.2f} s/image).")

    if len(all_metrics) > 0:
        df = pd.DataFrame(all_metrics)
        cols = ['filename'] + [c for c in df.columns if c != 'filename']
        df = df[cols]
        mean_row = df.select_dtypes(include=[np.number]).mean()

        mean_row['filename'] = 'AVERAGE'
        mean_row['Total_Time(s)'] = round(total_processing_time, 2)
        mean_row['Avg_Time/Img(s)'] = round(avg_time_per_img, 2)

        df_final = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        df_final.to_excel(excel_path, index=False)
        print(f"Saved metrics to {excel_path}")
    else:
        print("No metrics were generated.")


if __name__ == "__main__":
    main()
