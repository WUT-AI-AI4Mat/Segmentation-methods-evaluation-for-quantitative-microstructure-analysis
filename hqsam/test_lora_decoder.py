import argparse
import os
import sys
import cv2
import numpy as np
import torch
import pandas as pd
import tifffile
from tqdm import tqdm
from PIL import Image
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    from Myutils.visualizer import Visualizer
    from Myutils.metrics import Metric
    from Myutils.checkpoint import load_incremental_checkpoint
    from peft import LoraConfig, get_peft_model
    print("Loaded HQ-SAM and evaluation dependencies.")
except ImportError as e:
    print(f"Failed to import a required dependency: {e}")
    raise e

IMG_DIR = None
LBL_DIR = None
RESULT_ROOT = None
SAVE_PLOT_DIR = None
EXCEL_PATH = None

MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = None

FINETUNED_WEIGHT_PATH = None


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

POINTS_PER_SIDE = 32
PRED_IOU_THRESH = 0.78
STABILITY_SCORE = 0.8
CROP_BOTTOM_PIXELS = 0
MAX_MASK_AREA = 2000000

LABEL_RULES = [
    lambda x: x,
    lambda x: x.replace("RG", "RGMask"),
    lambda x: x + "_mask",
    lambda x: x + "_label",
    lambda x: x + "_seg",
]
SUPPORTED_EXTS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LoRA and mask-decoder fine-tuned HQ-SAM.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--finetuned-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--points-per-side", type=int, default=POINTS_PER_SIDE)
    parser.add_argument("--pred-iou-thresh", type=float, default=PRED_IOU_THRESH)
    parser.add_argument("--stability-score-thresh", type=float, default=STABILITY_SCORE)
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

    if img is None and file_path.lower().endswith(('.tif', '.tiff')):
        try:
            img = tifffile.imread(file_path)
            if flags == cv2.IMREAD_COLOR and img.ndim == 3:
                 img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except: pass

    if img is None:
        try:
            pil_img = Image.open(file_path)
            img = np.asarray(pil_img)
            if flags == cv2.IMREAD_COLOR and img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except: pass

    if img is None: return None

    if flags == cv2.IMREAD_UNCHANGED or flags == -1: return img
    if flags == cv2.IMREAD_GRAYSCALE:
        if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif flags == cv2.IMREAD_COLOR:
        if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def find_label_file(img_name, label_dir):
    base_name = os.path.splitext(img_name)[0]
    for rule_func in LABEL_RULES:
        try: target_name = rule_func(base_name)
        except: continue
        for ext in SUPPORTED_EXTS:
            test_path = os.path.join(label_dir, target_name + ext)
            if os.path.exists(test_path): return test_path
    return None

def main():
    global IMG_DIR, LBL_DIR, RESULT_ROOT, SAVE_PLOT_DIR, EXCEL_PATH
    global CHECKPOINT_PATH, FINETUNED_WEIGHT_PATH, DEVICE
    global POINTS_PER_SIDE, PRED_IOU_THRESH, STABILITY_SCORE
    global CROP_BOTTOM_PIXELS, MAX_MASK_AREA

    args = parse_args()
    IMG_DIR = os.path.join(args.dataset_root, "test", "images")
    LBL_DIR = os.path.join(args.dataset_root, "test", "masks")
    RESULT_ROOT = args.output_dir
    SAVE_PLOT_DIR = os.path.join(RESULT_ROOT, "plots")
    EXCEL_PATH = os.path.join(RESULT_ROOT, "metrics_summary.xlsx")
    CHECKPOINT_PATH = args.checkpoint
    FINETUNED_WEIGHT_PATH = args.finetuned_checkpoint
    POINTS_PER_SIDE = args.points_per_side
    PRED_IOU_THRESH = args.pred_iou_thresh
    STABILITY_SCORE = args.stability_score_thresh
    CROP_BOTTOM_PIXELS = args.crop_bottom_pixels
    MAX_MASK_AREA = args.max_mask_area
    DEVICE = args.device

    print("Starting fine-tuned HQ-SAM evaluation.")
    os.makedirs(SAVE_PLOT_DIR, exist_ok=True)

    print(f"Loading {MODEL_TYPE} base checkpoint.")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)

    if os.path.exists(FINETUNED_WEIGHT_PATH):
        lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["qkv", "proj"],
            lora_dropout=0.05, bias="none"
        )
        sam.image_encoder = get_peft_model(sam.image_encoder, lora_config)

        loaded_count = load_incremental_checkpoint(
            sam, FINETUNED_WEIGHT_PATH, map_location=DEVICE
        )
        sam.image_encoder = sam.image_encoder.merge_and_unload()
        print(f"Loaded {loaded_count} fine-tuned tensors.")
    else:
        raise FileNotFoundError(FINETUNED_WEIGHT_PATH)

    sam.to(device=DEVICE)
    sam.eval()

    original_forward = sam.mask_decoder.forward

    def patched_forward(*args, **kwargs):
        kwargs['hq_token_only'] = True
        kwargs['multimask_output'] = False
        return original_forward(*args, **kwargs)

    sam.mask_decoder.forward = patched_forward

    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=POINTS_PER_SIDE,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_SCORE,
        crop_n_layers=0
    )

    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(valid_extensions)]
    print(f"Found {len(img_files)} test images.")

    all_metrics_list = []
    start_time = time.time()

    for img_file in tqdm(img_files, desc="Processing"):
        img_path = os.path.join(IMG_DIR, img_file)

        try:
            image = cv_imread(img_path, cv2.IMREAD_COLOR)
            if image is None: continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            h_raw_origin, w_raw_origin = image_rgb.shape[:2]

            if CROP_BOTTOM_PIXELS > 0:
                if h_raw_origin > CROP_BOTTOM_PIXELS:
                    image_rgb = image_rgb[:-CROP_BOTTOM_PIXELS, :]

            label_path = find_label_file(img_file, LBL_DIR)
            gt_label = None

            if label_path:
                gt_label = cv_imread(label_path, cv2.IMREAD_UNCHANGED)
                if gt_label is not None and gt_label.ndim == 3:
                    gt_label = cv2.cvtColor(gt_label, cv2.COLOR_BGR2GRAY)

            if gt_label is not None:
                h_lbl_origin, w_lbl_origin = gt_label.shape[:2]
                if CROP_BOTTOM_PIXELS > 0:
                    if abs(h_lbl_origin - h_raw_origin) < 2:
                        gt_label = gt_label[:-CROP_BOTTOM_PIXELS, :]

                h_img_curr, w_img_curr = image_rgb.shape[:2]
                h_lbl_curr, w_lbl_curr = gt_label.shape[:2]

                common_h = min(h_img_curr, h_lbl_curr)
                if h_img_curr > common_h: image_rgb = image_rgb[:common_h, :]
                if h_lbl_curr > common_h: gt_label = gt_label[:common_h, :]

                common_w = min(w_img_curr, w_lbl_curr)
                if w_img_curr > common_w: image_rgb = image_rgb[:, :common_w]
                if w_lbl_curr > common_w: gt_label = gt_label[:, :common_w]

            try:
                raw_masks = generator.generate(image_rgb)
            except Exception as e:
                print(f"\nErr: {img_file} {e}")
                continue

            filtered_masks = []
            for mask in raw_masks:
                if mask['area'] <= MAX_MASK_AREA:
                    filtered_masks.append(mask)
            masks = filtered_masks

            if gt_label is not None:
                current_metrics = Metric.compute_all(gt_label, masks)
                current_metrics['filename'] = img_file
                all_metrics_list.append(current_metrics)
            else:
                print(f"No label found for {img_file}; metrics were skipped.")

            base_name = os.path.splitext(img_file)[0]
            raw_save_path = os.path.join(SAVE_PLOT_DIR, f"{base_name}_raw_mask.png")

            Visualizer.save_raw_prediction(
                image_shape=image_rgb.shape,
                pred_result=masks,
                save_path=raw_save_path
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Evaluation failed for {img_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    end_time = time.time()
    total_processing_time = end_time - start_time
    avg_time_per_img = total_processing_time / len(img_files) if len(img_files) > 0 else 0

    print(f"Evaluation completed in {total_processing_time:.2f} s ({avg_time_per_img:.2f} s/image).")

    if len(all_metrics_list) > 0:
        df = pd.DataFrame(all_metrics_list)
        cols = ['filename'] + [c for c in df.columns if c != 'filename']
        df = df[cols]
        mean_row = df.select_dtypes(include=[np.number]).mean()

        mean_row['filename'] = 'AVERAGE'
        mean_row['Total_Time(s)'] = round(total_processing_time, 2)
        mean_row['Avg_Time/Img(s)'] = round(avg_time_per_img, 2)

        df_final = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        df_final.to_excel(EXCEL_PATH, index=False)

        print(f"Saved metrics to {EXCEL_PATH}")
    else:
        print("No metrics were generated.")

if __name__ == "__main__":
    main()
