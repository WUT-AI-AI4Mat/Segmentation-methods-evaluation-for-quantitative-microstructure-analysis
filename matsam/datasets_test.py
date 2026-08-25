import sys
import os
import argparse
import cv2
import numpy as np
import torch
import pandas as pd
import tifffile
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from unittest.mock import MagicMock
import time

current_dir = os.path.dirname(os.path.abspath(__file__))


sys.modules["gala"] = MagicMock()
sys.modules["gala.evaluate"] = MagicMock()


project_root = os.path.dirname(current_dir)

if current_dir not in sys.path:
    sys.path.append(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)


from utils.segment_anything_ import sam_model_registry, SamAutomaticMaskGenerator
from utils.prompt_generator import PromptGenerator
from Myutils.visualizer import Visualizer
from Myutils.metrics import Metric
print("Loaded MatSAM and evaluation dependencies.")

IMG_DIR = None
LBL_DIR = None
RESULT_ROOT = None
SAVE_PLOT_DIR = None
EXCEL_PATH = None



CHECKPOINT_PATH = None
MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


METHOD_TYPE = 2
CROP_BOTTOM_PIXELS = 0
PROMPT_LAYERS = 0
PROMPT_SCALES = 3
N_PER_SIDE_BASE = 54
PRED_IOU_THRESH = 0.88
STABILITY_SCORE_THRESH = 0.9
BOX_NMS_THRESH = 0.7
POINTS_PER_BATCH = 256
MAX_MASK_AREA = 2000000


LABEL_RULES = [
    lambda x: x,
    lambda x: x.replace("RG", "RGMask"),
    lambda x: x + "_mask",
    lambda x: x + "_label",
    lambda x: x + "_seg",
]
SUPPORTED_EXTS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']



def cv_imread(file_path, flags=cv2.IMREAD_COLOR):
    """Read an image while preserving class indices when requested."""
    if not os.path.exists(file_path): return None

    img = None


    try:

        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, flags)
    except: pass


    if img is None and file_path.lower().endswith(('.tif', '.tiff')):
        try:
            import tifffile
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




    if flags == cv2.IMREAD_UNCHANGED or flags == -1:
        return img


    if flags == cv2.IMREAD_GRAYSCALE:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    elif flags == cv2.IMREAD_COLOR:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img

def find_label_file(img_name, label_dir):
    """Find the label file associated with an image."""
    base_name = os.path.splitext(img_name)[0]
    for rule_func in LABEL_RULES:
        try: target_name = rule_func(base_name)
        except: continue
        for ext in SUPPORTED_EXTS:
            test_path = os.path.join(label_dir, target_name + ext)
            if os.path.exists(test_path): return test_path
    return None

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the original MatSAM model.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-type", type=int, choices=(1, 2), default=METHOD_TYPE)
    parser.add_argument("--crop-bottom-pixels", type=int, default=CROP_BOTTOM_PIXELS)
    parser.add_argument("--prompt-layers", type=int, default=PROMPT_LAYERS)
    parser.add_argument("--prompt-scales", type=int, default=PROMPT_SCALES)
    parser.add_argument("--n-per-side-base", type=int, default=N_PER_SIDE_BASE)
    parser.add_argument("--pred-iou-threshold", type=float, default=PRED_IOU_THRESH)
    parser.add_argument("--stability-score-threshold", type=float, default=STABILITY_SCORE_THRESH)
    parser.add_argument("--box-nms-threshold", type=float, default=BOX_NMS_THRESH)
    parser.add_argument("--points-per-batch", type=int, default=POINTS_PER_BATCH)
    parser.add_argument("--max-mask-area", type=int, default=MAX_MASK_AREA)
    parser.add_argument("--device", default=DEVICE)
    return parser.parse_args()


def main():
    global IMG_DIR, LBL_DIR, RESULT_ROOT, SAVE_PLOT_DIR, EXCEL_PATH
    global CHECKPOINT_PATH, METHOD_TYPE, CROP_BOTTOM_PIXELS, DEVICE
    global PROMPT_LAYERS, PROMPT_SCALES, N_PER_SIDE_BASE
    global PRED_IOU_THRESH, STABILITY_SCORE_THRESH, BOX_NMS_THRESH
    global POINTS_PER_BATCH, MAX_MASK_AREA

    args = parse_args()
    test_dir = os.path.join(args.dataset_root, "test")
    IMG_DIR = os.path.join(test_dir, "images")
    LBL_DIR = os.path.join(test_dir, "masks")
    RESULT_ROOT = args.output_dir
    SAVE_PLOT_DIR = os.path.join(RESULT_ROOT, "plots")
    EXCEL_PATH = os.path.join(RESULT_ROOT, "metrics_summary.xlsx")
    CHECKPOINT_PATH = args.checkpoint
    METHOD_TYPE = args.method_type
    CROP_BOTTOM_PIXELS = args.crop_bottom_pixels
    PROMPT_LAYERS = args.prompt_layers
    PROMPT_SCALES = args.prompt_scales
    N_PER_SIDE_BASE = args.n_per_side_base
    PRED_IOU_THRESH = args.pred_iou_threshold
    STABILITY_SCORE_THRESH = args.stability_score_threshold
    BOX_NMS_THRESH = args.box_nms_threshold
    POINTS_PER_BATCH = args.points_per_batch
    MAX_MASK_AREA = args.max_mask_area
    DEVICE = args.device

    print("Starting MatSAM evaluation.")
    print(f"Method type: {METHOD_TYPE}; crop bottom: {CROP_BOTTOM_PIXELS} px")
    print(f"Maximum retained mask area: {MAX_MASK_AREA} px")


    os.makedirs(SAVE_PLOT_DIR, exist_ok=True)


    print(f"Loading {MODEL_TYPE} checkpoint.")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint not found: {CHECKPOINT_PATH}")
        return
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)


    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(valid_extensions)]
    print(f"Found {len(img_files)} test images.")

    all_metrics_list = []




    start_time = time.time()


    for img_file in tqdm(img_files, desc="Processing"):
        img_path = os.path.join(IMG_DIR, img_file)


        image = cv_imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Could not read image: {img_file}")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        h_raw_origin, w_raw_origin = image_rgb.shape[:2]


        if CROP_BOTTOM_PIXELS > 0 and h_raw_origin > CROP_BOTTOM_PIXELS:
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
            common_w = min(w_img_curr, w_lbl_curr)

            if h_img_curr > common_h: image_rgb = image_rgb[:common_h, :]
            if h_lbl_curr > common_h: gt_label = gt_label[:common_h, :]
            if w_img_curr > common_w: image_rgb = image_rgb[:, :common_w]
            if w_lbl_curr > common_w: gt_label = gt_label[:, :common_w]


        try:

            prompter = PromptGenerator(
                image_rgb,
                layers=PROMPT_LAYERS,
                scales=PROMPT_SCALES,
                n_per_side_base=N_PER_SIDE_BASE,
                method_type=METHOD_TYPE
            )
            points_layers = prompter.generate_prompt_points()


            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=None,
                point_grids=points_layers,
                pred_iou_thresh=PRED_IOU_THRESH,
                crop_n_layers=0,
                crop_n_points_downscale_factor=3,
                box_nms_thresh=BOX_NMS_THRESH,
                crop_nms_thresh=BOX_NMS_THRESH,
                stability_score_thresh=STABILITY_SCORE_THRESH,
                points_per_batch=POINTS_PER_BATCH,
                min_mask_region_area=0
            )

            raw_masks = mask_generator.generate(image_rgb)


            filtered_masks = []
            for mask in raw_masks:
                if mask['area'] <= MAX_MASK_AREA:
                    filtered_masks.append(mask)

            masks = filtered_masks

        except Exception as e:
            print(f"\nInference failed for {img_file}: {e}")
            continue



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

    print("Evaluation completed.")

if __name__ == "__main__":
    main()
