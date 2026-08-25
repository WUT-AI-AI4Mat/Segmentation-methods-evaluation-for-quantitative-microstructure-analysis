import sys
import os
import argparse
import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import time


try:
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
except NameError:
    current_dir = os.getcwd()

parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)


try:
    from Myutils.visualizer import Visualizer
    from Myutils.metrics import Metric
    print("Loaded evaluation utilities.")
except ImportError as e:
    print(f"Failed to import evaluation utilities: {e}")

IMG_DIR = None
LBL_DIR = None
RESULT_ROOT = None
MODEL_PATH = None
ENCODER = 'resnet50'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = 512

NUM_CLASSES = 5

CROP_BOTTOM_PIXELS = 0

def cv_imread(file_path, flags=cv2.IMREAD_COLOR):
    if not os.path.exists(file_path): return None
    try:
        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, flags)
        return img
    except:
        return None

def find_label_file(img_name, label_dir):
    base_name = os.path.splitext(img_name)[0]
    candidates = [
        base_name.replace("RG", "RGMask"),
        base_name,
        base_name + "_mask",
        base_name + "_label"
    ]
    exts = ['.png', '.jpg', '.tif', '.bmp']
    for t in candidates:
        for ext in exts:
            path = os.path.join(label_dir, t + ext)
            if os.path.exists(path): return path
    return None

def get_preprocessing():
    return albu.Compose([
        albu.Resize(INPUT_SIZE, INPUT_SIZE),
        albu.Normalize(),
        ToTensorV2()
    ])

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multiclass DeepLabV3+.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--device", default=DEVICE)
    return parser.parse_args()


def main():
    global IMG_DIR, LBL_DIR, RESULT_ROOT, MODEL_PATH, NUM_CLASSES, DEVICE

    args = parse_args()
    IMG_DIR = os.path.join(args.dataset_root, "test", "images")
    LBL_DIR = os.path.join(args.dataset_root, "test", "masks")
    RESULT_ROOT = args.output_dir
    MODEL_PATH = args.checkpoint
    NUM_CLASSES = args.num_classes
    DEVICE = args.device

    print(f"Starting {NUM_CLASSES}-class DeepLabV3+ evaluation.")
    print(f"Saving results to {RESULT_ROOT}")

    MAX_MASK_AREA = 2000000
    print(f"Maximum retained mask area: {MAX_MASK_AREA} pixels.")

    save_plot_dir = os.path.join(RESULT_ROOT, "plots")
    os.makedirs(save_plot_dir, exist_ok=True)
    excel_path = os.path.join(RESULT_ROOT, "metrics_summary.xlsx")


    print(f"Loading DeepLabV3+ with {ENCODER} encoder.")


    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None
    )

    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
            print("Loaded checkpoint.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return
    else:
        print(f"Checkpoint not found: {MODEL_PATH}")
        return


    valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(valid_exts)]
    print(f"Found {len(img_files)} test images.")

    all_metrics = []
    preprocessing_fn = get_preprocessing()
    start_time = time.time()


    for img_file in tqdm(img_files, desc="Processing"):
        img_path = os.path.join(IMG_DIR, img_file)

        try:

            image_raw = cv_imread(img_path, cv2.IMREAD_COLOR)
            if image_raw is None:
                print(f"Could not read image: {img_file}")
                continue

            label_path = find_label_file(img_file, LBL_DIR)
            gt_label = None

            if label_path:
                gt_label = cv_imread(label_path, cv2.IMREAD_UNCHANGED)

                if gt_label is not None:
                    h_img, w_img = image_raw.shape[:2]
                    h_lbl, w_lbl = gt_label.shape[:2]

                    if h_img > h_lbl:
                        image_raw = image_raw[:h_lbl, :]

            image_rgb_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
            h_raw, w_raw = image_rgb_raw.shape[:2]


            if gt_label is not None:
                if gt_label.ndim == 3:

                    gt_label = gt_label[:, :, 0]


                gt_label = gt_label.astype(np.uint8)


            sample = preprocessing_fn(image=image_rgb_raw)
            input_tensor = sample['image'].unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(input_tensor)


                pred_mask_512 = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)


            pred_mask_multi = cv2.resize(pred_mask_512, (w_raw, h_raw), interpolation=cv2.INTER_NEAREST)


            filtered_pred = np.zeros_like(pred_mask_multi, dtype=np.uint8)

            for class_id in range(1, NUM_CLASSES):

                class_binary_mask = (pred_mask_multi == class_id).astype(np.uint8)


                if class_binary_mask.sum() == 0:
                    continue

                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_binary_mask, connectivity=8)


                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area <= MAX_MASK_AREA:

                        filtered_pred[labels == i] = class_id

            pred_mask_instance = filtered_pred


            if gt_label is not None:
                if gt_label.shape != pred_mask_instance.shape:
                    gt_label = cv2.resize(gt_label, (w_raw, h_raw), interpolation=cv2.INTER_NEAREST)


                current_metric = Metric.compute_all(gt_label, pred_mask_instance, num_classes=NUM_CLASSES)
                current_metric['filename'] = img_file
                all_metrics.append(current_metric)
            else:
                print(f"No label found for {img_file}; metrics were skipped.")


            base_name = os.path.splitext(img_file)[0]
            raw_save_path = os.path.join(save_plot_dir, f"{base_name}_raw_mask.png")

            Visualizer.save_raw_prediction(
                image_shape=image_rgb_raw.shape,
                pred_result=pred_mask_instance,
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
        print(f"Average mIoU: {mean_row.get('miou', 0):.4f}")
    else:
        print("No metrics were generated.")

    print("Evaluation completed.")

if __name__ == "__main__":
    main()
