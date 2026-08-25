import sys
import os
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from Myutils.visualizer import Visualizer
from Myutils.metrics import Metric
import cv2
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config
import time


BASE_RESULT_DIR = None
CHECKPOINT_PATH = None


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

class Args:
    cfg = "configs/swin_tiny_patch4_window7_224_lite.yaml"
    opts = None
    zip = False
    cache_mode = 'part'
    resume = None
    accumulation_steps = None
    use_checkpoint = False
    amp_opt_level = 'O1'
    tag = None
    eval = True
    throughput = False
    batch_size = 32
    base_lr = 0.01
    output_dir = None
    img_size = IMG_SIZE

args = Args()


def cv_imread(file_path, flags=cv2.IMREAD_COLOR):
    if not os.path.exists(file_path): return None
    try:
        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, flags)
    except: return None
    return img

def find_label_file(img_name, label_dir):
    base_name = os.path.splitext(img_name)[0]
    rules = [
        lambda x: x,
        lambda x: x.replace("RG", "RGMask"),
        lambda x: x + "_label",
        lambda x: x + "_mask",
        lambda x: x + "_seg",
        lambda x: x.replace("img", "msk"),
        lambda x: x.replace("image", "mask")
    ]
    exts = ['.png', '.jpg', '.tif', '.bmp']
    for rule in rules:
        try: target = rule(base_name)
        except: continue
        for ext in exts:
            path = os.path.join(label_dir, target + ext)
            if os.path.exists(path): return path
    return None


def test_single_dataset(dataset_name, dataset_root, train_num_classes):
    print(f"\n{'='*10}  Testing (Binary Mode): {dataset_name} {'='*10}")


    MAX_MASK_AREA = 2000000
    print(f"Maximum retained mask area: {MAX_MASK_AREA} pixels.")


    img_dir = os.path.join(dataset_root, "test", "images")
    lbl_dir = os.path.join(dataset_root, "test", "masks")

    if not os.path.exists(img_dir):
        print(f" Test set not found, checking Validation set...")
        img_dir = os.path.join(dataset_root, "val", "images")
        lbl_dir = os.path.join(dataset_root, "val", "masks")
        if not os.path.exists(img_dir):
            print(f" No images found for {dataset_name}, skipping.")
            return

    save_root = os.path.join(BASE_RESULT_DIR, dataset_name)
    save_plot_dir = os.path.join(save_root, "plots")
    os.makedirs(save_plot_dir, exist_ok=True)
    excel_path = os.path.join(save_root, "metrics_summary.xlsx")


    model_path = CHECKPOINT_PATH
    if not os.path.exists(model_path):
        print(f" Model weight not found at {model_path}")
        return

    print(f" Loading Model (Trained with {train_num_classes} classes)...")
    config = get_config(args)

    model = ViT_seg(config, img_size=IMG_SIZE, num_classes=train_num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.tif', '.bmp', '.jpeg', '.tiff'))])
    all_metrics = []


    start_time = time.time()


    for img_file in tqdm(img_files, desc=f"Infer {dataset_name}"):
        img_path = os.path.join(img_dir, img_file)

        try:

            image = cv_imread(img_path, cv2.IMREAD_COLOR)
            if image is None: continue

            h_raw, w_raw = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_input = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))

            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = (image_input.astype(np.float32) / 255.0 - mean) / std
            inp_tensor = torch.from_numpy(inp.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)


            with torch.no_grad():
                outputs = model(inp_tensor)

                if train_num_classes > 1:
                    outputs = torch.softmax(outputs, dim=1)
                    pred_map = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
                else:
                    pred_map = (torch.sigmoid(outputs) > 0.5).float().squeeze(0).cpu().numpy()


            pred_binary = np.where(pred_map > 0, 1, 0).astype(np.uint8)


            pred_binary_raw = cv2.resize(pred_binary.astype(np.float32), (w_raw, h_raw), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_binary_raw, connectivity=8)
            filtered_pred = np.zeros_like(pred_binary_raw, dtype=np.int32)

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area <= MAX_MASK_AREA:
                    filtered_pred[labels == i] = i

            pred_mask_instance = filtered_pred


            label_path = find_label_file(img_file, lbl_dir)
            gt_label = None
            if label_path:
                gt_label = cv_imread(label_path, cv2.IMREAD_UNCHANGED)
                if gt_label is not None:
                    if gt_label.ndim == 3: gt_label = np.max(gt_label, axis=2)
                    if gt_label.shape != (h_raw, w_raw):
                        gt_label = cv2.resize(gt_label, (w_raw, h_raw), interpolation=cv2.INTER_NEAREST)


                    gt_binary = np.where(gt_label > 0, 1, 0).astype(np.uint8)


                    current_metric = Metric.compute_all(
                        gt_label=gt_binary,
                        pred_input=pred_mask_instance,
                        num_classes=2
                    )
                    current_metric['filename'] = img_file
                    all_metrics.append(current_metric)
            else:
                print(f"No label found for {img_file}; metrics were skipped.")


            base_name = os.path.splitext(img_file)[0]
            raw_save_path = os.path.join(save_plot_dir, f"{base_name}_raw_mask.png")

            Visualizer.save_raw_prediction(
                image_shape=image_rgb.shape,
                pred_result=pred_mask_instance,
                save_path=raw_save_path
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f" Error {img_file}: {e}")
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

        print(f" Excel saved: {excel_path}")
        print(f"   Average MBSS: {mean_row.get('mbss', 0):.4f} | mIoU: {mean_row.get('miou', 0):.4f}")
    else:
        print(" No metrics generated.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Swin-Unet on one dataset.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=Args.cfg)
    parser.add_argument("--device", default=DEVICE)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    BASE_RESULT_DIR = cli_args.output_dir
    CHECKPOINT_PATH = cli_args.checkpoint
    DEVICE = cli_args.device
    args.cfg = cli_args.config
    args.output_dir = BASE_RESULT_DIR
    dataset_name = cli_args.dataset_name or os.path.basename(
        os.path.normpath(cli_args.dataset_root)
    )

    print("Starting Swin-Unet evaluation.")
    test_single_dataset(
        dataset_name, cli_args.dataset_root, cli_args.num_classes
    )

    print("\n All tasks finished!")
