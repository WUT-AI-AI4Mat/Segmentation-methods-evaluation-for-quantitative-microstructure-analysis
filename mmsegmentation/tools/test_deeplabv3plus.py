import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mmseg.apis import inference_model, init_model


try:
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
except NameError:
    current_dir = os.getcwd()

project_root = os.path.dirname(current_dir)
parent_dir = os.path.dirname(project_root)

for path in (project_root, parent_dir):
    if path not in sys.path:
        sys.path.insert(0, path)

print(f"Current execution path: {current_dir}")
print(f"Detected project root: {project_root}")

try:
    from Myutils.visualizer import Visualizer
    from Myutils.metrics import Metric
    print("Successfully imported Myutils")
except ImportError as exc:
    print(f"Myutils import failed: {exc}")
    print(f"Current sys.path search list: {sys.path[:3]} ...")
    raise


# Fill these paths before running.
IMG_DIR = ''
LBL_DIR = ''
RESULT_ROOT = ''
CONFIG_PATH = (
    'configs/deeplabv3plus/'
    'deeplabv3plus_r50-d8_4xb4-80k_ade20k-512x512.py')
CHECKPOINT_PATH = ''

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
CLASSES = 2


def cv_imread(file_path, flags=cv2.IMREAD_COLOR):
    if not os.path.exists(file_path):
        return None
    img = None
    try:
        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, flags)
    except Exception:
        pass
    if img is None:
        return None

    if flags == cv2.IMREAD_GRAYSCALE and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif flags == cv2.IMREAD_COLOR and img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def find_label_file(img_name, label_dir):
    base_name = os.path.splitext(img_name)[0]
    target_name = base_name.replace('RG', 'RGMask')
    candidates = [base_name, target_name, base_name + '_mask']

    exts = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
    for candidate in candidates:
        for ext in exts:
            path = os.path.join(label_dir, candidate + ext)
            if os.path.exists(path):
                return path
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch test DeepLabV3+ on a semantic segmentation dataset.')
    parser.add_argument('--config', default=CONFIG_PATH)
    parser.add_argument('--checkpoint', default=CHECKPOINT_PATH)
    parser.add_argument('--img-dir', default=IMG_DIR)
    parser.add_argument('--label-dir', default=LBL_DIR)
    parser.add_argument('--result-root', default=RESULT_ROOT)
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--num-classes', type=int, default=CLASSES)
    return parser.parse_args()


def main():
    args = parse_args()
    print('Start DeepLabV3+ batch multi-class test...')
    print(f'Results will be saved to: {args.result_root}')

    if not args.img_dir or not os.path.isdir(args.img_dir):
        print(f'Error: invalid IMG_DIR: {args.img_dir}')
        return
    if not args.label_dir or not os.path.isdir(args.label_dir):
        print(f'Error: invalid LBL_DIR: {args.label_dir}')
        return
    if not args.result_root:
        print('Error: RESULT_ROOT is empty.')
        return
    if not args.config or not os.path.exists(args.config):
        print(f'Error: config not found: {args.config}')
        return
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        print(f'Error: checkpoint not found: {args.checkpoint}')
        return

    save_plot_dir = os.path.join(args.result_root, 'plots')
    excel_path = os.path.join(args.result_root, 'metrics_summary.xlsx')
    os.makedirs(save_plot_dir, exist_ok=True)

    # Model-specific section: DeepLabV3+ is loaded through MMSegmentation.
    print(f'Loading DeepLabV3+ config: {args.config}')
    print(f'Loading checkpoint: {args.checkpoint}')
    model = init_model(args.config, args.checkpoint, device=args.device)
    print('Model loaded successfully')

    valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
    img_files = [
        f for f in os.listdir(args.img_dir)
        if f.lower().endswith(valid_exts)
    ]
    img_files = sorted(img_files)
    print(f'Found {len(img_files)} images')

    all_metrics = []
    start_time = time.time()

    for img_file in tqdm(img_files, desc='Processing'):
        img_path = os.path.join(args.img_dir, img_file)

        try:
            image_raw = cv_imread(img_path, cv2.IMREAD_COLOR)
            if image_raw is None:
                continue
            image_rgb_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
            h_raw, w_raw = image_rgb_raw.shape[:2]

            label_path = find_label_file(img_file, args.label_dir)
            gt_label = None
            if label_path:
                gt_label = cv_imread(label_path, cv2.IMREAD_UNCHANGED)
                if gt_label is not None and gt_label.ndim == 3:
                    gt_label = cv2.cvtColor(gt_label, cv2.COLOR_BGR2GRAY)
                    print(
                        f'\nWarning: RGB label converted to grayscale: '
                        f'{label_path}')
                if gt_label is not None:
                    gt_label = gt_label.astype(np.uint8)

            # Model-specific section: MMSeg inference returns class-ID mask.
            with torch.no_grad():
                result = inference_model(model, img_path)
                pred_mask_semantic = result.pred_sem_seg.data.squeeze()
                pred_mask_semantic = pred_mask_semantic.cpu().numpy()
                pred_mask_semantic = pred_mask_semantic.astype(np.uint8)

            if pred_mask_semantic.shape != (h_raw, w_raw):
                pred_mask_semantic = cv2.resize(
                    pred_mask_semantic,
                    (w_raw, h_raw),
                    interpolation=cv2.INTER_NEAREST)

            if gt_label is not None:
                if gt_label.shape != pred_mask_semantic.shape:
                    gt_label = cv2.resize(
                        gt_label,
                        (w_raw, h_raw),
                        interpolation=cv2.INTER_NEAREST)

                current_metric = Metric.compute_all(
                    gt_label,
                    pred_mask_semantic,
                    num_classes=args.num_classes)
                current_metric['filename'] = img_file
                all_metrics.append(current_metric)
            else:
                print(
                    f'\nWarning: label file for {img_file} was not found; '
                    'metrics are skipped.')

            base_name = os.path.splitext(img_file)[0]
            raw_save_path = os.path.join(
                save_plot_dir, f'{base_name}_raw_mask.png')

            Visualizer.save_raw_prediction(
                image_shape=image_rgb_raw.shape,
                pred_result=pred_mask_semantic,
                save_path=raw_save_path)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as exc:
            print(f'\nError processing {img_file}: {exc}')
            import traceback
            traceback.print_exc()
            continue

    total_processing_time = time.time() - start_time
    avg_time_per_img = (
        total_processing_time / len(img_files) if len(img_files) > 0 else 0)

    print(
        f'\nPrediction finished. Total time: {total_processing_time:.2f}s '
        f'(average {avg_time_per_img:.2f}s/image)')

    print('\nExporting Excel report...')
    if len(all_metrics) > 0:
        df = pd.DataFrame(all_metrics)
        cols = ['filename'] + [c for c in df.columns if c != 'filename']
        df = df[cols]

        mean_row = df.select_dtypes(include=[np.number]).mean()
        mean_row['filename'] = 'AVERAGE'
        mean_row['Total_Time(s)'] = round(total_processing_time, 2)
        mean_row['Avg_Time/Img(s)'] = round(avg_time_per_img, 2)

        df_final = pd.concat(
            [df, pd.DataFrame([mean_row.to_dict()])],
            ignore_index=True)

        df_final.to_excel(excel_path, index=False)
        print(f'Excel saved to: {excel_path}')
        print(f'Average metrics (DeepLabV3+ multi-class):\n{mean_row}')
    else:
        print('No valid metric data was generated.')

    print('DeepLabV3+ test finished.')


if __name__ == '__main__':
    main()
