import argparse
import glob
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
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.getcwd()

PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)

for path in (PROJECT_ROOT, PARENT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from Myutils.visualizer import Visualizer
    from Myutils.metrics import Metric
    print('Successfully imported Myutils')
except ImportError as exc:
    print(f'Myutils import failed: {exc}')
    print(f'Current sys.path search list: {sys.path[:3]} ...')
    raise


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


def find_label_file(img_name, label_dir, label_exts):
    base_name = os.path.splitext(img_name)[0]
    target_name = base_name.replace('RG', 'RGMask')
    candidates = [base_name, target_name, base_name + '_mask']

    for candidate in candidates:
        for ext in label_exts:
            path = os.path.join(label_dir, candidate + ext)
            if os.path.exists(path):
                return path
    return None


def find_best_checkpoint(work_dir):
    patterns = [
        os.path.join(work_dir, 'best_mIoU*.pth'),
        os.path.join(work_dir, 'best_*.pth'),
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    candidates = sorted(set(candidates), key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else None


def build_parser(defaults):
    parser = argparse.ArgumentParser(
        description=f"Batch test SegFormer-B0 on {defaults['dataset_name']}.")
    parser.add_argument('--config', default=defaults['config'])
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--work-dir', default=defaults['work_dir'])
    parser.add_argument('--dataset-root', default=defaults.get('dataset_root'))
    parser.add_argument('--img-dir', default=defaults.get('img_dir'))
    parser.add_argument('--label-dir', default=defaults.get('label_dir'))
    parser.add_argument('--result-root', default=defaults['result_root'])
    parser.add_argument(
        '--device',
        default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument(
        '--num-classes', type=int, default=defaults['num_classes'])
    parser.add_argument(
        '--batch-size', type=int, default=defaults.get('batch_size', 16))
    parser.add_argument(
        '--image-exts',
        nargs='+',
        default=defaults['image_exts'])
    parser.add_argument(
        '--label-exts',
        nargs='+',
        default=defaults['label_exts'])
    return parser


def run(defaults):
    args = build_parser(defaults).parse_args()
    if args.dataset_root:
        args.img_dir = os.path.join(args.dataset_root, 'test', 'images')
        args.label_dir = os.path.join(args.dataset_root, 'test', 'masks')
    if not args.img_dir or not args.label_dir:
        raise ValueError(
            'Pass --dataset-root, or provide both --img-dir and --label-dir.')
    print(f"Start SegFormer batch test on {defaults['dataset_name']}...")
    print(f'Results will be saved to: {args.result_root}')
    print(f'Inference batch size: {args.batch_size}')

    save_plot_dir = os.path.join(args.result_root, 'plots')
    excel_path = os.path.join(args.result_root, 'metrics_summary.xlsx')
    os.makedirs(save_plot_dir, exist_ok=True)

    checkpoint = args.checkpoint or find_best_checkpoint(args.work_dir)
    if checkpoint is None or not os.path.exists(checkpoint):
        print(f'Error: checkpoint not found. work_dir={args.work_dir}')
        print('Pass it explicitly, for example: --checkpoint path/to/best.pth')
        return

    print(f'Loading SegFormer config: {args.config}')
    print(f'Loading checkpoint: {checkpoint}')
    model = init_model(args.config, checkpoint, device=args.device)
    print('Model loaded successfully')

    valid_exts = tuple(ext.lower() for ext in args.image_exts)
    label_exts = tuple(args.label_exts)
    img_files = [
        f for f in os.listdir(args.img_dir)
        if f.lower().endswith(valid_exts)
    ]
    img_files = sorted(img_files)
    print(f'Found {len(img_files)} images')

    all_metrics = []
    start_time = time.time()

    for start_idx in tqdm(
            range(0, len(img_files), args.batch_size), desc='Processing'):
        batch_files = img_files[start_idx:start_idx + args.batch_size]
        batch_paths = [os.path.join(args.img_dir, f) for f in batch_files]

        try:
            with torch.no_grad():
                batch_results = inference_model(model, batch_paths)

            if not isinstance(batch_results, (list, tuple)):
                batch_results = [batch_results]

            for img_file, img_path, result in zip(
                    batch_files, batch_paths, batch_results):
                image_raw = cv_imread(img_path, cv2.IMREAD_COLOR)
                if image_raw is None:
                    continue
                image_rgb_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
                h_raw, w_raw = image_rgb_raw.shape[:2]

                label_path = find_label_file(
                    img_file, args.label_dir, label_exts)
                gt_label = None
                if label_path:
                    gt_label = cv_imread(label_path, cv2.IMREAD_UNCHANGED)
                    if gt_label is not None and gt_label.ndim == 3:
                        gt_label = cv2.cvtColor(
                            gt_label, cv2.COLOR_BGR2GRAY)
                    if gt_label is not None:
                        gt_label = gt_label.astype(np.uint8)

                pred_mask = result.pred_sem_seg.data.squeeze()
                pred_mask = pred_mask.cpu().numpy().astype(np.uint8)

                if pred_mask.shape != (h_raw, w_raw):
                    pred_mask = cv2.resize(
                        pred_mask, (w_raw, h_raw),
                        interpolation=cv2.INTER_NEAREST)

                if gt_label is not None:
                    if gt_label.shape != pred_mask.shape:
                        gt_label = cv2.resize(
                            gt_label, (w_raw, h_raw),
                            interpolation=cv2.INTER_NEAREST)

                    current_metric = Metric.compute_all(
                        gt_label, pred_mask, num_classes=args.num_classes)
                    current_metric['filename'] = img_file
                    all_metrics.append(current_metric)
                else:
                    print(
                        f'\nWarning: label file for {img_file} was not '
                        'found; metrics are skipped.')

                base_name = os.path.splitext(img_file)[0]
                raw_save_path = os.path.join(
                    save_plot_dir, f'{base_name}_raw_mask.png')

                Visualizer.save_raw_prediction(
                    image_shape=image_rgb_raw.shape,
                    pred_result=pred_mask,
                    save_path=raw_save_path)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as exc:
            print(f'\nError processing batch {batch_files}: {exc}')
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
        print(f'Average metrics:\n{mean_row}')
    else:
        print('No valid metric data was generated.')

    print('SegFormer test finished.')
