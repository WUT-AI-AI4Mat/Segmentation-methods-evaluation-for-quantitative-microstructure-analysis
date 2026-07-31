import argparse
import csv
import os
from collections import defaultdict

import cv2
import numpy as np


CLASS_NAMES = {
    0: 'background',
    1: 'material_1',
    2: 'material_2',
    3: 'material_3',
    4: 'material_4',
    255: 'ignore',
}


def cv_imread(file_path, flags=cv2.IMREAD_UNCHANGED):
    if not os.path.exists(file_path):
        return None
    img_array = np.fromfile(file_path, dtype=np.uint8)
    return cv2.imdecode(img_array, flags)


def analyze_split(data_root, split, mask_dir, mask_suffix):
    split_mask_dir = os.path.join(data_root, split, mask_dir)
    stats = {
        'split': split,
        'mask_dir': split_mask_dir,
        'mask_files': 0,
        'total_pixels': 0,
        'counts': defaultdict(int),
        'image_hits': defaultdict(int),
        'unexpected_counts': defaultdict(int),
        'unexpected_image_hits': defaultdict(int),
        'empty_or_unreadable': [],
    }

    if not os.path.isdir(split_mask_dir):
        stats['missing_dir'] = True
        return stats

    mask_files = sorted([
        name for name in os.listdir(split_mask_dir)
        if name.lower().endswith(mask_suffix.lower())
    ])
    stats['mask_files'] = len(mask_files)

    valid_labels = set(CLASS_NAMES)
    for mask_name in mask_files:
        mask_path = os.path.join(split_mask_dir, mask_name)
        mask = cv_imread(mask_path)
        if mask is None:
            stats['empty_or_unreadable'].append(mask_path)
            continue
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(np.uint8)

        values, counts = np.unique(mask, return_counts=True)
        stats['total_pixels'] += int(mask.size)
        for value, count in zip(values.tolist(), counts.tolist()):
            value = int(value)
            count = int(count)
            if value in valid_labels:
                stats['counts'][value] += count
                stats['image_hits'][value] += 1
            else:
                stats['unexpected_counts'][value] += count
                stats['unexpected_image_hits'][value] += 1

    return stats


def print_split_report(stats):
    split = stats['split']
    print(f'\n[{split}]')
    print(f"mask_dir: {stats['mask_dir']}")
    if stats.get('missing_dir', False):
        print('ERROR: mask directory does not exist.')
        return

    print(f"mask files: {stats['mask_files']}")
    print(f"total pixels: {stats['total_pixels']}")
    if stats['empty_or_unreadable']:
        print(f"unreadable masks: {len(stats['empty_or_unreadable'])}")

    total = stats['total_pixels']
    print('label,class,pixels,percent,images_with_label')
    for label, name in CLASS_NAMES.items():
        count = stats['counts'][label]
        percent = count / total * 100 if total > 0 else 0.0
        hits = stats['image_hits'][label]
        print(f'{label},{name},{count},{percent:.6f},{hits}')

    if stats['unexpected_counts']:
        print('unexpected labels:')
        for label in sorted(stats['unexpected_counts']):
            count = stats['unexpected_counts'][label]
            percent = count / total * 100 if total > 0 else 0.0
            hits = stats['unexpected_image_hits'][label]
            print(f'{label},unexpected,{count},{percent:.6f},{hits}')


def save_csv(all_stats, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                'split', 'label', 'class_name', 'pixels', 'percent',
                'images_with_label', 'mask_files', 'total_pixels'
            ])
        writer.writeheader()
        for stats in all_stats:
            total = stats['total_pixels']
            for label, name in CLASS_NAMES.items():
                count = stats['counts'][label]
                writer.writerow({
                    'split': stats['split'],
                    'label': label,
                    'class_name': name,
                    'pixels': count,
                    'percent': count / total * 100 if total > 0 else 0.0,
                    'images_with_label': stats['image_hits'][label],
                    'mask_files': stats['mask_files'],
                    'total_pixels': total,
                })
            for label in sorted(stats['unexpected_counts']):
                count = stats['unexpected_counts'][label]
                writer.writerow({
                    'split': stats['split'],
                    'label': label,
                    'class_name': 'unexpected',
                    'pixels': count,
                    'percent': count / total * 100 if total > 0 else 0.0,
                    'images_with_label': stats['unexpected_image_hits'][label],
                    'mask_files': stats['mask_files'],
                    'total_pixels': total,
                })


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze MetalDAM semantic mask class distribution.')
    parser.add_argument('--data-root', default='MetalDAM')
    parser.add_argument('--mask-dir', default='masks')
    parser.add_argument('--mask-suffix', default='.png')
    parser.add_argument(
        '--output',
        default='work_dirs/metaldam_class_distribution.csv')
    return parser.parse_args()


def main():
    args = parse_args()
    all_stats = []
    for split in ('train', 'val', 'test'):
        stats = analyze_split(
            args.data_root, split, args.mask_dir, args.mask_suffix)
        all_stats.append(stats)
        print_split_report(stats)

    save_csv(all_stats, args.output)
    print(f'\nCSV saved to: {args.output}')


if __name__ == '__main__':
    main()
