import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

try:
    from .metrics import Metric
except ImportError:
    from metrics import Metric

class Visualizer:
    """Utilities for saving masks and visualizing segmentation results."""

    @staticmethod
    def _ensure_2d(data):
        """Return a two-dimensional label array."""
        if isinstance(data, np.ndarray) and data.ndim == 3:
            return data[:, :, 0]
        return data

    @staticmethod
    def _to_instance_map(shape, data, is_multiclass=False):
        """Convert model output to a two-dimensional instance or class map."""
        h, w = shape[:2]
        instance_map = np.zeros((h, w), dtype=np.int32)

        data = Visualizer._ensure_2d(data)

        if isinstance(data, list):
            sorted_masks = sorted(data, key=(lambda x: x['area'] if isinstance(x, dict) else np.sum(x)), reverse=True)
            for i, ann in enumerate(sorted_masks):
                m = ann['segmentation'] if isinstance(ann, dict) else ann
                if m.shape[:2] != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                instance_map[m] = i + 1

        elif isinstance(data, np.ndarray):
            if data.shape[:2] != (h, w):
                data = cv2.resize(data.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


            if is_multiclass:
                instance_map = data.astype(np.int32)
            else:

                binary = (data > 0).astype(np.uint8)
                _, instance_map = cv2.connectedComponents(binary, connectivity=4)

        return instance_map

    @staticmethod
    def draw_style_black_bg(shape, mask_data, num_classes=2, fill_color=(110, 180, 130)):
        """Render masks with colored regions and black boundaries."""
        h, w = shape[:2]
        vis_img = np.zeros((h, w, 3), dtype=np.uint8)

        mask_data = Visualizer._ensure_2d(mask_data)


        is_multiclass = num_classes > 2


        inst_map = Visualizer._to_instance_map((h, w), mask_data, is_multiclass=is_multiclass)


        boundaries = find_boundaries(inst_map, mode='thick')

        if is_multiclass:


            unique_classes = np.unique(inst_map)
            unique_classes = unique_classes[unique_classes > 0]

            cmap = plt.get_cmap('tab20')
            for class_id in unique_classes:

                color = (np.array(cmap(class_id % 20)[:3]) * 255).astype(np.uint8)
                class_fill_mask = np.logical_and(inst_map == class_id, ~boundaries)
                vis_img[class_fill_mask] = color

        else:


            foreground_mask = (inst_map > 0)
            final_fill_mask = np.logical_and(foreground_mask, ~boundaries)

            if np.any(final_fill_mask):
                vis_img[final_fill_mask] = fill_color


        vis_img[boundaries] = (0, 0, 0)
        return vis_img

    @classmethod
    def save_raw_prediction(cls, image_shape, pred_result, save_path):
        """Save a prediction as an instance map or class-index mask."""
        h, w = image_shape[:2]

        if isinstance(pred_result, list):
            mask_to_save = np.zeros((h, w), dtype=np.int32)
            sorted_masks = sorted(pred_result, key=lambda x: x['area'] if isinstance(x, dict) else np.sum(x), reverse=True)

            for i, item in enumerate(sorted_masks):
                m = item['segmentation'] if isinstance(item, dict) else item
                if m.shape[:2] != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                mask_to_save[m] = i + 1

        elif isinstance(pred_result, np.ndarray):
            pred_result = cls._ensure_2d(pred_result)
            if pred_result.shape[:2] != (h, w):
                pred_result = cv2.resize(pred_result.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            mask_to_save = pred_result.astype(np.int32)

        save_dtype = np.uint16 if mask_to_save.max() > 255 else np.uint8
        mask_to_save = mask_to_save.astype(save_dtype)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, mask_to_save)

    @classmethod
    def plot_comparison(cls, model_name, image, gt_label, pred_result, metrics=['mbss', 'miou', 'dice', 'hd', 'mae'], save_path=None, num_classes=2):
        """Plot the image, ground truth, prediction, and selected metrics."""

        scores = Metric.compute_all(gt_label, pred_result, metrics, num_classes=num_classes)

        gt_c = scores.pop('gt_count', 0)
        pred_c = scores.pop('pred_count', 0)

        metric_strs = []
        for k, v in scores.items():
            k_name = k.upper()
            val_str = f"{v:.3f}" if isinstance(v, float) else f"{v}"
            metric_strs.append(f"{k_name}:{val_str}")

        full_metrics_str = " | ".join(metric_strs)

        UNIFIED_COLOR = (110, 180, 130)


        vis_gt = cls.draw_style_black_bg(image.shape, gt_label, num_classes=num_classes, fill_color=UNIFIED_COLOR)
        vis_pred = cls.draw_style_black_bg(image.shape, pred_result, num_classes=num_classes, fill_color=UNIFIED_COLOR)


        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(image)
        axes[0].set_title("Original Image", fontsize=12, color="#26F10B", fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(vis_gt)
        axes[1].set_title(f"Ground Truth\nCount: {gt_c}", fontsize=12, color="#26F10B", fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(vis_pred)
        title_str = f"{model_name}\n{full_metrics_str}\nPred Count: {pred_c}"
        axes[2].set_title(title_str, fontsize=11, color="#26F10B", fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#202020')

        plt.close(fig)
