import cv2
import numpy as np
from scipy.spatial import cKDTree

class Metric:
    """Segmentation metrics used by all evaluation scripts."""


    @classmethod
    def compute_all(cls, gt_label, pred_input, metrics=['mbss', 'miou', 'dice', 'precision', 'recall', 'acc', 'hd95', 'mae', 'hd', 'nsd', 'mbss_add'], num_classes=2):
        """Compute selected metrics for binary or multiclass predictions."""
        results = {}


        gt_label = cls._ensure_2d(gt_label)
        if isinstance(pred_input, np.ndarray):
            pred_input = cls._ensure_2d(pred_input)

        is_multiclass = num_classes > 2


        if not is_multiclass:

            gt_bin, pred_bin = cls._preprocess_to_binary(gt_label, pred_input)

            gt_raw, pred_raw = gt_label, pred_input
            if isinstance(pred_input, list):
                pred_raw = cls._list_to_instance_map(gt_label.shape[:2], pred_input)

            context = {
                'gt_binary': gt_bin,
                'pred_binary': pred_bin,
                'gt_raw': gt_raw,
                'pred_raw': pred_raw,
                'gt_count': cls._get_instance_count(gt_raw),
                'pred_count': cls._get_instance_count(pred_raw),
                'num_classes': num_classes
            }
        else:

            if isinstance(pred_input, list):
                pred_input = cls._list_to_instance_map(gt_label.shape[:2], pred_input)

            context = {
                'gt_map': gt_label,
                'pred_map': pred_input,
                'num_classes': num_classes,
                'gt_count': cls._get_instance_count(gt_label),
                'pred_count': cls._get_instance_count(pred_input)
            }


        results['gt_count'] = context.get('gt_count', 0)
        results['pred_count'] = context.get('pred_count', 0)


        for metric_name in metrics:
            func_name = f"calculate_{metric_name}"
            func = getattr(cls, func_name, None)

            if func and callable(func):
                try:
                    score = func(**context)
                    results[metric_name] = score
                except Exception as e:
                    print(f" 指标 {metric_name} 计算出错: {e}")
                    results[metric_name] = 0.0

        return results


    @staticmethod
    def calculate_acc(**kwargs):
        num_classes = kwargs.get('num_classes', 2)
        if num_classes <= 2:
            gt = kwargs['gt_binary']
            pred = kwargs['pred_binary']
        else:
            gt = kwargs['gt_map']
            pred = kwargs['pred_map']

        if gt is None or pred is None: return 0.0
        correct = (gt == pred).sum()
        total = gt.size
        return correct / total if total > 0 else 0.0

    @staticmethod
    def calculate_miou(**kwargs):
        num_classes = kwargs.get('num_classes', 2)
        if num_classes <= 2:
            gt_b = kwargs['gt_binary'] > 0
            pred_b = kwargs['pred_binary'] > 0
            inter = np.logical_and(gt_b, pred_b).sum()
            union = np.logical_or(gt_b, pred_b).sum()
            return inter / union if union > 0 else 0.0
        else:
            gt = kwargs['gt_map']
            pred = kwargs['pred_map']
            ious = []


            for c in range(1, num_classes):
                gt_c = (gt == c)
                pred_c = (pred == c)


                if not gt_c.any() and not pred_c.any():
                    continue

                inter = np.logical_and(gt_c, pred_c).sum()
                union = np.logical_or(gt_c, pred_c).sum()
                ious.append(inter / union if union > 0 else 0.0)

            return np.mean(ious) if len(ious) > 0 else 1.0

    @staticmethod
    def calculate_dice(**kwargs):
        num_classes = kwargs.get('num_classes', 2)
        if num_classes <= 2:
            gt_b = kwargs['gt_binary'] > 0
            pred_b = kwargs['pred_binary'] > 0
            inter = np.logical_and(gt_b, pred_b).sum()
            sums = gt_b.sum() + pred_b.sum()
            return 2 * inter / sums if sums > 0 else 0.0
        else:
            gt = kwargs['gt_map']
            pred = kwargs['pred_map']
            dices = []

            for c in range(1, num_classes):
                gt_c = (gt == c)
                pred_c = (pred == c)
                if not gt_c.any() and not pred_c.any():
                    continue

                inter = np.logical_and(gt_c, pred_c).sum()
                sums = gt_c.sum() + pred_c.sum()
                dices.append(2 * inter / sums if sums > 0 else 0.0)

            return np.mean(dices) if len(dices) > 0 else 1.0

    @staticmethod
    def calculate_precision(**kwargs):
        num_classes = kwargs.get('num_classes', 2)
        if num_classes <= 2:
            gt_b = kwargs['gt_binary'] > 0
            pred_b = kwargs['pred_binary'] > 0
            tp = np.logical_and(gt_b, pred_b).sum()
            pred_pos = pred_b.sum()
            return tp / pred_pos if pred_pos > 0 else 0.0
        else:
            gt = kwargs['gt_map']
            pred = kwargs['pred_map']
            precisions = []
            for c in range(1, num_classes):
                gt_c = (gt == c)
                pred_c = (pred == c)
                if not gt_c.any() and not pred_c.any(): continue

                tp = np.logical_and(gt_c, pred_c).sum()
                pred_pos = pred_c.sum()
                precisions.append(tp / pred_pos if pred_pos > 0 else 0.0)
            return np.mean(precisions) if len(precisions) > 0 else 1.0

    @staticmethod
    def calculate_recall(**kwargs):
        num_classes = kwargs.get('num_classes', 2)
        if num_classes <= 2:
            gt_b = kwargs['gt_binary'] > 0
            pred_b = kwargs['pred_binary'] > 0
            tp = np.logical_and(gt_b, pred_b).sum()
            gt_pos = gt_b.sum()
            return tp / gt_pos if gt_pos > 0 else 0.0
        else:
            gt = kwargs['gt_map']
            pred = kwargs['pred_map']
            recalls = []
            for c in range(1, num_classes):
                gt_c = (gt == c)
                pred_c = (pred == c)
                if not gt_c.any() and not pred_c.any(): continue

                tp = np.logical_and(gt_c, pred_c).sum()
                gt_pos = gt_c.sum()
                recalls.append(tp / gt_pos if gt_pos > 0 else 0.0)
            return np.mean(recalls) if len(recalls) > 0 else 1.0


    @staticmethod
    def calculate_hd95(**kwargs):
        return Metric._dispatch_distance_metric(percentile=95, **kwargs)

    @staticmethod
    def calculate_hd(**kwargs):
        return Metric._dispatch_distance_metric(percentile=100, **kwargs)

    @staticmethod
    def _dispatch_distance_metric(percentile, **kwargs):
        num_classes = kwargs.get('num_classes', 2)
        if num_classes <= 2:
            return Metric._compute_single_hd(kwargs['gt_binary'], kwargs['pred_binary'], percentile)
        else:
            gt, pred = kwargs['gt_map'], kwargs['pred_map']
            hds = []
            for c in range(1, num_classes):
                gt_c = (gt == c).astype(np.uint8)
                pred_c = (pred == c).astype(np.uint8)
                if gt_c.sum() == 0 and pred_c.sum() == 0: continue
                hds.append(Metric._compute_single_hd(gt_c, pred_c, percentile))
            return float(np.mean(hds)) if len(hds) > 0 else 0.0

    @staticmethod
    def calculate_mae(**kwargs):
        num_classes = kwargs.get('num_classes', 2)

        if num_classes <= 2:
            return abs(kwargs['gt_count'] - kwargs['pred_count'])
        else:

            gt_map, pred_map = kwargs['gt_map'], kwargs['pred_map']
            maes = []
            for c in range(1, num_classes):
                gt_c_count = Metric._get_instance_count(gt_map, class_id=c)
                pred_c_count = Metric._get_instance_count(pred_map, class_id=c)
                maes.append(abs(gt_c_count - pred_c_count))
            return float(np.mean(maes)) if len(maes) > 0 else 0.0

    @staticmethod
    def calculate_nsd(tolerance=5, **kwargs):
        num_classes = kwargs.get('num_classes', 2)
        if num_classes <= 2:
            return Metric._compute_single_nsd(kwargs['gt_binary'], kwargs['pred_binary'], tolerance)
        else:
            gt, pred = kwargs['gt_map'], kwargs['pred_map']
            nsds = []
            for c in range(1, num_classes):
                gt_c = (gt == c).astype(np.uint8)
                pred_c = (pred == c).astype(np.uint8)
                if gt_c.sum() == 0 and pred_c.sum() == 0: continue
                nsds.append(Metric._compute_single_nsd(gt_c, pred_c, tolerance))
            return float(np.mean(nsds)) if len(nsds) > 0 else 1.0

    @staticmethod
    def calculate_mbss(boundary_tolerance=5, **kwargs):
        num_classes = kwargs.get('num_classes', 2)

        if num_classes <= 2:

            return Metric._compute_single_mbss(
                kwargs['gt_binary'], kwargs['pred_binary'],
                kwargs['gt_count'], kwargs['pred_count'],
                boundary_tolerance
            )
        else:

            gt_map, pred_map = kwargs['gt_map'], kwargs['pred_map']
            mbss_scores = []

            for c in range(1, num_classes):
                gt_c = (gt_map == c).astype(np.uint8)
                pred_c = (pred_map == c).astype(np.uint8)


                if gt_c.sum() == 0 and pred_c.sum() == 0:
                    continue


                N_gt_c = Metric._get_instance_count(gt_map, class_id=c)
                N_pred_c = Metric._get_instance_count(pred_map, class_id=c)

                score = Metric._compute_single_mbss(gt_c, pred_c, N_gt_c, N_pred_c, boundary_tolerance)
                mbss_scores.append(score)

            return float(np.mean(mbss_scores)) if len(mbss_scores) > 0 else 1.0

    @staticmethod
    def _compute_single_mbss(gt_bin, pred_bin, N_gt, N_pred, boundary_tolerance=5):
        """MBSS 单类别计算逻辑"""

        inter = np.logical_and(gt_bin > 0, pred_bin > 0).sum()
        union = np.logical_or(gt_bin > 0, pred_bin > 0).sum()
        iou = inter / union if union > 0 else 0.0


        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gt_border = cv2.morphologyEx(gt_bin, cv2.MORPH_GRADIENT, kernel)
        pred_border = cv2.morphologyEx(pred_bin, cv2.MORPH_GRADIENT, kernel)

        tol_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*boundary_tolerance+1, 2*boundary_tolerance+1))
        pred_border_dilated = cv2.dilate(pred_border, tol_kernel)

        covered_gt_pixels = np.logical_and(gt_border > 0, pred_border_dilated > 0).sum()
        total_gt_pixels = (gt_border > 0).sum()

        if total_gt_pixels == 0:

            boundary_recall = 1.0 if (pred_border > 0).sum() == 0 else 0.0
        else:
            boundary_recall = covered_gt_pixels / total_gt_pixels


        if max(N_gt, N_pred) == 0:
            count_score = 1.0
        else:
            count_score = np.sqrt(min(N_gt, N_pred) / max(N_gt, N_pred))

        return iou * boundary_recall * count_score

    @staticmethod
    def _compute_single_hd(gt_bin, pred_bin, percentile=95):
        if gt_bin.sum() == 0 and pred_bin.sum() == 0:
            return 0.0

        h, w = gt_bin.shape
        max_dist = np.sqrt(h**2 + w**2)


        if gt_bin.sum() == 0 or pred_bin.sum() == 0:
            return max_dist

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gt_border = cv2.morphologyEx(gt_bin, cv2.MORPH_GRADIENT, kernel)
        pred_border = cv2.morphologyEx(pred_bin, cv2.MORPH_GRADIENT, kernel)

        gt_pts = np.argwhere(gt_border > 0)
        pred_pts = np.argwhere(pred_border > 0)

        if len(gt_pts) == 0 or len(pred_pts) == 0: return max_dist

        tree_gt = cKDTree(gt_pts)
        tree_pred = cKDTree(pred_pts)
        d_p2g, _ = tree_gt.query(pred_pts)
        d_g2p, _ = tree_pred.query(gt_pts)

        if percentile == 100:
            return max(np.max(d_p2g), np.max(d_g2p))
        else:
            return max(np.percentile(d_p2g, percentile), np.percentile(d_g2p, percentile))

    @staticmethod
    def _compute_single_nsd(gt_bin, pred_bin, tolerance=5):
        if gt_bin.sum() == 0 and pred_bin.sum() == 0: return 1.0
        if gt_bin.sum() == 0 or pred_bin.sum() == 0: return 0.0

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        gt_eroded = cv2.erode(gt_bin, kernel, iterations=1)
        pred_eroded = cv2.erode(pred_bin, kernel, iterations=1)

        surface_gt = gt_bin ^ gt_eroded
        surface_pred = pred_bin ^ pred_eroded

        inv_surface_gt = np.where(surface_gt > 0, 0, 255).astype(np.uint8)
        inv_surface_pred = np.where(surface_pred > 0, 0, 255).astype(np.uint8)

        dt_gt = cv2.distanceTransform(inv_surface_gt, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        dt_pred = cv2.distanceTransform(inv_surface_pred, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        valid_pred_pts = np.sum((surface_pred > 0) & (dt_gt <= tolerance))
        valid_gt_pts = np.sum((surface_gt > 0) & (dt_pred <= tolerance))

        total_pred_pts = np.sum(surface_pred > 0)
        total_gt_pts = np.sum(surface_gt > 0)

        if total_pred_pts + total_gt_pts == 0: return 1.0
        return (valid_pred_pts + valid_gt_pts) / (total_pred_pts + total_gt_pts)

    @staticmethod
    def _ensure_2d(data):
        if data is None: return None
        if isinstance(data, np.ndarray) and data.ndim == 3:
            return data[:, :, 0]
        return data

    @staticmethod
    def _list_to_instance_map(shape, data_list):
        h, w = shape
        instance_map = np.zeros((h, w), dtype=np.int32)
        if not data_list: return instance_map
        sorted_masks = sorted(data_list, key=lambda x: np.sum(x['segmentation']) if isinstance(x, dict) else np.sum(x), reverse=True)
        for i, item in enumerate(sorted_masks):
            m = item['segmentation'] if isinstance(item, dict) else item
            if m.shape[:2] != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            instance_map[m] = i + 1
        return instance_map

    @staticmethod
    def _preprocess_to_binary(gt_label, pred_input):
        h, w = gt_label.shape[:2]
        gt_binary = (gt_label > 0).astype(np.uint8)
        pred_binary = np.zeros((h, w), dtype=np.uint8)

        if isinstance(pred_input, list):
            for item in pred_input:
                m = item['segmentation'] if isinstance(item, dict) else item
                if m.shape[:2] != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                pred_binary = np.maximum(pred_binary, m.astype(np.uint8))
        elif isinstance(pred_input, np.ndarray):
            pred_input = Metric._ensure_2d(pred_input)
            if pred_input.shape[:2] != (h, w):
                pred_input = cv2.resize(pred_input.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            pred_binary = (pred_input > 0).astype(np.uint8)

        return gt_binary, pred_binary

    @staticmethod
    def _get_instance_count(data, class_id=None, min_size=10):
        """
        计算实例数/连通域数量。如果传入 class_id，则只统计该类别的实例数。
        """
        if data is None: return 0

        if isinstance(data, list):
            count = 0
            for item in data:
                area = item['area'] if isinstance(item, dict) else np.sum(item)
                if area >= min_size: count += 1
            return count

        if isinstance(data, np.ndarray):
            data = Metric._ensure_2d(data)
            if data.size == 0 or data.sum() == 0: return 0

            total_instances = 0

            classes_to_check = [class_id] if class_id is not None else np.unique(data)

            for cid in classes_to_check:
                if cid == 0: continue

                mask = (data == cid).astype(np.uint8)
                if mask.sum() == 0: continue

                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                num, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

                for i in range(1, num):
                    if stats[i, cv2.CC_STAT_AREA] >= min_size:
                        total_instances += 1

            return total_instances
        return 0
    @staticmethod
    def calculate_mbss_add(boundary_tolerance=5, **kwargs):
        """MBSS 改进版：相加求平均"""
        num_classes = kwargs.get('num_classes', 2)

        if num_classes <= 2:

            return Metric._compute_single_mbss_add(
                kwargs['gt_binary'], kwargs['pred_binary'],
                kwargs['gt_count'], kwargs['pred_count'],
                boundary_tolerance
            )
        else:

            gt_map, pred_map = kwargs['gt_map'], kwargs['pred_map']
            mbss_scores = []

            for c in range(1, num_classes):
                gt_c = (gt_map == c).astype(np.uint8)
                pred_c = (pred_map == c).astype(np.uint8)


                if gt_c.sum() == 0 and pred_c.sum() == 0:
                    continue


                N_gt_c = Metric._get_instance_count(gt_map, class_id=c)
                N_pred_c = Metric._get_instance_count(pred_map, class_id=c)

                score = Metric._compute_single_mbss_add(gt_c, pred_c, N_gt_c, N_pred_c, boundary_tolerance)
                mbss_scores.append(score)

            return float(np.mean(mbss_scores)) if len(mbss_scores) > 0 else 1.0
    @staticmethod
    def _compute_single_mbss_add(gt_bin, pred_bin, N_gt, N_pred, boundary_tolerance=5):
        """MBSS_ADD 单类别计算逻辑 (IoU + Boundary_Recall + Count_Score) / 3"""

        inter = np.logical_and(gt_bin > 0, pred_bin > 0).sum()
        union = np.logical_or(gt_bin > 0, pred_bin > 0).sum()
        iou = inter / union if union > 0 else 0.0


        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gt_border = cv2.morphologyEx(gt_bin, cv2.MORPH_GRADIENT, kernel)
        pred_border = cv2.morphologyEx(pred_bin, cv2.MORPH_GRADIENT, kernel)

        tol_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*boundary_tolerance+1, 2*boundary_tolerance+1))
        pred_border_dilated = cv2.dilate(pred_border, tol_kernel)

        covered_gt_pixels = np.logical_and(gt_border > 0, pred_border_dilated > 0).sum()
        total_gt_pixels = (gt_border > 0).sum()

        if total_gt_pixels == 0:

            boundary_recall = 1.0 if (pred_border > 0).sum() == 0 else 0.0
        else:
            boundary_recall = covered_gt_pixels / total_gt_pixels


        if max(N_gt, N_pred) == 0:
            count_score = 1.0
        else:
            count_score = np.sqrt(min(N_gt, N_pred) / max(N_gt, N_pred))


        return (iou + boundary_recall + count_score) / 3.0
