import numpy as np
from mmengine.logging import MMLogger
from mmdet3d.registry import METRICS
from mmdet3d.evaluation.metrics import SegMetric


@METRICS.register_module()
class InstanceOnlyMetric(SegMetric):
    """Lightweight instance-only metric for point masks.

    Computes precision/recall and mean IoU using a greedy one-to-one match
    on mask IoU with a fixed threshold.
    """

    def __init__(self, iou_thr=0.5, logger_keys=('inst_precision', 'inst_recall', 'inst_miou'), **kwargs):
        self.iou_thr = iou_thr
        self.logger_keys = logger_keys
        super().__init__(**kwargs)

    @staticmethod
    def _to_mask_array(mask):
        if isinstance(mask, np.ndarray):
            return mask
        return np.asarray(mask)

    def compute_metrics(self, results):
        logger: MMLogger = MMLogger.get_current_instance()

        total_tp = 0
        total_pred = 0
        total_gt = 0
        matched_ious = []

        for eval_ann, pred in results:
            # GT instance ids per point
            gt_ids = eval_ann['pts_instance_mask']
            gt_ids = np.asarray(gt_ids)
            gt_valid = gt_ids >= 0
            gt_instance_ids = np.unique(gt_ids[gt_valid])
            gt_masks = [(gt_ids == gid) for gid in gt_instance_ids]

            # Pred masks: expect (num_pred, num_points) bool or list of masks
            pred_masks = pred.get('pts_instance_mask', None)
            if pred_masks is None:
                continue
            if isinstance(pred_masks, list):
                # For some outputs it is a list; use first element if nested
                pred_masks = pred_masks[0]
            pred_masks = np.asarray(pred_masks)
            if pred_masks.ndim == 1:
                pred_masks = pred_masks[None, :]

            num_pred = pred_masks.shape[0]
            num_gt = len(gt_masks)

            total_pred += num_pred
            total_gt += num_gt

            if num_pred == 0 or num_gt == 0:
                continue

            # Compute IoU matrix
            iou_mat = np.zeros((num_pred, num_gt), dtype=np.float32)
            for i in range(num_pred):
                pred_mask = pred_masks[i].astype(bool)
                pred_sum = pred_mask.sum()
                if pred_sum == 0:
                    continue
                for j in range(num_gt):
                    gt_mask = gt_masks[j]
                    inter = np.logical_and(pred_mask, gt_mask).sum()
                    union = pred_sum + gt_mask.sum() - inter
                    if union > 0:
                        iou_mat[i, j] = inter / union

            # Greedy one-to-one matching
            used_pred = set()
            used_gt = set()
            flat = [(iou_mat[i, j], i, j) for i in range(num_pred) for j in range(num_gt)]
            flat.sort(reverse=True, key=lambda x: x[0])
            for iou, i, j in flat:
                if iou < self.iou_thr:
                    break
                if i in used_pred or j in used_gt:
                    continue
                used_pred.add(i)
                used_gt.add(j)
                total_tp += 1
                matched_ious.append(iou)

        precision = total_tp / total_pred if total_pred > 0 else 0.0
        recall = total_tp / total_gt if total_gt > 0 else 0.0
        miou = float(np.mean(matched_ious)) if matched_ious else 0.0

        metrics = {
            'inst_precision': precision,
            'inst_recall': recall,
            'inst_miou': miou,
        }
        logger.info(f"Instance-only metrics: precision={precision:.4f}, recall={recall:.4f}, miou={miou:.4f}")
        return metrics
