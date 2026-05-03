"""
Shared decoding, NMS, and detection metric utilities.
"""

import numpy as np
import torch

from config import CLASS_NAMES, NUM_CLASSES


def activate_output(pred):
    """
    Convert raw model logits to bounded box/objectness predictions.
    Class channels remain as logits.
    """
    activated = pred.clone()
    activated[0:5, :, :] = torch.sigmoid(activated[0:5, :, :])
    return activated


def decode_predictions(
    pred,
    score_thresh=0.3,
    is_target=False,
    normalized=False,
    image_size=128,
    grid_size=8,
):
    """
    Decode model output to boxes.

    Returns:
        list of [x1, y1, x2, y2, score, class_idx, gi, gj]
    """
    if not is_target:
        pred = activate_output(pred)
    else:
        pred = pred.clone()

    scale = 1.0 if normalized else float(image_size)
    boxes = []

    for gj in range(grid_size):
        for gi in range(grid_size):
            obj_conf = float(pred[4, gj, gi].item())
            cls_logits = pred[5:, gj, gi]

            if is_target:
                if obj_conf <= 0.0:
                    continue
                class_idx = int(torch.argmax(cls_logits).item())
                class_score = float(cls_logits[class_idx].item())
            else:
                cls_probs = torch.softmax(cls_logits, dim=0)
                class_idx = int(torch.argmax(cls_probs).item())
                class_score = float(cls_probs[class_idx].item())

            score = obj_conf * class_score
            if score < score_thresh:
                continue

            tx = float(pred[0, gj, gi].item())
            ty = float(pred[1, gj, gi].item())
            tw = max(0.0, float(pred[2, gj, gi].item()))
            th = max(0.0, float(pred[3, gj, gi].item()))

            cx = (gi + tx) / grid_size * scale
            cy = (gj + ty) / grid_size * scale
            bw = tw * scale
            bh = th * scale

            x1 = max(0.0, cx - bw / 2.0)
            y1 = max(0.0, cy - bh / 2.0)
            x2 = min(scale, cx + bw / 2.0)
            y2 = min(scale, cy + bh / 2.0)

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2, score, class_idx, gi, gj])

    return boxes


def compute_iou(box1, box2):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / max(union, 1e-6)


def nms(boxes, iou_thresh=0.4):
    """Class-aware non-maximum suppression."""
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []

    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        remaining = []
        for box in boxes:
            if best[5] != box[5] or compute_iou(best[:4], box[:4]) < iou_thresh:
                remaining.append(box)
        boxes = remaining

    return keep


def init_metric_state(num_classes=NUM_CLASSES):
    return {
        'scores': [[] for _ in range(num_classes)],
        'tp_flags': [[] for _ in range(num_classes)],
        'fp_flags': [[] for _ in range(num_classes)],
        'gt_counts': [0 for _ in range(num_classes)],
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'ious': [],
    }


def update_metric_state(state, pred_boxes, gt_boxes, iou_thresh=0.5):
    """Accumulate dataset-wide detection statistics."""
    for gt in gt_boxes:
        state['gt_counts'][int(gt[5])] += 1

    pred_boxes = sorted(pred_boxes, key=lambda b: b[4], reverse=True)
    matched_gt = set()

    for pred in pred_boxes:
        pred_class = int(pred[5])
        state['scores'][pred_class].append(float(pred[4]))

        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt or int(gt[5]) != pred_class:
                continue

            iou = compute_iou(pred[:4], gt[:4])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_thresh and best_gt_idx >= 0:
            state['tp_flags'][pred_class].append(1)
            state['fp_flags'][pred_class].append(0)
            state['tp'] += 1
            state['ious'].append(best_iou)
            matched_gt.add(best_gt_idx)
        else:
            state['tp_flags'][pred_class].append(0)
            state['fp_flags'][pred_class].append(1)
            state['fp'] += 1

    state['fn'] += len(gt_boxes) - len(matched_gt)


def _compute_ap(scores, tp_flags, fp_flags, gt_count):
    if gt_count == 0:
        return None
    if not scores:
        return 0.0

    order = np.argsort(-np.asarray(scores, dtype=np.float32))
    tp = np.asarray(tp_flags, dtype=np.float32)[order]
    fp = np.asarray(fp_flags, dtype=np.float32)[order]

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / max(gt_count, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for idx in range(len(mpre) - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])

    change_points = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[change_points + 1] - mrec[change_points]) * mpre[change_points + 1])
    return float(ap)


def finalize_metric_state(state):
    """Turn accumulated state into precision/recall/F1/mAP50 metrics."""
    ap50_per_class = {}
    valid_aps = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        ap = _compute_ap(
            state['scores'][class_idx],
            state['tp_flags'][class_idx],
            state['fp_flags'][class_idx],
            state['gt_counts'][class_idx],
        )
        if ap is not None:
            ap50_per_class[class_name] = ap
            valid_aps.append(ap)

    precision = state['tp'] / max(state['tp'] + state['fp'], 1)
    recall = state['tp'] / max(state['tp'] + state['fn'], 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    avg_iou = float(np.mean(state['ious'])) if state['ious'] else 0.0
    map50 = float(np.mean(valid_aps)) if valid_aps else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_iou': avg_iou,
        'map50': map50,
        'tp': state['tp'],
        'fp': state['fp'],
        'fn': state['fn'],
        'ap50_per_class': ap50_per_class,
    }


@torch.no_grad()
def evaluate_model(model, loader, device, score_thresh=0.05, nms_iou=0.4, match_iou=0.5, normalized=False, image_size=128, grid_size=8):
    """Evaluate a model with IoU-aware matching and AP50."""
    model.eval()
    state = init_metric_state()

    for images, targets in loader:
        images = images.to(device)
        preds = model(images)

        for batch_idx in range(images.shape[0]):
            pred_boxes = decode_predictions(
                preds[batch_idx].cpu(),
                score_thresh=score_thresh,
                normalized=normalized,
                image_size=image_size,
                grid_size=grid_size,
            )
            pred_boxes = nms(pred_boxes, iou_thresh=nms_iou)

            gt_boxes = decode_predictions(
                targets[batch_idx].cpu(),
                score_thresh=0.5,
                is_target=True,
                normalized=normalized,
                image_size=image_size,
                grid_size=grid_size,
            )
            update_metric_state(state, pred_boxes, gt_boxes, iou_thresh=match_iou)

    return finalize_metric_state(state)
