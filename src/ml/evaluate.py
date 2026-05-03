"""
Evaluation & visualization for TinyDetector7Layer.

Usage:
    python evaluate.py                               # evaluate best.pth
    python evaluate.py --checkpoint checkpoints/last.pth
    python evaluate.py --visualize --num-vis 10       # save detection images
"""

import os
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from model import TinyDetector7Layer
from dataset import get_dataloaders, IMG_SIZE, GRID_SIZE
from config import CLASS_NAMES
from detection_utils import decode_predictions, nms, evaluate_model


MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def decode_grid(pred, conf_thresh=0.3, is_target=False):
    return decode_predictions(
        pred,
        score_thresh=conf_thresh,
        is_target=is_target,
        normalized=False,
        image_size=IMG_SIZE,
        grid_size=GRID_SIZE,
    )


def evaluate_map(model, loader, device, conf_thresh=0.3, iou_thresh=0.5):
    """Compute IoU-aware detection metrics, including mAP@0.5."""
    return evaluate_model(
        model,
        loader,
        device,
        score_thresh=conf_thresh,
        nms_iou=0.4,
        match_iou=iou_thresh,
        normalized=False,
        image_size=IMG_SIZE,
        grid_size=GRID_SIZE,
    )


def visualize_detections(model, loader, device, save_dir, num_images=10, conf_thresh=0.3):
    """Save images with predicted bounding boxes overlaid."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    count = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            preds = model(images)

            for i in range(images.shape[0]):
                if count >= num_images:
                    return

                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = img * STD + MEAN
                img = np.clip(img, 0, 1)

                pred_boxes = decode_grid(preds[i].cpu(), conf_thresh)
                pred_boxes = nms(pred_boxes)
                gt_boxes = decode_grid(targets[i].cpu(), conf_thresh=0.5, is_target=True)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

                # Ground truth
                ax1.imshow(img)
                ax1.set_title('Ground Truth')
                for box in gt_boxes:
                    rect = patches.Rectangle(
                        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                        linewidth=2, edgecolor='lime', facecolor='none'
                    )
                    ax1.add_patch(rect)
                    ax1.text(box[0], max(0, box[1] - 1), CLASS_NAMES[box[5]], color='lime', fontsize=8)
                ax1.axis('off')

                # Predictions
                ax2.imshow(img)
                ax2.set_title('Predictions')
                for box in pred_boxes:
                    rect = patches.Rectangle(
                        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    ax2.add_patch(rect)
                    ax2.text(
                        box[0], box[1] - 1, f'{CLASS_NAMES[box[5]]} {box[4]:.2f}',
                        color='red', fontsize=8, weight='bold',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7)
                    )
                ax2.axis('off')

                plt.tight_layout()
                path = os.path.join(save_dir, f'detection_{count:03d}.png')
                plt.savefig(path, dpi=100)
                plt.close()
                count += 1

    print(f"Saved {count} detection images to {save_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Evaluate TinyDetector7Layer')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--conf-thresh', type=float, default=0.25)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--num-vis', type=int, default=10)
    parser.add_argument('--vis-dir', type=str, default='vis_results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    model = TinyDetector7Layer().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    best_map50 = ckpt.get('best_map50', 'N/A')
    print(f"  Epoch: {ckpt['epoch']+1}, Best mAP50 from ckpt: {best_map50}")

    # Data
    _, val_loader = get_dataloaders(batch_size=args.batch_size, num_workers=2)

    # Evaluate
    print("\n── Evaluation ──")
    metrics = evaluate_map(model, val_loader, device, conf_thresh=args.conf_thresh)
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1 Score:   {metrics['f1']:.4f}")
    print(f"  mAP@0.5:    {metrics['map50']:.4f}")
    print(f"  Avg IoU:    {metrics['avg_iou']:.4f}")
    print(f"  TP: {metrics['tp']}  FP: {metrics['fp']}  FN: {metrics['fn']}")

    # Visualize
    if args.visualize:
        vis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.vis_dir)
        print(f"\n── Saving {args.num_vis} visualization images ──")
        visualize_detections(model, val_loader, device, vis_dir, args.num_vis, args.conf_thresh)

    print("\nDone!")


if __name__ == '__main__':
    main()
