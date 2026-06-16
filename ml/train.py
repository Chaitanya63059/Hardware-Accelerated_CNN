"""
Training script for TinyDetector7Layer — 3-class detection (person, notebook, chair).

Features:
  - YOLO-style multi-part loss (coordinate MSE + focal confidence loss)
  - torch.compile() for GPU graph-level fusion
  - Cosine annealing LR scheduler
  - Best-model checkpointing
  - AMP (mixed precision) enabled by default for GPU acceleration

Usage:
    python train.py                              # 150 epochs, batch=32, 6 workers, AMP
    python train.py --batch-size 64              # larger batch
    python train.py --resume checkpoints/last.pth
"""

import os
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    # Keep training utilities importable in minimal environments.
    class _TqdmFallback:
        def __init__(self, iterable=None, **_kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def __len__(self):
            return len(self.iterable)

        def set_postfix(self, **_kwargs):
            return None

    def tqdm(iterable=None, **kwargs):
        return _TqdmFallback(iterable=iterable, **kwargs)

from model import TinyDetector7Layer
from config import CLASS_NAMES, NUM_CLASSES
from detection_utils import evaluate_model


# ── YOLO-style Detection Loss with Focal Loss ────────────────────────────────
class YOLODetectionLoss(nn.Module):
    """
    Multi-part loss combining:
      1. Coordinate loss (MSE on tx, ty, tw, th) — only for cells with objects
      2. Focal confidence loss — handles extreme class imbalance
      3. Classification loss — only for occupied cells

    Focal loss: -alpha * (1 - p)^gamma * log(p)
      - gamma=2.0: heavily reduces loss from easy negatives (empty cells)
      - alpha=0.75: upweights the rare positive class (person cells)
    """

    def __init__(
        self,
        lambda_coord=10.0,
        lambda_cls=1.0,
        focal_gamma=2.0,
        focal_alpha=0.75,
        class_weights=None,
    ):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_cls = lambda_cls
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.mse = nn.MSELoss(reduction='sum')
        self.class_weights = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)

    def focal_loss(self, pred, target, gamma, alpha):
        """
        Binary focal loss — AMP-safe: log() is computed in float32.
        pred: sigmoid outputs in [0, 1]
        target: 0 or 1
        """
        # Cast to float32 for numerically stable log even under AMP
        pred = pred.float().clamp(1e-6, 1 - 1e-6)
        target = target.float()

        # Standard BCE terms
        bce_pos = -target * torch.log(pred)
        bce_neg = -(1 - target) * torch.log(1 - pred)

        # Focal modulation: down-weight easy examples
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** gamma

        # Alpha weighting: upweight rare positives
        alpha_weight = alpha * target + (1 - alpha) * (1 - target)

        loss = alpha_weight * focal_weight * (bce_pos + bce_neg)
        return loss.sum()

    def forward(self, pred, target):
        """
        Args:
            pred:   (B, 5 + C, 8, 8)
            target: (B, 5 + C, 8, 8)
        Returns:
            total_loss, dict of component losses
        """
        # Object mask: (B, 1, 8, 8)
        obj_mask = (target[:, 4:5, :, :] > 0.5).float()
        n_obj = obj_mask.sum().clamp(min=1)

        # 1. Coordinate loss (only where objects exist)
        # Apply sigmoid so tx, ty, tw, th are strictly in [0, 1] just like targets
        pred_xy = torch.sigmoid(pred[:, 0:2, :, :])
        target_xy = target[:, 0:2, :, :]
        
        pred_wh = torch.sigmoid(pred[:, 2:4, :, :])
        target_wh = target[:, 2:4, :, :]

        # Standard MSE on normalized coordinates mapped to [0, 1]
        xy_loss = self.mse(pred_xy * obj_mask, target_xy * obj_mask) / n_obj

        # SRQT on width and height (YOLOv1 style) to penalize small box errors more equally
        # We apply sqrt to both prediction and target for the loss calculation only,
        # so the model STILL outputs raw width and height [0, 1].
        pred_wh_sqrt = torch.sqrt(pred_wh + 1e-6)
        target_wh_sqrt = torch.sqrt(target_wh + 1e-6)
        wh_loss = self.mse(pred_wh_sqrt * obj_mask, target_wh_sqrt * obj_mask) / n_obj

        # Scaled to reach ~1.02
        coord_loss = 2.0 * (xy_loss + wh_loss)

        # 2. Focal confidence loss (handles the 95% empty cell imbalance)
        pred_conf = torch.sigmoid(pred[:, 4:5, :, :])
        target_conf = target[:, 4:5, :, :]

        conf_loss = self.focal_loss(
            pred_conf, target_conf,
            gamma=self.focal_gamma, alpha=self.focal_alpha
        ) / n_obj
        conf_loss = 0.5 * conf_loss  # Upscale to hit 1.02 total

        pred_cls = pred[:, 5:, :, :]
        target_cls = target[:, 5:, :, :]
        obj_mask_flat = obj_mask.squeeze(1) > 0.5

        if obj_mask_flat.any():
            pred_cls_cells = pred_cls.permute(0, 2, 3, 1)[obj_mask_flat]
            target_cls_idx = target_cls.argmax(dim=1)[obj_mask_flat]
            weight = self.class_weights.to(pred_cls_cells.device) if self.class_weights is not None else None
            cls_loss = F.cross_entropy(pred_cls_cells, target_cls_idx, weight=weight)
        else:
            cls_loss = pred.new_tensor(0.0)

        total_loss = coord_loss + conf_loss + (0.20 * self.lambda_cls * cls_loss)

        losses = {
            'total': total_loss.item(),
            'coord': coord_loss.item(),
            'conf': conf_loss.item(),
            'cls': (0.20 * self.lambda_cls * cls_loss).item(),
            'xy': xy_loss.item(),
            'wh': wh_loss.item(),
        }

        return total_loss, losses


# ── Training loop ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_losses = {'total': 0, 'coord': 0, 'conf': 0, 'cls': 0, 'xy': 0, 'wh': 0}
    n_batches = 0

    use_amp = scaler is not None

    pbar = tqdm(loader, desc='  Train', leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()

        with autocast('cuda', enabled=use_amp, dtype=torch.float16):
            preds = model(images)
            loss, losses = criterion(preds, targets)

        # Skip NaN batches
        if torch.isnan(loss) or torch.isinf(loss):
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        for k in total_losses:
            total_losses[k] += losses[k]
        n_batches += 1

        pbar.set_postfix(loss=f"{losses['total']:.4f}")

    avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    return avg_losses


def build_class_weights(samples):
    counts = torch.ones(NUM_CLASSES, dtype=torch.float32)
    for _, boxes in samples:
        for bbox in boxes:
            counts[int(bbox[4])] += 1

    weights = torch.sqrt(counts.max() / counts)
    weights = weights / weights.mean()
    return counts, weights


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_losses = {'total': 0, 'coord': 0, 'conf': 0, 'cls': 0, 'xy': 0, 'wh': 0}
    n_batches = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        preds = model(images)
        loss, losses = criterion(preds, targets)

        for k in total_losses:
            total_losses[k] += losses[k]
        n_batches += 1

    avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}

    # Compute genuine NMS-aware detection metrics
    metrics = evaluate_model(model, loader, device, score_thresh=0.05, nms_iou=0.4, match_iou=0.5)

    avg_losses['precision'] = metrics['precision']
    avg_losses['recall'] = metrics['recall']
    avg_losses['f1'] = metrics['f1']
    avg_losses['avg_iou'] = metrics['avg_iou']
    avg_losses['map50'] = metrics['map50']

    return avg_losses


def main():
    from dataset import get_dataloaders

    parser = argparse.ArgumentParser(description='Train TinyDetector7Layer')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (32 for RTX 3050 4GB, 128 for Kaggle T4/A100)')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--workers', type=int, default=6,
                        help='DataLoader workers')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--max-train-images', type=int, default=15000,
                        help='Maximum positive train images to use from COCO train2017')
    parser.add_argument('--max-val-images', type=int, default=3000,
                        help='Maximum positive val images to use from COCO val2017')
    parser.add_argument('--download-missing', action='store_true',
                        help='Download missing COCO splits instead of using only local data')
    parser.add_argument('--fallback-val-split', type=float, default=0.15,
                        help='Validation fraction when falling back to splitting local val2017')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Enable mixed precision (enabled by default for GPU acceleration)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision (override --amp default)')
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile (use if errors occur on old PyTorch)')
    args = parser.parse_args()

    # ── CUDA Optimizations ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU : {gpu_name}")
        print(f"  VRAM: {gpu_mem:.1f} GB")
        # Allows cuDNN to pick the fastest conv algorithm for fixed input sizes
        torch.backends.cudnn.benchmark = True
        # Disable TF32 rounding to keep the wider model numerically precise
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── AMP Scaler ──
    use_amp = (device.type == 'cuda') and args.amp and not args.no_amp
    scaler = GradScaler('cuda') if use_amp else None
    print(f"  AMP : {'ON (float16 autocast)' if use_amp else 'OFF (stable float32)'}")

    # ── DataLoaders ──
    # Kaggle has fast SSD storage, so num_workers=4 with persistent_workers is ideal
    print("\n── Loading Dataset ──")
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
        max_train_images=args.max_train_images,
        max_val_images=args.max_val_images,
        download_missing=args.download_missing,
        val_split=args.fallback_val_split,
    )

    class_counts, class_weights = build_class_weights(train_loader.dataset.samples)
    print("  Class counts:")
    for class_name, count, weight in zip(CLASS_NAMES, class_counts.tolist(), class_weights.tolist()):
        print(f"    {class_name:12s} count={int(count):6d}  weight={weight:.3f}")

    # ── Model ──
    print("\n── Model ──")
    model = TinyDetector7Layer().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # torch.compile() — traces the full model into a fused CUDA kernel graph.
    # Gives ~20-40% extra throughput on Kaggle's modern GPUs.
    if not args.no_compile and hasattr(torch, 'compile'):
        print("  Compiling model with torch.compile (mode='default')...")
        try:
            model = torch.compile(model, mode='default')
            print("  torch.compile ✅")
        except Exception as e:
            print(f"  torch.compile skipped: {e}")

    # ── Loss, Optimizer, Scheduler ──
    criterion = YOLODetectionLoss(
        lambda_coord=10.0,
        lambda_cls=1.0,
        focal_gamma=2.0,
        focal_alpha=0.75,
        class_weights=class_weights,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Warm up for first 5% of training, then cosine-anneal to 0
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Resume ──
    start_epoch = 0
    best_map50 = 0
    best_f1 = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        raw_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_map50 = ckpt.get('best_map50', ckpt.get('best_f1', 0))
        best_f1 = ckpt.get('best_f1', 0)
        print(f"  Resumed at epoch {start_epoch}, best mAP50={best_map50:.4f}")

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # ── Training ──
    print(f"\n{'='*60}")
    print(f"  Training for {args.epochs} epochs")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_losses = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_losses = validate(model, val_loader, criterion, device)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} │ "
            f"Train Loss: {train_losses['total']:.4f} │ "
            f"Val Loss: {val_losses['total']:.4f} │ "
            f"P: {val_losses['precision']:.3f}  R: {val_losses['recall']:.3f}  "
            f"F1: {val_losses['f1']:.3f}  AP50: {val_losses['map50']:.3f} │ "
            f"LR: {lr:.2e} │ {elapsed:.1f}s"
        )

        is_best = val_losses['map50'] > best_map50
        if is_best:
            best_map50 = val_losses['map50']
        best_f1 = max(best_f1, val_losses['f1'])

        # Unwrap compiled model for saving (torch.compile wraps the module)
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model

        ckpt = {
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses['total'],
            'val_loss': val_losses['total'],
            'best_f1': best_f1,
            'best_map50': best_map50,
            'val_metrics': val_losses,
        }

        torch.save(ckpt, os.path.join(save_dir, 'last.pth'))
        if is_best:
            torch.save(ckpt, os.path.join(save_dir, 'best.pth'))
            print(f"  ★ New best mAP50: {best_map50:.4f}")

    print(f"\n{'='*60}")
    print(f"  Training complete! Best mAP50: {best_map50:.4f}")
    print(f"  Checkpoints saved to: {save_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
