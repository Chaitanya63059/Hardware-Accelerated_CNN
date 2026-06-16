"""
COCO Multi-Class Dataset — Uses local COCO 2017 images/annotations when
available, filters for a fixed 3-class subset (person, notebook, chair),
and creates training-ready 128×128 grid targets.

Default strategy:
  - Never download missing COCO splits unless explicitly requested
  - Prefer local train2017 for training and local val2017 for validation
  - If train2017 images are absent, fall back to splitting local val2017

Usage:
    python dataset.py              # uses local data, shows sample
    python dataset.py --show 5     # show 5 samples with grid overlay
"""

import os
import json
import zipfile
import urllib.request
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
IMG_SIZE = 128
GRID_SIZE = 8
MAX_TRAIN_IMAGES = 15000
MAX_VAL_IMAGES = None
NEG_RATIO = 0.2

from config import CLASS_NAMES, COCO_CATEGORY_ID_TO_CLASS_IDX, NUM_OUTPUTS

# COCO 2017 URLs
COCO_IMAGE_URLS = {
    'train2017': 'http://images.cocodataset.org/zips/train2017.zip',
    'val2017': 'http://images.cocodataset.org/zips/val2017.zip',
}
COCO_ANN_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
LOCAL_MIN_FILES = {'train2017': 100000, 'val2017': 1000}


def download_file(url, dest_path):
    """Download a file with progress reporting. Skips if already exists."""
    if os.path.exists(dest_path):
        print(f"  Already exists: {os.path.basename(dest_path)}")
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"  Downloading {os.path.basename(dest_path)} ...")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r    {pct}% ({mb:.1f}/{total_mb:.1f} MB)", end='', flush=True)

    urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
    print()


def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    print(f"  Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    print(f"  Done.")


def download_coco_split(split, data_dir=DATA_DIR):
    """Download a COCO 2017 split and matching annotations."""
    if split not in COCO_IMAGE_URLS:
        raise ValueError(f"Unsupported split: {split}")

    os.makedirs(data_dir, exist_ok=True)
    img_dir = os.path.join(data_dir, split)
    ann_file = os.path.join(data_dir, 'annotations', f'instances_{split}.json')
    min_files = LOCAL_MIN_FILES[split]

    # Check if data is already fully present
    if os.path.isdir(img_dir) and len(os.listdir(img_dir)) > min_files and os.path.isfile(ann_file):
        print(f"COCO {split} data already present.")
        return img_dir, ann_file

    print("=" * 60)
    print(f"Downloading COCO 2017 {split}")
    print("=" * 60)

    # Download annotations
    ann_zip = os.path.join(data_dir, 'annotations_trainval2017.zip')
    download_file(COCO_ANN_URL, ann_zip)
    if not os.path.isfile(ann_file):
        extract_zip(ann_zip, data_dir)

    # Download images (single bulk zip — much faster than individual downloads)
    img_zip = os.path.join(data_dir, f'{split}.zip')
    download_file(COCO_IMAGE_URLS[split], img_zip)
    if not os.path.isdir(img_dir) or len(os.listdir(img_dir)) < min_files:
        extract_zip(img_zip, data_dir)

    return img_dir, ann_file


def find_local_coco_split(split, data_dir=DATA_DIR):
    """Return local split paths if a complete split already exists on disk."""
    img_dir = os.path.join(data_dir, split)
    ann_file = os.path.join(data_dir, 'annotations', f'instances_{split}.json')
    min_files = LOCAL_MIN_FILES[split]

    if os.path.isdir(img_dir) and len(os.listdir(img_dir)) > min_files and os.path.isfile(ann_file):
        return img_dir, ann_file
    return None, None

def load_multiclass_annotations(ann_file, img_dir):
    """
    Parse COCO annotations and return all valid samples.
    Returns:
        positive_samples: list of (img_path, list_of_bboxes)
        negative_samples: list of (img_path, [])
    """
    print(f"Loading annotations from {os.path.basename(ann_file)}...")
    with open(ann_file, 'r') as f:
        coco = json.load(f)

    # Build image id → filename mapping
    id_to_info = {}
    for img_info in coco['images']:
        id_to_info[img_info['id']] = {
            'file_name': img_info['file_name'],
            'width': img_info['width'],
            'height': img_info['height'],
        }

    # Collect target bboxes per image
    img_bboxes = {}
    for ann in coco['annotations']:
        class_idx = COCO_CATEGORY_ID_TO_CLASS_IDX.get(ann['category_id'])
        if class_idx is None:
            continue
        if ann.get('iscrowd', 0):
            continue

        img_id = ann['image_id']
        if img_id not in id_to_info:
            continue

        info = id_to_info[img_id]
        x, y, w, h = ann['bbox']  # COCO format: [x_min, y_min, width, height]
        iw, ih = info['width'], info['height']

        # Convert to absolute coordinates for clipping
        x1 = max(0, min(iw, x))
        y1 = max(0, min(ih, y))
        x2 = max(0, min(iw, x + w))
        y2 = max(0, min(ih, y + h))

        # Recompute width and height
        w_clipped = x2 - x1
        h_clipped = y2 - y1

        # Skip tiny/invalid boxes
        if w_clipped / iw < 0.02 or h_clipped / ih < 0.02:
            continue

        # Convert back to normalized center format
        cx = (x1 + w_clipped / 2) / iw
        cy = (y1 + h_clipped / 2) / ih
        nw = w_clipped / iw
        nh = h_clipped / ih

        if img_id not in img_bboxes:
            img_bboxes[img_id] = []
        img_bboxes[img_id].append([cx, cy, nw, nh, class_idx])

    positive_samples = []
    for img_id, bboxes in img_bboxes.items():
        info = id_to_info[img_id]
        img_path = os.path.join(img_dir, info['file_name'])
        if os.path.isfile(img_path):
            positive_samples.append((img_path, bboxes))

    # Add negative samples (images WITHOUT target classes)
    positive_img_ids = set(img_bboxes.keys())
    all_img_ids = set(id_to_info.keys())
    negative_img_ids = list(all_img_ids - positive_img_ids)
    
    negative_samples = []
    for img_id in negative_img_ids:
        info = id_to_info[img_id]
        img_path = os.path.join(img_dir, info['file_name'])
        if os.path.isfile(img_path):
            negative_samples.append((img_path, []))
            
    return positive_samples, negative_samples


def build_train_split(positive_samples, negative_samples, max_images=None, person_img_cap=None):
    """
    Subsample and balance dataset safely for a given split.
    Uses classification focal weights, so no exact duplicate oversampling is done.
    """
    random.seed(42)
    positives = list(positive_samples)
    random.shuffle(positives)
    
    if person_img_cap is not None:
        filtered_positives = []
        person_only_count = 0
        for sample in positives:
            bboxes = sample[1]
            if all(b[4] == 0 for b in bboxes):
                if person_only_count >= person_img_cap:
                    continue
                person_only_count += 1
            filtered_positives.append(sample)
        positives = filtered_positives

    if max_images and len(positives) > max_images:
        positives = positives[:max_images]

    negatives = list(negative_samples)
    random.shuffle(negatives)
    n_neg = int(len(positives) * NEG_RATIO)
    negatives = negatives[:n_neg]
    
    samples = positives + negatives
    random.shuffle(samples)
    
    # Split stats
    n_pos = sum(1 for _, b in samples if len(b) > 0)
    n_neg_final = sum(1 for _, b in samples if len(b) == 0)
    final_class_counts = {name: 0 for name in CLASS_NAMES}
    for _, boxes in samples:
        for bbox in boxes:
            final_class_counts[CLASS_NAMES[int(bbox[4])]] += 1

    print(f"  Positive images: {n_pos}, Negative samples: {n_neg_final}, Total: {len(samples)}")
    for class_name, count in final_class_counts.items():
        print(f"    {class_name}: {count} boxes")
        
    return samples


def build_eval_split(positive_samples, negative_samples, max_images=None):
    """
    Build an evaluation split without class balancing, duplicate oversampling,
    or synthetic negative-ratio adjustments.
    """
    random.seed(42)
    positives = list(positive_samples)
    negatives = list(negative_samples)
    random.shuffle(positives)
    random.shuffle(negatives)

    total_available = len(positives) + len(negatives)
    if max_images is None or max_images >= total_available:
        samples = positives + negatives
    else:
        pos_fraction = len(positives) / max(total_available, 1)
        n_pos = min(len(positives), max(1 if positives else 0, int(round(max_images * pos_fraction))))
        n_neg = min(len(negatives), max_images - n_pos)

        samples = positives[:n_pos] + negatives[:n_neg]
        remainder = max_images - len(samples)
        if remainder > 0:
            extra = positives[n_pos:] + negatives[n_neg:]
            random.shuffle(extra)
            samples.extend(extra[:remainder])

    random.shuffle(samples)

    n_pos = sum(1 for _, b in samples if len(b) > 0)
    n_neg_final = sum(1 for _, b in samples if len(b) == 0)
    final_class_counts = {name: 0 for name in CLASS_NAMES}
    for _, boxes in samples:
        for bbox in boxes:
            final_class_counts[CLASS_NAMES[int(bbox[4])]] += 1

    print(f"  Positive images: {n_pos}, Negative images: {n_neg_final}, Total: {len(samples)}")
    for class_name, count in final_class_counts.items():
        print(f"    {class_name}: {count} boxes")

    return samples


# ── Grid Target Builder ──────────────────────────────────────────────────────
def build_grid_target(bboxes, grid_size=GRID_SIZE, num_outputs=NUM_OUTPUTS, num_classes=None):
    """
    Convert list of normalised bboxes to YOLO-style grid target.

    Args:
        bboxes: list of [cx, cy, w, h, class_idx] in [0, 1]

    Returns:
        target: (num_outputs, grid_size, grid_size) tensor
                channels: [tx, ty, tw, th, conf, one_hot_class...]
    """
    if num_classes is None:
        from config import NUM_CLASSES
        num_classes = NUM_CLASSES

    target = torch.zeros(grid_size, grid_size, num_outputs)

    for bbox in bboxes:
        if len(bbox) == 4:
            cx, cy, w, h = bbox
            class_idx = 0
        else:
            cx, cy, w, h, class_idx = bbox[0], bbox[1], bbox[2], bbox[3], int(bbox[4])

        gi = min(int(cx * grid_size), grid_size - 1)
        gj = min(int(cy * grid_size), grid_size - 1)

        tx = cx * grid_size - gi
        ty = cy * grid_size - gj

        # Only assign if cell is empty (first object wins)
        if target[gj, gi, 4] == 0:
            target[gj, gi, 0] = tx
            target[gj, gi, 1] = ty
            target[gj, gi, 2] = w
            target[gj, gi, 3] = h
            target[gj, gi, 4] = 1.0  # confidence
            # One-hot class vector
            target[gj, gi, 5:] = 0
            target[gj, gi, 5 + class_idx] = 1.0

    return target


# ── Dataset ──────────────────────────────────────────────────────────────────
class COCOMultiClassDataset(Dataset):
    """
    PyTorch Dataset for COCO multi-class detection.

    Returns:
        image: (3, 128, 128) float tensor, normalised
        target: (NUM_OUTPUTS, 8, 8) grid target tensor
    """

    def __init__(self, samples, img_size=IMG_SIZE, augment=True):
        self.samples = samples
        self.img_size = img_size
        self.augment = augment

        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['labels'],
                min_visibility=0.3,
            ))
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['labels'],
                min_visibility=0.3,
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, bboxes = self.samples[idx]

        img = np.array(Image.open(img_path).convert('RGB'))

        valid_bboxes = []
        class_labels = []
        for b in bboxes:
            cx, cy, w, h = b[0], b[1], b[2], b[3]
            class_idx = int(b[4]) if len(b) > 4 else 0

            cx = max(0.001, min(0.999, cx))
            cy = max(0.001, min(0.999, cy))
            w = max(0.001, min(0.999, w))
            h = max(0.001, min(0.999, h))
            if cx - w / 2 < 0:
                w = cx * 2
            if cy - h / 2 < 0:
                h = cy * 2
            if cx + w / 2 > 1:
                w = (1 - cx) * 2
            if cy + h / 2 > 1:
                h = (1 - cy) * 2
            if w > 0.001 and h > 0.001:
                valid_bboxes.append([cx, cy, w, h])
                class_labels.append(class_idx)

        try:
            transformed = self.transform(
                image=img,
                bboxes=valid_bboxes,
                labels=class_labels,
            )
            image = transformed['image']
            out_bboxes = transformed['bboxes']
            out_labels = transformed['labels']
        except Exception:
            image = self.transform(image=img, bboxes=[], labels=[])['image']
            out_bboxes = valid_bboxes
            out_labels = class_labels

        # Reconstruct [cx, cy, w, h, class_idx] tuples
        full_bboxes = []
        for box, lbl in zip(out_bboxes, out_labels):
            full_bboxes.append([box[0], box[1], box[2], box[3], lbl])

        # Build grid target → (NUM_OUTPUTS, 8, 8)
        target = build_grid_target(full_bboxes)
        target = target.permute(2, 0, 1)

        return image, target


def get_dataloaders(
    data_dir=DATA_DIR,
    batch_size=16,
    num_workers=4,
    max_train_images=MAX_TRAIN_IMAGES,
    max_val_images=MAX_VAL_IMAGES,
    download_missing=False,
    val_split=0.15,
):
    train_img_dir, train_ann_file = find_local_coco_split('train2017', data_dir)
    val_img_dir, val_ann_file = find_local_coco_split('val2017', data_dir)

    if train_img_dir is None and download_missing:
        train_img_dir, train_ann_file = download_coco_split('train2017', data_dir)
    if val_img_dir is None and download_missing:
        val_img_dir, val_ann_file = download_coco_split('val2017', data_dir)

    if train_img_dir is not None and val_img_dir is not None:
        print("Using local train2017 for training and local val2017 for validation.")
        t_pos, t_neg = load_multiclass_annotations(train_ann_file, train_img_dir)
        v_pos, v_neg = load_multiclass_annotations(val_ann_file, val_img_dir)
        
        print("\n--- Train Split ---")
        train_samples = build_train_split(t_pos, t_neg, max_images=max_train_images, person_img_cap=40000)
        print("\n--- Val Split ---")
        val_samples = build_eval_split(v_pos, v_neg, max_images=max_val_images)
    elif val_img_dir is not None:
        print("train2017 images are not available locally. Falling back to a train/val split from local val2017 only.")
        pos_samples, neg_samples = load_multiclass_annotations(val_ann_file, val_img_dir)
        
        # Split purely by images BEFORE adding negatives or applying caps
        random.seed(42)
        random.shuffle(pos_samples)
        random.shuffle(neg_samples)
        
        n_val_pos = max(1, int(len(pos_samples) * val_split))
        n_val_neg = max(1, int(len(neg_samples) * val_split))
        
        t_pos, v_pos = pos_samples[:-n_val_pos], pos_samples[-n_val_pos:]
        t_neg, v_neg = neg_samples[:-n_val_neg], neg_samples[-n_val_neg:]
        
        print("\n--- Train Split ---")
        train_samples = build_train_split(t_pos, t_neg, max_images=None, person_img_cap=20000)
        print("\n--- Val Split ---")
        val_samples = build_eval_split(v_pos, v_neg, max_images=max_val_images)
    else:
        raise FileNotFoundError(
            "No usable local COCO split found. Put val2017 in data/ or rerun with download_missing=True."
        )

    print(f"\n  Final Train: {len(train_samples)} images")
    print(f"  Final Val:   {len(val_samples)} images")

    train_ds = COCOMultiClassDataset(train_samples, augment=True)
    val_ds = COCOMultiClassDataset(val_samples, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader



# ─── Self-test ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    parser = argparse.ArgumentParser()
    parser.add_argument('--show', type=int, default=3, help='Number of samples to visualize')
    args = parser.parse_args()

    train_loader, val_loader = get_dataloaders(batch_size=4)
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    # Show some samples
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    images, targets = next(iter(train_loader))
    n_show = min(args.show, len(images))

    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 5))
    if n_show == 1:
        axes = [axes]

    for i in range(n_show):
        img = images[i].permute(1, 2, 0).numpy()
        img = img * STD + MEAN
        img = np.clip(img, 0, 1)

        target = targets[i]

        ax = axes[i]
        ax.imshow(img)
        ax.set_title(f"Sample {i}")

        cell_w = IMG_SIZE / GRID_SIZE
        for gi in range(GRID_SIZE):
            for gj in range(GRID_SIZE):
                conf = target[4, gj, gi].item()
                if conf > 0.5:
                    class_idx = int(torch.argmax(target[5:, gj, gi]).item())
                    tx = target[0, gj, gi].item()
                    ty = target[1, gj, gi].item()
                    tw = target[2, gj, gi].item()
                    th = target[3, gj, gi].item()

                    cx = (gi + tx) / GRID_SIZE * IMG_SIZE
                    cy = (gj + ty) / GRID_SIZE * IMG_SIZE
                    bw = tw * IMG_SIZE
                    bh = th * IMG_SIZE

                    from matplotlib.patches import Rectangle
                    rect = Rectangle(
                        (cx - bw/2, cy - bh/2), bw, bh,
                        linewidth=2, edgecolor='lime', facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.plot(cx, cy, 'r+', markersize=10)
                    ax.text(cx, max(0, cy - 4), CLASS_NAMES[class_idx], color='yellow', fontsize=8)

        for g in range(GRID_SIZE + 1):
            ax.axhline(g * cell_w, color='white', alpha=0.3, linewidth=0.5)
            ax.axvline(g * cell_w, color='white', alpha=0.3, linewidth=0.5)

        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_grid.png'), dpi=100)
    print(f"\nSaved sample visualization to sample_grid.png")
    plt.show()
