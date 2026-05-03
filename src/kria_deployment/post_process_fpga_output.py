"""
Post-process raw FPGA fabric outputs offline (on CPU, without FPGA).

This demonstrates loading the raw layer7 output from FPGA and applying
the exact same post-processing that run_fpga.py does, but completely
on the CPU without needing FPGA hardware.

Usage:
    python post_process_fpga_output.py [input_image_path]
"""

import numpy as np
import os
import cv2
import argparse

CLASS_NAMES = ["person"]
MAX_DETECTIONS = 1
GRID_STRIDE = 16.0
TRAINING_IMAGE_SIZE = 128.0

# Must match run_fpga.py. These constants map the saved raw layer7 values
# back into the logit range used by the detector decoder.
LAYER7_DEQUANT_ZERO_POINT = 88.52
LAYER7_DEQUANT_SCALE = 31.99


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-6)


def decode_and_draw(feature_map, out_img, orig_w, orig_h, score_thresh=0.15):
    """
    Decode final layer7 output (raw from FPGA [0, 127]) into bounding boxes.
    """
    _, grid_h, grid_w = feature_map.shape

    print(f"\n  Detection grid: {grid_h}x{grid_w}")
    print(f"  Layer7 raw output range: [{feature_map.min():.1f}, {feature_map.max():.1f}]")

    if np.all(feature_map == 127):
        print("  ERROR: Layer7 raw output is all 127. The FPGA output is saturated.")
        print("         Check that the bitstream includes OUT_SHIFT wiring and was regenerated.")
        cv2.putText(out_img, "ERROR: saturated FPGA output - no valid decode",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return out_img
    if np.std(feature_map) < 1e-6:
        print("  ERROR: Layer7 raw output is flat; skipping bounding-box decode.")
        cv2.putText(out_img, "ERROR: flat FPGA output - no valid decode",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return out_img

    feature_map_scaled = (feature_map - LAYER7_DEQUANT_ZERO_POINT) / LAYER7_DEQUANT_SCALE

    boxes = []
    for gj in range(grid_h):
        for gi in range(grid_w):
            obj_conf = sigmoid(feature_map_scaled[4, gj, gi])

            cls_logits = feature_map_scaled[5:6, gj, gi]
            cls_probs = softmax(cls_logits)
            class_idx = int(np.argmax(cls_probs))
            class_score = float(cls_probs[class_idx])

            score = obj_conf * class_score
            if score < score_thresh:
                continue

            tx = sigmoid(feature_map_scaled[0, gj, gi])
            ty = sigmoid(feature_map_scaled[1, gj, gi])
            tw = sigmoid(feature_map_scaled[2, gj, gi])
            th = sigmoid(feature_map_scaled[3, gj, gi])

            cx = (gi + tx) * GRID_STRIDE / TRAINING_IMAGE_SIZE * orig_w
            cy = (gj + ty) * GRID_STRIDE / TRAINING_IMAGE_SIZE * orig_h
            bw = tw * orig_w
            bh = th * orig_h

            x1 = int(max(0, cx - bw / 2.0))
            y1 = int(max(0, cy - bh / 2.0))
            x2 = int(min(orig_w, cx + bw / 2.0))
            y2 = int(min(orig_h, cy + bh / 2.0))

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2, score, class_idx])

    # NMS
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes
                 if b[5] != best[5] or compute_iou(best[:4], b[:4]) < 0.4]

    keep = keep[:MAX_DETECTIONS]

    print(f"  Detected {len(keep)} object(s):")
    for b in keep:
        x1, y1, x2, y2, score, c_idx = b
        class_name = CLASS_NAMES[c_idx]
        green = int(255 * score)
        red = int(255 * (1 - score))
        color = (0, green, red)

        print(f"    {class_name:>10}: {score:.3f}  [{x1},{y1},{x2},{y2}]")
        cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {score:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out_img, (x1, y1 - label_size[1] - 6), (x1 + label_size[0], y1), color, -1)
        cv2.putText(out_img, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return out_img


def main():
    parser = argparse.ArgumentParser(description='Post-process raw FPGA outputs')
    parser.add_argument('image', nargs='?', default='/home/ubuntu/cnn_accelerator/person_test.jpg',
                        help='Original image path')
    parser.add_argument('--raw-layer7', type=str, default='/home/ubuntu/cnn_accelerator/fpga_output_raw_layer7.npy',
                        help='Path to raw layer7 output from FPGA')
    parser.add_argument('--weight-dir', type=str, default='/home/ubuntu/cnn_accelerator',
                        help='Directory to save results')
    args = parser.parse_args()

    print("=" * 60)
    print("  Offline FPGA Output Post-Processing (CPU only)")
    print("=" * 60)

    # Load image
    print(f"\n[1/3] Loading image...")
    if not os.path.exists(args.image):
        print(f"  ERROR: Image not found: {args.image}")
        return
    img_bgr = cv2.imread(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]
    print(f"  {args.image}")
    print(f"  Size: {orig_w}x{orig_h}")

    # Load raw layer7 output from FPGA
    print(f"\n[2/3] Loading raw FPGA layer7 output...")
    if not os.path.exists(args.raw_layer7):
        print(f"  ERROR: Raw output not found: {args.raw_layer7}")
        print(f"        Run run_fpga.py first to generate raw outputs")
        return
    feature_map_raw = np.load(args.raw_layer7)
    print(f"  {args.raw_layer7}")
    print(f"  Shape: {feature_map_raw.shape}")
    print(f"  Data type: {feature_map_raw.dtype}")
    print(f"  Range: [{feature_map_raw.min():.1f}, {feature_map_raw.max():.1f}]")

    # Decode and visualize
    print(f"\n[3/3] Decoding detections...")
    out_img = cv2.resize(img_rgb, (orig_w, orig_h))
    out_img = decode_and_draw(feature_map_raw, out_img, orig_w, orig_h)

    # Save result
    cv2.putText(out_img, "FPGA Output (Post-processed via CPU)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    result_path = os.path.join(args.weight_dir, "fpga_output_postprocessed.png")
    cv2.imwrite(result_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    print(f"\n  Result saved: {result_path}")

    # Also display if GUI available
    try:
        cv2.imshow("FPGA Output (Post-processed Offline)", cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
        print("  (Press any key to close)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("  (No display available)")


if __name__ == '__main__':
    main()
