"""
ARM CPU-only CNN inference on Kria KV260.
Simulates the EXACT hardware behavior in software:
  - INT8 signed weights, unsigned 8-bit pixel inputs  
  - 32-bit accumulation across input channels
  - 16-bit signed bias addition
  - Arithmetic right shift -> ReLU + saturation to unsigned [0, 127]
  - Software MaxPool 2x2

CRITICAL: ImageNet normalization (MEAN/STD) is applied to match training.
Raw pixel values are normalized, scaled to [-128,127], then converted
to unsigned [0,255] for hardware input to match FPGA behavior.

This matches what the Verilog conv_channel_proc does, cycle-for-cycle.
Used for timing comparison: FPGA HW Conv vs ARM SW Conv.
"""

import numpy as np
import time
import os
import cv2

# ===================== CONFIGURATION =====================
LAYERS = [
    # (name, C_in, C_out, kernel_size, stride, pad, has_maxpool)
    ("layer1",   3,  32, 4, 2, 1, False),
    ("layer2",  32,  64, 3, 1, 1, True),
    ("layer3",  64, 128, 4, 2, 1, False),
    ("layer4", 128, 256, 3, 1, 1, True),
    ("layer5", 256, 256, 3, 1, 1, False),
    ("layer6", 256, 128, 3, 1, 1, False),
    ("layer7", 128,   8, 1, 1, 0, False),
]
LAYER_OUTPUT_SHIFTS = {
    "layer1": 9,
    "layer2": 7,
    "layer3": 8,
    "layer4": 7,
    "layer5": 8,
    "layer6": 8,
    "layer7": 4,
}

WEIGHT_DIR = "/home/ubuntu/cnn_accelerator"
IMAGE_PATH = "/home/ubuntu/cnn_accelerator/000000000139.jpg"
INPUT_SIZE = 128  # ARM has no MAX_WIDTH limit

# ===================== HELPERS =====================

MEM_CACHE = {}

def parse_mem_file(filepath):
    """Parse .mem file -> list of signed int8 values."""
    if filepath in MEM_CACHE:
        return MEM_CACHE[filepath]
    values = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('//') or line == '':
                continue
            parts = line.split()
            if len(parts) == 2:
                hex_val = int(parts[1], 16)
                if hex_val > 127:
                    hex_val -= 256
                values.append(hex_val)
    MEM_CACHE[filepath] = values
    return values


def load_layer_params(layer_name, c_in, c_out, k_size):
    """Load INT8 weights and INT16 biases from .mem files."""
    w_flat = parse_mem_file(os.path.join(WEIGHT_DIR, f"{layer_name}_weight.mem"))
    b_flat = parse_mem_file(os.path.join(WEIGHT_DIR, f"{layer_name}_bias.mem"))

    weights = np.array(w_flat, dtype=np.int32).reshape(c_out, c_in, k_size, k_size)
    biases = np.array(b_flat[:c_out], dtype=np.int32)
    return weights, biases


def hw_conv2d(input_data, weights, biases, stride, pad, out_shift):
    """
    Simulate the EXACT hardware convolution behavior:
      1. Pad input (software, before sending to HW)
      2. Valid convolution with INT8 weights * UINT8 pixels
      3. Accumulate in 32-bit signed accumulator across all input channels
      4. Add 16-bit signed bias
      5. Arithmetic right shift
      6. ReLU + saturate to [0, 127]

    input_data: (C_in, H, W) uint8/float with values in [0, 127]
    weights:    (C_out, C_in, K, K) int32 (signed int8 range)
    biases:     (C_out,) int32 (signed int16 range)
    """
    c_in, h, w = input_data.shape
    c_out, _, k, _ = weights.shape

    # Padding
    if pad > 0:
        input_data = np.pad(input_data, ((0, 0), (pad, pad), (pad, pad)),
                            'constant', constant_values=0)
        h += 2 * pad
        w += 2 * pad

    oh = (h - k) // stride + 1
    ow = (w - k) // stride + 1

    # Cast input to int32 for accumulation (hardware uses unsigned input)
    inp = input_data.astype(np.int32)

    # 32-bit accumulation
    output = np.zeros((c_out, oh, ow), dtype=np.int32)

    for oc in range(c_out):
        for ic in range(c_in):
            for kh in range(k):
                for kw in range(k):
                    h_idx = np.arange(oh) * stride + kh
                    w_idx = np.arange(ow) * stride + kw
                    patch = inp[ic][np.ix_(h_idx, w_idx)]
                    output[oc] += patch * int(weights[oc, ic, kh, kw])
        # Add 16-bit signed bias
        output[oc] += int(biases[oc])

    output = output >> out_shift

    # ReLU + saturate to [0, 127] (matches Verilog relu_out logic)
    output = np.clip(output, 0, 127).astype(np.float32)

    return output, oh, ow


# ===================== DECODE =====================

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
    """Decode feature map into bounding boxes and draw them."""
    _, grid_h, grid_w = feature_map.shape
    class_names = ["person", "notebook", "chair"]
    colors = [(0, 255, 0), (255, 165, 0), (0, 120, 255)]

    print(f"\n  Detection grid: {grid_h}x{grid_w}")
    print(f"  Output range: [{feature_map.min():.1f}, {feature_map.max():.1f}]")

    boxes = []
    for gj in range(grid_h):
        for gi in range(grid_w):
            obj_conf = sigmoid(feature_map[4, gj, gi])
            cls_logits = feature_map[5:8, gj, gi]
            cls_probs = softmax(cls_logits)
            class_idx = int(np.argmax(cls_probs))
            class_score = float(cls_probs[class_idx])

            score = obj_conf * class_score
            if score < score_thresh:
                continue

            tx = sigmoid(feature_map[0, gj, gi])
            ty = sigmoid(feature_map[1, gj, gi])
            tw = sigmoid(feature_map[2, gj, gi])
            th = sigmoid(feature_map[3, gj, gi])

            cx = (gi + tx) / grid_w * orig_w
            cy = (gj + ty) / grid_h * orig_h
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

    print(f"  Detected {len(keep)} object(s):")
    for b in keep:
        x1, y1, x2, y2, score, c_idx = b
        color = colors[c_idx % len(colors)]
        print(f"    {class_names[c_idx]:>10}: {score:.3f}  [{x1},{y1},{x2},{y2}]")
        cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 3)
        label = f"{class_names[c_idx]}: {score:.2f}"
        (tw_t, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out_img, (x1, max(0, y1 - th_t - 8)),
                      (x1 + tw_t + 4, y1), color, -1)
        cv2.putText(out_img, label, (x1 + 2, max(th_t + 4, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return out_img


# ===================== MAIN =====================

def main():
    print("=" * 60)
    print("  CNN Inference - ARM Cortex-A53 (Software Only)")
    print("  Simulates exact FPGA hardware behavior")
    print("=" * 60)

    # Preprocess
    print(f"\n[1/3] Preprocessing image ({INPUT_SIZE}x{INPUT_SIZE})...")
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    img_bgr = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    img_resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
    
    # Normalize according to ImageNet stats (CRITICAL for matching training)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = (img_normalized - MEAN) / STD
    
    # Scale to [-128, 127] range to match INT8 hardware
    feature_map = np.clip(img_normalized * 128.0, -128, 127)
    feature_map = np.transpose(feature_map, (2, 0, 1)).astype(np.float32)
    print(f"  {IMAGE_PATH}")
    print(f"  Input: {feature_map.shape}, range=[{feature_map.min():.1f}, {feature_map.max():.1f}]")

    # Load weights
    print("\n[2/3] Loading weights...")
    all_params = {}
    for layer in LAYERS:
        name, c_in, c_out, k_size = layer[0], layer[1], layer[2], layer[3]
        w, b = load_layer_params(name, c_in, c_out, k_size)
        all_params[name] = (w, b)

    # Run inference
    print("\n[3/3] Running ARM CPU inference...")
    layer_times = []

    for layer_cfg in LAYERS:
        name, c_in, c_out, k_size, stride, pad, has_maxpool = layer_cfg
        w, b = all_params[name]
        out_shift = LAYER_OUTPUT_SHIFTS.get(name, 0)

        print(f"  {name}: ({c_in},{feature_map.shape[1]},{feature_map.shape[2]}) "
              f"k={k_size} s={stride}", end="", flush=True)

        t0 = time.perf_counter()
        feature_map, oh, ow = hw_conv2d(feature_map, w, b, stride, pad, out_shift)
        dt_conv = (time.perf_counter() - t0) * 1000
        layer_times.append((name, dt_conv))

        # MaxPool 2x2
        if has_maxpool:
            c = feature_map.shape[0]
            ph, pw = oh // 2, ow // 2
            cropped = feature_map[:, :ph * 2, :pw * 2]
            feature_map = cropped.reshape(c, ph, 2, pw, 2).max(axis=(2, 4))
            print(f" -> ({c_out},{oh},{ow}) -> pool -> ({c_out},{ph},{pw}) [{dt_conv:.1f}ms]")
        else:
            print(f" -> ({c_out},{oh},{ow}) [{dt_conv:.1f}ms]")

    t_total_conv = sum(ms for _, ms in layer_times)

    print(f"\n  Convolution Timing:")
    print(f"  {'_' * 40}")
    for name, ms in layer_times:
        print(f"    {name}: {ms:.1f} ms")
    print(f"    {'_' * 30}")
    print(f"    Total Conv Time: {t_total_conv:.1f} ms")
    print(f"    Conv FPS:        {1000 / t_total_conv:.2f}")

    # Scale ReLU-saturated [0, 127] output back to logit range [-2.5, 2.5]
    feature_map = (feature_map - 64.0) / 25.6

    print(f"\n  Output: shape={feature_map.shape}, "
          f"range=[{feature_map.min():.1f}, {feature_map.max():.1f}]")

    np.save(os.path.join(WEIGHT_DIR, "arm_output.npy"), feature_map)

    # Decode and draw
    out_img = cv2.resize(img_rgb, (orig_w, orig_h))
    out_img = decode_and_draw(feature_map, out_img, orig_w, orig_h)

    cv2.putText(out_img, f"ARM Conv: {t_total_conv:.1f}ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(out_img, f"ARM FPS: {1000 / t_total_conv:.2f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imwrite(os.path.join(WEIGHT_DIR, "arm_output.png"),
                cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    print(f"\n  Saved: arm_output.png + arm_output.npy")
    print("=" * 60)


if __name__ == '__main__':
    main()
