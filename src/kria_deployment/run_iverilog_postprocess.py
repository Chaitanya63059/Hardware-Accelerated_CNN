"""
Run the patched Verilog accelerator with iverilog on one image, then decode
the simulated layer7 output into a post-processed detection image.
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = REPO_ROOT / "kria_deployment"
DEFAULT_IMAGE = DEPLOY_DIR / "person_test.jpg"
WEIGHT_DIR = DEPLOY_DIR / "exported_weights"
VERILOG_DIR = REPO_ROOT / "cnn_accel_16parallel" / "src"
BEHAVIORAL_TB_PATH = DEPLOY_DIR / "tb_behavioral_conv.v"
SIM_DIR = DEPLOY_DIR / "iverilog_sim"

LAYERS = [
    ("layer1", 3, 32, 4, 2, 1, False),
    ("layer2", 32, 64, 3, 1, 1, True),
    ("layer3", 64, 128, 4, 2, 1, False),
    ("layer4", 128, 256, 3, 1, 1, True),
    ("layer5", 256, 256, 3, 1, 1, False),
    ("layer6", 256, 128, 3, 1, 1, False),
    ("layer7", 128, 6, 1, 1, 0, False),
]

KERNEL_MODE = {1: 0, 3: 1, 4: 2}
WEIGHT_SLOT_TO_COMPACT_IDX = {
    1: [0] + [None] * 15,
    3: [0, 1, 2, None, 3, 4, 5, None, 6, 7, 8, None, None, None, None, None],
    4: list(range(16)),
}

INPUT_SIZE = 126
GRID_STRIDE = 16.0
TRAINING_IMAGE_SIZE = 128.0
LAYER7_DEQUANT_ZERO_POINT = 88.52
LAYER7_DEQUANT_SCALE = 31.99


def to_hex(value, bits):
    value = int(value)
    if bits == 8:
        return f"{value & 0xFF:02X}"
    if bits == 16:
        return f"{value & 0xFFFF:04X}"
    raise ValueError(bits)


def parse_mem_file(path, bits):
    sign_bit = 1 << (bits - 1)
    full_range = 1 << bits
    values = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            value = int(parts[1], 16)
            if value >= sign_bit:
                value -= full_range
            values.append(value)
    return np.array(values, dtype=np.int32)


def hw_output_dims(in_h, in_w, k_size, stride):
    if k_size == 1:
        return ((in_h >> 1) if stride == 2 else in_h,
                (in_w >> 1) if stride == 2 else in_w)
    if k_size == 3:
        return (((in_h - 2) >> 1) if stride == 2 else (in_h - 2),
                ((in_w - 2) >> 1) if stride == 2 else (in_w - 2))
    if k_size == 4:
        return (((in_h - 3) >> 1) if stride == 2 else (in_h - 3),
                ((in_w - 3) >> 1) if stride == 2 else (in_w - 3))
    raise ValueError(k_size)


def load_output_shifts():
    with open(WEIGHT_DIR / "scale_factors.json", "r") as f:
        scale_info = json.load(f)
    return {
        name: int(scale_info[f"{name}_weight"]["out_shift"])
        for name, *_ in LAYERS
    }


def load_layer_params(name, c_in, c_out, k_size):
    weights = parse_mem_file(WEIGHT_DIR / f"{name}_weight.mem", bits=8)
    biases = parse_mem_file(WEIGHT_DIR / f"{name}_bias.mem", bits=16)
    expected_weights = c_out * c_in * k_size * k_size
    if weights.size != expected_weights:
        raise RuntimeError(f"{name} has {weights.size} weights, expected {expected_weights}")
    if biases.size != c_out:
        raise RuntimeError(f"{name} has {biases.size} biases, expected {c_out}")
    return weights.reshape(c_out, c_in, k_size, k_size), biases


def prepare_sim_dir():
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BEHAVIORAL_TB_PATH, SIM_DIR / "tb_behavioral_conv.v")

    cmd = [
        "iverilog", "-g2012", "-o", "sim.vvp",
        "tb_behavioral_conv.v",
    ]
    subprocess.run(cmd, cwd=SIM_DIR, check=True)


def write_verilog_inputs(input_data, weights, biases, k_size):
    c_out, c_in, _, _ = weights.shape
    slot_to_compact = WEIGHT_SLOT_TO_COMPACT_IDX[k_size]

    with open(SIM_DIR / "weights.hex", "w") as f:
        for lane in range(16):
            for ic in range(c_in):
                for slot, compact_idx in enumerate(slot_to_compact):
                    value = 0
                    if lane < c_out and compact_idx is not None:
                        kh = compact_idx // k_size
                        kw = compact_idx % k_size
                        value = weights[lane, ic, kh, kw]
                    f.write(to_hex(value, 8) + "\n")

    with open(SIM_DIR / "biases.hex", "w") as f:
        for lane in range(16):
            value = int(biases[lane]) if lane < biases.size else 0
            f.write(to_hex(value, 16) + "\n")

    with open(SIM_DIR / "image_in.hex", "w") as f:
        for value in input_data.flatten():
            f.write(to_hex(int(value), 8) + "\n")


def run_verilog_block(input_data, weights, biases, k_size, stride, out_shift, out_h, out_w):
    write_verilog_inputs(input_data, weights, biases, k_size)
    out_path = SIM_DIR / "image_out.hex"
    if out_path.exists():
        out_path.unlink()

    cmd = [
        "vvp", "sim.vvp",
        f"+IMG_WIDTH={input_data.shape[2]}",
        f"+IMG_HEIGHT={input_data.shape[1]}",
        f"+IN_CHANNELS={input_data.shape[0]}",
        f"+KERNEL_MODE={KERNEL_MODE[k_size]}",
        f"+STRIDE={stride}",
        f"+WEIGHT_CNT={input_data.shape[0] * 16}",
        f"+IN_PIXELS={input_data.size}",
        f"+OUT_SHIFT={out_shift}",
    ]
    with open(SIM_DIR / "vvp_out.log", "a") as log:
        log.write(" ".join(cmd) + "\n")
        subprocess.run(cmd, cwd=SIM_DIR, stdout=log, stderr=subprocess.STDOUT, check=True)

    values = []
    with open(out_path, "r") as f:
        for line in f:
            text = line.strip().replace("x", "0").replace("X", "0")
            if text:
                values.append(int(text, 16))

    expected = 16 * out_h * out_w
    if len(values) < expected:
        raise RuntimeError(f"Verilog emitted {len(values)} bytes, expected {expected}")
    return np.array(values[:expected], dtype=np.uint8).reshape(16, out_h, out_w)


def run_python_fixed_layer(input_data, weights, biases, k_size, stride, out_shift):
    c_out, c_in, _, _ = weights.shape
    _, in_h, in_w = input_data.shape
    out_h, out_w = hw_output_dims(in_h, in_w, k_size, stride)
    output = np.zeros((c_out, out_h, out_w), dtype=np.int32)
    inp = input_data.astype(np.int32)

    for oc in range(c_out):
        acc = np.zeros((out_h, out_w), dtype=np.int32)
        for ic in range(c_in):
            for kh in range(k_size):
                rows = np.arange(out_h) * stride + kh
                for kw in range(k_size):
                    cols = np.arange(out_w) * stride + kw
                    patch = inp[ic][np.ix_(rows, cols)]
                    acc += patch * int(weights[oc, ic, kh, kw])
        acc += int(biases[oc])
        output[oc] = np.clip(acc >> out_shift, 0, 127)

    return output.astype(np.uint8)


def preprocess_image(image_path):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise RuntimeError(f"Image not found: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
    chw = np.transpose(resized, (2, 0, 1)).astype(np.uint8)
    return chw, img_rgb


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    return inter / max(area1 + area2 - inter, 1e-6)


def decode_and_draw(raw_layer7, img_rgb, score_thresh):
    out_img = img_rgb.copy()
    orig_h, orig_w = out_img.shape[:2]

    if np.all(raw_layer7 == 127) or np.std(raw_layer7) < 1e-6:
        cv2.putText(out_img, "ERROR: flat/saturated Verilog output",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return out_img, []

    feature_map = (raw_layer7.astype(np.float32) - LAYER7_DEQUANT_ZERO_POINT) / LAYER7_DEQUANT_SCALE
    _, grid_h, grid_w = feature_map.shape
    boxes = []

    for gj in range(grid_h):
        for gi in range(grid_w):
            score = float(sigmoid(feature_map[4, gj, gi]))
            if score < score_thresh:
                continue

            tx = sigmoid(feature_map[0, gj, gi])
            ty = sigmoid(feature_map[1, gj, gi])
            tw = sigmoid(feature_map[2, gj, gi])
            th = sigmoid(feature_map[3, gj, gi])

            cx = (gi + tx) * GRID_STRIDE / TRAINING_IMAGE_SIZE * orig_w
            cy = (gj + ty) * GRID_STRIDE / TRAINING_IMAGE_SIZE * orig_h
            bw = tw * orig_w
            bh = th * orig_h

            x1 = int(max(0, cx - bw / 2.0))
            y1 = int(max(0, cy - bh / 2.0))
            x2 = int(min(orig_w, cx + bw / 2.0))
            y2 = int(min(orig_h, cy + bh / 2.0))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2, score, 0])

    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if compute_iou(best[:4], b[:4]) < 0.4]
    keep = keep[:1]

    for x1, y1, x2, y2, score, _ in keep:
        green = int(255 * score)
        red = int(255 * (1 - score))
        color = (0, green, red)
        cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
        label = f"person {score:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out_img, (x1, y1 - label_size[1] - 6), (x1 + label_size[0], y1), color, -1)
        cv2.putText(out_img, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return out_img, keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default=str(DEFAULT_IMAGE))
    parser.add_argument("--score-thresh", type=float, default=0.15)
    args = parser.parse_args()

    prepare_sim_dir()
    shifts = load_output_shifts()
    feature_map, img_rgb = preprocess_image(args.image)
    h = INPUT_SIZE
    w = INPUT_SIZE

    print(f"Input image: {args.image}")
    print(f"Input tensor: {feature_map.shape}, range=[{feature_map.min()}, {feature_map.max()}]")

    for name, c_in, c_out, k_size, stride, pad, has_maxpool in LAYERS:
        weights, biases = load_layer_params(name, c_in, c_out, k_size)
        if pad > 0:
            layer_input = np.pad(feature_map, ((0, 0), (pad, pad), (pad, pad)),
                                 "constant", constant_values=0)
        else:
            layer_input = feature_map

        conv_h, conv_w = hw_output_dims(layer_input.shape[1], layer_input.shape[2], k_size, stride)
        if name == "layer7":
            raw_block = run_verilog_block(
                layer_input,
                weights,
                biases,
                k_size,
                stride,
                shifts[name],
                conv_h,
                conv_w,
            )
            feature_map = raw_block[:c_out]
            print(f"{name}: iverilog raw {feature_map.shape}, range=[{feature_map.min()}, {feature_map.max()}]")
        else:
            feature_map = run_python_fixed_layer(
                layer_input,
                weights,
                biases,
                k_size,
                stride,
                shifts[name],
            )
            print(f"{name}: python fixed raw {feature_map.shape}, range=[{feature_map.min()}, {feature_map.max()}]")

        if has_maxpool:
            ph, pw = conv_h // 2, conv_w // 2
            feature_map = feature_map[:, :ph * 2, :pw * 2].reshape(c_out, ph, 2, pw, 2).max(axis=(2, 4))
            print(f"{name}: pool {feature_map.shape}, range=[{feature_map.min()}, {feature_map.max()}]")

        h, w = feature_map.shape[1:]

    raw_layer7 = feature_map.astype(np.uint8)
    scaled_layer7 = (raw_layer7.astype(np.float32) - LAYER7_DEQUANT_ZERO_POINT) / LAYER7_DEQUANT_SCALE
    np.save(DEPLOY_DIR / "iverilog_output_raw_layer7.npy", raw_layer7)
    np.save(DEPLOY_DIR / "iverilog_output.npy", scaled_layer7)

    out_img, boxes = decode_and_draw(raw_layer7, img_rgb, args.score_thresh)
    out_path = DEPLOY_DIR / "iverilog_postprocessed_output.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

    print(f"Layer7 raw: shape={raw_layer7.shape}, range=[{raw_layer7.min()}, {raw_layer7.max()}], std={raw_layer7.std():.3f}")
    print(f"Detections: {len(boxes)}")
    for box in boxes:
        print(f"  person {box[4]:.3f} [{box[0]}, {box[1]}, {box[2]}, {box[3]}]")
    print(f"Saved: {out_path}")
    print(f"Saved: {DEPLOY_DIR / 'iverilog_output_raw_layer7.npy'}")
    print(f"Saved: {DEPLOY_DIR / 'iverilog_output.npy'}")


if __name__ == "__main__":
    main()
