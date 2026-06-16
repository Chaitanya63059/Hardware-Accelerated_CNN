"""
FPGA CNN inference on Kria KV260.
Hardware convolution + software MaxPool/decode.

OUTPUTS:
  - fpga_raw_output.png:    Heatmap visualization of raw FPGA fabric output (no post-processing)
  - fpga_postprocessed_output.png: Final detection image (scaled + decoded + NMS + bounding boxes)
  - fpga_output.npy:        Processed layer7 output (mapped back to logits)
  - fpga_output_raw_layer7.npy: Raw layer7 output [0, 127] for offline post-processing
  - fpga_output_raw_all_layers.npz: All layer outputs [0, 127] for complete reconstruction

To post-process raw outputs on CPU without FPGA:
  python post_process_fpga_output.py [image_path]

CRITICAL: ImageNet normalization (MEAN/STD) is folded into layer1 weights by
export_weights.py. Send raw uint8 pixels to hardware.

CRITICAL HARDWARE CONSTRAINTS:
  - Line buffers: MAX_WIDTH=128 -> all padded inputs must be <= 128
  - Accumulator: 32-bit signed, with 16-bit bias added
  - Output: arithmetic right shift -> ReLU -> saturate to unsigned 8-bit [0, 127]
  - Layer7 (detection head) output is converted back to logit range for decode
  - Input pixels are treated as UNSIGNED 8-bit by the hardware
"""

import pynq
from pynq import Overlay
import numpy as np
import time
import os
import sys
import argparse
import json
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
    ("layer7", 128,   6, 1, 1, 0, False),
]

# AXI-Lite Register offsets
REG_CTRL         = 0x00
REG_KERNEL_MODE  = 0x04
REG_STRIDE       = 0x08
REG_IMG_WIDTH    = 0x0C
REG_IMG_HEIGHT   = 0x10
REG_NUM_IN_CH    = 0x14
REG_W_BRAM_ADDR  = 0x18
REG_W_BRAM_DATA  = 0x1C
REG_W_BRAM_WEN   = 0x20
REG_BIAS_IDX     = 0x24
REG_BIAS_DATA    = 0x28
REG_BIAS_WEN     = 0x2C
REG_STATUS       = 0x30
REG_OUT_SHIFT    = 0x34

KERNEL_MODE_MAP = {1: 0, 3: 1, 4: 2}
WEIGHT_SLOT_TO_COMPACT_IDX = {
    1: [0] + [None] * 15,
    3: [0, 1, 2, None, 3, 4, 5, None, 6, 7, 8, None, None, None, None, None],
    4: list(range(16)),
}
# Fallback per-layer arithmetic right shift before ReLU clamp.
# scale_factors.json overrides these at runtime when it is present.
# Wrong shifts cause total accumulator saturation (all outputs -> 127).
LAYER_OUTPUT_SHIFTS = {
    "layer1": 13,
    "layer2": 13,
    "layer3": 14,
    "layer4": 13,
    "layer5": 14,
    "layer6": 14,
    "layer7": 4,
}

WEIGHT_DIR = "/home/ubuntu/cnn_accelerator"
IMAGE_PATH = "/home/ubuntu/cnn_accelerator/person_test.jpg"

# Resize to 126 so that after pad=1, layer1 gets exactly 128 (MAX_WIDTH limit)
INPUT_SIZE = 126

# Must match run_iverilog_postprocess.py and the exported calibration metadata.
# These map raw layer7 bytes [0, 127] back into the logit range used by decode.
LAYER7_DEQUANT_ZERO_POINT = 88.52
LAYER7_DEQUANT_SCALE = 31.99

WEIGHTS_CACHE = {}

# ===================== HELPERS =====================

def hw_output_dims(in_h, in_w, k_size, stride):
    """
    Match the EXACT formulas from cnn_compute_unit.v S_INIT state:
      k=1x1: out = W (s=1) or W>>1 (s=2)
      k=3x3: out = W-2 (s=1) or (W-2)>>1 (s=2)
      k=4x4: out = W-3 (s=1) or (W-3)>>1 (s=2)
    """
    if k_size == 1:
        oh = (in_h >> 1) if stride == 2 else in_h
        ow = (in_w >> 1) if stride == 2 else in_w
    elif k_size == 3:
        oh = ((in_h - 2) >> 1) if stride == 2 else (in_h - 2)
        ow = ((in_w - 2) >> 1) if stride == 2 else (in_w - 2)
    elif k_size == 4:
        oh = ((in_h - 3) >> 1) if stride == 2 else (in_h - 3)
        ow = ((in_w - 3) >> 1) if stride == 2 else (in_w - 3)
    return oh, ow


MEM_CACHE = {}

def parse_mem_file(filepath, bits):
    """Parse a .mem file -> list of signed integer values."""
    cache_key = (filepath, bits)
    if cache_key in MEM_CACHE:
        return MEM_CACHE[cache_key]

    if bits not in (8, 16):
        raise ValueError(f"Unsupported .mem bit width: {bits}")

    header_bits = None
    sign_bit = 1 << (bits - 1)
    full_range = 1 << bits
    values = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('//'):
                if 'Bits:' in line:
                    try:
                        header_bits = int(line.rsplit('Bits:', 1)[1].strip())
                    except ValueError:
                        pass
                continue
            if line == '':
                continue
            parts = line.split()
            if len(parts) == 2:
                hex_val = int(parts[1], 16)
                if hex_val >= sign_bit:
                    hex_val -= full_range
                values.append(hex_val)

    if header_bits is not None and header_bits != bits:
        raise RuntimeError(
            f"{filepath} declares Bits: {header_bits}, but run_fpga expected {bits}-bit values"
        )

    MEM_CACHE[cache_key] = values
    return values


def load_output_shifts(weight_dir):
    """Load per-layer output shifts from the exported scale_factors.json."""
    global LAYER_OUTPUT_SHIFTS

    scale_path = os.path.join(weight_dir, "scale_factors.json")
    if not os.path.exists(scale_path):
        print(f"  WARNING: {scale_path} not found; using built-in output shifts.")
        return

    with open(scale_path, 'r') as f:
        scale_info = json.load(f)

    loaded = {}
    for name, c_in, c_out, k_size, *_ in LAYERS:
        key = f"{name}_weight"
        info = scale_info.get(key)
        if info is None:
            raise RuntimeError(f"{scale_path} is missing {key}")

        expected_shape = [c_out, c_in, k_size, k_size]
        actual_shape = info.get("shape")
        if actual_shape is not None and actual_shape != expected_shape:
            raise RuntimeError(
                f"{scale_path} has {key} shape {actual_shape}, expected {expected_shape}. "
                "This usually means stale weights were copied."
            )

        if "out_shift" not in info:
            raise RuntimeError(f"{scale_path} is missing out_shift for {key}")
        loaded[name] = int(info["out_shift"])

    LAYER_OUTPUT_SHIFTS.update(loaded)
    print("  Output shifts loaded from scale_factors.json:")
    for name, *_ in LAYERS:
        print(f"    {name}: {LAYER_OUTPUT_SHIFTS[name]}")


def preload_all_weights():
    """Load all .mem files into RAM at startup."""
    print("\n[0/5] Pre-loading weights to RAM...")
    for name, c_in, c_out, k_size, *_ in LAYERS:
        w_path = os.path.join(WEIGHT_DIR, f"{name}_weight.mem")
        b_path = os.path.join(WEIGHT_DIR, f"{name}_bias.mem")

        weights = parse_mem_file(w_path, bits=8)
        biases = parse_mem_file(b_path, bits=16)

        expected_weights = c_out * c_in * k_size * k_size
        if len(weights) != expected_weights:
            raise RuntimeError(
                f"{w_path} has {len(weights)} values, expected {expected_weights}. "
                "Check that the single-class exported weights were copied."
            )
        if len(biases) != c_out:
            raise RuntimeError(
                f"{b_path} has {len(biases)} values, expected {c_out}. "
                "Check that the current 16-bit bias files were copied."
            )

        WEIGHTS_CACHE[f"{name}_w"] = weights
        WEIGHTS_CACHE[f"{name}_b"] = biases
    print("  Done.")


def load_weights_to_bram(accel_ip, layer_name, lane_start, num_lanes, c_in, k_size):
    weights = WEIGHTS_CACHE[f"{layer_name}_w"]
    biases = WEIGHTS_CACHE[f"{layer_name}_b"]
    compact_kernel_size = k_size * k_size
    weights_per_filter = c_in * compact_kernel_size
    slot_to_compact = WEIGHT_SLOT_TO_COMPACT_IDX[k_size]

    for lane in range(num_lanes):
        oc = lane_start + lane
        base_idx = oc * weights_per_filter
        for ic in range(c_in):
            channel_base_idx = base_idx + ic * compact_kernel_size
            bram_base_addr = ic * 16
            for slot, compact_idx in enumerate(slot_to_compact):
                if compact_idx is None:
                    w_val = 0
                else:
                    compact_offset = channel_base_idx + compact_idx
                    w_val = weights[compact_offset] if compact_offset < len(weights) else 0
                if w_val < 0:
                    w_val = (1 << 8) + w_val
                bram_addr = (lane << 12) | ((bram_base_addr + slot) & 0xFFF)
                accel_ip.write(REG_W_BRAM_ADDR, bram_addr)
                accel_ip.write(REG_W_BRAM_DATA, w_val & 0xFF)
                accel_ip.write(REG_W_BRAM_WEN, 1)

        bias_val = biases[oc] if oc < len(biases) else 0
        if bias_val < 0:
            bias_val = (1 << 16) + bias_val
        accel_ip.write(REG_BIAS_IDX, lane)
        accel_ip.write(REG_BIAS_DATA, bias_val & 0xFFFF)
        accel_ip.write(REG_BIAS_WEN, 1)


# ===================== INFERENCE =====================

def run_fpga_layer(accel_ip, dma, input_data, layer_cfg, in_h, in_w, in_buffer, out_buffer):
    """Run one CNN layer on FPGA. Returns (feature_map, out_h, out_w, conv_time_ms)."""
    name, c_in, c_out, k_size, stride, pad, has_maxpool = layer_cfg
    out_shift = LAYER_OUTPUT_SHIFTS.get(name, 0)

    # Software padding (all padded sizes stay <= 128)
    if pad > 0:
        input_data = np.pad(input_data, ((0, 0), (pad, pad), (pad, pad)),
                            'constant', constant_values=0)
        in_h += 2 * pad
        in_w += 2 * pad

    conv_h, conv_w = hw_output_dims(in_h, in_w, k_size, stride)
    print(f"  {name}: ({c_in},{in_h},{in_w}) k={k_size} s={stride} -> ({c_out},{conv_h},{conv_w})", end="", flush=True)

    all_outputs = []
    conv_time_sec = 0.0
    hw_cycles = 0

    for oc_start in range(0, c_out, 16):
        num_lanes = min(16, c_out - oc_start)
        load_weights_to_bram(accel_ip, name, oc_start, num_lanes, c_in, k_size)

        accel_ip.write(REG_KERNEL_MODE, KERNEL_MODE_MAP[k_size])
        accel_ip.write(REG_STRIDE, stride)
        accel_ip.write(REG_IMG_WIDTH, in_w)
        accel_ip.write(REG_IMG_HEIGHT, in_h)
        accel_ip.write(REG_NUM_IN_CH, c_in)
        accel_ip.write(REG_OUT_SHIFT, out_shift)

        # Input data is unsigned uint8:
        #   - Layer 1: raw pixels [0, 255] from preprocessing
        #   - Layers 2-7: hardware output [0, 127] from previous layer
        input_flat = input_data.flatten().astype(np.uint8)
        in_buffer[:len(input_flat)] = input_flat

        # Hardware always outputs 16 lanes
        out_size = 16 * conv_h * conv_w

        dma.recvchannel.transfer(out_buffer, nbytes=out_size)
        dma.sendchannel.transfer(in_buffer, nbytes=len(input_flat))

        t_conv_start = time.perf_counter()
        accel_ip.write(REG_CTRL, 0)
        accel_ip.write(REG_CTRL, 1)

        dma.sendchannel.wait()
        dma.recvchannel.wait()
        conv_time_sec += time.perf_counter() - t_conv_start

        # Calculate theoretical hardware clock cycles (input stream + output stream)
        # cnn_compute_unit processes 1 input byte per clock cycle, then outputs 16 bytes.
        in_cycles = c_in * in_h * in_w
        out_cycles = 16 * conv_h * conv_w
        hw_cycles += (in_cycles + out_cycles)

        # Hardware output is unsigned 8-bit [0, 127] (ReLU + saturation)
        out_all = np.array(out_buffer[:out_size], dtype=np.uint8).reshape(16, conv_h, conv_w)
        all_outputs.append(out_all[:num_lanes].copy())

    feature_map = np.concatenate(all_outputs, axis=0).astype(np.float32)
    feature_map_raw = feature_map.copy()  # Save raw [0, 127] output
    
    # CRITICAL: Do NOT scale intermediate layer outputs!
    # Raw [0, 127] values must flow directly to the next layer as uint8,
    # matching exactly what the hardware does. Scaling to logit range
    # is applied ONLY ONCE to the final layer7 output in main().
    # (Previous bug: scaling here caused negatives -> uint8 wrap to 253-255
    #  -> accumulator overflow -> every output saturated to 127)
    # Vectorized software MaxPool 2x2
    out_h, out_w = conv_h, conv_w
    if has_maxpool:
        ph, pw = conv_h // 2, conv_w // 2
        cropped = feature_map[:, :ph * 2, :pw * 2]
        feature_map = cropped.reshape(c_out, ph, 2, pw, 2).max(axis=(2, 4))
        out_h, out_w = ph, pw
        print(f" -> pool -> ({c_out},{out_h},{out_w})")
    else:
        print()

    hw_limit_ms = (hw_cycles / 100_000_000.0) * 1000.0
    return feature_map, feature_map_raw, out_h, out_w, conv_time_sec * 1000.0, hw_limit_ms


# ===================== PREPROCESSING =====================

def preprocess_image(image_path):
    """
    Load image -> raw uint8 tensor (3, INPUT_SIZE, INPUT_SIZE).
    CRITICAL: ImageNet normalization (MEAN/STD) is already FOLDED into the
    Layer 1 weights by export_weights.py. The hardware expects RAW 0-255
    pixel values. Do NOT normalize here — that would double-normalize!
    """
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    img_resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
    
    # Send raw 0-255 pixels directly to hardware (normalization is in the weights)
    img_chw = np.transpose(img_resized, (2, 0, 1)).astype(np.uint8)  # HWC -> CHW

    return img_chw, orig_w, orig_h, img_rgb


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
    """
    Decode final feature map into bounding boxes.
    NOTE: Hardware outputs [0, 127] (ReLU saturated) and are scaled to logit range [-2.5, 2.5].
    We decode relative to the actual grid size.
    """
    _, grid_h, grid_w = feature_map.shape
    class_names = ["person"]

    print(f"\n  Detection grid: {grid_h}x{grid_w}")
    print(f"  Layer7 output range: [{feature_map.min():.1f}, {feature_map.max():.1f}]")

    if np.std(feature_map) < 1e-6:
        print("  ERROR: Layer7 output is flat; skipping bounding-box decode.")
        print("         A flat output usually means the FPGA layer output saturated.")
        cv2.putText(out_img, "ERROR: flat FPGA output - no valid decode",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return out_img, 0

    boxes = []
    for gj in range(grid_h):
        for gi in range(grid_w):
            obj_conf = sigmoid(feature_map[4, gj, gi])

            cls_logits = feature_map[5:6, gj, gi]
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

            # The model was trained with a 128x128 input and 8x8 grid (stride 16).
            # The FPGA uses 126x126 input and produces a 7x7 grid, losing the edges.
            # We must map coordinates using the training scale, not the dynamic grid size.
            cx = (gi + tx) * 16.0 / 128.0 * orig_w
            cy = (gj + ty) * 16.0 / 128.0 * orig_h
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
    
    keep = keep[:1]  # Single best detection only

    print(f"  Detected {len(keep)} object(s):")

    if keep:
        b = keep[0]
        x1, y1, x2, y2, score, c_idx = b
        class_name = class_names[c_idx] if c_idx < len(class_names) else f"cls{c_idx}"
        print(f"    {class_name:>10}: {score:.3f}  [{x1},{y1},{x2},{y2}]")

        # Thick green bounding box
        for o in range(4):
            cv2.rectangle(out_img, (x1-o, y1-o), (x2+o, y2+o), (0, 255, 0), 1)

        # Clean label with percentage confidence
        label = f"{class_name} {score:.0%}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        lw, lh = label_size[0] + 12, label_size[1] + 12
        ly = max(0, y1 - lh - 4)
        if ly < 5:
            ly = y1 + 6

        cv2.rectangle(out_img, (x1, ly), (x1 + lw, ly + lh), (0, 255, 0), -1)
        cv2.putText(out_img, label, (x1 + 6, ly + lh - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    return out_img, len(keep)


# ===================== RAW VISUALIZATION =====================

def generate_raw_visualization(raw_layer7, img_rgb, orig_w, orig_h):
    """
    Generate a heatmap visualization of the RAW FPGA fabric output.
    No post-processing, no scaling, no bounding boxes — just the raw
    unsigned [0, 127] activations overlaid on the original image.
    
    Shows what the FPGA hardware actually produces before any software
    post-processing happens.
    """
    n_channels, grid_h, grid_w = raw_layer7.shape
    
    # Create a side-by-side visualization
    # Left: original image
    # Right: channel heatmaps of layer7 output
    
    img_resized = cv2.resize(img_rgb, (320, 320))
    
    # Create heatmap panel: show key channels as colored overlays
    channel_names = ["tx", "ty", "tw", "th", "conf", "person"]
    n_show = min(n_channels, 6)
    
    # Build grid of channel heatmaps (2 rows x 4 cols)
    cell_size = 80
    n_cols = 4
    n_rows = (n_show + n_cols - 1) // n_cols
    heatmap_panel_w = n_cols * cell_size
    heatmap_panel_h = n_rows * (cell_size + 18)  # 18px for label
    
    heatmap_panel = np.zeros((heatmap_panel_h, heatmap_panel_w, 3), dtype=np.uint8)
    
    for ch_idx in range(n_show):
        row = ch_idx // n_cols
        col = ch_idx % n_cols
        
        # Get raw channel data [0, 127]
        ch_data = raw_layer7[ch_idx]  # (grid_h, grid_w)
        
        # Normalize to [0, 255] for visualization
        if ch_data.max() > ch_data.min():
            ch_norm = ((ch_data - ch_data.min()) / (ch_data.max() - ch_data.min()) * 255).astype(np.uint8)
        else:
            ch_norm = np.zeros_like(ch_data, dtype=np.uint8)
        
        # Resize to cell_size with nearest-neighbor (to show grid pixels clearly)
        ch_resized = cv2.resize(ch_norm, (cell_size, cell_size), interpolation=cv2.INTER_NEAREST)
        
        # Apply colormap
        ch_colored = cv2.applyColorMap(ch_resized, cv2.COLORMAP_JET)
        ch_colored = cv2.cvtColor(ch_colored, cv2.COLOR_BGR2RGB)
        
        # Place in panel
        y_start = row * (cell_size + 18)
        x_start = col * cell_size
        heatmap_panel[y_start:y_start + cell_size, x_start:x_start + cell_size] = ch_colored
        
        # Add label
        label = channel_names[ch_idx] if ch_idx < len(channel_names) else f"ch{ch_idx}"
        range_str = f"{ch_data.min():.0f}-{ch_data.max():.0f}"
        cv2.putText(heatmap_panel, f"{label}: {range_str}",
                    (x_start + 2, y_start + cell_size + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    # Create combined visualization
    # Resize heatmap panel to match image height
    heatmap_panel_resized = cv2.resize(heatmap_panel, (320, 320))
    
    # Also create a confidence heatmap overlay on the original image
    if n_channels > 4:
        conf_raw = raw_layer7[4]  # objectness channel, raw [0, 127]
        conf_norm = ((conf_raw / 127.0) * 255).astype(np.uint8)
        conf_heatmap = cv2.resize(conf_norm, (320, 320), interpolation=cv2.INTER_NEAREST)
        conf_colored = cv2.applyColorMap(conf_heatmap, cv2.COLORMAP_HOT)
        conf_colored = cv2.cvtColor(conf_colored, cv2.COLOR_BGR2RGB)
        conf_overlay = cv2.addWeighted(img_resized, 0.5, conf_colored, 0.5, 0)
    else:
        conf_overlay = img_resized.copy()
    
    # Stack: [Original | Conf Overlay | Channel Heatmaps]
    combined = np.hstack([img_resized, conf_overlay, heatmap_panel_resized])
    
    # Add title bar
    title_bar = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, "RAW FPGA OUTPUT (No Post-Processing)",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(title_bar, "Original",
                (120, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(title_bar, "Conf Overlay",
                (440, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(title_bar, "Channel Heatmaps",
                (760, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Add stats footer
    footer = np.zeros((30, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(footer, f"Layer7 shape: {raw_layer7.shape} | "
                f"Range: [{raw_layer7.min():.0f}, {raw_layer7.max():.0f}] | "
                f"ReLU-saturated unsigned [0, 127]",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    
    result = np.vstack([title_bar, combined, footer])
    return result


# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser(description='FPGA CNN Inference on Kria KV260')
    parser.add_argument('image', nargs='?', default=IMAGE_PATH,
                        help='Path to input image (default: person_test.jpg)')
    args = parser.parse_args()
    image_path = args.image

    print("=" * 60)
    print("  CNN Accelerator - Kria KV260 FPGA Inference")
    print("  (Dual Output: Raw + Post-Processed)")
    print("=" * 60)

    load_output_shifts(WEIGHT_DIR)
    preload_all_weights()

    print("\n[1/5] Loading FPGA Bitstream...")
    t0 = time.perf_counter()
    overlay = Overlay('./design_123.bit')
    accel_ip = overlay.cnn_pipeline_top_1
    dma = overlay.axi_dma_0
    print(f"  Loaded in {(time.perf_counter() - t0) * 1000:.1f} ms")

    MAX_BUFFER_SIZE = 1024 * 1024
    in_buffer = pynq.allocate(shape=(MAX_BUFFER_SIZE,), dtype=np.uint8)
    out_buffer = pynq.allocate(shape=(MAX_BUFFER_SIZE,), dtype=np.uint8)

    print(f"\n[2/5] Preprocessing image ({INPUT_SIZE}x{INPUT_SIZE})...")
    img_normalized, orig_w, orig_h, img_rgb = preprocess_image(image_path)
    print(f"  {image_path}")
    print(f"  Input: {img_normalized.shape}, range=[{img_normalized.min()}, {img_normalized.max()}]")

    print("\n[3/5] Running FPGA inference...")
    t_start = time.perf_counter()
    feature_map = img_normalized.astype(np.float32)
    h, w = INPUT_SIZE, INPUT_SIZE
    layer_times = []
    raw_outputs = {}  # Store raw FPGA fabric outputs before post-processing

    for layer_cfg in LAYERS:
        feature_map, feature_map_raw, h, w, dt_conv, hw_ms = run_fpga_layer(
            accel_ip, dma, feature_map, layer_cfg, h, w, in_buffer, out_buffer
        )
        layer_times.append((layer_cfg[0], dt_conv, hw_ms))
        raw_outputs[layer_cfg[0]] = feature_map_raw  # Save raw output

    t_total_end2end = (time.perf_counter() - t_start) * 1000
    t_total_conv = sum(ms for _, _, _ in layer_times)
    t_total_hw = sum(hw for _, _, hw in layer_times)

    print(f"\n[4/5] Convolution Timing")
    print(f"\n  Details (Measured with OS Interrupt Latency vs Theoretical Hardware Limit @ 100MHz):")
    for name, ms, hw_ms in layer_times:
        print(f"    {name:<10}: Measured = {ms:>5.1f} ms  |  Hardware Limit = {hw_ms:>5.1f} ms")
    print(f"    {'_' * 60}")
    print(f"    Total Measured Time (with PYNQ overhead): {t_total_conv:.1f} ms")
    print(f"    Total Theoretical HW Limit (@100MHz):     {t_total_hw:.1f} ms")
    print(f"    Measured FPS:      {1000 / max(1e-6, t_total_conv):.1f}")
    print(f"    Theoretical HW FPS:{1000 / max(1e-6, t_total_hw):.1f}")

    print(f"\n  Raw output: shape={feature_map.shape}, "
          f"range=[{feature_map.min():.1f}, {feature_map.max():.1f}]")

    # Dequantize ONLY the final layer7 output for decode.
    # This matches run_iverilog_postprocess.py and the saved iverilog_output.npy.
    feature_map = (feature_map.astype(np.float32) - LAYER7_DEQUANT_ZERO_POINT) / LAYER7_DEQUANT_SCALE
    print(f"  Dequantized output: range=[{feature_map.min():.3f}, {feature_map.max():.3f}]")

    # Save processed output (post-scaled)
    np.save(os.path.join(WEIGHT_DIR, "fpga_output.npy"), feature_map)
    
    # Save raw FPGA fabric outputs (no post-processing) for later offline post-processing
    print(f"\n  Saving raw FPGA fabric outputs...")
    raw_layer7 = raw_outputs["layer7"]  # Final layer before decoding
    np.save(os.path.join(WEIGHT_DIR, "fpga_output_raw_layer7.npy"), raw_layer7)
    print(f"    Layer7 raw output: shape={raw_layer7.shape}, range=[{raw_layer7.min():.1f}, {raw_layer7.max():.1f}]")
    if np.all(raw_layer7 == 127):
        print("    ERROR: Layer7 raw output is all 127. The FPGA output is saturated.")
        print("           Check that the bitstream includes OUT_SHIFT wiring and was regenerated.")
    elif np.std(raw_layer7) < 1e-6:
        print("    ERROR: Layer7 raw output is flat. Bounding boxes will not be meaningful.")
    
    # Save all intermediate raw outputs for complete reconstruction
    np.savez(os.path.join(WEIGHT_DIR, "fpga_output_raw_all_layers.npz"), **raw_outputs)
    print(f"    Saved all raw layer outputs to fpga_output_raw_all_layers.npz")

    # ===== IMAGE 1: Raw FPGA output (no post-processing) =====
    print(f"\n[5/5] Generating output images...")
    print(f"\n  --- Image 1: Raw FPGA Output (no post-processing) ---")
    raw_vis = generate_raw_visualization(raw_layer7, img_rgb, orig_w, orig_h)
    raw_path = os.path.join(WEIGHT_DIR, "fpga_raw_output.png")
    cv2.imwrite(raw_path, cv2.cvtColor(raw_vis, cv2.COLOR_RGB2BGR))
    print(f"    Saved: {raw_path}")

    # ===== IMAGE 2: Post-processed FPGA output (bounding boxes) =====
    print(f"\n  --- Image 2: Post-Processed FPGA Output ---")
    out_img = cv2.resize(img_rgb, (orig_w, orig_h))
    out_img, n_detections = decode_and_draw(feature_map, out_img, orig_w, orig_h)



    postproc_path = os.path.join(WEIGHT_DIR, "fpga_postprocessed_output.png")
    cv2.imwrite(postproc_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    print(f"    Saved: {postproc_path}")

    # Also save legacy fpga_output.png for backward compatibility
    cv2.imwrite(os.path.join(WEIGHT_DIR, "fpga_output.png"),
                cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

    print(f"\n  ✓ Saved 2 output images:")
    print(f"    1. fpga_raw_output.png          (raw FPGA heatmaps, no post-processing)")
    print(f"    2. fpga_postprocessed_output.png (decoded detections with bounding boxes)")
    print(f"    + fpga_output.npy, fpga_output_raw_layer7.npy, fpga_output_raw_all_layers.npz")
    print("=" * 60)

    in_buffer.freebuffer()
    out_buffer.freebuffer()


if __name__ == '__main__':
    main()
