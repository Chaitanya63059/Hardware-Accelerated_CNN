#!/usr/bin/env python3
"""
End-to-End INT8 CNN Simulation Pipeline
========================================
Simulates the full FPGA quantized inference pipeline:
  1. Load single-person image, preprocess to 128x128
  2. Quantize weights with ImageNet-normalization folded into Layer 1
  3. Run Layer 1 in Python INT8 (bit-exact hardware emulation)
  4. Run Layers 2-7 in PyTorch (fused BN model)
  5. Decode, NMS, draw single best bounding box, save final image
"""
import os, sys, math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TinyDetector7Layer, fuse_model
from detection_utils import decode_predictions, nms
from config import CLASS_NAMES

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT  = os.path.join(PROJECT_DIR, "checkpoints", "best.pth")
INPUT_IMAGE = os.path.join(PROJECT_DIR, "data/val2017/000000411938.jpg")
OUTPUT_IMAGE = os.path.join(PROJECT_DIR, "iverilog_e2e_output.png")
IMG_SIZE = 128


def load_quantized_weights():
    """Load model, fuse BN, quantize all layer weights."""
    print("=" * 60)
    print("  Step 1: Loading model and quantizing weights")
    print("=" * 60)
    device = torch.device('cpu')
    model = TinyDetector7Layer().to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    fused = fuse_model(model)
    fused.eval()

    layers_cfg = {}
    for nm in ['layer1','layer2','layer3','layer4','layer5','layer6','layer7']:
        l = getattr(fused, nm)
        conv = l[0] if isinstance(l, nn.Sequential) else l
        w_data = conv.weight.data.clone()
        b_data = conv.bias.data.clone() if conv.bias is not None else torch.zeros(conv.weight.shape[0])

        if nm == 'layer1':
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            b_adj = (w_data * (mean / std)).sum(dim=(1, 2, 3))
            b_data = b_data - b_adj
            w_data = w_data / (255.0 * std)

        w_max = max(torch.max(torch.abs(w_data)).item(), 1e-6)
        w_scale = w_max / 127.0
        b_max = torch.max(torch.abs(b_data)).item()
        if b_max / 32767.0 > w_scale:
            w_scale = b_max / 32767.0

        w_q = torch.round(w_data / w_scale).clamp(-127, 127)
        x_max = 255.0 if nm == 'layer1' else 127.0
        worst = torch.max(torch.sum(torch.abs(w_q), dim=(1, 2, 3))).item()
        req = x_max * worst / 127.0
        out_shift = int(math.ceil(math.log2(req))) if req > 1 else 0
        out_shift = max(0, min(31, out_shift - 3))

        layers_cfg[nm] = {
            'w': w_q.detach().numpy(),
            'b': torch.round(b_data / w_scale).clamp(-32767, 32767).detach().numpy(),
            'w_scale': w_scale, 'out_shift': out_shift,
            'shape': list(conv.weight.shape)
        }
        print(f"  {nm}: {list(conv.weight.shape)}  scale={w_scale:.6f}  shift={out_shift}")
    return layers_cfg, fused


def python_int8_conv_layer1(img_arr, layers_cfg):
    """Bit-exact INT8 convolution matching hardware."""
    print(f"\n{'='*60}")
    print(f"  Step 3: INT8 Layer 1 (bit-exact HW emulation)")
    print(f"{'='*60}")

    cfg = layers_cfg['layer1']
    w_q = cfg['w'].astype(np.int64)
    b_q = cfg['b'].astype(np.int64)
    out_shift = cfg['out_shift']
    C_out, C_in, K, _ = w_q.shape

    padded = np.pad(img_arr.astype(np.int64), ((0,0),(1,1),(1,1)), 'constant', constant_values=0)
    H_p, W_p = padded.shape[1], padded.shape[2]
    H_out = (H_p - K) // 2 + 1
    W_out = (W_p - K) // 2 + 1
    output = np.zeros((C_out, H_out, W_out), dtype=np.int64)

    for oc in range(C_out):
        for oh in range(H_out):
            for ow in range(W_out):
                acc = np.int64(0)
                ih, iw = oh * 2, ow * 2
                for ic in range(C_in):
                    for kr in range(K):
                        for kc in range(K):
                            acc += np.int64(padded[ic, ih+kr, iw+kc]) * np.int64(w_q[oc, ic, kr, kc])
                acc += b_q[oc]
                shifted = acc >> out_shift
                if shifted < 0: shifted = 0
                if shifted > 127: shifted = 127
                output[oc, oh, ow] = shifted

    print(f"  Output: {output.shape}  range=[{output.min()}, {output.max()}]")
    print(f"  Non-zero: {np.count_nonzero(output)}/{output.size} ({100*np.count_nonzero(output)/output.size:.1f}%)")
    return output.astype(np.uint8)


def main():
    print("\n" + "="*60)
    print("  INT8 CNN END-TO-END SIMULATION")
    print("  TinyDetector7Layer — Single Person Detection")
    print("="*60)

    # Step 1: Load & quantize
    layers_cfg, fused = load_quantized_weights()

    # Step 2: Preprocess
    print(f"\n{'='*60}\n  Step 2: Preprocessing\n{'='*60}")
    orig_img = Image.open(INPUT_IMAGE).convert('RGB')
    orig_w, orig_h = orig_img.size
    img_resized = orig_img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    img_arr = np.array(img_resized, dtype=np.int32).transpose(2, 0, 1)
    print(f"  {orig_w}x{orig_h} -> {IMG_SIZE}x{IMG_SIZE}, range=[{img_arr.min()},{img_arr.max()}]")

    # Step 3: INT8 Layer 1
    int8_out = python_int8_conv_layer1(img_arr, layers_cfg)

    # Step 4: Layers 2-7 in PyTorch
    print(f"\n{'='*60}\n  Step 4: Layers 2-7 in PyTorch\n{'='*60}")
    w_scale = layers_cfg['layer1']['w_scale']
    out_shift = layers_cfg['layer1']['out_shift']
    hw_tensor = torch.from_numpy(int8_out.astype(np.float32)).unsqueeze(0)
    hw_tensor = hw_tensor * (w_scale * (2 ** out_shift))

    with torch.no_grad():
        mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
        std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
        norm = (torch.from_numpy(img_arr.astype(np.float32)).unsqueeze(0)/255.0 - mean) / std
        sw_l1 = fused.layer1(norm)
        print(f"  INT8 L1: [{hw_tensor.min():.3f}, {hw_tensor.max():.3f}]")
        print(f"  Float L1: [{sw_l1.min():.3f}, {sw_l1.max():.3f}]")

        x = fused.layer2(hw_tensor)
        x = fused.layer3(x)
        x = fused.layer4(x)
        x = fused.layer5(x)
        x = fused.layer6(x)
        final = fused.layer7(x)
        print(f"  Final output: {final.shape} [{final.min():.3f}, {final.max():.3f}]")

    # Step 5: Post-processing — single best detection
    print(f"\n{'='*60}\n  Step 5: Post-processing & Bounding Box\n{'='*60}")
    boxes = decode_predictions(final[0], score_thresh=0.05, normalized=True, image_size=IMG_SIZE)
    boxes = nms(boxes, 0.4)
    boxes = boxes[:4]  # Top 4 detections

    print(f"  Detection: {len(boxes)}")
    for i, b in enumerate(boxes):
        cn = CLASS_NAMES[int(b[5])] if int(b[5]) < len(CLASS_NAMES) else f"cls{int(b[5])}"
        print(f"    {cn:<10} conf={b[4]:.3f}  box=({b[0]:.3f},{b[1]:.3f},{b[2]:.3f},{b[3]:.3f})")

    # Draw — clean single bounding box, no title bar
    draw_img = orig_img.copy()
    draw = ImageDraw.Draw(draw_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()

    colors = ["#00FF00", "#00CCFF", "#FF6600", "#FF00FF"]
    for i, b in reversed(list(enumerate(boxes))):
        cn = CLASS_NAMES[int(b[5])] if int(b[5]) < len(CLASS_NAMES) else f"cls{int(b[5])}"
        x1, y1 = max(0, int(b[0]*orig_w)), max(0, int(b[1]*orig_h))
        x2, y2 = min(orig_w-1, int(b[2]*orig_w)), min(orig_h-1, int(b[3]*orig_h))
        c = colors[i % len(colors)]

        # Thick bounding box
        w = 4 if i == 0 else 3
        for o in range(w):
            draw.rectangle([x1-o, y1-o, x2+o, y2+o], outline=c)

        # Label
        label = f"{cn} {b[4]:.0%}"
        bb = draw.textbbox((x1, y1-28), label, font=font)
        lh = bb[3]-bb[1]+10
        lw = bb[2]-bb[0]+14
        if i == 0:
            ly = max(0, y1-lh-4)
            if ly < 5:
                ly = y1 + 6
        else:
            # Place second label at bottom of its box to avoid overlap
            ly = max(0, y2 - lh - 4)
        draw.rectangle([x1, ly, x1+lw, ly+lh], fill=c)
        draw.text((x1+6, ly+4), label, fill="black", font=font)

    draw_img.save(OUTPUT_IMAGE)
    print(f"\n  ✓ Saved: {OUTPUT_IMAGE}")

    # Reference: Pure PyTorch
    print(f"\n{'='*60}\n  Reference: Pure PyTorch\n{'='*60}")
    with torch.no_grad():
        ref = fused(norm)
    rb = decode_predictions(ref[0], score_thresh=0.05, normalized=True, image_size=IMG_SIZE)
    rb = nms(rb, 0.4)[:4]
    for b in rb:
        cn = CLASS_NAMES[int(b[5])] if int(b[5]) < len(CLASS_NAMES) else f"cls{int(b[5])}"
        print(f"    {cn:<10} conf={b[4]:.3f}  box=({b[0]:.3f},{b[1]:.3f},{b[2]:.3f},{b[3]:.3f})")

    print(f"\n{'='*60}\n  ✓ SIMULATION COMPLETE\n{'='*60}")


if __name__ == '__main__':
    main()
