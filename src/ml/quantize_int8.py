"""
INT8 Post-Training Static Quantization for KRIA KV260 DPU.

Workflow:
  1. Load trained float32 checkpoint
  2. Fuse BatchNorm into Conv layers
  3. Calibrate quantization scales using real COCO data
  4. Apply symmetric per-tensor INT8 quantization
  5. Export quantized ONNX model + BRAM-ready weight files
  6. Validate quantized vs float32 accuracy

The Xilinx DPU on KV260 uses symmetric INT8 quantization with per-tensor
scale factors. This script produces:
  - model_int8.onnx — quantized ONNX for Vitis AI / DPU compilation
  - INT8 weight .coe/.mem files — direct BRAM initialization
  - scale_factors_int8.json — scale/zero-point per layer for Verilog

Usage:
    python quantize_int8.py                                     # default
    python quantize_int8.py --checkpoint checkpoints/best.pth   # custom
    python quantize_int8.py --num-calibration 200               # more calibration data
    python quantize_int8.py --skip-export                       # quantize + validate only
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.quantization as tq

from model import TinyDetector7Layer, fuse_model
from dataset import get_dataloaders, IMG_SIZE, GRID_SIZE
from detection_utils import evaluate_model


# ── Quantization-aware wrapper ────────────────────────────────────────────────
class QuantizedTinyDetector(nn.Module):
    """
    Wraps the fused TinyDetector for PyTorch static quantization.

    The DPU expects symmetric INT8, so we use:
      - qint8 for weights (symmetric, per-tensor)
      - quint8 for activations (affine, per-tensor)

    The quant/dequant stubs mark the quantization boundary.
    """

    def __init__(self, float_model):
        super().__init__()
        self.quant = tq.QuantStub()
        self.model = float_model
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        # Run through each layer manually so quantization observers see them
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.layer5(x)
        x = self.model.layer6(x)
        x = self.model.layer7(x)
        x = self.dequant(x)
        return x


# ── Calibration ───────────────────────────────────────────────────────────────
def calibrate(quantized_model, data_loader, device, num_batches=50):
    """
    Run calibration data through the model so quantization observers
    collect activation statistics (min/max ranges).
    """
    quantized_model.eval()
    count = 0
    print(f"  Calibrating with {num_batches} batches...")

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            quantized_model(images)
            count += 1
            if count >= num_batches:
                break

    print(f"  Calibration complete ({count} batches processed)")


# ── INT8 Weight Export ────────────────────────────────────────────────────────
def quantize_weight_symmetric(weight_tensor, bits=8):
    """
    Symmetric quantization: maps [-max, +max] to [-127, +127].
    This is what the Xilinx DPU expects.
    """
    max_val = weight_tensor.abs().max().item()
    if max_val == 0:
        max_val = 1e-8

    qmax = (2 ** (bits - 1)) - 1  # 127 for INT8
    scale = max_val / qmax
    quantized = torch.round(weight_tensor / scale).clamp(-qmax, qmax).to(torch.int32)

    return quantized.numpy(), scale


def to_hex(value, bits=8):
    """Convert signed integer to hex (two's complement)."""
    if bits == 8:
        return f"{value & 0xFF:02X}" if value < 0 else f"{value & 0xFF:02X}"
    elif bits == 16:
        return f"{value & 0xFFFF:04X}"
    else:
        raise ValueError(f"Unsupported: {bits}")


def export_coe(data, filepath, bits=8):
    """Write quantized weights as Vivado .coe file."""
    flat = data.flatten()
    with open(filepath, 'w') as f:
        f.write(f"; INT{bits} quantized for KRIA KV260\n")
        f.write(f"; Shape: {data.shape}\n")
        f.write(f"memory_initialization_radix=16;\n")
        f.write(f"memory_initialization_vector=\n")
        for i, val in enumerate(flat):
            hex_str = to_hex(int(val), bits)
            sep = ',' if i < len(flat) - 1 else ';'
            f.write(f"{hex_str}{sep}\n")


def export_mem(data, filepath, bits=8):
    """Write quantized weights as Vivado .mem file."""
    flat = data.flatten()
    with open(filepath, 'w') as f:
        f.write(f"// INT{bits} quantized for KRIA KV260\n")
        f.write(f"// Shape: {data.shape}\n")
        for i, val in enumerate(flat):
            hex_str = to_hex(int(val), bits)
            f.write(f"@{i:08X} {hex_str}\n")


def export_int8_weights(fused_model, out_dir):
    """Export INT8 quantized weights as .coe and .mem files for FPGA BRAM."""
    os.makedirs(out_dir, exist_ok=True)
    scale_info = {}
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'layer7']

    for name in layer_names:
        layer = getattr(fused_model, name)
        conv = layer[0] if isinstance(layer, nn.Sequential) else layer

        # Quantize weights
        w_q, w_scale = quantize_weight_symmetric(conv.weight.data, bits=8)
        scale_info[f"{name}_weight"] = {
            'shape': list(conv.weight.shape),
            'scale': w_scale,
            'zero_point': 0,  # symmetric = zero_point is always 0
            'dtype': 'int8',
        }

        prefix = os.path.join(out_dir, f"{name}_weight")
        export_coe(w_q, f"{prefix}.coe", bits=8)
        export_mem(w_q, f"{prefix}.mem", bits=8)

        ks = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
        print(f"  {name}: Conv{ks[0]}×{ks[1]}({conv.in_channels}→{conv.out_channels})  "
              f"scale={w_scale:.6f}")

        # Quantize bias
        if conv.bias is not None:
            b_q, b_scale = quantize_weight_symmetric(conv.bias.data, bits=8)
            scale_info[f"{name}_bias"] = {
                'shape': list(conv.bias.shape),
                'scale': b_scale,
                'zero_point': 0,
                'dtype': 'int8',
            }

            prefix = os.path.join(out_dir, f"{name}_bias")
            export_coe(b_q, f"{prefix}.coe", bits=8)
            export_mem(b_q, f"{prefix}.mem", bits=8)

    # Save scale factors
    scale_path = os.path.join(out_dir, 'scale_factors_int8.json')
    with open(scale_path, 'w') as f:
        json.dump(scale_info, f, indent=2)

    return scale_info


# ── ONNX Export ───────────────────────────────────────────────────────────────
def export_quantized_onnx(fused_model, out_path):
    """Export the BN-fused float model to ONNX (for Vitis AI INT8 quantization)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dummy = torch.randn(1, 3, 128, 128)

    torch.onnx.export(
        fused_model,
        dummy,
        out_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  ONNX exported: {out_path} ({size_mb:.2f} MB)")


# ── Validation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate_model(model, loader, device, label="Model"):
    """Compute IoU-aware detection metrics on the validation set."""
    metrics = evaluate_model(model, loader, device, score_thresh=0.05, nms_iou=0.4, match_iou=0.5)
    print(
        f"  {label}:  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
        f"F1={metrics['f1']:.4f}  AP50={metrics['map50']:.4f}"
    )
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='INT8 Quantization for KRIA KV260')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth',
                        help='Path to trained checkpoint')
    parser.add_argument('--out-dir', type=str, default='exported_weights_int8',
                        help='Output directory for quantized weights')
    parser.add_argument('--num-calibration', type=int, default=100,
                        help='Number of calibration batches')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--skip-export', action='store_true',
                        help='Skip ONNX/weight export, just validate')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(base_dir, args.checkpoint)
    out_dir = os.path.join(base_dir, args.out_dir)

    device = torch.device('cpu')  # quantization runs on CPU
    print(f"Device: {device} (quantization requires CPU)")

    # ── Step 1: Load trained model ──
    print(f"\n{'='*60}")
    print(f"  Step 1: Loading trained checkpoint")
    print(f"{'='*60}")
    model = TinyDetector7Layer()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Epoch: {ckpt['epoch'] + 1}")
    print(f"  Parameters: {total_params:,}")

    # ── Step 2: Fuse BatchNorm ──
    print(f"\n{'='*60}")
    print(f"  Step 2: Fusing BatchNorm into Conv layers")
    print(f"{'='*60}")
    fused_model = fuse_model(model)
    fused_model.eval()

    # Verify fusion
    dummy = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out_orig = model(dummy)
        out_fused = fused_model(dummy)
    diff = (out_orig - out_fused).abs().max().item()
    print(f"  BN fusion max error: {diff:.2e}")
    assert diff < 1e-4, f"BN fusion error too large: {diff}"
    print(f"  Fusion verified ✓")

    # ── Step 3: PyTorch static quantization ──
    print(f"\n{'='*60}")
    print(f"  Step 3: Static INT8 Quantization")
    print(f"{'='*60}")

    # Configure quantization — symmetric INT8 for DPU compatibility
    qconfig = tq.QConfig(
        activation=tq.HistogramObserver.with_args(dtype=torch.quint8),
        weight=tq.default_per_channel_weight_observer,
    )

    # Prepare quantized model
    q_model = QuantizedTinyDetector(fused_model)
    q_model.eval()
    q_model.qconfig = qconfig
    tq.prepare(q_model, inplace=True)

    # Load calibration data
    print(f"\n  Loading calibration data...")
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=2,
    )

    # Calibrate
    calibrate(q_model, train_loader, device, num_batches=args.num_calibration)

    # Convert to quantized
    tq.convert(q_model, inplace=True)
    print(f"  INT8 conversion complete ✓")

    # ── Step 4: Validate quantized model ──
    print(f"\n{'='*60}")
    print(f"  Step 4: Accuracy Comparison (float32 vs INT8)")
    print(f"{'='*60}")

    print(f"\n  Float32 model:")
    float_metrics = validate_model(model, val_loader, device, label="Float32")

    print(f"\n  INT8 quantized model:")
    int8_metrics = validate_model(q_model, val_loader, device, label="INT8   ")

    # Print comparison
    f1_drop = float_metrics['f1'] - int8_metrics['f1']
    map50_drop = float_metrics['map50'] - int8_metrics['map50']
    print(f"\n  F1 drop from quantization:    {f1_drop:.4f}")
    print(f"  mAP50 drop from quantization: {map50_drop:.4f} "
          f"({'acceptable' if abs(map50_drop) < 0.05 else 'WARNING: significant'})")

    if args.skip_export:
        print(f"\n  Skipping export (--skip-export flag set)")
        return

    # ── Step 5: Export ──
    print(f"\n{'='*60}")
    print(f"  Step 5: Exporting INT8 weights for KV260")
    print(f"{'='*60}")

    # Export INT8 weights as .coe/.mem
    print(f"\n  Exporting BRAM weight files...")
    scale_info = export_int8_weights(fused_model, out_dir)

    # Export ONNX (BN-fused float model — Vitis AI does its own INT8 quantization)
    onnx_path = os.path.join(out_dir, 'model_fused.onnx')
    print(f"\n  Exporting ONNX model...")
    export_quantized_onnx(fused_model, onnx_path)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  ✓ INT8 Quantization Complete")
    print(f"{'='*60}")
    print(f"  Output directory: {out_dir}/")
    print(f"  Weight files:     .coe + .mem for each layer")
    print(f"  Scale factors:    scale_factors_int8.json")
    print(f"  ONNX model:       model_fused.onnx")
    print(f"  Target:           KRIA KV260 DPU (Xilinx INT8)")
    print(f"  Model size (INT8): {total_params / 1024 / 1024:.2f} MB")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
