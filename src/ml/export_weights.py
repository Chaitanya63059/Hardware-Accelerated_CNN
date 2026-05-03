"""
Export trained TinyDetector7Layer weights to .coe/.mem files for Vivado BRAM.

Steps:
  1. Load trained checkpoint
  2. Fuse BatchNorm into Conv weights
  3. Quantize to INT8 or INT16
  4. Export each layer as a .coe file for Vivado Block RAM initialization

Usage:
    python export_weights.py                                    # default INT8
    python export_weights.py --bits 16 --checkpoint best.pth   # INT16
"""

import os
import argparse
import numpy as np
import torch

from model import TinyDetector7Layer, fuse_model


def quantize_weights(weight_tensor, bits=8, downscale=1.0):
    """
    Symmetric quantization of weight tensor to fixed-point integer.

    Args:
        weight_tensor: float tensor
        bits: 8 or 16
        downscale: artificially divide quantized weights by this factor to prevent overflow

    Returns:
        quantized: numpy int array
        scale: float scale factor (for dequantization: float_val = int_val * scale)
    """
    max_val = weight_tensor.abs().max().item()
    if max_val == 0:
        max_val = 1e-8

    # Artificially increase max_val so the quantized integers become smaller
    max_val = max_val * downscale

    qmax = (2 ** (bits - 1)) - 1
    scale = max_val / qmax
    quantized = torch.round(weight_tensor / scale).clamp(-qmax, qmax).to(torch.int32)

    return quantized.numpy(), scale


def to_hex_string(value, bits=8):
    """Convert signed integer to hex string of appropriate width."""
    if bits == 8:
        if value < 0:
            value = (1 << 8) + value  # two's complement
        return f"{value & 0xFF:02X}"
    elif bits == 16:
        if value < 0:
            value = (1 << 16) + value
        return f"{value & 0xFFFF:04X}"
    else:
        raise ValueError(f"Unsupported bit width: {bits}")


def export_coe(data, filepath, bits=8, radix=16):
    """
    Write quantized weights as a Vivado .coe file.
    Format: one coefficient per line, hex radix.
    """
    flat = data.flatten()
    with open(filepath, 'w') as f:
        f.write(f"; Exported from TinyDetector7Layer\n")
        f.write(f"; Shape: {data.shape}, Bits: {bits}\n")
        f.write(f"memory_initialization_radix={radix};\n")
        f.write(f"memory_initialization_vector=\n")
        for i, val in enumerate(flat):
            hex_str = to_hex_string(int(val), bits)
            if i < len(flat) - 1:
                f.write(f"{hex_str},\n")
            else:
                f.write(f"{hex_str};\n")


def export_mem(data, filepath, bits=8):
    """
    Write quantized weights as a Vivado .mem file.
    Format: @address hex_value
    """
    flat = data.flatten()
    with open(filepath, 'w') as f:
        f.write(f"// Exported from TinyDetector7Layer\n")
        f.write(f"// Shape: {data.shape}, Bits: {bits}\n")
        for i, val in enumerate(flat):
            hex_str = to_hex_string(int(val), bits)
            f.write(f"@{i:08X} {hex_str}\n")


def main():
    parser = argparse.ArgumentParser(description='Export weights for FPGA')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth')
    parser.add_argument('--out-dir', type=str, default='exported_weights')
    parser.add_argument('--bits', type=int, default=8, choices=[8, 16])
    parser.add_argument('--format', type=str, default='both', choices=['coe', 'mem', 'both'])
    parser.add_argument('--downscale', type=float, default=8.0, help='Factor to reduce weight magnitude for overflow prevention')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(base_dir, args.checkpoint)
    out_dir = os.path.join(base_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {ckpt_path}")
    model = TinyDetector7Layer()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Fuse BN
    print("Fusing BatchNorm into Conv layers...")
    fused_model = fuse_model(model)

    # Verify fusion correctness
    dummy = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out_orig = model(dummy)
        out_fused = fused_model(dummy)
    diff = (out_orig - out_fused).abs().max().item()
    print(f"  BN fusion max error: {diff:.2e} (should be < 1e-5)")
    assert diff < 1e-4, f"BN fusion error too large: {diff}"

    # Export each layer
    print(f"\nExporting weights (INT{args.bits})...")
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'layer7']
    scale_info = {}

    for name in layer_names:
        layer = getattr(fused_model, name)
        if isinstance(layer, torch.nn.Sequential):
            conv = layer[0]
        else:
            conv = layer

        layer_num = int(name[-1])
        # Downscale weights to artificially increase w_scale, which reduces the required out_shift.
        if layer_num >= 4:
            ds_factor = 4.0
        elif layer_num >= 2:
            ds_factor = 2.0
        else:
            ds_factor = 1.0

        # --- ImageNet Normalization Folding for Layer 1 ---
        if name == 'layer1':
            # The model was trained on: x_norm = (x_int / 255.0 - mean) / std
            # Verilog computes: y = W * x_int + b
            # To make them equivalent:
            # y = W_float * ((x_int / 255) - mean) / std + b_float
            # y = [W_float / (255 * std)] * x_int + [b_float - sum(W_float * mean / std)]
            
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            
            # Adjust bias first using the ORIGINAL W_float
            b_adj = (conv.weight.data * (mean / std)).sum(dim=(1, 2, 3))
            if conv.bias is None:
                conv.bias = nn.Parameter(-b_adj)
            else:
                conv.bias.data = conv.bias.data - b_adj
                
            # Adjust weights
            conv.weight.data = conv.weight.data / (255.0 * std)

        # (Logit Shift for Layer 7 is now handled directly in the integer quantization below)

        # Calculate base scale for 8-bit weights
        w_max = torch.max(torch.abs(conv.weight.data)).item() * ds_factor
        if w_max == 0: w_max = 1e-6
        w_scale = w_max / 127.0
        
        # Ensure bias fits in 16 bits!
        if conv.bias is not None:
            b_max = torch.max(torch.abs(conv.bias.data)).item()
            min_w_scale = b_max / 32767.0
            if min_w_scale > w_scale:
                w_scale = min_w_scale
                
        w_q_temp = torch.round(conv.weight.data / w_scale).clamp(-127, 127)
        
        # Calculate out_shift to prevent 127 accumulator saturation!
        x_max = 255.0 if name == 'layer1' else 127.0
        worst_channel_sum = torch.max(torch.sum(torch.abs(w_q_temp), dim=(1, 2, 3))).item()
        max_acc = x_max * worst_channel_sum
        
        import math
        req_shift_val = max_acc / 127.0
        out_shift = int(math.ceil(math.log2(req_shift_val))) if req_shift_val > 1 else 0
        out_shift = max(0, min(31, out_shift))  # clamp
        
        # --- HARDCODE LAYER 7 SHIFT ---
        # Theoretical out_shift is too pessimistic and crushes precision.
        # We manually force out_shift=4 to preserve precision and allow bias shifting.
        if name == 'layer7':
            out_shift = 4

        w_q = w_q_temp.to(torch.int32).numpy()
        scale_info[f"{name}_weight"] = {'shape': list(conv.weight.shape), 'scale': w_scale, 'out_shift': out_shift}

        prefix = os.path.join(out_dir, f"{name}_weight")
        if args.format in ('coe', 'both'):
            export_coe(w_q, f"{prefix}.coe", bits=args.bits)
        if args.format in ('mem', 'both'):
            export_mem(w_q, f"{prefix}.mem", bits=args.bits)

        print(f"  {name} weight: shape={list(conv.weight.shape)}, scale={w_scale:.6f}, out_shift={out_shift}")

        # Quantize bias (Biases must match the MAC accumulator scale and be 16-bit)
        if conv.bias is not None:
            mac_scale = w_scale
            b_qmax = (2 ** 15) - 1
            
            # --- HARDWARE BIAS OFFSET FOR LAYER 7 ---
            b_q_float = conv.bias.data / mac_scale
            
            if name == 'layer7':
                # We want to shift the output up by 64.
                # out_shift = 4, so 64 * 2^4 = 1024.
                hw_offset = 64 * (2 ** out_shift)
                b_q_float += hw_offset
            
            b_q = torch.round(b_q_float).clamp(-b_qmax, b_qmax).to(torch.int32).numpy()
            b_scale = mac_scale
            
            scale_info[f"{name}_bias"] = {'shape': list(conv.bias.shape), 'scale': b_scale, 'downscale': 1.0}

            prefix = os.path.join(out_dir, f"{name}_bias")
            if args.format in ('coe', 'both'):
                export_coe(b_q, f"{prefix}.coe", bits=16)
            if args.format in ('mem', 'both'):
                export_mem(b_q, f"{prefix}.mem", bits=16)

            print(f"  {name} bias:   shape={list(conv.bias.shape)}, scale={b_scale:.6f}")

    # Save scale factors for dequantization in Verilog
    import json
    scale_path = os.path.join(out_dir, 'scale_factors.json')
    with open(scale_path, 'w') as f:
        json.dump(scale_info, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Exported {len(layer_names)} layers to {out_dir}/")
    print(f"  Scale factors saved to {scale_path}")
    print(f"  Bit width: INT{args.bits}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
