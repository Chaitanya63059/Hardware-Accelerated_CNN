"""
Inspect weight and bias files to verify they're being loaded correctly.
"""

import numpy as np
import os

def parse_mem_file(filepath, is_bias=False):
    """Parse .mem file -> list of signed int8 or int16 values."""
    values = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('//') or line == '':
                continue
            parts = line.split()
            if len(parts) == 2:
                hex_val = int(parts[1], 16)
                if is_bias:
                    if hex_val > 32767:
                        hex_val -= 65536
                else:
                    if hex_val > 127:
                        hex_val -= 256
                values.append(hex_val)
    dtype = np.int16 if is_bias else np.int8
    return np.array(values, dtype=dtype)

def main():
    WEIGHT_DIR = "../exported_weights"
    LAYERS = [
        ("layer1",   3,  32, 4),
        ("layer2",  32,  64, 3),
        ("layer3",  64, 128, 4),
        ("layer4", 128, 256, 3),
        ("layer5", 256, 256, 3),
        ("layer6", 256, 128, 3),
        ("layer7", 128,   6, 1),
    ]
    
    print("=" * 60)
    print("  Weight and Bias File Inspection")
    print("=" * 60)
    
    for layer_name, c_in, c_out, k_size in LAYERS:
        w_path = os.path.join(WEIGHT_DIR, f"{layer_name}_weight.mem")
        b_path = os.path.join(WEIGHT_DIR, f"{layer_name}_bias.mem")
        
        print(f"\n{layer_name}:")
        
        if os.path.exists(w_path):
            weights = parse_mem_file(w_path)
            expected = c_out * c_in * k_size * k_size
            print(f"  Weights: {len(weights)} values (expected {expected})")
            print(f"    Range: [{weights.min()}, {weights.max()}]")
            print(f"    Mean: {weights.mean():.2f}, Std: {weights.std():.2f}")
            if len(weights) != expected:
                print(f"    WARNING: Size mismatch!")
        else:
            print(f"  Weights: NOT FOUND at {w_path}")
        
        if os.path.exists(b_path):
            biases = parse_mem_file(b_path, is_bias=True)
            print(f"  Biases: {len(biases)} values (expected {c_out})")
            print(f"    Range: [{biases.min()}, {biases.max()}]")
            print(f"    Mean: {biases.mean():.2f}, Std: {biases.std():.2f}")
            if len(biases) < c_out:
                print(f"    WARNING: Not enough biases!")
        else:
            print(f"  Biases: NOT FOUND at {b_path}")
    
    print("\n" + "=" * 60)
    print("  FPGA Hardware Parameters:")
    print("=" * 60)
    print("  Input pixel range (unsigned): [0, 255]")
    print("  Weight range (signed): [-128, 127]")
    print("  Bias range (signed): [-32768, 32767]")
    print("  Accumulator: 32-bit signed")
    print("  Output (post-ReLU): [0, 127]")
    print("\n  For NO saturation, need:")
    print("    |sum(weight[i] * input[i]) + bias| < 128")

if __name__ == '__main__':
    main()
