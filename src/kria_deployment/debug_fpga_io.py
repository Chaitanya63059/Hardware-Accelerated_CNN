"""
Debug script to verify FPGA input preprocessing and compare with CPU simulation.

This helps identify if the saturation is due to:
1. Input preprocessing (image normalization)
2. Weight loading issues
3. Hardware accumulation overflow
"""

import numpy as np
import cv2
import os

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def main():
    IMAGE_PATH = "/home/ubuntu/cnn_accelerator/000000000139.jpg"
    INPUT_SIZE = 126
    WEIGHT_DIR = "/home/ubuntu/cnn_accelerator"
    
    print("=" * 60)
    print("  FPGA I/O Debug - Check Input Preprocessing")
    print("=" * 60)
    
    # === Check Input Preprocessing ===
    print("\n[1/3] Input Preprocessing Analysis")
    print("=" * 60)
    
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    img_bgr = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]
    
    img_resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
    print(f"  Raw resized image: dtype={img_resized.dtype}, range=[{img_resized.min()}, {img_resized.max()}]")
    
    # Step 1: Normalize to [0, 1]
    img_norm_01 = img_resized.astype(np.float32) / 255.0
    print(f"  After /255: dtype={img_norm_01.dtype}, range=[{img_norm_01.min():.3f}, {img_norm_01.max():.3f}]")
    
    # Step 2: Apply ImageNet normalization
    img_normalized = (img_norm_01 - MEAN) / STD
    print(f"  After ImageNet norm: dtype={img_normalized.dtype}, range=[{img_normalized.min():.3f}, {img_normalized.max():.3f}]")
    print(f"    Std of normalized: {img_normalized.std():.3f} (should be ~1.0)")
    
    # Step 3: Scale to int8 range
    img_scaled = img_normalized * 128.0
    print(f"  After *128: dtype={img_scaled.dtype}, range=[{img_scaled.min():.1f}, {img_scaled.max():.1f}]")
    
    # Step 4: Clip to int8 range
    img_int8 = np.clip(img_scaled, -128, 127).astype(np.int8)
    print(f"  After clip to int8: dtype={img_int8.dtype}, range=[{img_int8.min()}, {img_int8.max()}]")
    
    # Step 5: Convert to unsigned for hardware
    img_uint8_hw = img_int8.astype(np.int16) + 128
    print(f"  After +128 (for HW): dtype={img_uint8_hw.dtype}, range=[{img_uint8_hw.min()}, {img_uint8_hw.max()}]")
    
    img_uint8_final = img_uint8_hw.astype(np.uint8)
    print(f"  Final for HW: dtype={img_uint8_final.dtype}, range=[{img_uint8_final.min()}, {img_uint8_final.max()}]")
    
    # === Load and Display Raw FPGA Outputs ===
    print("\n[2/3] Raw FPGA Output Analysis")
    print("=" * 60)
    
    raw_layer7_path = os.path.join(WEIGHT_DIR, "fpga_output_raw_layer7.npy")
    if os.path.exists(raw_layer7_path):
        layer7_raw = np.load(raw_layer7_path)
        print(f"  Layer7 raw output: shape={layer7_raw.shape}")
        print(f"  Range: [{layer7_raw.min()}, {layer7_raw.max()}]")
        print(f"  All values equal? {np.allclose(layer7_raw, layer7_raw.flat[0])}")
        print(f"  Unique values: {len(np.unique(layer7_raw))}")
        
        if len(np.unique(layer7_raw)) > 1:
            print(f"  Distribution:")
            for ch in range(layer7_raw.shape[0]):
                ch_data = layer7_raw[ch]
                print(f"    Channel {ch}: min={ch_data.min()}, max={ch_data.max()}, mean={ch_data.mean():.1f}")
        else:
            print(f"  WARNING: All values are identical ({layer7_raw.flat[0]})")
            print(f"  This indicates 100% saturation!")
    else:
        print(f"  ERROR: fpga_output_raw_layer7.npy not found at {raw_layer7_path}")
        print(f"  Run run_fpga.py first")
        return
    
    # === Compare scaling ===
    print("\n[3/3] Scaling Analysis for Post-Processing")
    print("=" * 60)
    
    print(f"  Raw ReLU output: [0, 127]")
    print(f"  Scaled by (x - 64) / 25.6:")
    print(f"    0 -> {(0 - 64) / 25.6:.2f}")
    print(f"    64 -> {(64 - 64) / 25.6:.2f}")
    print(f"    127 -> {(127 - 64) / 25.6:.2f}")
    
    if len(np.unique(layer7_raw)) == 1:
        val = layer7_raw.flat[0]
        scaled = (val - 64.0) / 25.6
        sig = sigmoid(scaled)
        print(f"\n  Current saturation point: {val}")
        print(f"  Scaled value: {scaled:.3f}")
        print(f"  Sigmoid(scaled): {sig:.3f}")
    
    print("\n" + "=" * 60)
    print("  RECOMMENDATIONS:")
    print("=" * 60)
    if len(np.unique(layer7_raw)) == 1 and layer7_raw.flat[0] == 127:
        print("  ✗ Layer7 is 100% saturated at 127")
        print("  Possible causes:")
        print("    1. Input values too large → check preprocessing")
        print("    2. Weights not loaded correctly → check BRAM writes")
        print("    3. Accumulator overflow → check weight ranges")
        print("    4. Bias values wrong → check bias loading")
    else:
        print("  ✓ Output looks reasonable")


if __name__ == '__main__':
    main()
