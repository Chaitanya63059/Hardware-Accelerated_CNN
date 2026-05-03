"""
Compare FPGA and ARM CPU simulation outputs to identify where divergence occurs.

Usage:
    After running both run_fpga.py and run_arm.py, this compares their outputs.
"""

import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Compare FPGA and ARM outputs')
    parser.add_argument('--fpga-raw', type=str, default='/home/ubuntu/cnn_accelerator/fpga_output_raw_layer7.npy',
                        help='Path to FPGA raw layer7 output')
    parser.add_argument('--arm-raw', type=str, default='/home/ubuntu/cnn_accelerator/arm_output.npy',
                        help='Path to ARM output')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  FPGA vs ARM Output Comparison")
    print("=" * 60)
    
    # Load FPGA output
    if not os.path.exists(args.fpga_raw):
        print(f"ERROR: FPGA output not found: {args.fpga_raw}")
        return
    
    fpga_out = np.load(args.fpga_raw)
    print(f"\nFPGA Output (Layer7):")
    print(f"  Shape: {fpga_out.shape}")
    print(f"  Dtype: {fpga_out.dtype}")
    print(f"  Range: [{fpga_out.min()}, {fpga_out.max()}]")
    print(f"  Mean: {fpga_out.mean():.2f}")
    print(f"  Std: {fpga_out.std():.2f}")
    
    # Load ARM output
    if not os.path.exists(args.arm_raw):
        print(f"\nWARNING: ARM output not found: {args.arm_raw}")
        print("  Run run_arm.py first to generate ARM output")
        return
    
    arm_out = np.load(args.arm_raw)
    print(f"\nARM Output (Layer7):")
    print(f"  Shape: {arm_out.shape}")
    print(f"  Dtype: {arm_out.dtype}")
    print(f"  Range: [{arm_out.min()}, {arm_out.max()}]")
    print(f"  Mean: {arm_out.mean():.2f}")
    print(f"  Std: {arm_out.std():.2f}")
    
    # Compare
    print("\n" + "=" * 60)
    print("  Comparison:")
    print("=" * 60)
    
    if fpga_out.shape != arm_out.shape:
        print("ERROR: Shapes don't match!")
        print(f"  FPGA: {fpga_out.shape}")
        print(f"  ARM:  {arm_out.shape}")
        return
    
    # Calculate statistics
    if fpga_out.dtype == np.float32 and arm_out.dtype == np.float32:
        diff = fpga_out - arm_out
        abs_diff = np.abs(diff)
        rel_diff = np.abs(diff) / (np.abs(arm_out) + 1e-6)
        
        print(f"  Absolute difference:")
        print(f"    Max: {abs_diff.max():.4f}")
        print(f"    Mean: {abs_diff.mean():.4f}")
        print(f"    Median: {np.median(abs_diff):.4f}")
        
        print(f"  Relative difference (%):")
        print(f"    Max: {rel_diff.max() * 100:.2f}%")
        print(f"    Mean: {rel_diff.mean() * 100:.2f}%")
        
        # Find most different elements
        top_k = 5
        flat_idx = np.argsort(abs_diff.flat)[-top_k:][::-1]
        print(f"\n  Top {top_k} most different values:")
        for rank, idx in enumerate(flat_idx, 1):
            pos = np.unravel_index(idx, fpga_out.shape)
            print(f"    {rank}. [{pos}]: FPGA={fpga_out[pos]:.1f}, ARM={arm_out[pos]:.1f}, diff={abs_diff[pos]:.1f}")
    else:
        # Just compare raw values for uint8
        if np.array_equal(fpga_out, arm_out):
            print("  ✓ Outputs are IDENTICAL")
        else:
            diff_count = np.sum(fpga_out != arm_out)
            print(f"  ✗ Outputs DIFFER in {diff_count}/{fpga_out.size} values ({100*diff_count/fpga_out.size:.1f}%)")
    
    # Check for saturation
    print("\n" + "=" * 60)
    print("  Saturation Check:")
    print("=" * 60)
    
    fpga_saturated = np.sum(fpga_out == fpga_out.max())
    arm_saturated = np.sum(arm_out == arm_out.max())
    
    print(f"  FPGA saturated values: {fpga_saturated}/{fpga_out.size} ({100*fpga_saturated/fpga_out.size:.1f}%)")
    print(f"  ARM saturated values: {arm_saturated}/{arm_out.size} ({100*arm_saturated/arm_out.size:.1f}%)")


if __name__ == '__main__':
    main()
