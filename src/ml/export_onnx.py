import os
import argparse
import torch
import torch.onnx
from model import TinyDetector7Layer, fuse_model
from config import NUM_OUTPUTS

def main():
    parser = argparse.ArgumentParser(description='Export weights to ONNX')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth')
    parser.add_argument('--out', type=str, default='exported_weights/model_fused.onnx')
    parser.add_argument('--fuse-bn', action='store_true', default=True, help='Fuse BatchNorm before export')
    args = parser.parse_args()

    device = torch.device('cpu')
    model = TinyDetector7Layer()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    if args.fuse_bn:
        print("Fusing BatchNorm into Conv layers before ONNX export...")
        model = fuse_model(model)
        model.eval()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Export to ONNX
    print(f"Exporting model to {args.out} ...")
    dummy_input = torch.randn(1, 3, 128, 128, device=device)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        args.out, 
        export_params=True,
        opset_version=11, 
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"Success! ONNX model size: {size_mb:.2f} MB, outputs per cell: {NUM_OUTPUTS}")

if __name__ == '__main__':
    main()
