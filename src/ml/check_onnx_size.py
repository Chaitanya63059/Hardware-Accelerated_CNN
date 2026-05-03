import os
import torch
from model import TinyDetector7Layer

model = TinyDetector7Layer()
dummy_input = torch.randn(1, 3, 128, 128)
torch.onnx.export(model, dummy_input, "temp_model.onnx", 
                  opset_version=11, 
                  input_names=['input'], 
                  output_names=['output'])

size_mb = os.path.getsize("temp_model.onnx") / (1024 * 1024)
print(f"ONNX Size (Float32): {size_mb:.2f} MB")
