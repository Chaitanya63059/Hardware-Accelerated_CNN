import time
import torch
from model import TinyDetector7Layer

# 1. Load model and dummy input
model = TinyDetector7Layer()
model.eval()

# Input shape observed from model.py
dummy_input = torch.randn(1, 3, 128, 128)

# 2. Warm-up
print("Warming up...")
for _ in range(10):
    with torch.no_grad():
        _ = model(dummy_input)

# 3. Actual Measurement
print("Starting measurement...")
start_time = time.perf_counter()
num_iterations = 100

with torch.no_grad():
    for _ in range(num_iterations):
        _ = model(dummy_input)

end_time = time.perf_counter()

total_time = end_time - start_time
software_latency = (total_time / num_iterations) * 1000 # in milliseconds
software_throughput = 1000 / software_latency           # in FPS

print("-" * 30)
print(f"Software Latency: {software_latency:.2f} ms")
print(f"Software Throughput: {software_throughput:.2f} FPS")
print(f"Total time for {num_iterations} inferences: {total_time:.2f} s")
