# 16-Parallel CNN Hardware Accelerator for Kria KV260

## Overview
This repository contains the software and hardware source codes for an end-to-end Machine Learning pipeline featuring a **16-Parallel CNN Accelerator**, customized for deployment on the **Xilinx Kria KV260 Vision AI Starter Kit**. 

The pipeline begins with a PyTorch-based custom `TinyDetector` model, followed by Post-Training Quantization (PTQ) to strict 8-bit integers (INT8), and then complete deployment onto custom Verilog RTL hardware featuring highly parallelized AXI-Stream processing elements.

## Features
- **Custom CNN Model (TinyDetector)**: A lightweight convolutional neural network optimized for fast inference and bounding-box level object detection.
- **Robust INT8 Quantization**: Advanced symmetric INT8 quantization preserving model fidelity, handling zero-points, and tuning scaling factors explicitly for hardware bitwise parity.
- **Verilog RTL Accelerator**: Fully custom pipelined hardware implementation with 16-parallel MAC units, optimized line buffers, and parameterizable layer generation.
- **Kria KV260 Deployment Ready**: Designed with Vivado, exporting block designs and `.mem` weight files for real-time inference on the FPGA fabric via PYNQ on Ubuntu 22.04.

## Repository Structure
- `src/ml/`: PyTorch training, evaluation, INT8 quantization, scale calibration, and real-time Python test scripts.
- `src/verilog/`: Complete RTL source code including `cnn_compute_unit.v`, `conv_channel_proc.v`, `line_buffer.v`, etc., and the top-level AXI-stream wrapper.
- `src/kria_deployment/`: Target deployment scripts utilizing the PYNQ framework for the Kria board.

## Getting Started

### Prerequisites
- Python 3.10+
- PyTorch
- Icarus Verilog (for simulation parity tests)
- Vivado (for block design synthesis)

### Running Software Inference
```bash
cd src/ml
python detect_realtime.py
```

### Running Hardware Simulation
To perform a bit-accurate simulation of the first layer and compare against PyTorch outputs:
```bash
python run_iverilog_e2e.py
```

## Authors
- Chaitanya63059
