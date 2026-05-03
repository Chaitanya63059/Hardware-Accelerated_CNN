# CNN Accelerator on fpga

## Project Details and End-to-End Pipeline
This project presents an end-to-end framework for deploying a highly parallelized Convolutional Neural Network (CNN) onto an FPGA. The goal is to accelerate object detection inferences on edge devices by moving complex tensor calculations from the software domain (ARM processor) directly into dedicated hardware (Programmable Logic). 

The pipeline begins with a lightweight PyTorch CNN called `TinyDetector`, specifically designed for bounding-box regression with minimal parameters. Once trained on a custom dataset, the model undergoes Post-Training Quantization (PTQ) to convert all floating-point weights and activations into 8-bit integers (INT8). This step is critical, as fixed-point arithmetic significantly reduces the hardware footprint and latency on the FPGA. Symmetric quantization and custom scaling factors are employed to maintain model accuracy and ensure seamless zero-point handling.

After quantization, the neural network architecture is translated into Verilog RTL. The hardware design is highly customized around a 16-parallel Processing Element (PE) matrix, accelerating channel-wise convolutions. Vivado is used to package these modules, attach AXI-Stream interfaces, and synthesize the bitstream (`.bit`) for the Xilinx Kria KV260 platform. Finally, the PYNQ framework is utilized on the KV260 board running Ubuntu 22.04 to stream images to the hardware accelerator, retrieve the computed feature maps, apply Non-Maximum Suppression (NMS), and render the final bounding boxes onto the image in real-time.

## Hardware Modules

### `cnn_pipeline_top.v`
This top-level module serves as the primary wrapper for the entire CNN accelerator, interfacing directly with the outside world via AXI-Stream protocols. It orchestrates the flow of incoming image data and outgoing detection bounding boxes, maintaining pipeline synchronization and ensuring no data loss occurs between the processor and the programmable logic.

The module connects the internal line buffers, window generators, and the core compute unit. It handles all control signals, ensuring data validity across the clock boundaries and seamlessly managing backpressure from downstream AXI components so the accelerator does not stall unexpectedly.

### `cnn_compute_unit.v`
This module acts as the primary parallel processing engine of the CNN accelerator. It instantiates multiple individual channel processing elements and aggregates their results to compute the final output activations, making it the mathematical core of the design.

By employing 16-parallelism, the compute unit drastically increases throughput compared to a traditional serial approach. It reads localized feature map windows and corresponding weights simultaneously, producing multiple channels of output in a single clock cycle to achieve maximum hardware utilization.

### `conv_channel_proc.v`
Operating at the lowest level of the compute hierarchy, this module is responsible for the actual multiply-accumulate (MAC) arithmetic for a single output channel. It performs precise 8-bit signed and unsigned multiplications between input activations and layer weights.

Additionally, this module integrates the crucial non-linear activation function (ReLU) and saturation logic. By accurately clamping mathematical results to the `[0, 127]` range, it ensures the INT8 datatype properties are strictly maintained, preventing overflow errors in deeper network layers.

### `conv_engine.v`
The convolution engine abstracts the sliding window process, systematically stepping through the feature map and providing the necessary operands to the downstream compute units. It coordinates the traversal logic, shifting the operation window correctly across rows and columns.

This module significantly simplifies the control logic overhead from the main processing path. By systematically and continuously feeding the MAC arrays with the correct sequences of data, it maintains the high utilization rate of the spatial multipliers.

### `line_buffer.v`
Crucial for optimizing memory bandwidth, the line buffer caches incoming sequential pixel data streams to construct localized 2D patches. This entirely eliminates the need to fetch the same pixel multiple times from external DDR memory, which would otherwise be a massive bottleneck.

By acting as a First-In-First-Out (FIFO) queue for entire rows of an image, it ensures that as a new pixel arrives, an entire column of the sliding window can be updated instantly. This allows continuous streaming operations without data stalling.

### `window_gen_3x3.v` / `window_gen_4x4.v`
The window generator modules work in tandem with the line buffers to extract the specific 3x3 or 4x4 spatial patch required by the convolution kernel at any given clock cycle. They assemble the newly cached rows into a contiguous, easily readable 2D array.

This hardware logic ensures that the `cnn_compute_unit` is always presented with valid, aligned data. It strictly manages the boundary padding, ensuring zero values are properly injected when the convolution filter operates at the edges of the image.

## Repository Structure
- **`src/ml/`**: Contains the core Python machine learning logic. Files include the neural network definition (`model.py`), `dataset.py` for loading the image data, `train.py` for optimizing the model, `quantize_int8.py` for conversion to fixed-point precision, and `detect_realtime.py` for running bounding box predictions.
- **`src/verilog/`**: Contains the RTL source code modules described above.
- **`src/kria_deployment/`**: Contains execution scripts (`run_arm.py` and `run_fpga.py`) along with the synthesized `.bit` and weight configuration files to run the accelerator on the physical Kria KV260 board.
