import os
import subprocess
import copy
from run_eval_float import decode_predictions, nms, CLASS_NAMES
from PIL import Image, ImageDraw
import numpy as np
import torch
from model import TinyDetector7Layer, fuse_model

def to_hex(val, bits=8):
    if bits == 8:
        if val < 0: val += 256
        return f"{int(val) & 0xFF:02X}"
    else:
        if val < 0: val += 65536
        return f"{int(val) & 0xFFFF:04X}"

def read_quantized_weights_and_biases():
    # We will use the fused PyTorch model to get the weights,
    # then quantize them exactly like export_weights.py did, 
    # but we hold them in memory to feed to Verilog.
    device = torch.device('cpu')
    model = TinyDetector7Layer().to(device)
    ckpt = torch.load('/home/chaitanya/Desktop/mini_proj_ml/checkpoints/best.pth', map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    fused = fuse_model(model)
    fused.eval()

    def q_w(w, bits=8, downscale=1.0):
        max_v = w.abs().max().item()
        if max_v == 0: max_v = 1e-8
        max_v = max_v * downscale
        qmax = (2**(bits-1))-1
        scale = max_v / qmax
        ans = torch.round(w / scale).clamp(-qmax, qmax).detach().numpy()
        return ans, scale

    layers_cfg = {}
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'layer7']
    for nm in layer_names:
        l = getattr(fused, nm)
        conv = l[0] if isinstance(l, torch.nn.Sequential) else l
        
        # We need a clone of weight/bias so we don't permanently modify the fused model in memory
        # since we still need it for layers 2-7 software pass
        w_data = conv.weight.data.clone()
        if conv.bias is not None:
            b_data = conv.bias.data.clone()
        else:
            b_data = torch.zeros(conv.weight.shape[0])
            
        layer_num = int(nm[-1])
        ds_factor = 4.0 if layer_num >= 4 else (2.0 if layer_num >= 2 else 1.0)
        
        # --- ImageNet Normalization Folding for Layer 1 ---
        if nm == 'layer1':
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            
            b_adj = (w_data * (mean / std)).sum(dim=(1, 2, 3))
            b_data = b_data - b_adj
            w_data = w_data / (255.0 * std)
            
        # Calculate base scale for 8-bit weights
        w_max = torch.max(torch.abs(w_data)).item()
        if w_max == 0: w_max = 1e-6
        w_scale = w_max / 127.0
        
        # Ensure bias fits in 16 bits!
        b_max = torch.max(torch.abs(b_data)).item()
        min_w_scale = b_max / 32767.0
        if min_w_scale > w_scale:
            w_scale = min_w_scale
            
        w_q_temp = torch.round(w_data / w_scale).clamp(-127, 127)
        
        # Calculate out_shift to prevent 127 accumulator saturation!
        x_max = 255.0 if nm == 'layer1' else 127.0
        worst_channel_sum = torch.max(torch.sum(torch.abs(w_q_temp), dim=(1, 2, 3))).item()
        max_acc = x_max * worst_channel_sum
        
        import math
        req_shift_val = max_acc / 127.0
        out_shift = int(math.ceil(math.log2(req_shift_val))) if req_shift_val > 1 else 0
        out_shift = max(0, min(31, out_shift))  # clamp
        
        w_q = w_q_temp.detach().numpy()
        b_q = torch.round(b_data / w_scale).clamp(-32767, 32767).detach().numpy()
            
        layers_cfg[nm] = {'w': w_q, 'b': b_q, 'w_scale': w_scale, 'out_shift': out_shift}
    return layers_cfg

def run_verilog_pass(padded_in, w_chunk, b_chunk, stride, k_mode, out_shift, new_folder_dir):
    C_in, H, W = padded_in.shape
    num_lanes = w_chunk.shape[0]
    K = w_chunk.shape[2]

    dyn_w = W
    dyn_h = H
    dyn_ch = C_in
    dyn_mode = k_mode
    dyn_stride = stride
    dyn_w_cnt = C_in * 16

    # Write weights.hex
    with open(os.path.join(new_folder_dir, "weights.hex"), "w") as f:
        for lane in range(16):
            for cin in range(C_in):
                for idx in range(16):
                    v = 0
                    if lane < num_lanes and idx < K*K:
                        r = idx // K
                        c = idx % K
                        v = w_chunk[lane, cin, r, c]
                    f.write(to_hex(v, 8) + "\n")
    
    # Write biases.hex
    with open(os.path.join(new_folder_dir, "biases.hex"), "w") as f:
        for lane in range(16):
            v = 0
            if lane < num_lanes:
                v = b_chunk[lane]
            f.write(to_hex(v, 16) + "\n")

    # Write image_in.hex
    flat_img = padded_in.flatten()
    with open(os.path.join(new_folder_dir, "image_in.hex"), "w") as f:
        for v in flat_img:
            v_int = int(v)
            if v_int < 0: v_int = 0
            if v_int > 255: v_int = 255
            f.write(to_hex(v_int, 8) + "\n")


    # Run
    cmd_run = [
        "vvp", "sim.vvp",
        f"+IMG_WIDTH={dyn_w}",
        f"+IMG_HEIGHT={dyn_h}",
        f"+IN_CHANNELS={dyn_ch}",
        f"+KERNEL_MODE={dyn_mode}",
        f"+STRIDE={dyn_stride}",
        f"+WEIGHT_CNT={dyn_w_cnt}",
        f"+IN_PIXELS={len(flat_img)}",
        f"+OUT_SHIFT={out_shift}"
    ]
    subprocess.run(cmd_run, cwd=new_folder_dir, check=True)

    # Parse image_out.hex
    out_lines = []
    with open(os.path.join(new_folder_dir, "image_out.hex"), "r") as f:
        for line in f:
            if line.strip():
                val = line.strip().replace('x', '0').replace('\x00', '0')
                if val == '' or val == '000': val = '0'
                try:
                    out_lines.append(int(val, 16))
                except ValueError:
                    out_lines.append(0)
    
    out_np = np.array(out_lines, dtype=np.uint8)
    L = 16
    out_size = len(out_np) // L
    side = int(np.sqrt(out_size))
    out_tensor = out_np.reshape(L, side, side)
    return out_tensor[:num_lanes] # Return valid channels


def run_full_iverilog_pipeline():
    new_folder_dir = "/home/chaitanya/Desktop/new_folder"
    layers_cfg = read_quantized_weights_and_biases()

    # We need the actual PyTorch model for layers 2-7
    device = torch.device('cpu')
    model = TinyDetector7Layer().to(device)
    ckpt = torch.load('/home/chaitanya/Desktop/mini_proj_ml/checkpoints/best.pth', map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    fused = fuse_model(model)
    fused.eval()

    # Compile simulation ONCE
    print("Compiling Verilog simulation...")
    cmd_compile = ["iverilog", "-g2012", "-o", "sim.vvp", "cnn_pipeline_top.v", "cnn_compute_unit.v", "conv_channel_proc.v", "conv_engine.v", "line_buffer.v", "window_gen_3x3.v", "window_gen_4x4.v", "tb_cnn_pipeline.v"]
    subprocess.run(cmd_compile, cwd=new_folder_dir, check=True)

    # Inputs
    img_path = 'person.jpg'
    orig_img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = orig_img.size
    img_pil = orig_img.resize((128, 128), Image.Resampling.BILINEAR)
    img_arr = np.array(img_pil, dtype=np.int32)
    img_arr = np.transpose(img_arr, (2, 0, 1)) # (3, 128, 128) -> 0 to 255 natively
    
    current_map = img_arr

    print(f"Starting Verilog simulation for Layer 1... Image input {current_map.shape}")

    # --- HARDWARE EXECUTION (Layer 1) ---
    lname = 'layer1'
    k_mode = 2 # 4x4
    stride = 2
    pad = 1

    print(f"Running {lname} in Verilog...")
    w_all = layers_cfg[lname]['w']
    b_all = layers_cfg[lname]['b']
    C_out, C_in, K, _ = w_all.shape

    padded_in = np.pad(current_map, ((0,0), (pad,pad), (pad,pad)), 'constant', constant_values=0)
    out_chunks = []
    
    for oc in range(0, C_out, 16):
        oc_end = min(oc + 16, C_out)
        w_chunk = w_all[oc:oc_end]
        b_chunk = b_all[oc:oc_end]
        
        out_shift_layer = layers_cfg[lname]['out_shift']
        res = run_verilog_pass(padded_in, w_chunk, b_chunk, stride, k_mode, out_shift_layer, new_folder_dir)
        out_chunks.append(res)
    
    hw_output = np.concatenate(out_chunks, axis=0)
    print(f"  {lname} Hardware Output Shape: {hw_output.shape}")

    # --- SOFTWARE EXECUTION (Layers 2-7) ---
    print("Verilog execution complete. Passing hardware output to PyTorch for remaining layers...")
    
    # Convert hardware output back to float PyTorch tensor
    hw_tensor = torch.from_numpy(hw_output.astype(np.float32)).unsqueeze(0)
    
    # Scale correction for the float network:
    # Verilog computed: y_int = (sum(x_int * W_new / w_scale) + b_new / w_scale) >> out_shift
    # To get back to true float activation: multiply by (w_scale * 2**out_shift)
    true_w_scale = layers_cfg['layer1']['w_scale']
    layer_out_shift = layers_cfg['layer1']['out_shift']
    hw_tensor = hw_tensor * (true_w_scale * (2 ** layer_out_shift)) 
    
    with torch.no_grad():
        # Evaluate PyTorch Layer 1 properly on NORMALIZED inputs to see the true expected float activation
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        norm_img = (torch.from_numpy(current_map.astype(np.float32)).unsqueeze(0) / 255.0 - mean) / std
        pytorch_l1 = fused.layer1(norm_img)
        print(f"HW Tensor Min/Max: {hw_tensor.min().item():.4f} / {hw_tensor.max().item():.4f}")
        print(f"PyTorch L1 Min/Max: {pytorch_l1.min().item():.4f} / {pytorch_l1.max().item():.4f}")
        
        x = fused.layer2(hw_tensor)
        x = fused.layer3(x)
        x = fused.layer4(x)
        x = fused.layer5(x)
        x = fused.layer6(x)
        final_tensor = fused.layer7(x)

    print("PyTorch pass complete. Running post-processing decoding...")
    
    boxes = decode_predictions(final_tensor[0], score_thresh=0.01, normalized=True, image_size=128)
    boxes = nms(boxes, 0.4)
    boxes = boxes[:2] # Restrict to top 2 detections
    
    print(f"HW+SW Detected {len(boxes)} objects.")
    draw = ImageDraw.Draw(orig_img)
    for b in boxes:
        print(f"  Class: {CLASS_NAMES[int(b[5])]:<12} Conf: {b[4]:.3f} Box: {b[:4]}")
        x1, y1 = int(b[0] * orig_w), int(b[1] * orig_h)
        x2, y2 = int(b[2] * orig_w), int(b[3] * orig_h)
        draw.rectangle([x1, y1, x2, y2], outline="cyan", width=2)
        draw.text((x1, max(0, y1-15)), f"{CLASS_NAMES[int(b[5])]}", fill="red")
        
    orig_img.save('verilog_full_pipeline_output.png')
    print("Saved HW+SW output to verilog_full_pipeline_output.png")

if __name__ == '__main__':
    run_full_iverilog_pipeline()
