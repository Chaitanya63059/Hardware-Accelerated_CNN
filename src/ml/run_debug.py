import os
import subprocess
from PIL import Image
import numpy as np
import torch
from sim_full_verilog import read_quantized_weights_and_biases, run_verilog_pass

def debug_layer1():
    new_folder_dir = "/home/chaitanya/Desktop/new_folder"
    layers_cfg = read_quantized_weights_and_biases()
    
    img_path = '/home/chaitanya/Desktop/mini_proj_ml/data/val2017/000000000139.jpg'
    img_pil = Image.open(img_path).convert('RGB').resize((128, 128), Image.Resampling.BILINEAR)
    img_arr = np.array(img_pil, dtype=np.int32)
    img_arr = np.transpose(img_arr, (2, 0, 1))
    
    current_map = img_arr
    pad = 1
    stride = 2
    k_mode = 2 # 4x4
    
    lname = 'layer1'
    w_all = layers_cfg[lname]['w']
    b_all = layers_cfg[lname]['b']
    
    padded_in = np.pad(current_map, ((0,0), (pad,pad), (pad,pad)), 'constant', constant_values=0)
    
    # Just chunk 1
    oc = 0
    oc_end = 16
    w_chunk = w_all[oc:oc_end]
    b_chunk = b_all[oc:oc_end]
    
    print("Running debug pass...")
    res = run_verilog_pass(padded_in, w_chunk, b_chunk, stride, k_mode, new_folder_dir)
    print("Output shape:", res.shape)

if __name__ == '__main__':
    debug_layer1()
