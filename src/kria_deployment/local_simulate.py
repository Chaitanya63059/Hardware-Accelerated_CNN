import numpy as np
import cv2
import os

# Local simulation of FPGA pipeline using exported_weights in repo
WEIGHT_DIR = 'kria_deployment/exported_weights'
IMAGE_PATH = 'kria_deployment/000000000139.jpg'
INPUT_SIZE = 126

LAYERS = [
    ("layer1",   3,  32, 4, 2, 1, False),
    ("layer2",  32,  64, 3, 1, 1, True),
    ("layer3",  64, 128, 4, 2, 1, False),
    ("layer4", 128, 256, 3, 1, 1, True),
    ("layer5", 256, 256, 3, 1, 1, False),
    ("layer6", 256, 128, 3, 1, 1, False),
    ("layer7", 128,   8, 1, 1, 0, False),
]
LAYER_OUTPUT_SHIFTS = {
    "layer1": 9,
    "layer2": 7,
    "layer3": 8,
    "layer4": 7,
    "layer5": 8,
    "layer6": 8,
    "layer7": 4,
}

def parse_mem_file(filepath, signed_bits=8):
    vals=[]
    with open(filepath) as fh:
        for line in fh:
            line=line.strip()
            if not line or line.startswith('//'): continue
            parts=line.split()
            if len(parts)==2:
                v=int(parts[1],16)
                if signed_bits==8:
                    if v>127: v-=256
                elif signed_bits==16:
                    if v>32767: v-=65536
                vals.append(v)
    return np.array(vals)


def load_layer_params(layer_name,c_in,c_out,k_size):
    w_path=os.path.join('kria_deployment','exported_weights',f"{layer_name}_weight.mem")
    b_path=os.path.join('kria_deployment','exported_weights',f"{layer_name}_bias.mem")
    w=parse_mem_file(w_path, signed_bits=8)
    b=parse_mem_file(b_path, signed_bits=8)
    weights=w.reshape(c_out,c_in,k_size,k_size).astype(np.int32)
    biases=b.astype(np.int32)
    return weights,biases


def hw_conv2d(input_data, weights, biases, stride, pad, out_shift):
    c_in,h,w=input_data.shape
    c_out,_,k,_=weights.shape
    if pad>0:
        input_data=np.pad(input_data,((0,0),(pad,pad),(pad,pad)),'constant',constant_values=0)
        h+=2*pad; w+=2*pad
    oh=(h-k)//stride+1
    ow=(w-k)//stride+1
    inp=input_data.astype(np.int32)
    output=np.zeros((c_out,oh,ow),dtype=np.int32)
    for oc in range(c_out):
        for ic in range(c_in):
            for kh in range(k):
                for kw in range(k):
                    h_idx=np.arange(oh)*stride+kh
                    w_idx=np.arange(ow)*stride+kw
                    patch=inp[ic][np.ix_(h_idx,w_idx)]
                    output[oc]+=patch*int(weights[oc,ic,kh,kw])
        output[oc]+=int(biases[oc] if oc < len(biases) else 0)
    output=(output >> out_shift)
    output=np.clip(output,0,127).astype(np.float32)
    return output,oh,ow


def preprocess_image(path):
    MEAN=np.array([0.485,0.456,0.406],dtype=np.float32)
    STD=np.array([0.229,0.224,0.225],dtype=np.float32)
    img_bgr=cv2.imread(path)
    if img_bgr is None:
        # Fallback: generate synthetic image with typical ImageNet-like values
        print(f"Warning: image not found at {path} - using synthetic input")
        img_resized = (np.clip(np.random.randn(INPUT_SIZE, INPUT_SIZE, 3) * 0.2 + 0.5, 0.0, 1.0) * 255).astype(np.uint8)
        img_rgb = img_resized.copy()
    else:
        img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
        img_resized=cv2.resize(img_rgb,(INPUT_SIZE,INPUT_SIZE))
    img_norm=img_resized.astype(np.float32)/255.0
    img_norm=(img_norm-MEAN)/STD
    img_scaled=np.clip(img_norm*128.0,-128,127).astype(np.int8)
    img_uint8=(img_scaled.astype(np.int16)+128).astype(np.uint8)
    # hardware expects CHW
    return img_uint8.transpose(2,0,1).astype(np.uint8), img_rgb.shape[1], img_rgb.shape[0]


def main():
    inp,orig_w,orig_h=preprocess_image(IMAGE_PATH)
    feat=inp.astype(np.uint8)
    h,w=INPUT_SIZE,INPUT_SIZE
    for layer in LAYERS:
        name,c_in,c_out,k_size,stride,pad,has_pool=layer
        weights,biases=load_layer_params(name,c_in,c_out,k_size)
        # feed as unsigned u8
        feat,oh,ow=hw_conv2d(feat,weights,biases,stride,pad,LAYER_OUTPUT_SHIFTS.get(name,0))
        print(name,'->',feat.shape, 'range', feat.min(), feat.max())
        if has_pool:
            c=feat.shape[0]
            ph, pw = oh//2, ow//2
            cropped = feat[:, :ph*2, :pw*2]
            feat = cropped.reshape(c,ph,2,pw,2).max(axis=(2,4))
            print(' pooled ->', feat.shape)
    np.save('kria_deployment/arm_sim_output.npy',feat)
    print('Saved arm sim output')

if __name__=='__main__':
    main()
