import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from model import TinyDetector7Layer, fuse_model
from detection_utils import decode_predictions, nms
from config import CLASS_NAMES

def main():
    device = torch.device('cpu')
    model = TinyDetector7Layer().to(device)
    ckpt = torch.load('/home/chaitanya/Desktop/mini_proj_ml/checkpoints/best.pth', map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    img_path = '/home/chaitanya/Desktop/mini_proj_ml/data/val2017/000000000139.jpg'

    img_pil = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img_pil.size
    
    # Preprocess
    img_resized = img_pil.resize((128, 128), Image.Resampling.BILINEAR)
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_arr = (img_arr - mean) / std
    img_arr = np.transpose(img_arr, (2, 0, 1))
    input_tensor = torch.from_numpy(img_arr).unsqueeze(0).to(device)

    import torch.quantization as tq
    from quantize_int8 import QuantizedTinyDetector, calibrate
    from dataset import get_dataloaders
    
    # INT8 inference
    model_fused = fuse_model(model)
    model_fused.eval()
    qconfig = tq.QConfig(activation=tq.HistogramObserver.with_args(dtype=torch.quint8), weight=tq.default_per_channel_weight_observer)
    model_q = QuantizedTinyDetector(model_fused)
    model_q.qconfig = qconfig
    tq.prepare(model_q, inplace=True)
    
    train_loader, _ = get_dataloaders(batch_size=8, num_workers=1)
    calibrate(model_q, train_loader, device, num_batches=1)
    tq.convert(model_q, inplace=True)

    with torch.no_grad():
        pred_int8 = model_q(input_tensor)[0]
    
    boxes = decode_predictions(pred_int8, score_thresh=0.20, normalized=True)
    boxes = nms(boxes, 0.4)
    
    print(f"Detected {len(boxes)} objects.")
    
    draw = ImageDraw.Draw(img_pil)
    for b in boxes:
        print(f"  Class: {CLASS_NAMES[int(b[5])]:<12} Conf: {b[4]:.3f} Box: {b[:4]}")
        x1, y1 = int(b[0] * orig_w), int(b[1] * orig_h)
        x2, y2 = int(b[2] * orig_w), int(b[3] * orig_h)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1-15)), f"{CLASS_NAMES[int(b[5])]} {b[4]:.2f}", fill="green")
        
    img_pil.save('full_pipeline_output.png')
    print("Saved to full_pipeline_output.png")

if __name__ == '__main__':
    main()
