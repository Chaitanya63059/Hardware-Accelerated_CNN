import torch
import cv2
import numpy as np
from model import TinyDetector7Layer, fuse_model
from detect_realtime import preprocess_frame, decode_predictions, nms, CLASS_NAMES
from quantize_int8 import QuantizedTinyDetector, calibrate
import torch.quantization as tq
from dataset import get_dataloaders

device = torch.device('cpu')
model = TinyDetector7Layer().to(device)
ckpt = torch.load('checkpoints/best.pth', map_location=device, weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

img_path = 'capture_000.png'
frame = cv2.imread(img_path)

input_tensor = preprocess_frame(frame).to(device)

print("--- FLOAT32 ---")
with torch.no_grad():
    pred_f32 = model(input_tensor)[0]
boxes = decode_predictions(pred_f32, conf_thresh=0.1)
boxes = nms(boxes, 0.4)
for b in boxes:
    print(f"  Class: {CLASS_NAMES[int(b[5])]:<12} Conf: {b[4]:.3f} Box: {b[:4]}")


print("\n--- INT8 ---")
model = fuse_model(model)
model.eval()
qconfig = tq.QConfig(
    activation=tq.HistogramObserver.with_args(dtype=torch.quint8),
    weight=tq.default_per_channel_weight_observer,
)
model = QuantizedTinyDetector(model)
model.eval()
model.qconfig = qconfig
tq.prepare(model, inplace=True)
train_loader, _ = get_dataloaders(batch_size=8, num_workers=1)
calibrate(model, train_loader, device, num_batches=10)
tq.convert(model, inplace=True)

with torch.no_grad():
    pred_int8 = model(input_tensor)[0]
boxes = decode_predictions(pred_int8, conf_thresh=0.1)
boxes = nms(boxes, 0.4)
for b in boxes:
    print(f"  Class: {CLASS_NAMES[int(b[5])]:<12} Conf: {b[4]:.3f} Box: {b[:4]}")
