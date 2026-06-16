"""
Real-time multi-class detection using TinyDetector7Layer and webcam.

Captures frames from the webcam, runs inference on each frame,
applies NMS, and draws bounding boxes with confidence scores.

Usage:
    python detect_realtime.py                           # default webcam (Float32)
    python detect_realtime.py --int8                    # run INT8 fixed-point simulation
    python detect_realtime.py --checkpoint best.pth     # custom checkpoint
    python detect_realtime.py --source video.mp4        # from video file

Controls:
    q / ESC  — quit
    +/-      — raise/lower confidence threshold
    s        — save current frame
"""

import os
import argparse
import time

import cv2
import numpy as np
import torch

from model import TinyDetector7Layer, fuse_model
from config import CLASS_NAMES
from detection_utils import decode_predictions as decode_model_predictions, nms, compute_iou

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = 128
GRID_SIZE = 8
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_frame(frame):
    """
    Preprocess a BGR OpenCV frame for model input.
    Returns: (1, 3, 128, 128) float tensor on CPU
    """
    # Resize to 128x128
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    # BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    # HWC → CHW → BCHW
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0)
    return tensor


def decode_predictions(pred, conf_thresh=0.5):
    """Decode raw model logits into normalized boxes."""
    return decode_model_predictions(
        pred,
        score_thresh=conf_thresh,
        normalized=True,
        image_size=IMG_SIZE,
        grid_size=GRID_SIZE,
    )


class BBoxTracker:
    """
    Simple temporal tracker to prevent bounding boxes from abruptly 
    disappearing, and keeps them smoothed via EMA.
    """
    def __init__(self, max_missed=60, iou_thresh=0.3):
        self.tracks = []  # List of {'box': [x1,y1,x2,y2,conf,class_idx], 'missed': 0}
        self.max_missed = max_missed
        self.iou_thresh = iou_thresh

    def update(self, new_boxes):
        if not new_boxes:
            for t in self.tracks:
                t['missed'] += 1
            self.tracks = [t for t in self.tracks if t['missed'] < self.max_missed]
            return [t['box'] for t in self.tracks]

        updated_tracks = []
        unmatched_new = list(new_boxes)

        for track in self.tracks:
            best_iou = 0
            best_match_idx = -1
            
            for i, nbox in enumerate(unmatched_new):
                iou = compute_iou(track['box'][:4], nbox[:4])
                if nbox[5] == track['box'][5] and iou > best_iou:
                    best_iou = iou
                    best_match_idx = i

            if best_iou > self.iou_thresh:
                # Match found -> update via EMA (Exponential Moving Average)
                nbox = unmatched_new.pop(best_match_idx)
                alpha = 0.6  # 60% new, 40% old (smoothing)
                smoothed = [
                    alpha * nbox[0] + (1 - alpha) * track['box'][0],
                    alpha * nbox[1] + (1 - alpha) * track['box'][1],
                    alpha * nbox[2] + (1 - alpha) * track['box'][2],
                    alpha * nbox[3] + (1 - alpha) * track['box'][3],
                    nbox[4],
                    nbox[5],
                ]
                updated_tracks.append({'box': smoothed, 'missed': 0})
            else:
                # No match -> increment missed
                track['missed'] += 1
                if track['missed'] < self.max_missed:
                    # Decay confidence slightly so it fades out
                    track['box'][4] *= 0.95
                    updated_tracks.append(track)
                    
        # Add entirely new boxes
        for nbox in unmatched_new:
            updated_tracks.append({'box': nbox, 'missed': 0})
            
        self.tracks = updated_tracks
        return [t['box'] for t in self.tracks]


def draw_boxes(frame, boxes, conf_thresh):
    """Draw bounding boxes on the frame."""
    h, w = frame.shape[:2]

    for box in boxes:
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)
        conf = box[4]
        class_name = CLASS_NAMES[int(box[5])]

        # Color: green → red based on confidence
        green = int(255 * conf)
        red = int(255 * (1 - conf))
        color = (0, green, red)

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"{class_name} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 6), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_grid_overlay(frame, active_cells, alpha=0.16):
    """Draw the detector's 8x8 grid and highlight active cells."""
    h, w = frame.shape[:2]
    cell_w = w / GRID_SIZE
    cell_h = h / GRID_SIZE

    overlay = frame.copy()
    line_color = (220, 220, 220)
    active_color = (30, 140, 255)

    for gj, gi in active_cells:
        x1 = int(gi * cell_w)
        y1 = int(gj * cell_h)
        x2 = int((gi + 1) * cell_w)
        y2 = int((gj + 1) * cell_h)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), active_color, -1)

    blended = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)

    for g in range(1, GRID_SIZE):
        x = int(g * cell_w)
        y = int(g * cell_h)
        cv2.line(blended, (x, 0), (x, h), line_color, 1, cv2.LINE_AA)
        cv2.line(blended, (0, y), (w, y), line_color, 1, cv2.LINE_AA)

    return blended


def draw_hud(frame, fps, conf_thresh, n_detections, is_int8=False):
    """Draw heads-up display info on frame."""
    h, w = frame.shape[:2]

    # Semi-transparent overlay for HUD
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    mode_text = "INT8 Fixed-Point" if is_int8 else "Float32"
    cv2.putText(frame, f"Mode: {mode_text}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0) if is_int8 else (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Score Thresh: {conf_thresh:.2f}  (+/-)", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Detections: {n_detections}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def main():
    parser = argparse.ArgumentParser(description='Real-time Multi-Class Detection')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth')
    parser.add_argument('--conf-thresh', type=float, default=0.54)
    parser.add_argument('--iou-thresh', type=float, default=0.4)
    parser.add_argument('--source', type=str, default='0',
                        help='Webcam index (0, 1, ...) or video file path')
    parser.add_argument('--save-output', type=str, default=None,
                        help='Save output video to file')
    parser.add_argument('--int8', action='store_true',
                        help='Run INT8 fixed-point simulation instead of Float32')
    parser.add_argument('--fps-limit', type=float, default=0.0,
                        help='Limit the maximum FPS (e.g. 15.0). 0 means unbounded.')
    parser.add_argument('--show-grid', action='store_true',
                        help='Show the 8x8 detector grid overlay')
    args = parser.parse_args()

    # Device
    device = torch.device('cpu') if args.int8 else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load base model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(base_dir, args.checkpoint)
    print(f"Loading model from {ckpt_path}")

    model = TinyDetector7Layer().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Float32 Model loaded (epoch {ckpt['epoch']+1})")

    # If INT8 requested, run quantization pipeline
    if args.int8:
        import torch.quantization as tq
        from quantize_int8 import QuantizedTinyDetector, calibrate
        from dataset import get_dataloaders

        print("\n  Preparing INT8 Fixed-Point Simulation...")
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
        
        print("  Loading calibration data for INT8...")
        train_loader, _ = get_dataloaders(batch_size=32, num_workers=2)
        calibrate(model, train_loader, device, num_batches=100)
        tq.convert(model, inplace=True)
        print("  INT8 Model ready!\n")

    # Open video source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.source}")
        return

    # Get video properties
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Source: {args.source} ({frame_w}x{frame_h})")

    # Video writer
    writer = None
    if args.save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_output, fourcc, 30.0, (frame_w, frame_h))

    conf_thresh = args.conf_thresh
    fps = 0
    frame_count = 0
    save_count = 0

    print(f"\n{'='*50}")
    print(f"  Real-time Detection Running")
    print(f"  Press 'q' or ESC to quit")
    print(f"  Press +/- to adjust confidence threshold")
    print(f"  Press 's' to save a frame")
    print(f"{'='*50}\n")

    tracker = BBoxTracker(max_missed=60)  # remembers boxes for ~2 seconds (60 frames)

    with torch.no_grad():
        while True:
            t0 = time.time()

            ret, frame = cap.read()
            if not ret:
                print("End of video / camera error")
                break

            # Preprocess
            input_tensor = preprocess_frame(frame).to(device)

            # Inference
            pred = model(input_tensor)
            pred = pred[0].cpu()

            # Decode + NMS
            boxes = decode_predictions(pred, conf_thresh)
            boxes = nms(boxes, args.iou_thresh)
            
            # Limit to top 4 most confident boxes BEFORE the tracker memorizes them
            boxes = sorted(boxes, key=lambda b: b[4], reverse=True)[:4]
            active_cells = {(int(box[7]), int(box[6])) for box in boxes if len(box) >= 8}
            
            # Apply temporal tracking/smoothing so boxes don't disappear instantly
            boxes = tracker.update(boxes)
            # Also cap tracker output to 4 total displayed detections
            boxes = sorted(boxes, key=lambda b: b[4], reverse=True)[:4]

            # Draw results
            if args.show_grid:
                frame = draw_grid_overlay(frame, active_cells)
            frame = draw_boxes(frame, boxes, conf_thresh)
            frame = draw_hud(frame, fps, conf_thresh, len(boxes), is_int8=args.int8)

            # Display
            cv2.imshow('TinyDetector - Multi-Class Detection', frame)

            # Save output
            if writer:
                writer.write(frame)

            # Optional FPS Limiter
            if args.fps_limit > 0:
                expected_time = 1.0 / args.fps_limit
                current_time = time.time() - t0
                if current_time < expected_time:
                    time.sleep(expected_time - current_time)

            # FPS
            elapsed = time.time() - t0
            fps = 0.9 * fps + 0.1 * (1.0 / max(elapsed, 1e-6))  # smoothed
            frame_count += 1

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('+') or key == ord('='):
                conf_thresh = min(0.999, conf_thresh + 0.01)
                print(f"  Confidence threshold: {conf_thresh:.3f}")
            elif key == ord('-') or key == ord('_'):
                conf_thresh = max(0.001, conf_thresh - 0.01)
                print(f"  Confidence threshold: {conf_thresh:.3f}")
            elif key == ord('s'):
                save_path = os.path.join(base_dir, f'capture_{save_count:03d}.png')
                cv2.imwrite(save_path, frame)
                print(f"  Saved: {save_path}")
                save_count += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames. Done!")


if __name__ == '__main__':
    main()
