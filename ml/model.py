"""
TinyDetector7Layer — FPGA-optimized multi-class detection CNN (mixed-kernel).

Architecture:
  - Mixed kernels: 4×4 (learned downsampling), 3×3 (features), 1×1 (head)
  - 128×128 input → 8×8 detection grid
  - BatchNorm after each conv (fused at export time = zero cost on FPGA)
  - Wide channels for better feature learning (~1.3M params)
  - Output: 5 + C channels per cell [tx, ty, tw, th, confidence, class_logits...]

Kernel layout:
  Layer 1: 4×4 stride-2 (learned downsampling, replaces conv+pool)
  Layer 2: 3×3 + MaxPool (classic feature extraction)
  Layer 3: 4×4 stride-2 (learned downsampling, replaces conv+pool)
  Layer 4: 3×3 + MaxPool (classic feature extraction)
  Layer 5: 3×3 (deep features, no pooling)
  Layer 6: 3×3 (channel reduction, no pooling)
  Layer 7: 1×1 (detection head — minimal compute)

FPGA (Kria KV260):
  - INT8 weights = ~1.3MB (fits in BRAM)
  - Needs 4×4, 3×3, and 1×1 PE designs
  - BN fused into conv at export = no runtime overhead
"""

import copy
import torch
import torch.nn as nn

from config import NUM_CLASSES, NUM_OUTPUTS


class TinyDetector7Layer(nn.Module):
    """
    7-layer mixed-kernel detector — wider channels for higher accuracy.

    Input:  (B, 3, 128, 128)
    Output: (B, 5 + C, 8, 8) → 8×8 grid with [tx, ty, tw, th, conf, class_logits]

    Layer structure:
      Layer 1: Conv4×4(s=2) + BN + ReLU          (128→64)
      Layer 2: Conv3×3 + BN + ReLU + MaxPool2×2   (64→32)
      Layer 3: Conv4×4(s=2) + BN + ReLU          (32→16)
      Layer 4: Conv3×3 + BN + ReLU + MaxPool2×2   (16→8)
      Layer 5: Conv3×3 + BN + ReLU                (8→8)
      Layer 6: Conv3×3 + BN + ReLU                (8→8)
      Layer 7: Conv1×1 → detection head            (8→8)

    Channels: 32→64→128→256→256→128→(5 + C)
    """

    def __init__(self):
        super().__init__()

        # --- Layer 1: 4×4 stride-2 learned downsampling (128→64) ---
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # --- Layer 2: 3×3 feature extraction + MaxPool (64→32) ---
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Layer 3: 4×4 stride-2 learned downsampling (32→16) ---
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # --- Layer 4: 3×3 feature extraction + MaxPool (16→8) ---
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Layer 5: 3×3 deep feature learning (no pooling) ---
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # --- Layer 6: 3×3 channel reduction (no pooling) ---
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # --- Layer 7: 1×1 detection head (no BN, has bias) ---
        self.layer7 = nn.Conv2d(128, NUM_OUTPUTS, kernel_size=1, stride=1, padding=0, bias=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Bias init for detection head — start with low confidence
        nn.init.constant_(self.layer7.bias, -2.0)

    def forward(self, x):
        """
        Args:
            x: (B, 3, 128, 128) input image tensor
        Returns:
            out: (B, 5 + C, 8, 8) detection grid
                 channels: [tx, ty, tw, th, conf, class_logits]
                 - tx, ty: raw center-offset logits
                 - tw, th: raw width/height logits
                 - conf: raw objectness logit
        """
        x = self.layer1(x)   # (B, 32, 64, 64)   — 4×4 stride-2
        x = self.layer2(x)   # (B, 64, 32, 32)   — 3×3 + pool
        x = self.layer3(x)   # (B, 128, 16, 16)  — 4×4 stride-2
        x = self.layer4(x)   # (B, 256, 8, 8)    — 3×3 + pool
        x = self.layer5(x)   # (B, 256, 8, 8)    — 3×3
        x = self.layer6(x)   # (B, 128, 8, 8)    — 3×3
        out = self.layer7(x)

        # Return raw logits. Sigmoid/Softmax is applied during loss computation 
        # or inference decoding, not inside the core model forward pass.
        return out


# ── BatchNorm Fusion ─────────────────────────────────────────────────────────
def fuse_bn_into_conv(conv, bn):
    """
    Fuse BatchNorm parameters into Conv2d weights.
    Result: single Conv2d with bias — zero extra FPGA cost.
    """
    fused = copy.deepcopy(conv)
    fused.bias = nn.Parameter(torch.zeros(conv.out_channels))

    # BN params
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    scale = gamma / torch.sqrt(var + eps)

    # Fuse: W_fused = W * scale, b_fused = beta - mean * scale
    fused.weight.data = conv.weight.data * scale.view(-1, 1, 1, 1)
    fused.bias.data = beta - mean * scale

    return fused


def fuse_model(model):
    """Fuse all BN layers into their preceding Conv layers."""
    fused_model = copy.deepcopy(model)

    for name in ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6']:
        layer = getattr(fused_model, name)
        conv = layer[0]
        bn = layer[1]
        fused_conv = fuse_bn_into_conv(conv, bn)
        new_layer = nn.Sequential(fused_conv, layer[2])  # conv + relu
        if len(layer) == 4:  # has maxpool
            new_layer = nn.Sequential(fused_conv, layer[2], layer[3])
        setattr(fused_model, name, new_layer)

    return fused_model


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model = TinyDetector7Layer()

    print(f"TinyDetector7Layer (mixed kernels: 4×4/3×3/1×1, 128×128 input)")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  INT8 size:        {total / 1024 / 1024:.2f} MB")

    print()
    kernel_labels = {
        'layer1': '4×4', 'layer2': '3×3', 'layer3': '4×4',
        'layer4': '3×3', 'layer5': '3×3', 'layer6': '3×3', 'layer7': '1×1',
    }
    for name in ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'layer7']:
        layer = getattr(model, name)
        n = sum(p.numel() for p in layer.parameters())
        kl = kernel_labels[name]
        if isinstance(layer, nn.Sequential):
            conv = layer[0]
            print(f"  {name}: Conv{kl}({conv.in_channels}→{conv.out_channels})  params={n:,}")
        else:
            print(f"  {name}: Conv{kl}({layer.in_channels}→{layer.out_channels})  params={n:,}")

    # Test forward
    x = torch.randn(1, 3, 128, 128)
    model.eval()
    with torch.no_grad():
        y = model(x)

    print(f"\n  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}  (expected: [1, {NUM_OUTPUTS}, 8, 8])")
    assert y.shape == (1, NUM_OUTPUTS, 8, 8), f"Wrong shape: {y.shape}"

    # Test BN fusion
    fused = fuse_model(model)
    with torch.no_grad():
        y_fused = fused(x)
    err = (y - y_fused).abs().max().item()
    print(f"  BN fusion max error: {err:.2e}  (should be < 1e-5)")

    assert err < 1e-4, f"Fusion error too high: {err}"
    print(f"\n✓ All checks passed!")
