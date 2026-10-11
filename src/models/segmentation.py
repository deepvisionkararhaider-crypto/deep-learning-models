"""U-Net semantic segmentation + pixel-level losses and metrics.

The U-Net has 4 down-sampling stages, so input side length must be a multiple of
16 (we use 224). Masks are produced at full input resolution as raw logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 8, base * 16)
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2); self.c4 = DoubleConv(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2); self.c3 = DoubleConv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2); self.c2 = DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2); self.c1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        s1 = self.d1(x)
        s2 = self.d2(self.pool(s1))
        s3 = self.d3(self.pool(s2))
        s4 = self.d4(self.pool(s3))
        b = self.bottleneck(self.pool(s4))
        x = self.c4(torch.cat([self.u4(b), s4], dim=1))
        x = self.c3(torch.cat([self.u3(x), s3], dim=1))
        x = self.c2(torch.cat([self.u2(x), s2], dim=1))
        x = self.c1(torch.cat([self.u1(x), s1], dim=1))
        return self.out(x)


def build_unet(n_classes: int = 1, in_ch: int = 3, base: int = 32, pretrained: bool = False) -> nn.Module:
    """Binary segmentation -> out_ch=1 logits; multiclass -> out_ch=n_classes."""
    out_ch = 1 if n_classes <= 1 else n_classes
    return UNet(in_ch=in_ch, out_ch=out_ch, base=base)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss for binary masks (logits: B,1,H,W; target: B,1,H,W in {0,1})."""
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(1, 2, 3)) + eps
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, bce_weight: float = 0.5) -> torch.Tensor:
    """Combined BCE + Dice — the loss used to train U-Net here."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return bce_weight * bce + (1 - bce_weight) * dice_loss(logits, target)


@torch.no_grad()
def dice_coef(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(logits) > thr).float()
    num = 2 * (pred * target).sum().item() + eps
    den = pred.sum().item() + target.sum().item() + eps
    return num / den


@torch.no_grad()
def iou_score(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item() - inter
    return (inter + eps) / (union + eps)
