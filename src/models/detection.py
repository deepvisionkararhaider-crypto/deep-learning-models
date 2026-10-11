"""Object detection: a compact anchor-free YOLO-style detector + torchvision Mask R-CNN.

Both detectors are trained on **genuine bounding-box annotations** decoded from the
COCO-format ``objects`` column of the source datasets. Nothing here is trained on
image-level class labels pretending to be localisation.

YOLO-style (``YoloLite``)
-------------------------
* single-scale anchor-free head on a GxG grid,
* per-cell targets: objectness, (cx, cy, w, h) normalised to [0, 1], class,
* loss: BCE(objectness) + L1 + (1 - IoU) on positive cells + CE(class),
* inference: decode + class-wise NMS.

Mask R-CNN
----------
* ``torchvision.models.detection.maskrcnn_resnet50_fpn`` with a replaced head,
* trained on COCO-format targets (boxes / labels / masks),
* inference returns boxes, labels, scores and per-instance masks.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between (N,4) and (M,4) boxes in xyxy."""
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]))
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


def xywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def xyxy_to_xywh(b: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = b.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1).clamp(min=0), (y2 - y1).clamp(min=0)], dim=-1)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float = 0.45) -> torch.Tensor:
    """Class-agnostic greedy NMS. Returns kept indices."""
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long)
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i].unsqueeze(0), boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return torch.tensor(keep, dtype=torch.long)


# --------------------------------------------------------------------------- #
# YOLO-style anchor-free detector
# --------------------------------------------------------------------------- #
class YoloLite(nn.Module):
    """Anchor-free single-scale detector. Output: (B, G, G, 5 + n_classes)."""

    def __init__(self, n_classes: int, in_ch: int = 3, grid: int = 14, width: int = 32):
        super().__init__()
        self.grid = grid
        self.n_classes = n_classes
        w = width
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, 2, 1), nn.BatchNorm2d(w), nn.ReLU(inplace=True),   # S/2
            nn.Conv2d(w, w * 2, 3, 2, 1), nn.BatchNorm2d(w * 2), nn.ReLU(inplace=True),  # S/4
            nn.Conv2d(w * 2, w * 4, 3, 2, 1), nn.BatchNorm2d(w * 4), nn.ReLU(inplace=True),  # S/8
            nn.Conv2d(w * 4, w * 4, 3, 1, 1), nn.BatchNorm2d(w * 4), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((grid, grid)),
        )
        self.head = nn.Conv2d(w * 4, 5 + n_classes, 1)

    def forward(self, x):
        y = self.head(self.backbone(x))                 # (B, 5+C, G, G)
        return y.permute(0, 2, 3, 1).contiguous()       # (B, G, G, 5+C)


def build_yolo(n_classes: int, in_ch: int = 3, grid: int = 14, width: int = 32,
               pretrained: bool = False) -> nn.Module:
    return YoloLite(n_classes, in_ch=in_ch, grid=grid, width=width)


def encode_targets(boxes_xyxy: Sequence[torch.Tensor], labels: Sequence[torch.Tensor],
                   grid: int, n_classes: int, obj_smooth: float = 0.0) -> torch.Tensor:
    """Build a (N, G, G, 5+C) target tensor from per-image xyxy boxes (pixel coords)."""
    n = len(boxes_xyxy)
    tgt = torch.zeros((n, grid, grid, 5 + n_classes), dtype=torch.float32)
    for i in range(n):
        b = boxes_xyxy[i]
        lab = labels[i]
        if b.numel() == 0:
            continue
        cxcywh = xyxy_to_xywh(b) / 1.0  # normalise below
        # assume boxes already normalised to [0,1] by the caller
        for j in range(b.shape[0]):
            cx, cy, bw, bh = [float(v) for v in cxcywh[j]]
            gx = min(grid - 1, max(0, int(cx * grid)))
            gy = min(grid - 1, max(0, int(cy * grid)))
            tgt[i, gy, gx, 0] = 1.0 - obj_smooth
            # store CELL-RELATIVE offsets so the sigmoid head in decode_predictions
            # (cx = (gx + sigmoid(tx)) / grid) is exactly what the loss regresses to.
            tgt[i, gy, gx, 1] = float(min(max(cx * grid - gx, 0.0), 1.0))
            tgt[i, gy, gx, 2] = float(min(max(cy * grid - gy, 0.0), 1.0))
            tgt[i, gy, gx, 3] = bw
            tgt[i, gy, gx, 4] = bh
            c = int(lab[j])
            if 0 <= c < n_classes:
                tgt[i, gy, gx, 5 + c] = 1.0
    return tgt


def yolo_loss(pred: torch.Tensor, target: torch.Tensor, pos_weight: float = 20.0) -> torch.Tensor:
    """BCE objectness + L1 box + (1-IoU) + CE class on positive cells."""
    obj_p = pred[..., 0]
    obj_t = target[..., 0]
    loss_obj = F.binary_cross_entropy_with_logits(
        obj_p, obj_t, pos_weight=torch.tensor(pos_weight))

    mask = obj_t > 0.5
    if mask.any():
        bp = torch.sigmoid(pred[..., 1:5])[mask]
        bt = target[..., 1:5][mask]
        loss_box = F.l1_loss(bp, bt)
        iou = box_iou(xywh_to_xyxy(bp), xywh_to_xyxy(bt)).diag()
        loss_box = loss_box + (1 - iou).mean()
        cp = pred[..., 5:][mask]
        ct = target[..., 5:][mask].argmax(-1)
        loss_cls = F.cross_entropy(cp, ct)
    else:
        loss_box = pred.sum() * 0.0
        loss_cls = pred.sum() * 0.0
    return loss_obj + 5.0 * loss_box + loss_cls


@torch.no_grad()
def decode_predictions(pred: torch.Tensor, conf_thr: float = 0.30, iou_thr: float = 0.45,
                       max_det: int = 30) -> list[dict]:
    """pred: (G,G,5+C) for one image -> list of {x1,y1,x2,y2,score,class} normalised [0,1]."""
    obj = torch.sigmoid(pred[..., 0])
    cls_logits = pred[..., 5:]
    cls_prob = torch.softmax(cls_logits, dim=-1)
    score, cls = cls_prob.max(-1)
    score = score * obj
    g = pred.shape[0]
    ys, xs = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
    cx = (xs + torch.sigmoid(pred[..., 1])) / g
    cy = (ys + torch.sigmoid(pred[..., 2])) / g
    bw = torch.sigmoid(pred[..., 3])
    bh = torch.sigmoid(pred[..., 4])
    boxes = xywh_to_xyxy(torch.stack([cx, cy, bw, bh], -1)).reshape(-1, 4)
    flat_score = score.reshape(-1)
    flat_cls = cls.reshape(-1)
    keep = flat_score >= conf_thr
    boxes, flat_score, flat_cls = boxes[keep], flat_score[keep], flat_cls[keep]
    if boxes.numel() == 0:
        return []
    # class-wise NMS
    out: list[dict] = []
    for c in flat_cls.unique():
        idx = (flat_cls == c).nonzero(as_tuple=True)[0]
        sel = nms(boxes[idx], flat_score[idx], iou_thr)
        for k in sel[:max_det]:
            j = int(idx[int(k)])
            out.append({"x1": float(boxes[j, 0]), "y1": float(boxes[j, 1]),
                        "x2": float(boxes[j, 2]), "y2": float(boxes[j, 3]),
                        "score": float(flat_score[j]), "class": int(flat_cls[j])})
    out.sort(key=lambda d: -d["score"])
    return out[:max_det]


# --------------------------------------------------------------------------- #
# Mask R-CNN (instance segmentation)
# --------------------------------------------------------------------------- #
def build_maskrcnn(n_classes: int, pretrained: bool = False) -> nn.Module:
    """n_classes must include background (i.e. len(classes) + 1)."""
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    m = maskrcnn_resnet50_fpn(weights=weights, weights_backbone=None)
    in_f = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_f, n_classes)
    in_m = m.roi_heads.mask_predictor.conv5_mask.in_channels
    m.roi_heads.mask_predictor = MaskRCNNPredictor(in_m, 256, n_classes)
    return m


def make_maskrcnn_target(boxes_xyxy: torch.Tensor, labels: torch.Tensor,
                         masks: torch.Tensor) -> dict:
    """boxes: (N,4) pixel xyxy; labels: (N,) 1..C (0=bg); masks: (N,H,W) uint8/float."""
    return {"boxes": boxes_xyxy.float(),
            "labels": labels.long(),
            "masks": masks.unsqueeze(1).float()}


@torch.no_grad()
def instances_to_predictions(instances: dict, score_thr: float = 0.4, max_det: int = 20) -> list[dict]:
    """Convert a torchvision Mask R-CNN instance dict into serialisable dicts."""
    out = []
    boxes = instances["boxes"]; labels = instances["labels"]; scores = instances["scores"]
    masks = instances.get("masks")
    keep = scores >= score_thr
    for i in keep.nonzero(as_tuple=True)[0][:max_det]:
        i = int(i)
        d = {"x1": float(boxes[i, 0]), "y1": float(boxes[i, 1]),
             "x2": float(boxes[i, 2]), "y2": float(boxes[i, 3]),
             "score": float(scores[i]), "class": int(labels[i])}
        if masks is not None:
            d["mask"] = (masks[i, 0].cpu().numpy() > 0.5).astype(np.uint8)
        out.append(d)
    return out
