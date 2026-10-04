# 16 — YOLO (Single-Stage Object Detection)

## Description

A from-scratch implementation of **YOLO** (You Only Look Once) — the single-stage
object detector that predicts bounding boxes and class probabilities directly
from a full image in one forward pass, instead of the two-stage
region-proposal pipeline used by R-CNN family detectors.

`model.py` implements the complete pipeline:

- Synthetic multi-object dataset generation (1–3 shapes per 64×64 image)
- Grid target encoding (`S × S` cells, one object per cell)
- A compact convolutional backbone with a **single `1×1` detection head**
- A custom multi-part **YOLO loss** (objectness + coordinate + classification)
- Prediction decoding + **non-maximum suppression (NMS)**
- Object-level evaluation: **Precision, Recall, F1, mean IoU**

## How to Run

```bash
pip install -r requirements.txt
python model.py
```

## Architecture

```
Input 64x64x1
   │
Conv2D(16,3) -> MaxPool        (32x32)
   │
Conv2D(32,3) -> MaxPool        (16x16)
   │
Conv2D(64,3) -> MaxPool        (8x8)
   │
Conv2D(64,3)                   (8x8)
   │
Conv2D(8,1)  ── 1x1 detection head
   │
8x8x8 output = [objectness, x, y, w, h, circle, square, triangle]
```

## Outputs

- `predictions.csv` — per-image detection summary (GT / predicted / matched / missed)
- `plots/yolo_analysis.png` — loss curves + metric bar chart
- `plots/yolo_detections.png` — ground-truth (green) vs predicted (red) boxes
- `data/objects.csv` — objects-per-image distribution

## Metrics

Detection is scored at **IoU ≥ 0.5** against ground truth:

| Metric | Meaning |
|---|---|
| Precision | matched detections / all detections |
| Recall | matched detections / all ground-truth objects |
| F1 | harmonic mean of precision and recall |
| mean IoU | average IoU of true-positive matches |

> Accuracy / confusion-matrix style metrics do not apply directly to detection;
> object-level P/R/F1 + IoU are the standard measures.

## Measured Results

Actually measured by running `python model.py` (TensorFlow 2.21 CPU, 80 epochs,
800 train / 200 test images, `16_yolo/model.py`):

| Metric | Value |
|---|---|
| True Positives (IoU ≥ 0.5) | 140 |
| False Positives | 264 |
| False Negatives | 268 |
| Precision | 0.347 |
| Recall | 0.343 |
| F1 Score | 0.345 |
| Mean IoU (true positives) | **0.812** |
| Final train loss | 0.015 |
| Final val loss | 0.079 |

**Reading these numbers honestly:** localization is strong (mean IoU 0.81 for
matched boxes), while detection P/R are modest. This is expected for a tiny
from-scratch detector on a 1-object-per-cell grid head with a fixed confidence
threshold — background cells leak false positives and the simple assignment
misses some objects. Training a larger backbone, using multiple anchors per
cell, and tuning the confidence/NMS thresholds would raise P/R substantially.

## Dataset

Fully **synthetic and reproducible** (seeded `numpy` generator inside `model.py`) —
circles, squares, and triangles rendered on a noisy background. No external
download is required, so the script runs offline and deterministically.

## Reference

- YOLO paper: <https://arxiv.org/abs/1506.02640>

