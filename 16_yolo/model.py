"""
YOLO — Single-Stage Object Detection (implemented from scratch)
===============================================================
Dataset: Synthetic multi-object detection dataset (generated, reproducible)
Reference: https://arxiv.org/abs/1506.02640  (You Only Look Once, Redmon et al.)

Framework: TensorFlow / Keras
Task: Real-time Object Detection — localize AND classify 1-3 shapes per image
Grid: 8x8 over 64x64 px images   |   Classes: circle, square, triangle
Head: per-cell [objectness, x, y, w, h, class(3)] = 8 values
Output: bounding boxes [class, x1, y1, x2, y2, confidence] + IoU / P / R / F1
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  YOLO — Single-Stage Object Detection (from scratch)")
print("=" * 60)

GRID = 8            # S x S grid
IMG = 64            # image size (square)
N_CLASSES = 3       # circle, square, triangle
CLASS_NAMES = ['circle', 'square', 'triangle']
MAX_OBJ = 3

# ─────────────────────────────────────────────────────────────
# 1. Synthetic detection dataset (shapes on a noisy background)
# ─────────────────────────────────────────────────────────────
def _draw_shape(img, cx, cy, r, shape_id, value):
    """Rasterize one shape onto a 2D image array."""
    yy, xx = np.ogrid[:IMG, :IMG]
    if shape_id == 0:                       # circle
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    elif shape_id == 1:                     # square
        mask = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
    else:                                   # triangle (upward)
        x1, y1 = cx, cy - r
        x2, y2 = cx - r, cy + r
        x3, y3 = cx + r, cy + r
        def side(ax, ay, bx, by):
            return (xx - bx) * (ay - by) - (ax - bx) * (yy - by)
        d1, d2, d3 = side(x1, y1, x2, y2), side(x2, y2, x3, y3), side(x3, y3, x1, y1)
        has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        mask = ~(has_neg & has_pos)
    img[mask] = value
    return img


def generate_dataset(n_samples=1000, seed=42):
    """Return (images [N,64,64,1] in [0,1], boxes list of [x1,y1,x2,y2,cls])."""
    rng = np.random.RandomState(seed)
    images = np.zeros((n_samples, IMG, IMG, 1), dtype=np.float32)
    all_boxes = []
    for i in range(n_samples):
        img = rng.rand(IMG, IMG).astype(np.float32) * 0.15          # background noise
        n_obj = rng.randint(1, MAX_OBJ + 1)
        boxes = []
        occupied = []
        for _ in range(n_obj):
            for _try in range(20):                                   # avoid overlap
                r = rng.randint(4, 8)
                cx = rng.randint(r + 1, IMG - r - 1)
                cy = rng.randint(r + 1, IMG - r - 1)
                if all((cx - ox) ** 2 + (cy - oy) ** 2 > (r + orr + 4) ** 2
                       for ox, oy, orr in occupied):
                    break
            else:
                continue
            shape_id = rng.randint(0, N_CLASSES)
            img = _draw_shape(img, cx, cy, r, shape_id, float(rng.uniform(0.7, 1.0)))
            occupied.append((cx, cy, r))
            boxes.append([cx - r, cy - r, cx + r, cy + r, shape_id])
        images[i, :, :, 0] = img
        all_boxes.append(np.array(boxes, dtype=np.float32))
    return images, all_boxes


def encode_targets(boxes_list):
    """Encode GT boxes into the (GRID, GRID, 8) YOLO target tensor."""
    targets = np.zeros((len(boxes_list), GRID, GRID, 1 + 4 + N_CLASSES), dtype=np.float32)
    for n, boxes in enumerate(boxes_list):
        for x1, y1, x2, y2, cls in boxes:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w, h = (x2 - x1), (y2 - y1)
            col = min(int(cx / IMG * GRID), GRID - 1)
            row = min(int(cy / IMG * GRID), GRID - 1)
            targets[n, row, col, 0] = 1.0
            targets[n, row, col, 1] = cx / IMG * GRID - col      # cell-relative
            targets[n, row, col, 2] = cy / IMG * GRID - row
            targets[n, row, col, 3] = w / IMG                    # image-relative
            targets[n, row, col, 4] = h / IMG
            targets[n, row, col, 5 + int(cls)] = 1.0
    return targets


X_all, boxes_all = generate_dataset(1000)
Y_all = encode_targets(boxes_all)
X_train, Y_train = X_all[:800], Y_all[:800]
X_test, Y_test = X_all[800:], Y_all[800:]
boxes_test = boxes_all[800:]

obj_counts = [len(b) for b in boxes_all]
pd.DataFrame({'objects_per_image': obj_counts}).to_csv('data/objects.csv', index=False)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print(f"Objects per image -> mean {np.mean(obj_counts):.2f}, max {np.max(obj_counts)}")

# ─────────────────────────────────────────────────────────────
# 2. Build the YOLO network (one-stage, single grid head)
# ─────────────────────────────────────────────────────────────
tf.random.set_seed(42)
inputs = tf.keras.Input(shape=(IMG, IMG, 1))
x = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
outputs = tf.keras.layers.Conv2D(1 + 4 + N_CLASSES, 1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs, name="tiny_yolo")

LAMBDA_COORD, LAMBDA_NOOBJ = 5.0, 0.5


def yolo_loss(y_true, y_pred):
    # Work with channel-squeezed tensors so every term is shaped [B, S, S].
    obj = y_true[..., 0]          # [B, S, S]
    coord = y_true[..., 1:5]      # [B, S, S, 4]  (x, y, w, h)
    cls = y_true[..., 5:]         # [B, S, S, C]
    p_obj = y_pred[..., 0]        # [B, S, S]
    p_coord = y_pred[..., 1:5]    # [B, S, S, 4]
    p_cls = y_pred[..., 5:]       # [B, S, S, C]

    eps = 1e-7

    def bce(target, prob):
        # Element-wise binary cross-entropy (NO reduction) -> [B, S, S].
        prob = tf.clip_by_value(prob, eps, 1.0 - eps)
        return -(target * tf.math.log(prob) + (1.0 - target) * tf.math.log(1.0 - prob))

    # Objectness: penalise missed objects and (more weakly) background cells.
    obj_bce = bce(obj, p_obj)                                                       # [B, S, S]
    obj_loss = tf.reduce_sum(obj * obj_bce + LAMBDA_NOOBJ * (1 - obj) * obj_bce)
    # Box regression: xy as squared error, wh as sqrt-space error (indices 2:4).
    xy = tf.reduce_sum((p_coord[..., 0:2] - coord[..., 0:2]) ** 2, axis=-1)        # [B, S, S]
    wh = tf.reduce_sum((tf.sqrt(p_coord[..., 2:4] + eps) -
                        tf.sqrt(coord[..., 2:4] + eps)) ** 2, axis=-1)             # [B, S, S]
    coord_loss = tf.reduce_sum(obj * (LAMBDA_COORD * xy + wh))
    # Classification: cross-entropy on the object-bearing cells only.
    cls_loss = -tf.reduce_sum(obj * tf.reduce_sum(cls * tf.math.log(p_cls + eps), axis=-1))
    return (obj_loss + coord_loss + cls_loss) / tf.cast(tf.shape(y_true)[0], tf.float32)


model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=yolo_loss)
print("\nYOLO Architecture:")
model.summary()

# ─────────────────────────────────────────────────────────────
# 3. Train
# ─────────────────────────────────────────────────────────────
history = model.fit(X_train, Y_train, epochs=80, batch_size=32,
                    validation_split=0.1, verbose=2)

# ─────────────────────────────────────────────────────────────
# 4. Decode + NMS
# ─────────────────────────────────────────────────────────────
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-8)


def decode(pred, conf_thr=0.4):
    """pred -> list of [x1,y1,x2,y2,cls,conf]."""
    boxes = []
    for row in range(GRID):
        for col in range(GRID):
            conf = float(pred[row, col, 0])
            if conf < conf_thr:
                continue
            cx = (col + float(pred[row, col, 1])) / GRID * IMG
            cy = (row + float(pred[row, col, 2])) / GRID * IMG
            w = float(pred[row, col, 3]) * IMG
            h = float(pred[row, col, 4]) * IMG
            cls = int(np.argmax(pred[row, col, 5:]))
            boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, cls, conf])
    return boxes


def nms(boxes, iou_thr=0.3):
    keep = []
    for cls in range(N_CLASSES):
        cand = [b for b in boxes if b[4] == cls]
        cand.sort(key=lambda b: b[5], reverse=True)
        while cand:
            best = cand.pop(0)
            keep.append(best)
            cand = [c for c in cand if iou(best, c) < iou_thr]
    return keep


y_pred_all = model.predict(X_test, verbose=0)

# ─────────────────────────────────────────────────────────────
# 5. Evaluation (object-level precision / recall / F1 + mean IoU)
# ─────────────────────────────────────────────────────────────
TP = FP = FN = 0
ious = []
rows = []
for i in range(len(X_test)):
    preds = nms(decode(y_pred_all[i]))
    gts = [list(b) for b in boxes_test[i]]
    matched_gt = set()
    for p in preds:
        best_j, best_iou = -1, 0.0
        for j, g in enumerate(gts):
            if j in matched_gt:
                continue
            if int(g[4]) != p[4]:
                continue
            v = iou(p, g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= 0.5:
            matched_gt.add(best_j)
            TP += 1
            ious.append(best_iou)
        else:
            FP += 1
    FN += len(gts) - len(matched_gt)
    rows.append({
        'Sample_ID': i, 'GT_Objects': len(gts), 'Predicted': len(preds),
        'Matched': len(matched_gt), 'Missed': len(gts) - len(matched_gt),
    })

precision = TP / (TP + FP + 1e-8)
recall = TP / (TP + FN + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
mean_iou = float(np.mean(ious)) if ious else 0.0

pd.DataFrame(rows).to_csv('predictions.csv', index=False)

print(f"\n── Detection Metrics (IoU>=0.5, {len(X_test)} test images) ──")
print(f"  True Positives  : {TP}")
print(f"  False Positives : {FP}")
print(f"  False Negatives : {FN}")
print(f"  Precision       : {precision:.4f}")
print(f"  Recall          : {recall:.4f}")
print(f"  F1 Score        : {f1:.4f}")
print(f"  Mean IoU (TP)   : {mean_iou:.4f}")

print("\n── Sample Predictions (first 10) ──")
print(f"{'#':>3} | {'GT':>3} | {'Pred':>5} | {'Matched':>8} | {'Missed':>7}")
print("-" * 44)
for r in rows[:10]:
    print(f"{r['Sample_ID']:>3} | {r['GT_Objects']:>3} | {r['Predicted']:>5} | {r['Matched']:>8} | {r['Missed']:>7}")

# ─────────────────────────────────────────────────────────────
# 6. Plots
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(history.history['loss'], label='Train', color='steelblue')
axes[0].plot(history.history['val_loss'], label='Val', color='coral')
axes[0].set_title('Loss — YOLO'); axes[0].set_xlabel('Epoch'); axes[0].legend()

axes[1].bar(['Precision', 'Recall', 'F1', 'mIoU'],
            [precision, recall, f1, mean_iou],
            color=['#4C72B0', '#55A868', '#C44E52', '#8172B3'])
axes[1].set_ylim(0, 1.05); axes[1].set_title('Detection Metrics — YOLO')

colors = ['#e74c3c', '#2ecc71', '#3498db']
for c in range(N_CLASSES):
    axes[2].scatter([], [], color=colors[c], label=CLASS_NAMES[c])
axes[2].legend(title='class'); axes[2].axis('off')
axes[2].set_title('Predicted class colors')

plt.tight_layout()
plt.savefig('plots/yolo_analysis.png', dpi=100, bbox_inches='tight')
plt.close()

# Detection visualisation: GT (green) vs Pred (red)
fig2, axes2 = plt.subplots(2, 5, figsize=(16, 7))
for idx, ax in enumerate(axes2.flat):
    ax.imshow(X_test[idx, :, :, 0], cmap='gray')
    for g in boxes_test[idx]:
        ax.add_patch(plt.Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1],
                                   fill=False, edgecolor='lime', lw=1.5))
    for p in nms(decode(y_pred_all[idx])):
        ax.add_patch(plt.Rectangle((p[0], p[1]), p[2] - p[0], p[3] - p[1],
                                   fill=False, edgecolor='red', lw=1.2, ls='--'))
        ax.text(p[0], p[1] - 1, CLASS_NAMES[p[4]], color='red', fontsize=6)
    ax.set_title(f'#{idx}  P{len(nms(decode(y_pred_all[idx])))}/G{len(boxes_test[idx])}', fontsize=8)
    ax.axis('off')
fig2.suptitle('YOLO Detections  (green = ground truth, red = prediction)', fontsize=13)
plt.tight_layout()
plt.savefig('plots/yolo_detections.png', dpi=100, bbox_inches='tight')
plt.close()

print("\nPlots saved to plots/\nPredictions saved to predictions.csv\nDone!")
