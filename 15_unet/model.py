"""
U-Net — Image Segmentation
============================
Dataset: Synthetic binary image segmentation dataset
Reference: https://arxiv.org/abs/1505.04597
Real use: https://www.kaggle.com/competitions/data-science-bowl-2018

Framework: TensorFlow / Keras
Task: Binary Image Segmentation — Segment foreground circles from background
Output: Per-pixel segmentation mask + IoU score
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  U-NET — Binary Image Segmentation")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1. Generate Synthetic Segmentation Dataset
# ─────────────────────────────────────────────────────────────
def generate_seg_data(n_samples=500, img_size=64, seed=42):
    np.random.seed(seed)
    imgs = np.zeros((n_samples, img_size, img_size, 1), dtype=np.float32)
    masks = np.zeros((n_samples, img_size, img_size, 1), dtype=np.float32)
    for i in range(n_samples):
        # Background noise
        noise = np.random.rand(img_size, img_size) * 0.2
        imgs[i, :, :, 0] = noise
        # Add 1-3 circles as foreground
        n_circles = np.random.randint(1, 4)
        for _ in range(n_circles):
            cx = np.random.randint(10, img_size-10)
            cy = np.random.randint(10, img_size-10)
            r  = np.random.randint(5, 15)
            y_g, x_g = np.ogrid[:img_size, :img_size]
            circle = (x_g - cx)**2 + (y_g - cy)**2 <= r**2
            imgs[i, circle, 0]  = np.random.uniform(0.6, 1.0)
            masks[i, circle, 0] = 1.0
    return imgs, masks

X_all, y_all = generate_seg_data(600, 64)
X_train, y_train = X_all[:500], y_all[:500]
X_test,  y_test  = X_all[500:], y_all[500:]

# Save sample info
pd.DataFrame({'sample_index': range(len(X_test)), 'n_foreground_pixels': y_test.reshape(len(X_test), -1).sum(axis=1)}).to_csv('data/segmentation_info.csv', index=False)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ─────────────────────────────────────────────────────────────
# 2. Build U-Net
# ─────────────────────────────────────────────────────────────
def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

tf.random.set_seed(42)
inputs = tf.keras.Input(shape=(64, 64, 1))

# Encoder (contracting path)
c1 = conv_block(inputs, 16); p1 = tf.keras.layers.MaxPooling2D()(c1)
c2 = conv_block(p1, 32);     p2 = tf.keras.layers.MaxPooling2D()(c2)
c3 = conv_block(p2, 64);     p3 = tf.keras.layers.MaxPooling2D()(c3)

# Bottleneck
bn = conv_block(p3, 128)

# Decoder (expansive path)
u4 = tf.keras.layers.UpSampling2D()(bn)
u4 = tf.keras.layers.Concatenate()([u4, c3])
c4 = conv_block(u4, 64)

u5 = tf.keras.layers.UpSampling2D()(c4)
u5 = tf.keras.layers.Concatenate()([u5, c2])
c5 = conv_block(u5, 32)

u6 = tf.keras.layers.UpSampling2D()(c5)
u6 = tf.keras.layers.Concatenate()([u6, c1])
c6 = conv_block(u6, 16)

outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(c6)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("\nU-Net Architecture:")
model.summary()

# ─────────────────────────────────────────────────────────────
# 3. Train
# ─────────────────────────────────────────────────────────────
history = model.fit(X_train, y_train, epochs=20, batch_size=16,
                    validation_split=0.1, verbose=1)

# ─────────────────────────────────────────────────────────────
# 4. Predictions (segmentation masks)
# ─────────────────────────────────────────────────────────────
y_pred_prob = model.predict(X_test, verbose=0)
y_pred_mask = (y_pred_prob > 0.5).astype(np.float32)

# Per-sample IoU (Intersection over Union)
def compute_iou(pred, true):
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return intersection / (union + 1e-8)

iou_scores = [compute_iou(y_pred_mask[i].flatten(), y_test[i].flatten()) for i in range(len(X_test))]
pixel_acc  = [accuracy_score(y_test[i].flatten(), y_pred_mask[i].flatten()) for i in range(len(X_test))]

pred_df = pd.DataFrame({
    'Sample_ID': range(len(X_test)),
    'IoU_Score': iou_scores,
    'Pixel_Accuracy': pixel_acc,
    'True_Foreground_Pixels': [y_test[i].sum() for i in range(len(X_test))],
    'Pred_Foreground_Pixels': [y_pred_mask[i].sum() for i in range(len(X_test))]
})
pred_df.to_csv('predictions.csv', index=False)

mean_iou = np.mean(iou_scores)
mean_acc = np.mean(pixel_acc)
print(f"\n── Segmentation Metrics ──")
print(f"  Mean IoU         : {mean_iou:.4f}")
print(f"  Mean Pixel Acc   : {mean_acc:.4f}")
print("  Note: Accuracy/Precision/F1/Confusion Matrix are per-pixel.")
print("        For segmentation, IoU is the standard metric.")

print("\n── Sample Predictions (per-image IoU) ──")
print(f"{'#':>3} | {'IoU':>8} | {'Pix Acc':>9} | {'True Px':>9} | {'Pred Px':>9}")
print("-" * 50)
for i in range(10):
    r = pred_df.iloc[i]
    print(f"{i:>3} | {r['IoU_Score']:>8.4f} | {r['Pixel_Accuracy']:>9.4f} | {int(r['True_Foreground_Pixels']):>9} | {int(r['Pred_Foreground_Pixels']):>9}")

# ─────────────────────────────────────────────────────────────
# 5. Plots
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history.history['loss'], label='Train Loss', color='steelblue')
axes[0].plot(history.history['val_loss'], label='Val Loss', color='coral')
axes[0].set_title('Loss — U-Net'); axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Train Acc', color='steelblue')
axes[1].plot(history.history['val_accuracy'], label='Val Acc', color='coral')
axes[1].set_title('Pixel Accuracy — U-Net'); axes[1].legend()

axes[2].hist(iou_scores, bins=20, color='steelblue', alpha=0.7)
axes[2].axvline(mean_iou, color='red', linestyle='--', label=f'Mean IoU={mean_iou:.3f}')
axes[2].set_xlabel('IoU Score')
axes[2].set_ylabel('Count')
axes[2].set_title('IoU Distribution — U-Net Segmentation')
axes[2].legend()

plt.tight_layout()
plt.savefig('plots/unet_metrics.png', dpi=100, bbox_inches='tight')
plt.close()

# Segmentation visualisation
fig2, axes2 = plt.subplots(4, 6, figsize=(18, 12))
for i in range(6):
    axes2[0, i].imshow(X_test[i, :, :, 0], cmap='gray')
    axes2[0, i].set_title(f'Input {i}', fontsize=8); axes2[0, i].axis('off')
    axes2[1, i].imshow(y_test[i, :, :, 0], cmap='Greens', vmin=0, vmax=1)
    axes2[1, i].set_title('True Mask', fontsize=8); axes2[1, i].axis('off')
    axes2[2, i].imshow(y_pred_mask[i, :, :, 0], cmap='Blues', vmin=0, vmax=1)
    axes2[2, i].set_title(f'Pred Mask', fontsize=8); axes2[2, i].axis('off')
    axes2[3, i].imshow(y_pred_prob[i, :, :, 0], cmap='hot', vmin=0, vmax=1)
    axes2[3, i].set_title(f'IoU={iou_scores[i]:.2f}', fontsize=8); axes2[3, i].axis('off')

row_labels = ['Input', 'True Mask', 'Pred Mask', 'Pred Prob']
for idx, lbl in enumerate(row_labels):
    axes2[idx, 0].set_ylabel(lbl, fontsize=10, rotation=90)
fig2.suptitle('U-Net Segmentation Results', fontsize=13)
plt.tight_layout()
plt.savefig('plots/unet_segmentation.png', dpi=100, bbox_inches='tight')
plt.close()

print("\nPlots saved to plots/\nPredictions saved to predictions.csv\nDone!")
