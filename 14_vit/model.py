"""
Vision Transformer (ViT)
=========================
Dataset: CIFAR-10 (via TensorFlow)
Source: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10
Original: https://www.cs.toronto.edu/~kriz/cifar.html

Framework: TensorFlow / Keras
Task: Multi-class Image Classification using patch-based Transformer
Architecture: ViT from scratch — patch embeddings + transformer encoder
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  VISION TRANSFORMER (ViT) — CIFAR-10")
print("=" * 60)

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
IMAGE_SIZE = 32
PATCH_SIZE = 4
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 64
EMBED_DIM = 64
NUM_HEADS = 4
TRANSFORMER_LAYERS = 4

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train, y_train = X_train[:6000], y_train[:6000].flatten()
X_test,  y_test  = X_test[:2000],  y_test[:2000].flatten()
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32') / 255.0

pd.DataFrame({'label': y_test, 'class': [class_names[i] for i in y_test]}).to_csv('data/cifar10_labels.csv', index=False)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print(f"Patch size: {PATCH_SIZE}×{PATCH_SIZE}, Num patches: {NUM_PATCHES}")

# ─────────────────────────────────────────────────────────────
# Patch + Position Embedding Layer
# ─────────────────────────────────────────────────────────────
class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = tf.keras.layers.Dense(embed_dim)
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, images):
        batch_size = tf.shape(images)[0]
        ps = self.patch_size
        # Extract patches
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, ps, ps, 1],
            strides=[1, ps, ps, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return self.norm(self.proj(patches))

# ─────────────────────────────────────────────────────────────
# Build ViT
# ─────────────────────────────────────────────────────────────
tf.random.set_seed(42)
inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

# Patch + position embeddings
patch_emb = PatchEmbedding(PATCH_SIZE, EMBED_DIM)(inputs)
positions = tf.range(start=0, limit=NUM_PATCHES)
pos_emb   = tf.keras.layers.Embedding(NUM_PATCHES, EMBED_DIM)(positions)
x = patch_emb + pos_emb

# Transformer encoder blocks
for _ in range(TRANSFORMER_LAYERS):
    x1 = tf.keras.layers.LayerNormalization()(x)
    x1 = tf.keras.layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=EMBED_DIM//NUM_HEADS)(x1, x1)
    x  = tf.keras.layers.Add()([x, x1])
    x2 = tf.keras.layers.LayerNormalization()(x)
    x2 = tf.keras.layers.Dense(EMBED_DIM*2, activation='gelu')(x2)
    x2 = tf.keras.layers.Dense(EMBED_DIM)(x2)
    x  = tf.keras.layers.Add()([x, x2])

x = tf.keras.layers.LayerNormalization()(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("\nViT Architecture:")
model.summary()

history = model.fit(X_train, y_train, epochs=15, batch_size=64,
                    validation_split=0.1, verbose=1)

y_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

pred_df = pd.DataFrame({
    'True_Label': y_test,
    'Predicted_Label': y_pred,
    'True_Class': [class_names[i] for i in y_test],
    'Pred_Class': [class_names[i] for i in y_pred],
    'Confidence': y_prob.max(axis=1),
    'Correct': y_test == y_pred
})
pred_df.to_csv('predictions.csv', index=False)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
f1   = f1_score(y_test, y_pred, average='weighted')
cm   = confusion_matrix(y_test, y_pred)

print(f"\n── Evaluation Metrics ──")
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  F1 Score  : {f1:.4f}")

print("\n── Sample Predictions (first 10) ──")
print(f"{'#':>3} | {'True':>12} | {'Predicted':>12} | {'Confidence':>11}")
print("-" * 50)
for i in range(10):
    r = pred_df.iloc[i]
    print(f"{i:>3} | {r['True_Class']:>12} | {r['Pred_Class']:>12} | {r['Confidence']:>11.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(history.history['loss'], label='Train', color='steelblue')
axes[0].plot(history.history['val_loss'], label='Val', color='coral')
axes[0].set_title('Loss — Vision Transformer'); axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Train', color='steelblue')
axes[1].plot(history.history['val_accuracy'], label='Val', color='coral')
axes[1].set_title('Accuracy — Vision Transformer'); axes[1].legend()

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=class_names, yticklabels=class_names)
axes[2].set_title('Confusion Matrix — ViT')
axes[2].set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
axes[2].set_yticklabels(class_names, rotation=0, fontsize=8)

plt.tight_layout()
plt.savefig('plots/vit_analysis.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nPlots saved to plots/\nPredictions saved to predictions.csv\nDone!")
