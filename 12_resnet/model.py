"""
ResNet — Residual Neural Network
==================================
Dataset: CIFAR-10 (via TensorFlow)
Source: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10
Original: https://www.cs.toronto.edu/~kriz/cifar.html

Framework: TensorFlow / Keras
Task: Multi-class Image Classification — 10 object categories
Architecture: ResNet-style with residual connections (from scratch)
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
print("  RESNET — CIFAR-10 Image Classification")
print("=" * 60)

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train, y_train = X_train[:8000], y_train[:8000].flatten()
X_test,  y_test  = X_test[:2000],  y_test[:2000].flatten()

X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32') / 255.0

pd.DataFrame({'label': y_test, 'class_name': [class_names[i] for i in y_test]}).to_csv('data/cifar10_labels.csv', index=False)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ─────────────────────────────────────────────────────────────
# Residual Block
# ─────────────────────────────────────────────────────────────
def residual_block(x, filters, strides=1):
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, 3, strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=strides, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x

tf.random.set_seed(42)
inputs = tf.keras.Input(shape=(32, 32, 3))
x = tf.keras.layers.Conv2D(32, 3, padding='same')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = residual_block(x, 32)
x = residual_block(x, 64, strides=2)
x = residual_block(x, 128, strides=2)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("\nResNet Architecture:")
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
axes[0].set_title('Loss — ResNet'); axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Train', color='steelblue')
axes[1].plot(history.history['val_accuracy'], label='Val', color='coral')
axes[1].set_title('Accuracy — ResNet'); axes[1].legend()

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=class_names, yticklabels=class_names)
axes[2].set_title('Confusion Matrix — ResNet (CIFAR-10)')
axes[2].set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
axes[2].set_yticklabels(class_names, rotation=0, fontsize=8)

plt.tight_layout()
plt.savefig('plots/resnet_analysis.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nPlots saved to plots/\nPredictions saved to predictions.csv\nDone!")
