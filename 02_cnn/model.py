"""
Convolutional Neural Network (CNN)
====================================
Dataset: MNIST Handwritten Digits (via TensorFlow)
Source: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist
Original: http://yann.lecun.com/exdb/mnist/

Framework: TensorFlow / Keras
Task: Multi-class Classification — Digit recognition (0-9)
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
print("  CNN — MNIST Handwritten Digits (0-9)")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────────────────────
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# Use subset for speed
X_train, y_train = X_train[:10000], y_train[:10000]
X_test,  y_test  = X_test[:2000],   y_test[:2000]

X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test  = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Save sample data info
pd.DataFrame({'label': y_test[:500]}).to_csv('data/mnist_test_labels.csv', index=False)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ─────────────────────────────────────────────────────────────
# 2. Build CNN
# ─────────────────────────────────────────────────────────────
tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("CNN Architecture:")
model.summary()

# ─────────────────────────────────────────────────────────────
# 3. Train
# ─────────────────────────────────────────────────────────────
history = model.fit(X_train, y_train, epochs=10, batch_size=64,
                    validation_split=0.1, verbose=1)

# ─────────────────────────────────────────────────────────────
# 4. Predictions
# ─────────────────────────────────────────────────────────────
y_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

pred_df = pd.DataFrame({
    'True_Digit': y_test,
    'Predicted_Digit': y_pred,
    'Confidence': y_prob.max(axis=1),
    'Correct': y_test == y_pred
})
pred_df.to_csv('predictions.csv', index=False)

# ─────────────────────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────────────────────
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
f1   = f1_score(y_test, y_pred, average='weighted')
cm   = confusion_matrix(y_test, y_pred)

print(f"\n── Evaluation Metrics ──")
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  F1 Score  : {f1:.4f}")

print("\n── Sample Predictions (first 10) ──")
print(f"{'#':>3} | {'True':>6} | {'Predicted':>10} | {'Confidence':>11} | {'Correct':>8}")
print("-" * 50)
for i in range(10):
    r = pred_df.iloc[i]
    print(f"{i:>3} | {int(r['True_Digit']):>6} | {int(r['Predicted_Digit']):>10} | {r['Confidence']:>11.4f} | {str(r['Correct']):>8}")

# ─────────────────────────────────────────────────────────────
# 6. Plots
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history.history['loss'], label='Train', color='steelblue')
axes[0].plot(history.history['val_loss'], label='Val', color='coral')
axes[0].set_title('Loss — CNN'); axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Cross-Entropy'); axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Train Acc', color='steelblue')
axes[1].plot(history.history['val_accuracy'], label='Val Acc', color='coral')
axes[1].set_title('Accuracy — CNN'); axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy'); axes[1].legend()

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2])
axes[2].set_title('Confusion Matrix — CNN (10×10)')
axes[2].set_ylabel('True'); axes[2].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('plots/cnn_analysis.png', dpi=100, bbox_inches='tight')
plt.close()

# Sample predictions grid
fig2, axes2 = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes2.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
    t = y_test[i]; p = y_pred[i]
    ax.set_title(f'T:{t} P:{p}', color='green' if t == p else 'red', fontsize=9)
    ax.axis('off')
fig2.suptitle('CNN — Sample Predictions (green=correct)', fontsize=12)
plt.tight_layout()
plt.savefig('plots/sample_predictions.png', dpi=100, bbox_inches='tight')
plt.close()

print("\nPlots saved to plots/\nPredictions saved to predictions.csv\nDone!")
