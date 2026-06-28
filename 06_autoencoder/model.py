"""
Autoencoder
============
Dataset: MNIST Handwritten Digits (via TensorFlow)
Source: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist
Original: http://yann.lecun.com/exdb/mnist/

Framework: TensorFlow / Keras
Task: Unsupervised — Dimensionality reduction + Reconstruction + Anomaly detection
Output: Reconstruction of input images, reconstruction error as anomaly score
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
print("  AUTOENCODER — MNIST Reconstruction + Anomaly Detection")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────────────────────
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train[:8000].reshape(-1, 784).astype('float32') / 255.0
X_test  = X_test[:2000].reshape(-1, 784).astype('float32') / 255.0
y_test_sub = y_test[:2000]

pd.DataFrame({'label': y_test_sub}).to_csv('data/mnist_labels.csv', index=False)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ─────────────────────────────────────────────────────────────
# 2. Build Autoencoder
# ─────────────────────────────────────────────────────────────
tf.random.set_seed(42)
ENCODING_DIM = 32

# Encoder
input_layer = tf.keras.Input(shape=(784,))
encoded = tf.keras.layers.Dense(256, activation='relu')(input_layer)
encoded = tf.keras.layers.Dense(128, activation='relu')(encoded)
encoded = tf.keras.layers.Dense(ENCODING_DIM, activation='relu')(encoded)

# Decoder
decoded = tf.keras.layers.Dense(128, activation='relu')(encoded)
decoded = tf.keras.layers.Dense(256, activation='relu')(decoded)
decoded = tf.keras.layers.Dense(784, activation='sigmoid')(decoded)

autoencoder = tf.keras.Model(input_layer, decoded)
encoder = tf.keras.Model(input_layer, encoded)

autoencoder.compile(optimizer='adam', loss='mse')
print("\nAutoencoder Architecture:")
autoencoder.summary()

# ─────────────────────────────────────────────────────────────
# 3. Train
# ─────────────────────────────────────────────────────────────
history = autoencoder.fit(
    X_train, X_train, epochs=20, batch_size=256,
    validation_split=0.1, verbose=1, shuffle=True
)

# ─────────────────────────────────────────────────────────────
# 4. Predictions (reconstruction + encoded representation)
# ─────────────────────────────────────────────────────────────
X_reconstructed = autoencoder.predict(X_test, verbose=0)
X_encoded       = encoder.predict(X_test, verbose=0)

# Per-sample reconstruction MSE = anomaly score
recon_errors = np.mean((X_test - X_reconstructed)**2, axis=1)

pred_df = pd.DataFrame({
    'True_Label': y_test_sub,
    'Reconstruction_MSE': recon_errors,
    'Anomaly_Score': recon_errors,
    'Is_Anomaly': (recon_errors > np.percentile(recon_errors, 95)).astype(int)
})
# Add first 5 encoding dims
for i in range(5):
    pred_df[f'Encoding_Dim_{i+1}'] = X_encoded[:, i]
pred_df.to_csv('predictions.csv', index=False)

# ─────────────────────────────────────────────────────────────
# 5. Evaluation (Reconstruction Metrics)
# ─────────────────────────────────────────────────────────────
overall_mse = np.mean(recon_errors)
overall_mae = np.mean(np.abs(X_test - X_reconstructed))
n_anomalies = pred_df['Is_Anomaly'].sum()

print(f"\n── Reconstruction Metrics ──")
print(f"  Reconstruction MSE : {overall_mse:.6f}")
print(f"  Reconstruction MAE : {overall_mae:.6f}")
print(f"  Encoding Dimensions: {ENCODING_DIM} (from 784)")
print(f"  Anomalies detected (top 5%): {n_anomalies}")
print("  Note: Accuracy/Precision/F1/Confusion Matrix not applicable (unsupervised).")

print("\n── Sample Predictions (Reconstruction Error) ──")
print(f"{'#':>3} | {'Digit':>6} | {'Recon MSE':>10} | {'Is Anomaly':>11} | {'Enc_Dim1':>10}")
print("-" * 55)
for i in range(10):
    r = pred_df.iloc[i]
    print(f"{i:>3} | {int(r['True_Label']):>6} | {r['Reconstruction_MSE']:>10.6f} | {str(bool(r['Is_Anomaly'])):>11} | {r['Encoding_Dim_1']:>10.4f}")

# ─────────────────────────────────────────────────────────────
# 6. Plots
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss
axes[0].plot(history.history['loss'], label='Train MSE', color='steelblue')
axes[0].plot(history.history['val_loss'], label='Val MSE', color='coral')
axes[0].set_title('Reconstruction Loss — Autoencoder')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE'); axes[0].legend()

# Reconstruction comparison
n_show = 5
orig_arr = X_test[:n_show].reshape(n_show, 28, 28)
recon_arr = X_reconstructed[:n_show].reshape(n_show, 28, 28)
combined = np.hstack([np.hstack(orig_arr), np.hstack(recon_arr)])
axes[1].imshow(combined, cmap='gray')
axes[1].set_title('Original (top) vs Reconstructed (bottom)\n[Left=Original, Right=Reconstructed]')
axes[1].axis('off')

# Reconstruction error distribution
axes[2].hist(recon_errors, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
axes[2].axvline(np.percentile(recon_errors, 95), color='red', linestyle='--', label='95th percentile')
axes[2].set_xlabel('Reconstruction MSE')
axes[2].set_ylabel('Count')
axes[2].set_title('Reconstruction Error Distribution (Anomaly Scores)')
axes[2].legend()

plt.tight_layout()
plt.savefig('plots/autoencoder_analysis.png', dpi=100, bbox_inches='tight')
plt.close()

# Show original vs reconstructed side by side
fig2, axes2 = plt.subplots(2, 8, figsize=(16, 4))
for i in range(8):
    axes2[0, i].imshow(X_test[i].reshape(28, 28), cmap='gray')
    axes2[0, i].set_title(f'Orig {y_test_sub[i]}', fontsize=8)
    axes2[0, i].axis('off')
    axes2[1, i].imshow(X_reconstructed[i].reshape(28, 28), cmap='gray')
    axes2[1, i].set_title(f'Recon {y_test_sub[i]}', fontsize=8)
    axes2[1, i].axis('off')
fig2.suptitle('Autoencoder: Original vs Reconstructed', fontsize=12)
plt.tight_layout()
plt.savefig('plots/reconstruction_samples.png', dpi=100, bbox_inches='tight')
plt.close()

print("\nPlots saved to plots/\nPredictions saved to predictions.csv\nDone!")
