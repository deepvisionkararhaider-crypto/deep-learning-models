"""
Variational Autoencoder (VAE)
==============================
Dataset: MNIST Handwritten Digits (via TensorFlow)
Source: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist
Original: http://yann.lecun.com/exdb/mnist/

Framework: TensorFlow / Keras
Task: Generative model — Learn latent space + reconstruct/generate digits
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
print("  VARIATIONAL AUTOENCODER (VAE) — MNIST")
print("=" * 60)

LATENT_DIM = 2

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train[:6000].reshape(-1, 784).astype('float32') / 255.0
X_test  = X_test[:1000].reshape(-1, 784).astype('float32') / 255.0
y_test_sub = y_test[:1000]
pd.DataFrame({'label': y_test_sub}).to_csv('data/mnist_labels.csv', index=False)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

tf.random.set_seed(42)

# ── Encoder ──
enc_input = tf.keras.Input(shape=(784,), name='enc_input')
h = tf.keras.layers.Dense(256, activation='relu')(enc_input)
h = tf.keras.layers.Dense(128, activation='relu')(h)
z_mean    = tf.keras.layers.Dense(LATENT_DIM, name='z_mean')(h)
z_log_var = tf.keras.layers.Dense(LATENT_DIM, name='z_log_var')(h)

encoder = tf.keras.Model(enc_input, [z_mean, z_log_var], name='encoder')

# ── Decoder ──
dec_input = tf.keras.Input(shape=(LATENT_DIM,), name='dec_input')
d = tf.keras.layers.Dense(128, activation='relu')(dec_input)
d = tf.keras.layers.Dense(256, activation='relu')(d)
dec_output = tf.keras.layers.Dense(784, activation='sigmoid')(d)
decoder = tf.keras.Model(dec_input, dec_output, name='decoder')

# ── VAE training step ──
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    def call(self, x, training=False):
        z_mean, z_log_var = self.encoder(x, training=training)
        eps = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * eps
        return self.decoder(z, training=training)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            eps = tf.random.normal(shape=tf.shape(z_mean))
            z = z_mean + tf.exp(0.5 * z_log_var) * eps
            reconstruction = self.decoder(z)
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconstruction)) * 784)
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = recon_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        return {"total_loss": self.total_loss_tracker.result()}

vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam())
print("\nVAE built (encoder + decoder).")

history = vae.fit(X_train, epochs=15, batch_size=128, verbose=1)

# Predictions
z_mean_pred, z_log_var_pred = encoder.predict(X_test, verbose=0)
eps = np.random.normal(size=z_mean_pred.shape)
z_samples = z_mean_pred + np.exp(0.5 * z_log_var_pred) * eps
X_reconstructed = decoder.predict(z_samples, verbose=0)
recon_errors = np.mean((X_test - X_reconstructed)**2, axis=1)

pred_df = pd.DataFrame({
    'True_Label': y_test_sub,
    'Latent_Z1': z_mean_pred[:, 0],
    'Latent_Z2': z_mean_pred[:, 1],
    'Reconstruction_MSE': recon_errors
})
pred_df.to_csv('predictions.csv', index=False)

print(f"\n── VAE Metrics ──")
print(f"  Reconstruction MSE : {np.mean(recon_errors):.6f}")
print(f"  Latent Dimensions  : {LATENT_DIM}")
print("  Note: Accuracy/Precision/F1/Confusion Matrix not applicable (generative).")

print("\n── Sample Latent Representations (first 10) ──")
print(f"{'#':>3} | {'Digit':>6} | {'Z1':>10} | {'Z2':>10} | {'Recon MSE':>10}")
print("-" * 50)
for i in range(10):
    r = pred_df.iloc[i]
    print(f"{i:>3} | {int(r['True_Label']):>6} | {r['Latent_Z1']:>10.4f} | {r['Latent_Z2']:>10.4f} | {r['Reconstruction_MSE']:>10.6f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['total_loss'], color='steelblue', label='Train Loss')
axes[0].set_title('VAE Total Loss (Recon+KL)'); axes[0].legend()

colors_map = plt.cm.tab10(np.linspace(0, 1, 10))
for digit in range(10):
    mask = y_test_sub == digit
    axes[1].scatter(z_mean_pred[mask, 0], z_mean_pred[mask, 1],
                    c=[colors_map[digit]], label=str(digit), alpha=0.5, s=10)
axes[1].set_title('VAE Latent Space (2D)')
axes[1].set_xlabel('Z1'); axes[1].set_ylabel('Z2')
axes[1].legend(title='Digit', ncol=2, fontsize=7)

plt.tight_layout()
plt.savefig('plots/vae_analysis.png', dpi=100, bbox_inches='tight')
plt.close()

fig2, ax2s = plt.subplots(2, 8, figsize=(16, 4))
for i in range(8):
    ax2s[0, i].imshow(X_test[i].reshape(28, 28), cmap='gray')
    ax2s[0, i].set_title(f'Orig {y_test_sub[i]}', fontsize=8); ax2s[0, i].axis('off')
    ax2s[1, i].imshow(X_reconstructed[i].reshape(28, 28), cmap='gray')
    ax2s[1, i].set_title('Recon', fontsize=8); ax2s[1, i].axis('off')
plt.tight_layout()
plt.savefig('plots/vae_reconstructions.png', dpi=100, bbox_inches='tight')
plt.close()

print("\nPlots saved to plots/\nPredictions saved to predictions.csv\nDone!")
