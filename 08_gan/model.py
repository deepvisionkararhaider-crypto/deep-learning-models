"""
Generative Adversarial Network (GAN)
=======================================
Dataset: MNIST Handwritten Digits (via TensorFlow)
Source: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist
Original: http://yann.lecun.com/exdb/mnist/

Framework: TensorFlow / Keras
Task: Generative — Generate new digit images
Output: Generated images + discriminator scores (real/fake probability)
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
print("  GAN — MNIST Digit Generation")
print("=" * 60)

LATENT_DIM = 64
EPOCHS = 20

# ─────────────────────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────────────────────
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train[:5000].reshape(-1, 784).astype('float32') / 127.5 - 1.0
X_test  = X_test[:500].reshape(-1, 784).astype('float32') / 127.5 - 1.0
pd.DataFrame({'label': y_test[:500]}).to_csv('data/mnist_labels.csv', index=False)
print(f"\nTraining on {X_train.shape[0]} real images.")

# ─────────────────────────────────────────────────────────────
# 2. Build Generator & Discriminator
# ─────────────────────────────────────────────────────────────
tf.random.set_seed(42)

# Generator
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(LATENT_DIM,)),
    tf.keras.layers.LeakyReLU(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.LeakyReLU(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(784, activation='tanh')
], name='generator')

# Discriminator
discriminator = tf.keras.Sequential([
    tf.keras.layers.Dense(512, input_shape=(784,)),
    tf.keras.layers.LeakyReLU(0.2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256),
    tf.keras.layers.LeakyReLU(0.2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
], name='discriminator')
discriminator.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                      loss='binary_crossentropy', metrics=['accuracy'])

# GAN (generator + frozen discriminator)
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(LATENT_DIM,))
gan_output = discriminator(generator(gan_input))
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy')

print("Generator and Discriminator built.")
generator.summary()

# ─────────────────────────────────────────────────────────────
# 3. Train GAN
# ─────────────────────────────────────────────────────────────
BATCH_SIZE = 128
d_losses, g_losses, d_accs = [], [], []
real_labels = np.ones((BATCH_SIZE, 1)) * 0.9   # label smoothing
fake_labels = np.zeros((BATCH_SIZE, 1))

for epoch in range(EPOCHS):
    # Train discriminator
    idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
    real_imgs = X_train[idx]
    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    fake_imgs = generator.predict(noise, verbose=0)
    d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
    d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])
    d_acc  = 0.5 * (d_loss_real[1] + d_loss_fake[1])

    # Train generator
    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    g_loss = gan.train_on_batch(noise, np.ones((BATCH_SIZE, 1)))

    d_losses.append(d_loss); g_losses.append(g_loss); d_accs.append(d_acc)
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS} — D_loss: {d_loss:.4f} | G_loss: {g_loss:.4f} | D_acc: {d_acc:.3f}")

# ─────────────────────────────────────────────────────────────
# 4. Predictions (discriminator scores on real vs generated)
# ─────────────────────────────────────────────────────────────
noise_test = np.random.normal(0, 1, (500, LATENT_DIM))
generated_imgs = generator.predict(noise_test, verbose=0)
real_scores = discriminator.predict(X_test, verbose=0).flatten()
fake_scores = discriminator.predict(generated_imgs, verbose=0).flatten()

pred_df = pd.DataFrame({
    'Image_Type': ['Real'] * len(real_scores) + ['Generated'] * len(fake_scores),
    'Discriminator_Score_Real': np.concatenate([real_scores, np.zeros(len(fake_scores))]),
    'Discriminator_Score_Fake': np.concatenate([np.zeros(len(real_scores)), fake_scores]),
    'Score': np.concatenate([real_scores, fake_scores]),
    'Classified_As': ['Real' if s > 0.5 else 'Fake' for s in np.concatenate([real_scores, fake_scores])]
})
pred_df.to_csv('predictions.csv', index=False)

# ─────────────────────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────────────────────
print(f"\n── GAN Metrics ──")
print(f"  Final Generator Loss     : {g_losses[-1]:.4f}")
print(f"  Final Discriminator Loss : {d_losses[-1]:.4f}")
print(f"  Final Discriminator Acc  : {d_accs[-1]:.4f}")
print(f"  Avg score (real images)  : {real_scores.mean():.4f}")
print(f"  Avg score (generated)    : {fake_scores.mean():.4f}")
print("  Note: Accuracy/Precision/F1/Confusion Matrix not applicable (generative model).")

print("\n── Discriminator Scores (first 10 real + first 10 generated) ──")
print(f"{'#':>3} | {'Image Type':>12} | {'D Score':>9} | {'Classified As':>14}")
print("-" * 45)
for i in range(10):
    print(f"{i:>3} | {'Real':>12} | {real_scores[i]:>9.4f} | {'Real' if real_scores[i]>0.5 else 'Fake':>14}")
for i in range(10):
    print(f"{i:>3} | {'Generated':>12} | {fake_scores[i]:>9.4f} | {'Real' if fake_scores[i]>0.5 else 'Fake':>14}")

# ─────────────────────────────────────────────────────────────
# 6. Plots
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(d_losses, label='Discriminator Loss', color='steelblue')
axes[0].plot(g_losses, label='Generator Loss', color='coral')
axes[0].set_title('GAN Training Losses'); axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss'); axes[0].legend()

axes[1].plot(d_accs, color='seagreen', label='Discriminator Accuracy')
axes[1].set_title('Discriminator Accuracy vs Epoch'); axes[1].legend()
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')

axes[2].hist(real_scores, bins=30, alpha=0.7, label='Real Images', color='steelblue')
axes[2].hist(fake_scores, bins=30, alpha=0.7, label='Generated Images', color='coral')
axes[2].set_title('Discriminator Score Distribution')
axes[2].set_xlabel('D(x) Score'); axes[2].legend()

plt.tight_layout()
plt.savefig('plots/gan_training.png', dpi=100, bbox_inches='tight')
plt.close()

# Generated image grid
fig2, axes2 = plt.subplots(2, 8, figsize=(16, 4))
for i, ax in enumerate(axes2.flat):
    if i < 16:
        ax.imshow(generated_imgs[i].reshape(28, 28), cmap='gray')
        ax.set_title(f'Gen-{i}', fontsize=7); ax.axis('off')
fig2.suptitle('GAN: Generated Digit Samples', fontsize=12)
plt.tight_layout()
plt.savefig('plots/gan_generated.png', dpi=100, bbox_inches='tight')
plt.close()

print("\nPlots saved to plots/\nPredictions saved to predictions.csv\nDone!")
