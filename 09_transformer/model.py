"""
Transformer (Encoder-based Text Classifier)
===========================================
Dataset: IMDB Sentiment Analysis
Source: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb
Original: https://ai.stanford.edu/~amaas/data/sentiment/

Framework: TensorFlow / Keras
Task: Binary Classification — Sentiment Analysis using Transformer encoder
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
print("  TRANSFORMER — IMDB Sentiment Analysis")
print("=" * 60)

MAX_FEATURES = 5000
MAX_LEN = 100
EMBED_DIM = 64
NUM_HEADS = 4
FF_DIM = 128

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=MAX_FEATURES)
X_train, y_train = X_train[:5000], y_train[:5000]
X_test,  y_test  = X_test[:1000],  y_test[:1000]
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=MAX_LEN)
X_test  = tf.keras.preprocessing.sequence.pad_sequences(X_test,  maxlen=MAX_LEN)
pd.DataFrame({'label': y_test}).to_csv('data/imdb_labels.csv', index=False)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ─────────────────────────────────────────────────────────────
# Multi-head Self-Attention Block
# ─────────────────────────────────────────────────────────────
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.pos_emb   = tf.keras.layers.Embedding(maxlen, embed_dim)

    def call(self, x):
        positions = tf.range(tf.shape(x)[-1])
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

# Build model
tf.random.set_seed(42)
inputs = tf.keras.Input(shape=(MAX_LEN,))
x = TokenAndPositionEmbedding(MAX_LEN, MAX_FEATURES, EMBED_DIM)(inputs)
x = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM)(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dropout(0.1)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("\nTransformer Architecture:")
model.summary()

history = model.fit(X_train, y_train, epochs=8, batch_size=64,
                    validation_split=0.1, verbose=1)

y_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_prob > 0.5).astype(int)

pred_df = pd.DataFrame({
    'True_Label': y_test,
    'Predicted_Label': y_pred,
    'Prob_Positive': y_prob,
    'True_Sentiment': ['Positive' if v == 1 else 'Negative' for v in y_test],
    'Pred_Sentiment': ['Positive' if v == 1 else 'Negative' for v in y_pred],
    'Correct': y_test == y_pred
})
pred_df.to_csv('predictions.csv', index=False)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)

print(f"\n── Evaluation Metrics ──")
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"\n── Confusion Matrix ──\n  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}")

print("\n── Sample Predictions (first 10) ──")
print(f"{'#':>3} | {'True':>10} | {'Predicted':>10} | {'Prob(Pos)':>10}")
print("-" * 45)
for i in range(10):
    r = pred_df.iloc[i]
    print(f"{i:>3} | {r['True_Sentiment']:>10} | {r['Pred_Sentiment']:>10} | {r['Prob_Positive']:>10.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(history.history['loss'], label='Train', color='steelblue')
axes[0].plot(history.history['val_loss'], label='Val', color='coral')
axes[0].set_title('Loss — Transformer'); axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Train Acc', color='steelblue')
axes[1].plot(history.history['val_accuracy'], label='Val Acc', color='coral')
axes[1].set_title('Accuracy — Transformer'); axes[1].legend()

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=['Neg','Pos'], yticklabels=['Neg','Pos'])
axes[2].set_title('Confusion Matrix — Transformer')

plt.tight_layout()
plt.savefig('plots/transformer_analysis.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nPlots saved to plots/\nPredictions saved to predictions.csv\nDone!")
