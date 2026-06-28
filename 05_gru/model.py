"""
Gated Recurrent Unit (GRU)
===========================
Dataset: IMDB Sentiment Analysis (via TensorFlow)
Source: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb
Original: https://ai.stanford.edu/~amaas/data/sentiment/

Framework: TensorFlow / Keras
Task: Binary Classification — Sentiment Analysis
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
print("  GRU — IMDB Sentiment Analysis")
print("=" * 60)

MAX_FEATURES = 5000
MAX_LEN = 100

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=MAX_FEATURES)
X_train, y_train = X_train[:5000], y_train[:5000]
X_test,  y_test  = X_test[:1000],  y_test[:1000]

X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=MAX_LEN)
X_test  = tf.keras.preprocessing.sequence.pad_sequences(X_test,  maxlen=MAX_LEN)

pd.DataFrame({'label': y_test}).to_csv('data/imdb_labels.csv', index=False)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(MAX_FEATURES, 64, input_length=MAX_LEN),
    tf.keras.layers.GRU(64, return_sequences=True),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("\nGRU Architecture:")
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
axes[0].set_title('Loss — GRU'); axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Train Acc', color='steelblue')
axes[1].plot(history.history['val_accuracy'], label='Val Acc', color='coral')
axes[1].set_title('Accuracy — GRU'); axes[1].legend()

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
axes[2].set_title('Confusion Matrix — GRU')
axes[2].set_ylabel('True'); axes[2].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('plots/gru_analysis.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nPlots saved to plots/\nPredictions saved to predictions.csv\nDone!")
