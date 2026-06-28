"""
Artificial Neural Network (ANN) — Multi-layer Perceptron
==========================================================
Dataset: Breast Cancer Wisconsin (sklearn built-in)
Source: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
Original: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

Framework: TensorFlow / Keras
Task: Binary Classification — Cancer detection
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import tensorflow as tf

print("=" * 60)
print("  ANN (Multi-Layer Perceptron) — Breast Cancer Dataset")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────────────────────
data = load_breast_cancer(as_frame=True)
X = data.data.values
y = data.target.values
df = pd.concat([data.data, pd.Series(data.target, name='label')], axis=1)
df.to_csv('data/breast_cancer.csv', index=False)
print(f"\nDataset: {X.shape}, Classes: malignant/benign")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────
# 2. Build Model
# ─────────────────────────────────────────────────────────────
tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("\nModel Architecture:")
model.summary()

# ─────────────────────────────────────────────────────────────
# 3. Train
# ─────────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train, epochs=50, batch_size=32,
    validation_split=0.15, verbose=0
)
print(f"\nTrained for {len(history.history['loss'])} epochs.")

# ─────────────────────────────────────────────────────────────
# 4. Predictions
# ─────────────────────────────────────────────────────────────
y_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_prob > 0.5).astype(int)

pred_df = pd.DataFrame({
    'True_Label': y_test,
    'Predicted_Label': y_pred,
    'Prob_Benign': y_prob,
    'True_Class': [data.target_names[i] for i in y_test],
    'Pred_Class': [data.target_names[i] for i in y_pred],
    'Correct': y_test == y_pred
})
pred_df.to_csv('predictions.csv', index=False)

# ─────────────────────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────────────────────
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
print(f"{'#':>3} | {'True':>10} | {'Predicted':>10} | {'Prob(Benign)':>12}")
print("-" * 45)
for i in range(10):
    print(f"{i:>3} | {pred_df['True_Class'].iloc[i]:>10} | {pred_df['Pred_Class'].iloc[i]:>10} | {pred_df['Prob_Benign'].iloc[i]:>12.4f}")

# ─────────────────────────────────────────────────────────────
# 6. Plots
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Training curves
axes[0].plot(history.history['loss'], label='Train Loss', color='steelblue')
axes[0].plot(history.history['val_loss'], label='Val Loss', color='coral')
axes[0].set_title('Loss Curve — ANN', fontsize=13)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Binary Cross-Entropy')
axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Train Acc', color='steelblue')
axes[1].plot(history.history['val_accuracy'], label='Val Acc', color='coral')
axes[1].set_title('Accuracy Curve — ANN', fontsize=13)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].legend()

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=data.target_names, yticklabels=data.target_names)
axes[2].set_title('Confusion Matrix — ANN', fontsize=13)
axes[2].set_ylabel('True Label'); axes[2].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('plots/ann_analysis.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nPlots saved to plots/\nPredictions saved to predictions.csv\nDone!")
