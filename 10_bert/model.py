"""
BERT — Bidirectional Encoder Representations from Transformers
==============================================================
Dataset: IMDB Sentiment Analysis
Source: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb
Pretrained model: distilbert-base-uncased (HuggingFace)
Model Hub: https://huggingface.co/distilbert-base-uncased

Framework: HuggingFace Transformers + PyTorch
Task: Binary Classification — Sentiment Analysis
Note: Uses DistilBERT (lighter BERT variant) for speed.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import (DistilBertTokenizerFast, DistilBertForSequenceClassification,
                          pipeline)
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  BERT (DistilBERT) — Sentiment Analysis")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1. Prepare Sample Dataset
# ─────────────────────────────────────────────────────────────
# Representative texts for demonstration (IMDB-style)
texts = [
    "This movie was absolutely fantastic! The acting was superb.",
    "Terrible film. Worst movie I have ever seen. Complete waste of time.",
    "An amazing cinematic experience. Highly recommended!",
    "Boring and predictable. Nothing new here.",
    "The story is beautiful and the performances are outstanding.",
    "I fell asleep watching this. Very dull and uninteresting.",
    "Great plot twists! Kept me on the edge of my seat.",
    "Poor acting, poor script, poor direction. Avoid.",
    "A masterpiece of storytelling. One of the best films this year.",
    "Completely disappointing. The trailer was much better than the film.",
    "Brilliant performances from all cast members. A must-watch!",
    "Overly long and tedious. Should have been 30 minutes shorter.",
    "Heartwarming and touching. Made me cry happy tears.",
    "Nonsensical plot with zero character development.",
    "One of the most creative films in recent memory.",
    "Cliche after cliche. Nothing original here.",
    "Genuinely moving story. Excellent production quality.",
    "Extremely slow pace. Struggles to maintain any interest.",
    "Perfect blend of humor and drama. Love every minute!",
    "Disappointing sequel. The original was far superior.",
]
true_labels = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]

df = pd.DataFrame({'text': texts, 'true_label': true_labels})
df.to_csv('data/sentiment_samples.csv', index=False)
print(f"\nDataset: {len(texts)} labeled review samples")
print("Using DistilBERT for inference (zero-shot, no fine-tuning for speed).")

# ─────────────────────────────────────────────────────────────
# 2. Load DistilBERT via pipeline
# ─────────────────────────────────────────────────────────────
print("\nLoading distilbert-base-uncased-finetuned-sst-2-english...")
classifier = pipeline(
    'text-classification',
    model='distilbert-base-uncased-finetuned-sst-2-english',
    device=-1   # CPU
)
print("Model loaded.")

# ─────────────────────────────────────────────────────────────
# 3. Predictions
# ─────────────────────────────────────────────────────────────
results = classifier(texts, truncation=True, max_length=128)
y_pred = [1 if r['label'] == 'POSITIVE' else 0 for r in results]
y_prob_pos = [r['score'] if r['label'] == 'POSITIVE' else 1 - r['score'] for r in results]

pred_df = pd.DataFrame({
    'Text': [t[:60] + '...' for t in texts],
    'True_Label': true_labels,
    'Predicted_Label': y_pred,
    'True_Sentiment': ['Positive' if v == 1 else 'Negative' for v in true_labels],
    'Pred_Sentiment': ['Positive' if v == 1 else 'Negative' for v in y_pred],
    'Prob_Positive': y_prob_pos,
    'Correct': [t == p for t, p in zip(true_labels, y_pred)]
})
pred_df.to_csv('predictions.csv', index=False)

# ─────────────────────────────────────────────────────────────
# 4. Evaluation
# ─────────────────────────────────────────────────────────────
acc  = accuracy_score(true_labels, y_pred)
prec = precision_score(true_labels, y_pred)
f1   = f1_score(true_labels, y_pred)
cm   = confusion_matrix(true_labels, y_pred)

print(f"\n── Evaluation Metrics ──")
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"\n── Confusion Matrix ──\n  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}")

print("\n── Sample Predictions (first 10) ──")
print(f"{'#':>3} | {'True':>10} | {'Predicted':>10} | {'Prob(Pos)':>10} | {'Correct':>8}")
print("-" * 55)
for i in range(10):
    r = pred_df.iloc[i]
    print(f"{i:>3} | {r['True_Sentiment']:>10} | {r['Pred_Sentiment']:>10} | {r['Prob_Positive']:>10.4f} | {str(r['Correct']):>8}")

# ─────────────────────────────────────────────────────────────
# 5. Plots
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
axes[0].set_title('Confusion Matrix — BERT (DistilBERT SST-2)', fontsize=12)
axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

colors = ['coral' if p == 0 else 'steelblue' for p in y_pred]
axes[1].barh(range(len(texts)), y_prob_pos, color=colors)
axes[1].axvline(0.5, color='black', linestyle='--')
axes[1].set_yticks(range(len(texts)))
axes[1].set_yticklabels([f'Review {i+1}' for i in range(len(texts))], fontsize=8)
axes[1].set_xlabel('Probability (Positive)')
axes[1].set_title('BERT Prediction Confidence per Review')

plt.tight_layout()
plt.savefig('plots/bert_analysis.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nPlots saved to plots/\nPredictions saved to predictions.csv\nDone!")
