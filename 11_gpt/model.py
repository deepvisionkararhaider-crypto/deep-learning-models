"""
GPT — Text Generation with GPT-2
==================================
Dataset: Custom prompts (demonstrating text generation capability)
Pretrained model: gpt2 (HuggingFace / OpenAI)
Model Hub: https://huggingface.co/gpt2

Framework: HuggingFace Transformers + PyTorch
Task: Causal Language Modeling — Text Generation
Output: Generated text + token-level probabilities for next tokens
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  GPT-2 — Text Generation")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1. Load GPT-2
# ─────────────────────────────────────────────────────────────
print("\nLoading GPT-2 (small, 117M params)...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()
tokenizer.pad_token = tokenizer.eos_token
print("GPT-2 loaded.")

# ─────────────────────────────────────────────────────────────
# 2. Prompts and generation
# ─────────────────────────────────────────────────────────────
prompts = [
    "Machine learning is a field of artificial intelligence that",
    "The future of deep learning involves",
    "Natural language processing allows computers to",
    "Neural networks are inspired by",
    "The transformer architecture revolutionized",
]

pd.DataFrame({'prompt': prompts}).to_csv('data/prompts.csv', index=False)

def generate_text(prompt, max_new_tokens=50, num_sequences=1):
    """Generate text continuation given a prompt."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_sequences,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in output]

def compute_perplexity(text):
    """Compute perplexity of a given text (lower = better)."""
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings['input_ids']
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss.item()
    return float(np.exp(loss))

# ─────────────────────────────────────────────────────────────
# 3. Generate and evaluate
# ─────────────────────────────────────────────────────────────
print("\nGenerating text continuations...")
records = []
for i, prompt in enumerate(prompts):
    generated = generate_text(prompt, max_new_tokens=40)[0]
    continuation = generated[len(prompt):]
    ppl = compute_perplexity(generated)
    records.append({
        'Prompt': prompt,
        'Generated_Continuation': continuation.strip(),
        'Full_Text': generated,
        'Perplexity': ppl
    })
    print(f"\n[{i+1}] Prompt   : {prompt}")
    print(f"     Continues: {continuation.strip()[:80]}...")
    print(f"     Perplexity: {ppl:.2f}")

pred_df = pd.DataFrame(records)
pred_df.to_csv('predictions.csv', index=False)

# ─────────────────────────────────────────────────────────────
# 4. Evaluation (Perplexity)
# ─────────────────────────────────────────────────────────────
avg_ppl = pred_df['Perplexity'].mean()
print(f"\n── GPT-2 Metrics ──")
print(f"  Model          : gpt2 (117M parameters)")
print(f"  Avg Perplexity : {avg_ppl:.2f} (lower = more fluent text)")
print(f"  Prompts tested : {len(prompts)}")
print("  Note: Accuracy/Precision/F1/Confusion Matrix not applicable (generative).")

print("\n── Sample Predictions (Generated Continuations) ──")
print(f"{'#':>3} | {'Prompt[:40]':>40} | {'Perplexity':>11}")
print("-" * 60)
for i, row in pred_df.iterrows():
    print(f"{i+1:>3} | {row['Prompt'][:40]:>40} | {row['Perplexity']:>11.2f}")

# ─────────────────────────────────────────────────────────────
# 5. Plots
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Perplexity bars
axes[0].bar(range(1, len(prompts)+1), pred_df['Perplexity'], color='steelblue', alpha=0.8)
axes[0].axhline(avg_ppl, color='red', linestyle='--', label=f'Mean={avg_ppl:.1f}')
axes[0].set_xlabel('Prompt #')
axes[0].set_ylabel('Perplexity (lower = better)')
axes[0].set_title('GPT-2: Perplexity per Generated Text')
axes[0].legend()

# Token probabilities for first prompt
prompt = prompts[0]
input_ids = tokenizer.encode(prompt, return_tensors='pt')
with torch.no_grad():
    outputs = model(input_ids)
logits = outputs.logits[0, -1, :]
probs = torch.softmax(logits, dim=-1).numpy()
top_k = 15
top_indices = np.argsort(probs)[-top_k:][::-1]
top_probs = probs[top_indices]
top_tokens = [tokenizer.decode([idx]).strip() for idx in top_indices]

axes[1].barh(range(top_k), top_probs[::-1], color='coral', alpha=0.8)
axes[1].set_yticks(range(top_k))
axes[1].set_yticklabels(top_tokens[::-1], fontsize=10)
axes[1].set_xlabel('Probability')
axes[1].set_title(f'GPT-2: Top-{top_k} Next Token Probs\n(after "{prompt[:30]}...")')

plt.tight_layout()
plt.savefig('plots/gpt_analysis.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nPlots saved to plots/\nPredictions saved to predictions.csv\nDone!")
