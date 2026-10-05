"""Shared, dependency-light inference runtime for every model demo.

Every generated Streamlit app imports this module. It loads the REAL trained
artifact (``<folder>/model/model.pt`` + ``preprocess.pkl`` + ``meta.json``) and
performs exactly the preprocessing the training pipeline used.

Keeping this in one file means each app folder holds only a tiny ``app.py``;
the heavy helper is committed once at ``app_shared.py`` and pulled into the
repository root, which enables Streamlit Community Cloud's multi-app mode.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(2)

REPO = Path(__file__).resolve().parent
_CACHE: dict[int, dict] = {}

# --------------------------------------------------------------------------- #
# Static model catalogue: the single source of truth for names, frameworks,
# tasks and the matching project folder. (id -> info)
# --------------------------------------------------------------------------- #
MODELS: dict[int, dict] = {
    1: {"folder": "01_ann", "name": "ANN / MLP", "framework": "PyTorch",
        "task": "Binary classification (malignant vs benign)",
        "dataset": "Breast Cancer Wisconsin (Diagnostic) — UCI",
        "input": "30 numeric cell-nucleus features"},
    2: {"folder": "02_cnn", "name": "CNN", "framework": "PyTorch",
        "task": "10-class handwritten-digit classification",
        "dataset": "MNIST (LeCun)", "input": "28x28 grayscale image"},
    3: {"folder": "03_rnn", "name": "RNN", "framework": "PyTorch",
        "task": "6-class activity recognition",
        "dataset": "UCI Smartphone HAR", "input": "93x6 sensor window"},
    4: {"folder": "04_lstm", "name": "LSTM", "framework": "PyTorch",
        "task": "6-class activity recognition",
        "dataset": "UCI Smartphone HAR", "input": "93x6 sensor window"},
    5: {"folder": "05_gru", "name": "GRU", "framework": "PyTorch",
        "task": "6-class activity recognition",
        "dataset": "UCI Smartphone HAR", "input": "93x6 sensor window"},
    6: {"folder": "06_autoencoder", "name": "Autoencoder", "framework": "PyTorch",
        "task": "Image reconstruction / anomaly score",
        "dataset": "MNIST", "input": "28x28 grayscale image"},
    7: {"folder": "07_vae", "name": "VAE", "framework": "PyTorch",
        "task": "Variational reconstruction (latent code)",
        "dataset": "MNIST", "input": "28x28 grayscale image"},
    8: {"folder": "08_gan", "name": "GAN", "framework": "PyTorch",
        "task": "Real vs generated digit (discriminator)",
        "dataset": "MNIST", "input": "28x28 grayscale image"},
    9: {"folder": "09_transformer", "name": "Transformer", "framework": "PyTorch",
        "task": "4-class topic classification",
        "dataset": "20 Newsgroups", "input": "free text"},
    10: {"folder": "10_bert", "name": "BERT-style", "framework": "PyTorch",
         "task": "4-class topic classification",
         "dataset": "20 Newsgroups", "input": "free text"},
    11: {"folder": "11_gpt", "name": "GPT-style", "framework": "PyTorch",
         "task": "Autoregressive text generation",
         "dataset": "20 Newsgroups corpus", "input": "free-text prompt"},
    12: {"folder": "12_resnet", "name": "ResNet", "framework": "PyTorch",
         "task": "10-class digit classification",
         "dataset": "MNIST", "input": "28x28 grayscale image"},
    13: {"folder": "13_densenet", "name": "DenseNet", "framework": "PyTorch",
         "task": "10-class digit classification",
         "dataset": "MNIST", "input": "28x28 grayscale image"},
    14: {"folder": "14_vit", "name": "ViT", "framework": "PyTorch",
         "task": "10-class digit classification",
         "dataset": "MNIST", "input": "28x28 grayscale image"},
    15: {"folder": "15_unet", "name": "U-Net", "framework": "PyTorch",
         "task": "Binary image segmentation",
         "dataset": "Synthetic circle segmentation", "input": "64x64 grayscale image"},
    16: {"folder": "16_yolo", "name": "YOLO-style", "framework": "PyTorch",
         "task": "Object detection + classification",
         "dataset": "Synthetic multi-shape", "input": "64x64 grayscale image"},
    17: {"folder": "17_siamese", "name": "Siamese Network", "framework": "PyTorch",
         "task": "Same/different digit verification",
         "dataset": "MNIST", "input": "two 28x28 grayscale images"},
    18: {"folder": "18_seq2seq", "name": "Seq2Seq", "framework": "PyTorch",
         "task": "Text reconstruction (token predictions)",
         "dataset": "20 Newsgroups corpus", "input": "free text"},
    19: {"folder": "19_diffusion", "name": "Diffusion MLP", "framework": "PyTorch",
         "task": "Tabular denoising (clean a feature row)",
         "dataset": "Breast Cancer Wisconsin — UCI", "input": "30 numeric features"},
    20: {"folder": "20_gnn", "name": "GNN", "framework": "PyTorch",
         "task": "Cora 7-class node classification",
         "dataset": "Cora citation network", "input": "1433-dim node features"},
}

CATEGORY = {1: "Tabular", 2: "Vision", 3: "Sequence", 4: "Sequence", 5: "Sequence",
            6: "Generative", 7: "Generative", 8: "Generative", 9: "NLP", 10: "NLP",
            11: "NLP", 12: "Vision", 13: "Vision", 14: "Vision", 15: "Vision",
            16: "Vision", 17: "Vision", 18: "NLP", 19: "Tabular", 20: "Graph"}


# --------------------------------------------------------------------------- #
# artifact loading
# --------------------------------------------------------------------------- #
def artifact_dir(model_id: int) -> Path:
    return REPO / MODELS[model_id]["folder"] / "model"


def artifacts_available(model_id: int) -> bool:
    return (artifact_dir(model_id) / "model.pt").exists()


def load(model_id: int) -> dict:
    """Load model + preprocessing once per process (Streamlit-friendly)."""
    if model_id in _CACHE:
        return _CACHE[model_id]
    d = artifact_dir(model_id)
    if not (d / "model.pt").exists():
        raise FileNotFoundError(
            f"Model artifacts are missing for {MODELS[model_id]['name']} "
            f"(expected {d}). Run `python tools/build_artifacts.py --only {model_id}`."
        )
    from models_arch import SPEC_BY_ID
    _, _, cls, _, _ = SPEC_BY_ID[model_id]
    model = cls()
    model.load_state_dict(torch.load(d / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    with open(d / "preprocess.pkl", "rb") as f:
        pre = pickle.load(f)
    with open(d / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    _CACHE[model_id] = {"model": model, "pre": pre, "meta": meta}
    return _CACHE[model_id]


# --------------------------------------------------------------------------- #
# preprocessing helpers (identical to the training pipeline)
# --------------------------------------------------------------------------- #
def scale_tabular(pre: dict, values: list[float]) -> torch.Tensor:
    mean = np.asarray(pre["scaler_mean"], dtype=np.float32)
    scale = np.asarray(pre["scaler_scale"], dtype=np.float32)
    x = (np.asarray(values, dtype=np.float32) - mean) / scale
    return torch.from_numpy(x.astype(np.float32))[None, :]


def _as_pil(obj):
    from PIL import Image
    if isinstance(obj, Image.Image):
        return obj
    if isinstance(obj, np.ndarray):
        return Image.fromarray(obj.astype("uint8"))
    return Image.open(obj)


def image_28(uploaded) -> tuple[torch.Tensor, np.ndarray]:
    img = _as_pil(uploaded).convert("L").resize((28, 28))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, None, :, :], arr


def image_64(uploaded) -> tuple[torch.Tensor, np.ndarray]:
    img = _as_pil(uploaded).convert("L").resize((64, 64))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, None, :, :], arr


def image_8(uploaded) -> tuple[torch.Tensor, np.ndarray]:
    img = _as_pil(uploaded).convert("L").resize((28, 28))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)[None, None]          # (1,1,28,28)
    return F.adaptive_avg_pool2d(t, (8, 8)), arr     # (1,1,8,8)


def tokenize_text(text: str, length: int = 32, vocab: int = 1024) -> torch.Tensor:
    words = text.lower().split()[:length]
    ids = [1 + int(hashlib.md5(w.encode("utf-8", "ignore")).hexdigest(), 16) % (vocab - 2)
           for w in words]
    ids = ids + [0] * (length - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def sequence_from_csv(data: bytes) -> torch.Tensor:
    """Read a CSV/whitespace file of numbers -> (1, 93, 6)."""
    import io
    import pandas as pd
    df = pd.read_csv(io.BytesIO(data), header=None)
    vals = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32).reshape(-1)
    if vals.size < 558:
        raise ValueError(f"Need at least 558 numeric values (93x6); received {vals.size}.")
    return torch.from_numpy(vals[:558].reshape(1, 93, 6))


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max()
    e = np.exp(logits)
    return (e / e.sum()).astype(np.float32)


# --------------------------------------------------------------------------- #
# prediction
# --------------------------------------------------------------------------- #
def predict(model_id: int, payload: dict) -> dict:
    """Run REAL inference. Returns dict with 'kind' plus kind-specific fields."""
    bundle = load(model_id)
    model, pre, meta = bundle["model"], bundle["pre"], bundle["meta"]
    kind = pre["kind"]

    with torch.no_grad():
        if kind == "tabular":
            x = scale_tabular(pre, payload["features"])
            if model_id == 19:                                  # diffusion denoiser
                t = torch.full((1, 1), float(payload.get("noise_level", 0.5)))
                noise_free = x - model(x, t)
                return {"kind": "denoise", "clean": [round(float(v), 4) for v in noise_free[0]],
                        "mse_to_input": round(float(F.mse_loss(noise_free, x)), 6)}
            logits = model(x)[0].numpy()
            probs = softmax(logits)
            idx = int(probs.argmax())
            names = pre["class_names"]
            return {"kind": "classification", "label": str(names[idx]),
                    "confidence": float(probs[idx]),
                    "probabilities": {str(n): float(p) for n, p in zip(names, probs)}}

        if model_id == 15:                                       # U-Net segmentation (64x64)
            x, _ = image_64(payload["image"])
            mask = (torch.sigmoid(model(x))[0, 0] > 0.5).numpy().astype(np.uint8)
            return {"kind": "segmentation", "mask": mask.tolist(),
                    "foreground_ratio": round(float(mask.mean()), 4)}

        if model_id == 16:                                       # YOLO detection (64x64)
            x, _ = image_64(payload["image"])
            out = model(x)[0]
            prob = torch.sigmoid(out[..., 0])
            conf = float(payload.get("conf", 0.5))
            boxes = []
            for r in range(8):
                for c in range(8):
                    if float(prob[r, c]) < conf:
                        continue
                    xd, yd, w, h = [float(v) for v in torch.sigmoid(out[r, c, 1:5])]
                    cls = int(out[r, c, 5:].argmax())
                    cx = (c + xd) / 8 * 64
                    cy = (r + yd) / 8 * 64
                    bw, bh = w * 64, h * 64
                    boxes.append({"x1": round(cx - bw / 2, 1), "y1": round(cy - bh / 2, 1),
                                  "x2": round(cx + bw / 2, 1), "y2": round(cy + bh / 2, 1),
                                  "score": round(float(prob[r, c]), 3), "class": cls})
            boxes.sort(key=lambda b: -b["score"])
            return {"kind": "detection", "boxes": boxes, "count": len(boxes),
                    "class_names": pre["class_names"]}

        if model_id == 17:                                       # Siamese pair (8x8 twins)
            xa, _ = image_8(payload["image_a"])
            xb, _ = image_8(payload["image_b"])
            za, zb = model(xa, xb)
            dist = float((za - zb).pow(2).sum(1).sqrt())
            thr = float(pre.get("threshold", 0.5))
            return {"kind": "pair", "distance": round(dist, 4), "threshold": thr,
                    "label": "same digit" if dist < thr else "different digits",
                    "confidence": round(min(1.0, abs(dist - thr) / max(thr, 1e-6)), 3)}

        if kind == "image" and pre.get("ossize") == 28:          # autoencoder / VAE / GAN
            x, _ = image_8(payload["image"])
            if model_id == 6:
                rec = model(x)
                mse = float(F.mse_loss(rec, x))
                return {"kind": "reconstruction", "image": rec[0, 0].numpy().tolist(),
                        "mse": round(mse, 5), "note": "low MSE = image resembles training digits"}
            if model_id == 7:
                rec, mu, _lv = model(x)
                return {"kind": "reconstruction", "image": rec[0, 0].numpy().tolist(),
                        "latent": mu[0].numpy().tolist(), "note": "VAE reconstruction"}
            if model_id == 8:
                score = float(torch.sigmoid(model(x))[0, 0])
                label = "real-like digit" if score >= 0.5 else "generated / unlike digits"
                return {"kind": "classification", "label": label, "confidence": max(score, 1 - score),
                        "probabilities": {"real-like": score, "generated": 1 - score}}

        if kind == "image":                                       # 28x28 classifier
            x, _ = image_28(payload["image"])
            probs = softmax(model(x)[0].numpy())
            idx = int(probs.argmax())
            names = pre["class_names"]
            return {"kind": "classification", "label": str(names[idx]),
                    "confidence": float(probs[idx]),
                    "probabilities": {str(n): float(p) for n, p in zip(names, probs)}}

        if kind == "sequence":
            x = sequence_from_csv(payload["csv"]) if payload.get("csv") else payload["tensor"]
            probs = softmax(model(x)[0].numpy())
            idx = int(probs.argmax())
            names = pre["class_names"]
            return {"kind": "classification", "label": str(names[idx]),
                    "confidence": float(probs[idx]),
                    "probabilities": {str(n): float(p) for n, p in zip(names, probs)}}

        if kind == "text":
            ids = tokenize_text(payload["text"], pre["length"], pre["vocab_size"])
            probs = softmax(model(ids[None, :])[0].numpy())
            idx = int(probs.argmax())
            names = pre["class_names"]
            return {"kind": "classification", "label": str(names[idx]),
                    "confidence": float(probs[idx]),
                    "probabilities": {str(n): float(p) for n, p in zip(names, probs)}}

        if kind in ("text_gen", "seq2seq"):
            ids = tokenize_text(payload["text"], pre["length"], pre["vocab_size"])
            out = model(ids[None, :])[0]                         # (32, 1024)
            probs = torch.softmax(out, dim=-1)
            nxt = int(probs[-1].argmax())
            top = torch.topk(probs[-1], 8)
            i2w = pre.get("id_to_word", {})
            return {"kind": "generation",
                    "next_token_id": nxt,
                    "next_token_word": i2w.get(str(nxt), f"<hash:{nxt}>"),
                    "top_tokens": [{"id": int(i), "word": i2w.get(str(int(i)), f"<hash:{int(i)}>"),
                                    "prob": round(float(p), 4)}
                                   for p, i in zip(top.values, top.indices)]}

        if kind == "segmentation" or (kind == "image" and pre.get("size") == 64 and model_id == 15):
            x, _ = image_64(payload["image"])
            mask = (torch.sigmoid(model(x))[0, 0] > 0.5).numpy().astype(np.uint8)
            return {"kind": "segmentation", "mask": mask.tolist(),
                    "foreground_ratio": round(float(mask.mean()), 4)}

        if kind == "detection":
            x, _ = image_64(payload["image"])
            out = model(x)[0]
            prob = torch.sigmoid(out[..., 0])
            conf = float(payload.get("conf", 0.5))
            boxes = []
            for r in range(8):
                for c in range(8):
                    if float(prob[r, c]) < conf:
                        continue
                    xd, yd, w, h = [float(v) for v in torch.sigmoid(out[r, c, 1:5])]
                    cls = int(out[r, c, 5:].argmax())
                    cx = (c + xd) / 8 * 64
                    cy = (r + yd) / 8 * 64
                    bw, bh = w * 64, h * 64
                    boxes.append({"x1": round(cx - bw / 2, 1), "y1": round(cy - bh / 2, 1),
                                  "x2": round(cx + bw / 2, 1), "y2": round(cy + bh / 2, 1),
                                  "score": round(float(prob[r, c]), 3), "class": cls})
            boxes.sort(key=lambda b: -b["score"])
            return {"kind": "detection", "boxes": boxes, "count": len(boxes),
                    "class_names": pre["class_names"]}

        if kind == "pair":
            xa, _ = image_28(payload["image_a"])
            xb, _ = image_28(payload["image_b"])
            za, zb = model(xa, xb)
            dist = float((za - zb).pow(2).sum(1).sqrt())
            thr = float(pre.get("threshold", 0.5))
            return {"kind": "pair", "distance": round(dist, 4), "threshold": thr,
                    "label": "same digit" if dist < thr else "different digits",
                    "confidence": round(min(1.0, abs(dist - thr) / max(thr, 1e-6)), 3)}

        if kind == "graph":
            x = torch.tensor([payload["features"]], dtype=torch.float32)
            a = torch.eye(1)
            probs = softmax(model(x, a)[0].numpy())
            idx = int(probs.argmax())
            names = pre["class_names"]
            return {"kind": "classification", "label": str(names[idx]),
                    "confidence": float(probs[idx]),
                    "probabilities": {str(n): float(p) for n, p in zip(names, probs)}}

    raise ValueError(f"Unsupported preprocessing kind: {kind}")


# --------------------------------------------------------------------------- #
# UI copy per model (kept here so every surface stays consistent)
# --------------------------------------------------------------------------- #
VERB: dict[int, str] = {
    1: "Classify tumour", 2: "Recognise digit", 3: "Recognise activity",
    4: "Recognise activity", 5: "Recognise activity", 6: "Reconstruct image",
    7: "Reconstruct image", 8: "Judge authenticity", 9: "Classify topic",
    10: "Classify topic", 11: "Generate continuation", 12: "Recognise digit",
    13: "Recognise digit", 14: "Recognise digit", 15: "Segment foreground",
    16: "Detect objects", 17: "Verify digit pair", 18: "Reconstruct text",
    19: "Denoise feature row", 20: "Classify node",
}
DEMO_LABEL: dict[int, str] = {
    1: "Predict with these values", 2: "Predict digit", 3: "Predict activity",
    4: "Predict activity", 5: "Predict activity", 6: "Reconstruct", 7: "Reconstruct",
    8: "Classify authenticity", 9: "Classify topic", 10: "Classify topic",
    11: "Generate next tokens", 12: "Predict digit", 13: "Predict digit",
    14: "Predict digit", 15: "Segment image", 16: "Detect objects",
    17: "Compare the two digits", 18: "Predict next tokens",
    19: "Denoise this row", 20: "Classify node",
}
SAMPLE_LABEL: dict[int, str] = {
    1: "Sample feature rows", 3: "Sample sensor windows", 9: "Sample articles",
    10: "Sample articles", 11: "Sample prompts", 18: "Sample text",
    20: "Sample node feature rows",
}
