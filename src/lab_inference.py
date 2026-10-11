"""Inference for the original 20-model laboratory.

Each lab model gets an input contract appropriate to its architecture — no shared
"generic input" hack. All functions load an artifact and run ``model.eval()`` +
``torch.inference_mode()``; none of them train.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import artifacts as A
from .imaging import load_image
from .registry import ModelSpec, build_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def is_ready(spec: ModelSpec) -> bool:
    return A.has_weights(spec.id, spec.area)


def load(spec: ModelSpec):
    model = A.load_model(spec.id, spec.area, build_model(spec))
    meta = A.read_metadata(spec.id) or {}
    return model, meta


def _softmax(logits: torch.Tensor) -> np.ndarray:
    return F.softmax(logits.float(), dim=1)[0].cpu().numpy()


def _mnist_tensor(image, size: int = 28) -> torch.Tensor:
    img = load_image(image, mode="L").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, None, :, :]


def _seq_tensor(values) -> torch.Tensor:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size < 128 * 9:
        raise ValueError(f"Need {128*9} numeric values (128 steps x 9 signals); received {arr.size}.")
    return torch.from_numpy(arr[: 128 * 9].reshape(1, 128, 9))


# --------------------------------------------------------------------------- #
# per-model predictors
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def predict_tabular(spec: ModelSpec, features: list[float]) -> dict:
    model, meta = load(spec)
    mean = np.asarray(meta["scaler"]["mean"], dtype=np.float32)
    scale = np.asarray(meta["scaler"]["scale"], dtype=np.float32)
    expected = int((meta.get("input") or {}).get("features", mean.size))
    if len(features) != expected:
        raise ValueError(f"Expected {expected} features, received {len(features)}.")
    x = (np.asarray(features, dtype=np.float32) - mean) / scale
    x = torch.from_numpy(x)[None, :]
    logits = model(x)
    probs = _softmax(logits)
    classes = meta.get("classes") or spec.classes
    return {"kind": "classification", "predicted": classes[int(probs.argmax())],
            "confidence": float(probs.max()),
            "probabilities": {c: float(p) for c, p in zip(classes, probs)}}


@torch.inference_mode()
def predict_image_class(spec: ModelSpec, image) -> dict:
    model, meta = load(spec)
    size = int((meta.get("input") or {}).get("size", 28))
    x = _mnist_tensor(image, size)
    probs = _softmax(model(x))
    classes = meta.get("classes") or spec.classes
    order = probs.argsort()[::-1][:5]
    return {"kind": "classification", "predicted": classes[int(probs.argmax())],
            "confidence": float(probs.max()),
            "top_k": [{"label": classes[int(i)], "prob": float(probs[int(i)])} for i in order],
            "probabilities": {c: float(p) for c, p in zip(classes, probs)}}


@torch.inference_mode()
def predict_sequence(spec: ModelSpec, values: list[float]) -> dict:
    model, meta = load(spec)
    x = _seq_tensor(values)
    probs = _softmax(model(x))
    classes = meta.get("classes") or spec.classes
    return {"kind": "classification", "predicted": classes[int(probs.argmax())],
            "confidence": float(probs.max()),
            "probabilities": {c: float(p) for c, p in zip(classes, probs)}}


@torch.inference_mode()
def reconstruct(spec: ModelSpec, image) -> dict:
    model, meta = load(spec)
    x = _mnist_tensor(image, 28)
    out = model(x)
    if isinstance(out, tuple):
        recon, mu, logvar = out
    else:
        recon, mu = out, None
    mse = float(F.mse_loss(recon, x))
    return {"kind": "reconstruction", "reconstruction": recon[0, 0].cpu().numpy(),
            "input": x[0, 0].cpu().numpy(), "mse": round(mse, 5),
            "latent": (mu[0].cpu().numpy().tolist() if mu is not None else None)}


@torch.inference_mode()
def gan_score(spec: ModelSpec, image) -> dict:
    model, meta = load(spec)
    x = _mnist_tensor(image, 28)
    score = float(torch.sigmoid(model(x))[0, 0])
    return {"kind": "binary", "predicted": "real" if score >= 0.5 else "generated",
            "confidence": max(score, 1 - score),
            "probabilities": {"real": score, "generated": 1 - score}}


@torch.inference_mode()
def generate_text(spec: ModelSpec, text: str, max_new: int = 24) -> dict:
    model, meta = load(spec)
    from .datasets.lab_datasets import encode_chars, _char_vocab

    vocab = _char_vocab()
    inv = {v: k for k, v in vocab.items()}
    length = int((meta.get("input") or {}).get("length", 96))
    ids = encode_chars(text, length)
    x = torch.tensor(ids, dtype=torch.long)[None, :]
    with torch.inference_mode():
        logits = model(x)[0]
        probs = torch.softmax(logits[-1], dim=-1)
        top = torch.topk(probs, 8)
    return {"kind": "generation", "next_tokens": [
        {"id": int(i), "char": inv.get(int(i), "?"), "prob": float(p)}
        for p, i in zip(top.values, top.indices)],
        "vocab_size": int(probs.shape[0])}


@torch.inference_mode()
def text_classify(spec: ModelSpec, text: str) -> dict:
    model, meta = load(spec)
    from .datasets.lab_datasets import encode_chars

    length = int((meta.get("input") or {}).get("length", 64))
    ids = encode_chars(text, length)
    x = torch.tensor(ids, dtype=torch.long)[None, :]
    probs = _softmax(model(x))
    classes = meta.get("classes") or spec.classes
    order = probs.argsort()[::-1]
    return {"kind": "classification", "predicted": classes[int(probs.argmax())],
            "confidence": float(probs.max()),
            "probabilities": {classes[int(i)]: float(probs[int(i)]) for i in order}}


@torch.inference_mode()
def siamese_pair(spec: ModelSpec, image_a, image_b) -> dict:
    model, meta = load(spec)
    xa = _mnist_tensor(image_a, 28)
    xb = _mnist_tensor(image_b, 28)
    za, zb = model(xa, xb)
    dist = float((za - zb).pow(2).sum(1).sqrt())
    thr = float(meta.get("threshold", 0.5))
    return {"kind": "pair", "distance": round(dist, 4), "threshold": thr,
            "predicted": "same digit" if dist < thr else "different digits",
            "confidence": round(min(1.0, abs(dist - thr) / max(thr, 1e-6)), 3)}


@torch.inference_mode()
def segmentation_28(spec: ModelSpec, image) -> dict:
    model, meta = load(spec)
    x = _mnist_tensor(image, 28)
    mask = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return {"kind": "segmentation", "mask": mask, "binary": (mask > 0.5).astype(np.uint8),
            "input": x[0, 0].cpu().numpy()}


@torch.inference_mode()
def detection_28(spec: ModelSpec, image, conf_thr: float = 0.3) -> dict:
    from .models.detection import decode_predictions

    model, meta = load(spec)
    x = _mnist_tensor(image, 28)
    out = model(x)[0].cpu()
    boxes = decode_predictions(out, conf_thr=conf_thr)
    classes = meta.get("classes") or spec.classes
    for b in boxes:
        b["x1"] *= 28; b["y1"] *= 28; b["x2"] *= 28; b["y2"] *= 28
        b["label"] = classes[b["class"]] if b["class"] < len(classes) else str(b["class"])
    return {"kind": "detection", "boxes": boxes, "count": len(boxes),
            "input": x[0, 0].cpu().numpy(), "size": 28}


@torch.inference_mode()
def denoise_tabular(spec: ModelSpec, features: list[float], noise_level: float = 0.5) -> dict:
    model, meta = load(spec)
    mean = np.asarray(meta["scaler"]["mean"], dtype=np.float32)
    scale = np.asarray(meta["scaler"]["scale"], dtype=np.float32)
    expected = int((meta.get("input") or {}).get("features", mean.size))
    if len(features) != expected:
        raise ValueError(f"Expected {expected} features, received {len(features)}.")
    x = (np.asarray(features, dtype=np.float32) - mean) / scale
    xt = torch.from_numpy(x)[None, :]
    t = torch.full((1, 1), float(noise_level))
    clean = xt - model(xt, t)
    return {"kind": "denoise", "clean": clean[0].cpu().numpy().tolist(),
            "mse_to_input": round(float(F.mse_loss(clean, xt)), 6)}


@torch.inference_mode()
def gnn_predict(spec: ModelSpec, features: list[float]) -> dict:
    model, meta = load(spec)
    f = int((meta.get("input") or {}).get("features", 1433))
    if len(features) != f:
        raise ValueError(f"Expected {f} node features, received {len(features)}.")
    x = torch.tensor([features], dtype=torch.float32)
    a = torch.eye(1)
    probs = _softmax(model(x, a))
    classes = meta.get("classes") or spec.classes
    return {"kind": "classification", "predicted": classes[int(probs.argmax())],
            "confidence": float(probs.max()),
            "probabilities": {c: float(p) for c, p in zip(classes, probs)}}


# --------------------------------------------------------------------------- #
# dispatch table (id -> callable)
# --------------------------------------------------------------------------- #
DISPATCH = {
    "lab01": predict_tabular,
    "lab02": predict_image_class, "lab12": predict_image_class, "lab13": predict_image_class,
    "lab14": predict_image_class,
    "lab03": predict_sequence, "lab04": predict_sequence, "lab05": predict_sequence,
    "lab06": reconstruct, "lab07": reconstruct,
    "lab08": gan_score,
    "lab09": text_classify, "lab10": text_classify,
    "lab11": generate_text,
    "lab15": segmentation_28,
    "lab16": detection_28,
    "lab17": siamese_pair,
    "lab18": generate_text,
    "lab19": denoise_tabular,
    "lab20": gnn_predict,
}

# input-kind labels for the UI
INPUT_KIND = {
    "lab01": "tabular", "lab19": "tabular_denoise", "lab20": "graph",
    "lab02": "image", "lab12": "image", "lab13": "image", "lab14": "image",
    "lab06": "image", "lab07": "image", "lab08": "image", "lab15": "image",
    "lab16": "image",
    "lab03": "sequence", "lab04": "sequence", "lab05": "sequence",
    "lab09": "text", "lab10": "text", "lab11": "text_prompt", "lab18": "text_prompt",
    "lab17": "image_pair",
}
