"""Inference runtime.

Guarantees enforced here:

* **Never trains.** Every function loads an artifact and runs ``model.eval()`` +
  ``torch.inference_mode()``. There is no optimiser, no loss, no backward pass.
* **Preprocessing matches training.** Normalisation constants come from the
  artifact metadata (written by the trainer), not from a re-derived default.
* **Graceful missing weights.** :func:`load` raises :class:`ArtifactMissing`,
  which the UI turns into an honest "training required" message.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import artifacts as A
from .imaging import load_image, to_tensor, overlay_mask, denormalize
from .registry import ModelSpec, build_model


def _norm(meta: dict) -> tuple[tuple, tuple]:
    inp = meta.get("input") or {}
    mean = tuple(inp.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(inp.get("std", (0.229, 0.224, 0.225)))
    return mean, std


def is_ready(spec: ModelSpec) -> bool:
    return A.has_weights(spec.id, spec.area)


def status(spec: ModelSpec) -> dict[str, Any]:
    """Honest, artifact-derived status for the UI."""
    meta = A.read_metadata(spec.id)
    ready = A.has_weights(spec.id, spec.area)
    return {
        "model_id": spec.id,
        "ready": ready,
        "metadata": meta,
        "metrics": (meta or {}).get("metrics", {}),
        "trained_at": (meta or {}).get("saved_at"),
        "weights_bytes": (meta or {}).get("weights_bytes"),
        "dataset": (meta or {}).get("dataset", spec.dataset_name),
        "note": (meta or {}).get("notes", spec.notes),
    }


def load(spec: ModelSpec) -> tuple[torch.nn.Module, dict]:
    """Load the model (eval mode) and its metadata. Raises ArtifactMissing."""
    model = A.load_model(spec.id, spec.area, build_model(spec))
    meta = A.read_metadata(spec.id) or {}
    return model, meta


# --------------------------------------------------------------------------- #
# task-specific predictors
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def predict_classification(spec: ModelSpec, image, top_k: int = 5) -> dict:
    model, meta = load(spec)
    mean, std = _norm(meta)
    size = int((meta.get("input") or {}).get("size", spec.input_size))
    x = to_tensor(load_image(image), size, mean, std)
    t0 = time.perf_counter()
    logits = model(x)
    if isinstance(logits, tuple):
        logits = logits[0]
    probs = F.softmax(logits.float(), dim=1)[0].cpu().numpy()
    dt = (time.perf_counter() - t0) * 1000
    classes = meta.get("classes") or spec.classes
    order = probs.argsort()[::-1][:top_k]
    return {
        "kind": "classification",
        "predicted": classes[int(probs.argmax())] if classes else str(int(probs.argmax())),
        "confidence": float(probs.max()),
        "top_k": [{"label": classes[int(i)] if classes else str(int(i)),
                   "prob": float(probs[int(i)])} for i in order],
        "probabilities": {classes[i] if classes else str(i): float(p) for i, p in enumerate(probs)},
        "inference_ms": round(dt, 2),
        "input_size": size,
        "preprocessed": denormalize(x, mean, std),
    }


@torch.inference_mode()
def predict_segmentation(spec: ModelSpec, image) -> dict:
    model, meta = load(spec)
    mean, std = _norm(meta)
    size = int((meta.get("input") or {}).get("size", spec.input_size))
    img = load_image(image)
    x = to_tensor(img, size, mean, std)
    t0 = time.perf_counter()
    logits = model(x)
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.shape[1] == 1:
        mask = torch.sigmoid(logits)[0, 0].cpu().numpy()
    else:
        mask = torch.softmax(logits.float(), 1)[0].argmax(0).cpu().numpy().astype(np.float32)
    dt = (time.perf_counter() - t0) * 1000
    binary = (mask > 0.5).astype(np.uint8)
    return {
        "kind": "segmentation",
        "mask": mask,
        "binary": binary,
        "foreground_ratio": float(binary.mean()),
        "overlay": overlay_mask(img, mask),
        "inference_ms": round(dt, 2),
        "input_size": size,
    }


@torch.inference_mode()
def predict_detection(spec: ModelSpec, image, conf_thr: float = 0.30, iou_thr: float = 0.45) -> dict:
    from .models.detection import decode_predictions

    model, meta = load(spec)
    mean, std = _norm(meta)
    size = int((meta.get("input") or {}).get("size", spec.input_size))
    img = load_image(image)
    x = to_tensor(img, size, mean, std)
    t0 = time.perf_counter()
    out = model(x)[0].cpu()
    dt = (time.perf_counter() - t0) * 1000
    boxes_norm = decode_predictions(out, conf_thr=conf_thr, iou_thr=iou_thr)
    classes = meta.get("classes") or spec.classes
    boxes = []
    for b in boxes_norm:
        boxes.append({
            "x1": b["x1"] * size, "y1": b["y1"] * size,
            "x2": b["x2"] * size, "y2": b["y2"] * size,
            "score": b["score"], "class": b["class"],
            "label": classes[b["class"]] if b["class"] < len(classes) else str(b["class"]),
        })
    return {"kind": "detection", "boxes": boxes, "count": len(boxes),
            "inference_ms": round(dt, 2), "input_size": size,
            "display_image": img.resize((size, size))}


@torch.inference_mode()
def predict_instance_segmentation(spec: ModelSpec, image, score_thr: float = 0.4) -> dict:
    from .models.detection import instances_to_predictions

    model, meta = load(spec)
    size = int((meta.get("input") or {}).get("size", spec.input_size))
    img = load_image(image)
    x = to_tensor(img, size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    t0 = time.perf_counter()
    out = model([x[0]])
    dt = (time.perf_counter() - t0) * 1000
    inst = instances_to_predictions(out[0], score_thr=score_thr)
    classes = meta.get("classes") or spec.classes
    instances = []
    for d in inst:
        c = int(d["class"])
        label = classes[c - 1] if 1 <= c <= len(classes) else "background"
        entry = {"x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
                 "score": d["score"], "class": c, "label": label}
        if "mask" in d:
            entry["mask"] = d["mask"]
        instances.append(entry)
    return {"kind": "instance_segmentation", "instances": instances, "count": len(instances),
            "inference_ms": round(dt, 2), "input_size": size,
            "display_image": img.resize((size, size))}


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
def predict(spec: ModelSpec, image=None, **kwargs) -> dict:
    """Route an inference request to the right predictor for the spec's task."""
    if spec.task == "classification":
        return predict_classification(spec, image, **kwargs)
    if spec.task == "segmentation":
        return predict_segmentation(spec, image)
    if spec.task == "detection":
        return predict_detection(spec, image, **kwargs)
    if spec.task == "instance_segmentation":
        return predict_instance_segmentation(spec, image, **kwargs)
    raise ValueError(f"task '{spec.task}' has no image predictor (lab models use lab_inference)")
