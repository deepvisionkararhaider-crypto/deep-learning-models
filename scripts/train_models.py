#!/usr/bin/env python3
"""Offline training pipeline for the medical & agriculture computer-vision models.

Usage
-----
    python scripts/train_models.py --model all
    python scripts/train_models.py --model brain_densenet
    python scripts/train_models.py --model plant_cnn --epochs 4
    python scripts/train_models.py --list

This is the ONLY place that trains. The Streamlit app never imports torch.optim.
Every run writes ``artifacts/<area>/<id>.pt`` + ``artifacts/metadata/<id>.json``
with the full input contract, class list and measured metrics.

Feasibility guards
------------------
Training a large detector (Mask R-CNN) on CPU in the free tier is documented as
too heavy; those models are skipped unless ``--allow-heavy`` is passed, and the
UI then reports them as "training required" rather than faking results.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import artifacts as A                       # noqa: E402
from src import inference as I                       # noqa: E402
from src.datasets import loaders as L                # noqa: E402
from src.imaging import IMAGENET_MEAN, IMAGENET_STD, make_transform  # noqa: E402
from src.models.detection import (encode_targets, yolo_loss, decode_predictions,  # noqa: E402
                                  box_iou, xywh_to_xyxy)
from src.models.segmentation import bce_dice_loss, dice_coef, iou_score  # noqa: E402
from src.registry import BY_ID, CV_MODELS, ModelSpec, build_model  # noqa: E402

SEED = 42
HEAVY = {"brain_maskrcnn", "plant_maskrcnn"}


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def _collate_classification(items, transform, batch_size, shuffle):
    from src.datasets.loaders import ImageClassificationDataset
    ds = ImageClassificationDataset(items, transform, [])
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def evaluate_classifier(model, loader, n_classes) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        ps.append(out.argmax(1).numpy())
        ys.append(y.numpy())
    y_true = np.concatenate(ys) if ys else np.array([])
    y_pred = np.concatenate(ps) if ps else np.array([])
    if len(y_true) == 0:
        return {}
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist()
    return {"accuracy": round(acc, 4), "precision_macro": round(float(p), 4),
            "recall_macro": round(float(r), 4), "f1_macro": round(float(f1), 4),
            "confusion_matrix": cm, "n_test": int(len(y_true))}


def train_classifier(spec: ModelSpec, epochs: int, batch_size: int, lr: float) -> dict:
    n = len(spec.classes)
    if spec.dataset_key == "brain_mri_cls":
        data = L.load_brain_mri()
    elif spec.dataset_key == "plant_cls":
        data = L.load_plant_cls()
    else:
        raise ValueError(f"no classifier loader for {spec.dataset_key}")

    size = spec.input_size
    train_tf = make_transform(size, IMAGENET_MEAN, IMAGENET_STD, train=True)
    eval_tf = make_transform(size, IMAGENET_MEAN, IMAGENET_STD, train=False)
    train_ld = _collate_classification(data["train"], train_tf, batch_size, True)
    val_ld = _collate_classification(data["val"], eval_tf, batch_size, False)
    test_ld = _collate_classification(data["test"], eval_tf, batch_size, False)

    builder = build_model(spec)
    # transfer-learned backbones get ImageNet weights ONLY during offline training
    if spec.builder == "densenet":
        from src.models.classifiers import build_densenet
        model = build_densenet(n, in_ch=spec.in_ch, pretrained=True)
    elif spec.builder == "vit":
        from src.models.classifiers import build_vit
        model = build_vit(n, in_ch=spec.in_ch, pretrained=False, image_size=size)
    else:
        model = builder()
    for p in model.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    hist = []
    for ep in range(epochs):
        model.train()
        tot, seen = 0.0, 0
        for x, y in train_ld:
            opt.zero_grad()
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            tot += float(loss) * x.size(0); seen += x.size(0)
        sched.step()
        hist.append(round(tot / max(seen, 1), 4))
        print(f"    epoch {ep+1}/{epochs} loss={hist[-1]:.4f}", flush=True)

    metrics = evaluate_classifier(model, test_ld, n)
    metrics["val"] = evaluate_classifier(model, val_ld, n)
    metrics["train_loss_curve"] = hist
    metrics["split_sizes"] = {k: len(v) for k, v in data.items()}
    meta = _base_meta(spec, size, IMAGENET_MEAN, IMAGENET_STD, metrics,
                      extra={"epochs": epochs, "lr": lr, "batch_size": batch_size,
                             "split_seed": SEED})
    return {"model": model, "metrics": metrics, "meta": meta}


# --------------------------------------------------------------------------- #
# segmentation (U-Net)
# --------------------------------------------------------------------------- #
def train_unet(spec: ModelSpec, epochs: int, batch_size: int, lr: float) -> dict:
    size = spec.input_size
    data = L.load_plant_seg()
    from src.datasets.loaders import SegmentationDataset
    tr = SegmentationDataset(data["train"], size, IMAGENET_MEAN, IMAGENET_STD, train=True)
    va = SegmentationDataset(data["val"], size, IMAGENET_MEAN, IMAGENET_STD, train=False)
    te = SegmentationDataset(data["test"], size, IMAGENET_MEAN, IMAGENET_STD, train=False)
    tl = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True)
    vl = torch.utils.data.DataLoader(va, batch_size=batch_size)
    el = torch.utils.data.DataLoader(te, batch_size=batch_size)

    model = build_model(spec)()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    hist = []
    for ep in range(epochs):
        model.train(); tot, seen = 0.0, 0
        for x, m in tl:
            opt.zero_grad()
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = bce_dice_loss(logits, m)
            loss.backward(); opt.step()
            tot += float(loss) * x.size(0); seen += x.size(0)
        hist.append(round(tot / max(seen, 1), 4))
        print(f"    epoch {ep+1}/{epochs} loss={hist[-1]:.4f}", flush=True)

    metrics = _eval_unet(model, el)
    metrics["val"] = _eval_unet(model, vl)
    metrics["train_loss_curve"] = hist
    metrics["split_sizes"] = {"train": len(tr), "val": len(va), "test": len(te)}
    meta = _base_meta(spec, size, IMAGENET_MEAN, IMAGENET_STD, metrics,
                      extra={"epochs": epochs, "lr": lr, "loss": "bce_dice", "mask_threshold": 0.5})
    return {"model": model, "metrics": metrics, "meta": meta}


@torch.no_grad()
def _eval_unet(model, loader) -> dict:
    model.eval(); dices, ious = [], []
    for x, m in loader:
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        dices.append(dice_coef(logits, m))
        ious.append(iou_score(logits, m))
    if not dices:
        return {}
    return {"dice": round(float(np.mean(dices)), 4), "iou": round(float(np.mean(ious)), 4),
            "n_test": len(dices)}


# --------------------------------------------------------------------------- #
# detection (YOLO-style)
# --------------------------------------------------------------------------- #
def train_yolo(spec: ModelSpec, epochs: int, batch_size: int, lr: float) -> dict:
    size = spec.input_size
    grid = spec.build_kwargs.get("grid", 14)
    data = L.load_detection(spec.dataset_key)
    tf = make_transform(size, IMAGENET_MEAN, IMAGENET_STD, train=False)

    def make_batch(rows):
        xs, tgts, gts = [], [], []
        for r in rows:
            from src.imaging import decode_image
            xs.append(tf(decode_image(r["image"])))
            w, h = r["width"], r["height"]
            b = r["boxes"].copy()
            if len(b):
                b[:, [0, 2]] /= w; b[:, [1, 3]] /= h
            gts.append((b.astype(np.float32), r["labels"].astype(np.int64)))
            tgts.append(encode_targets([torch.from_numpy(b)], [torch.from_numpy(r["labels"])],
                                       grid, len(spec.classes))[0])
        return torch.stack(xs), torch.stack(tgts), gts

    model = build_model(spec)()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    hist = []
    for ep in range(epochs):
        model.train(); random.shuffle(data["train"]); tot, seen = 0.0, 0
        for i in range(0, len(data["train"]), batch_size):
            xb, tb, _ = make_batch(data["train"][i:i + batch_size])
            opt.zero_grad()
            out = model(xb)
            loss = yolo_loss(out, tb)
            loss.backward(); opt.step()
            tot += float(loss) * xb.size(0); seen += xb.size(0)
        hist.append(round(tot / max(seen, 1), 4))
        print(f"    epoch {ep+1}/{epochs} loss={hist[-1]:.4f}", flush=True)

    metrics = _eval_detection(model, data["test"], tf, spec, grid)
    metrics["val"] = _eval_detection(model, data["val"], tf, spec, grid)
    metrics["train_loss_curve"] = hist
    metrics["split_sizes"] = {k: len(v) for k, v in data.items()}
    meta = _base_meta(spec, size, IMAGENET_MEAN, IMAGENET_STD, metrics,
                      extra={"epochs": epochs, "lr": lr, "grid": grid,
                             "box_format": "normalised xyxy", "conf_threshold": 0.30})
    return {"model": model, "metrics": metrics, "meta": meta}


@torch.no_grad()
def _eval_detection(model, rows, tf, spec: ModelSpec, grid: int) -> dict:
    from src.imaging import decode_image
    model.eval()
    aps = []
    tp = fp = fn = 0
    for r in rows:
        x = tf(decode_image(r["image"])).unsqueeze(0)
        preds = decode_predictions(model(x)[0].cpu(), conf_thr=0.30)
        w, h = r["width"], r["height"]
        gt = r["boxes"].astype(np.float32).copy()
        if len(gt):
            gt[:, [0, 2]] /= w; gt[:, [1, 3]] /= h
        gt_t = torch.from_numpy(gt)
        pred_t = torch.tensor([[p["x1"], p["y1"], p["x2"], p["y2"]] for p in preds]) if preds else torch.zeros((0, 4))
        if gt_t.numel() == 0 and pred_t.numel() == 0:
            continue
        if gt_t.numel() == 0:
            fp += pred_t.shape[0]; continue
        if pred_t.numel() == 0:
            fn += gt_t.shape[0]; continue
        iou = box_iou(pred_t, gt_t)
        best = iou.max(1).values
        matched = int((best >= 0.5).sum())
        tp += matched
        fp += pred_t.shape[0] - matched
        fn += gt_t.shape[0] - matched
        aps.append(float((best >= 0.5).float().mean()))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"mAP@50": round(float(np.mean(aps)) if aps else 0.0, 4),
            "precision@0.5": round(precision, 4), "recall@0.5": round(recall, 4),
            "n_test": len(rows)}


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #
def _base_meta(spec: ModelSpec, size: int, mean, std, metrics: dict, extra: dict | None = None) -> dict:
    meta = {
        "name": spec.name,
        "area": spec.area,
        "task": spec.task,
        "framework": spec.framework,
        "architecture": spec.builder,
        "dataset": spec.dataset_name,
        "dataset_key": spec.dataset_key,
        "classes": spec.classes,
        "n_classes": len(spec.classes),
        "input": {"size": size, "channels": spec.in_ch, "mean": list(mean), "std": list(std),
                  "desc": spec.input_desc},
        "outputs": spec.outputs,
        "metric_names": spec.metric_names,
        "metrics": metrics,
        "notes": spec.notes,
        "trained": True,
    }
    if extra:
        meta["training"] = extra
    return meta


def save(spec: ModelSpec, result: dict):
    meta = A.save_model(result["model"], spec.id, spec.area, result["meta"])
    print(f"  saved {spec.id}: {meta['weights_bytes']/1e6:.2f}MB | "
          f"metrics={ {k: v for k, v in result['metrics'].items() if isinstance(v, (int, float))} }",
          flush=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", help="model id, 'all', 'medical', 'agriculture', or 'lab'")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--allow-heavy", action="store_true", help="attempt Mask R-CNN (very slow on CPU)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for m in CV_MODELS:
            print(f"{m.id:18s} {m.area:12s} {m.task:22s} {m.dataset_name}")
        return

    if args.model == "all":
        targets = [m for m in CV_MODELS]
    elif args.model == "medical":
        targets = [m for m in CV_MODELS if m.area == "medical"]
    elif args.model == "agriculture":
        targets = [m for m in CV_MODELS if m.area == "agriculture"]
    else:
        targets = [BY_ID[args.model]]

    set_seed()
    summary = []
    for spec in targets:
        print(f"\n=== {spec.id} ({spec.task}) ===", flush=True)
        if spec.id in HEAVY and not args.allow_heavy:
            print("  SKIP: heavy detector (Mask R-CNN) — rerun with --allow-heavy to attempt.", flush=True)
            summary.append((spec.id, "skipped-heavy"))
            continue
        if spec.task == "instance_segmentation":
            print("  SKIP: instance segmentation trainer not enabled in this run.", flush=True)
            summary.append((spec.id, "skipped-not-implemented"))
            continue
        t0 = time.time()
        try:
            if spec.task == "classification":
                res = train_classifier(spec, args.epochs, args.batch_size, args.lr)
            elif spec.task == "segmentation":
                res = train_unet(spec, args.epochs, args.batch_size, args.lr)
            elif spec.task == "detection":
                res = train_yolo(spec, args.epochs, args.batch_size, args.lr)
            else:
                raise ValueError(spec.task)
            save(spec, res)
            summary.append((spec.id, f"ok ({time.time()-t0:.0f}s)"))
        except Exception as exc:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"  FAILED {spec.id}: {type(exc).__name__}: {exc}", flush=True)
            summary.append((spec.id, f"failed: {type(exc).__name__}"))

    print("\n===== TRAINING SUMMARY =====")
    for mid, status in summary:
        print(f"  {mid:18s} {status}")


if __name__ == "__main__":
    main()
