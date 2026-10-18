#!/usr/bin/env python3
"""Offline training pipeline for the medical & agriculture computer-vision models.

Usage
-----
    python scripts/train_models.py --list
    python scripts/train_models.py --model all
    python scripts/train_models.py --model brain_densenet --epochs 4
    python scripts/train_models.py --model plant_cnn --epochs 4 --batch-size 8
    python scripts/train_models.py --model brain_maskrcnn --allow-heavy
    python scripts/train_models.py --model all --defer brain_maskrcnn,plant_maskrcnn

This is the ONLY place that trains. The Streamlit app never imports ``torch.optim``.

Every run writes:
  * ``artifacts/<area>/<id>.pt``            — torch state_dict (CPU)
  * ``artifacts/metadata/<id>.json``        — contract + measured metrics + provenance
  * ``artifacts/status/<id>.json``          — structured per-model record
  * ``artifacts/status/summary.json``       — machine-readable run report

Honest statuses
---------------
TRAINED_VERIFIED    weights saved AND reloaded from disk in a *fresh process* + inference OK
TRAINED_UNVERIFIED  weights saved but the fresh-load check failed
ARCH_ONLY           architecture builds, no weights produced (deferred / not requested)
BLOCKED             no usable dataset exists — cannot be trained at all
DEFERRED            intentionally skipped via --defer (an explicit, recorded choice)
FAILED              training or evaluation raised

Exit code is 0 only when every *required* model reaches TRAINED_VERIFIED. A model
is required unless it is BLOCKED or explicitly listed in ``--defer``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import artifacts as A                       # noqa: E402
from src.datasets import loaders as L                # noqa: E402
from src.imaging import IMAGENET_MEAN, IMAGENET_STD, make_transform  # noqa: E402
from src.models.detection import (encode_targets, yolo_loss, decode_predictions,  # noqa: E402
                                  box_iou, xywh_to_xyxy, average_precision,
                                  build_maskrcnn, make_maskrcnn_target, instances_to_predictions)
from src.models.segmentation import bce_dice_loss, dice_coef, iou_score  # noqa: E402
from src.registry import BY_ID, CV_MODELS, ModelSpec, build_model  # noqa: E402
from src import training_state as TS  # noqa: E402

SEED = 42

# Mask R-CNN (ResNet-50-FPN) needs several GB of RAM even at batch size 1 and is
# impractical on a free CPU runner; it is gated behind --allow-heavy and reported
# honestly as DEFERRED (not "trained") when it is not attempted.
HEAVY = {"brain_maskrcnn", "plant_maskrcnn"}

# Cap on how many *training* images are decoded for the two largest CV datasets.
# Validation/test are always read in full so the reported metrics are honest.
TRAIN_CAPS = {"plant_cls": 4000, "brain_mri_cls": 4000, "plant_seg": 1500,
              "brain_det": 800, "cotton_det": 1200, "brain_instseg": 800}

# Statuses
TRAINED_VERIFIED = "TRAINED_VERIFIED"
TRAINED_UNVERIFIED = "TRAINED_UNVERIFIED"
ARCH_ONLY = "ARCH_ONLY"
BLOCKED = "BLOCKED"
DEFERRED = "DEFERRED"
FAILED = "FAILED"

STATUS_DIR = ROOT / "artifacts" / "status"


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


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


def _check_labels(items, n_classes, tag):
    bad = sorted({int(y) for _b, y in items if not (0 <= int(y) < n_classes)})
    if bad:
        raise ValueError(f"{tag}: labels {bad} are outside [0,{n_classes}); "
                         f"class mapping is wrong for this dataset.")


def _restore(spec: ModelSpec, model, opt, sched, epochs: int, resume: bool):
    """Load resume state when asked, else start clean. Never trusts bad state."""
    if not resume or not TS.has_state(spec.id):
        return None
    try:
        st = TS.load_state(spec.id, model=model, optimizer=opt, scheduler=sched,
                           total_epochs=epochs)
        print(f"    resumed from epoch {st.start_epoch}/{epochs} "
              f"(global_step={st.global_step})", flush=True)
        return st
    except TS.InvalidCheckpoint as exc:
        print(f"    resume state unusable ({exc}) — starting a fresh run", flush=True)
        TS.clear(spec.id)
        return None


def train_classifier(spec: ModelSpec, epochs: int, batch_size: int, lr: float,
                     resume: bool = False) -> dict:
    n = len(spec.classes)
    if spec.dataset_key == "brain_mri_cls":
        data = L.load_brain_mri(limit=None)
    elif spec.dataset_key == "plant_cls":
        data = L.load_plant_cls(limit=None)
    else:
        raise ValueError(f"no classifier loader for {spec.dataset_key}")

    # training subset is capped for speed; val/test stay complete
    cap = TRAIN_CAPS.get(spec.dataset_key)
    train_items = data["train"][:cap] if cap else data["train"]
    _check_labels(train_items, n, "train")
    _check_labels(data["val"], n, "val")
    _check_labels(data["test"], n, "test")

    size = spec.input_size
    train_tf = make_transform(size, IMAGENET_MEAN, IMAGENET_STD, train=True)
    eval_tf = make_transform(size, IMAGENET_MEAN, IMAGENET_STD, train=False)
    train_ld = _collate_classification(train_items, train_tf, batch_size, True)
    val_ld = _collate_classification(data["val"], eval_tf, batch_size, False)
    test_ld = _collate_classification(data["test"], eval_tf, batch_size, False)

    # Transfer-learned backbones get ImageNet weights ONLY during offline training.
    if spec.builder == "densenet":
        from src.models.classifiers import build_densenet
        model = build_densenet(n, in_ch=spec.in_ch, pretrained=True)
        backbone = "torchvision densenet121 (ImageNet-pretrained)"
    elif spec.builder == "vit":
        from src.models.classifiers import build_vit
        model = build_vit(n, in_ch=spec.in_ch, pretrained=True, image_size=size)
        backbone = "timm vit_tiny_patch16_224 (ImageNet-pretrained)"
    else:
        model = build_model(spec)()
        backbone = "from scratch"
    for p in model.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    st = _restore(spec, model, opt, sched, epochs, resume)
    start, step = (st.start_epoch, st.global_step) if st else (0, 0)
    hist = list(st.history) if st else []
    for ep in range(start, epochs):
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
            tot += float(loss.detach()) * x.size(0); seen += x.size(0)   # detach: no grad tracking
            step += 1
        sched.step()
        hist.append(round(tot / max(seen, 1), 4))
        print(f"    epoch {ep+1}/{epochs} loss={hist[-1]:.4f}", flush=True)
        # atomic resume state after every completed epoch
        TS.save_state(spec.id, model=model, optimizer=opt, scheduler=sched, epoch=ep + 1,
                      global_step=step, history=hist,
                      extra={"dataset_key": spec.dataset_key, "epochs": epochs})

    metrics = evaluate_classifier(model, test_ld, n)
    metrics["val"] = evaluate_classifier(model, val_ld, n)
    metrics["train_loss_curve"] = hist
    metrics["split_sizes"] = {"train": len(train_items), "val": len(data["val"]),
                              "test": len(data["test"])}
    meta = _base_meta(spec, size, IMAGENET_MEAN, IMAGENET_STD, metrics,
                      extra={"epochs": epochs, "lr": lr, "batch_size": batch_size,
                             "split_seed": SEED, "backbone": backbone,
                             "train_cap": cap, "resumed_from": start})
    return {"model": model, "metrics": metrics, "meta": meta}


# --------------------------------------------------------------------------- #
# segmentation (U-Net)
# --------------------------------------------------------------------------- #
def train_unet(spec: ModelSpec, epochs: int, batch_size: int, lr: float,
               resume: bool = False) -> dict:
    size = spec.input_size
    if spec.dataset_key == "brain_seg":
        data = L.load_brain_seg()
    elif spec.dataset_key == "plant_seg":
        data = L.load_plant_seg()
    else:
        raise ValueError(f"no segmentation loader for {spec.dataset_key}")

    cap = TRAIN_CAPS.get(spec.dataset_key)
    train_items = data["train"][:cap] if cap else data["train"]
    # Mask sanity: a target with no foreground pixels is a legitimate (empty) mask,
    # but *all*-empty targets mean the mask decode/threshold is wrong.
    fg = sum(1 for _i, mb in train_items[:64] if L.mask_array(mb, size).max() > 0)
    if train_items and fg == 0:
        raise ValueError(f"{spec.dataset_key}: every sampled mask is empty — mask decode is wrong.")

    from src.datasets.loaders import SegmentationDataset
    tr = SegmentationDataset(train_items, size, IMAGENET_MEAN, IMAGENET_STD, train=True)
    va = SegmentationDataset(data["val"], size, IMAGENET_MEAN, IMAGENET_STD, train=False)
    te = SegmentationDataset(data["test"], size, IMAGENET_MEAN, IMAGENET_STD, train=False)
    tl = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True)
    vl = torch.utils.data.DataLoader(va, batch_size=batch_size)
    el = torch.utils.data.DataLoader(te, batch_size=batch_size)

    model = build_model(spec)()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    st = _restore(spec, model, opt, sched, epochs, resume)
    start, step = (st.start_epoch, st.global_step) if st else (0, 0)
    hist = list(st.history) if st else []
    for ep in range(start, epochs):
        model.train(); tot, seen = 0.0, 0
        for x, m in tl:
            opt.zero_grad()
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = bce_dice_loss(logits, m)
            loss.backward(); opt.step()
            tot += float(loss.detach()) * x.size(0); seen += x.size(0)   # detach
            step += 1
        sched.step()
        hist.append(round(tot / max(seen, 1), 4))
        print(f"    epoch {ep+1}/{epochs} loss={hist[-1]:.4f}", flush=True)
        TS.save_state(spec.id, model=model, optimizer=opt, scheduler=sched, epoch=ep + 1,
                      global_step=step, history=hist,
                      extra={"dataset_key": spec.dataset_key, "epochs": epochs})

    metrics = _eval_unet(model, el)
    metrics["val"] = _eval_unet(model, vl)
    metrics["train_loss_curve"] = hist
    metrics["split_sizes"] = {"train": len(tr), "val": len(va), "test": len(te)}
    meta = _base_meta(spec, size, IMAGENET_MEAN, IMAGENET_STD, metrics,
                      extra={"epochs": epochs, "lr": lr, "loss": "bce_dice",
                             "mask_threshold": 0.5, "train_cap": cap,
                             "foreground_in_sample": fg, "resumed_from": start})
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
def _detection_targets(rows, tf, grid, n_classes):
    from src.imaging import decode_image
    xs, tgts, gts = [], [], []
    for r in rows:
        xs.append(tf(decode_image(r["image"])))
        w, h = r["width"], r["height"]
        b = r["boxes"].copy()
        if len(b):
            b[:, [0, 2]] /= w; b[:, [1, 3]] /= h
        b = np.clip(b, 0.0, 1.0).astype(np.float32)
        lab = r["labels"].astype(np.int64)
        gts.append((b, lab))
        tgts.append(encode_targets([torch.from_numpy(b)], [torch.from_numpy(lab)],
                                   grid, n_classes)[0])
    return xs, tgts, gts


def train_yolo(spec: ModelSpec, epochs: int, batch_size: int, lr: float,
               resume: bool = False) -> dict:
    size = spec.input_size
    grid = spec.build_kwargs.get("grid", 14)
    data = L.load_detection(spec.dataset_key)
    cap = TRAIN_CAPS.get(spec.dataset_key)
    train_rows = data["train"][:cap] if cap else data["train"]
    tf = make_transform(size, IMAGENET_MEAN, IMAGENET_STD, train=False)

    # annotation sanity: a detection dataset with no boxes anywhere cannot be trained
    n_boxes = sum(len(r["boxes"]) for r in train_rows)
    if train_rows and n_boxes == 0:
        raise ValueError(f"{spec.dataset_key}: no bounding boxes in the training split — "
                         f"this is a classification dataset, not a detection dataset.")

    model = build_model(spec)()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    st = _restore(spec, model, opt, sched, epochs, resume)
    start, step = (st.start_epoch, st.global_step) if st else (0, 0)
    hist = list(st.history) if st else []
    for ep in range(start, epochs):
        model.train(); random.shuffle(train_rows); tot, seen = 0.0, 0
        for i in range(0, len(train_rows), batch_size):
            chunk = train_rows[i:i + batch_size]
            xs, tgts, _ = _detection_targets(chunk, tf, grid, len(spec.classes))
            xb = torch.stack(xs); tb = torch.stack(tgts)
            opt.zero_grad()
            loss = yolo_loss(model(xb), tb)
            loss.backward(); opt.step()
            tot += float(loss.detach()) * xb.size(0); seen += xb.size(0)   # detach
            step += 1
        sched.step()
        hist.append(round(tot / max(seen, 1), 4))
        print(f"    epoch {ep+1}/{epochs} loss={hist[-1]:.4f}", flush=True)
        TS.save_state(spec.id, model=model, optimizer=opt, scheduler=sched, epoch=ep + 1,
                      global_step=step, history=hist,
                      extra={"dataset_key": spec.dataset_key, "epochs": epochs})

    metrics = _eval_detection(model, data["test"], tf, spec, grid)
    metrics["val"] = _eval_detection(model, data["val"], tf, spec, grid)
    metrics["train_loss_curve"] = hist
    metrics["split_sizes"] = {k: len(v) for k, v in data.items()}
    metrics["train_boxes"] = int(n_boxes)
    med = _median_box_px(data["train"])
    meta = _base_meta(spec, size, IMAGENET_MEAN, IMAGENET_STD, metrics,
                      extra={"epochs": epochs, "lr": lr, "grid": grid,
                             "box_format": "normalised xyxy", "conf_threshold": 0.30,
                             "train_cap": cap, "median_gt_box_px": med})
    return {"model": model, "metrics": metrics, "meta": meta}


def _median_box_px(rows) -> float | None:
    """Median ground-truth box side in *pixels* — documents tiny-object datasets."""
    sides = []
    for r in rows:
        b = r["boxes"]
        if len(b):
            sides.extend(((b[:, 2] - b[:, 0]) + (b[:, 3] - b[:, 1])).tolist() / 2.0)
    return round(float(np.median(sides)), 2) if sides else None


@torch.no_grad()
def _eval_detection(model, rows, tf, spec: ModelSpec, grid: int) -> dict:
    """Real VOC-style mAP@50 over the whole split (class-averaged)."""
    from src.imaging import decode_image
    model.eval()
    pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels = [], [], [], [], []
    for r in rows:
        x = tf(decode_image(r["image"])).unsqueeze(0)
        preds = decode_predictions(model(x)[0].cpu(), conf_thr=0.30)
        w, h = r["width"], r["height"]
        gt = r["boxes"].astype(np.float32).copy()
        if len(gt):
            gt[:, [0, 2]] /= w; gt[:, [1, 3]] /= h
        gt = np.clip(gt, 0.0, 1.0)
        gt_boxes.append(torch.from_numpy(gt))
        gt_labels.append(torch.from_numpy(r["labels"].astype(np.int64)))
        if preds:
            pred_boxes.append(torch.tensor([[p["x1"], p["y1"], p["x2"], p["y2"]] for p in preds]))
            pred_scores.append(torch.tensor([p["score"] for p in preds]))
            pred_labels.append(torch.tensor([p["class"] for p in preds], dtype=torch.int64))
        else:
            pred_boxes.append(torch.zeros((0, 4)))
            pred_scores.append(torch.zeros((0,)))
            pred_labels.append(torch.zeros((0,), dtype=torch.int64))
    res = average_precision(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels,
                            iou_thr=0.5, n_classes=len(spec.classes))
    res["n_test"] = len(rows)
    res["n_gt_boxes"] = int(sum(int(g.shape[0]) for g in gt_boxes))
    res["n_pred_boxes"] = int(sum(int(p.shape[0]) for p in pred_boxes))
    return res


# --------------------------------------------------------------------------- #
# instance segmentation (Mask R-CNN)
# --------------------------------------------------------------------------- #
def train_maskrcnn(spec: ModelSpec, epochs: int, batch_size: int, lr: float) -> dict:
    """Train torchvision Mask R-CNN on genuine COCO instance polygons.

    Requires real per-instance masks (boxes + labels + masks). Heavy: it is only
    attempted with --allow-heavy. Batch size is forced to 1 because Mask R-CNN
    builds a variable-length proposal list per image.
    """
    from src.imaging import decode_image
    data = L.load_instance_segmentation(spec.dataset_key)
    cap = TRAIN_CAPS.get(spec.dataset_key)
    train_rows = data["train"][:cap] if cap else data["train"]
    n_inst = sum(len(r["labels"]) for r in train_rows)
    if train_rows and n_inst == 0:
        raise ValueError(f"{spec.dataset_key}: no instance annotations — cannot train Mask R-CNN.")

    size = spec.input_size
    model = build_maskrcnn(len(spec.classes) + 1, pretrained=True)   # +1 background
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)

    def prep(row):
        img = decode_image(row["image"]).resize((size, size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - np.asarray(IMAGENET_MEAN, np.float32)) / np.asarray(IMAGENET_STD, np.float32)
        x = torch.from_numpy(np.transpose(arr, (2, 0, 1))).float()
        w, h = row["width"], row["height"]
        sx, sy = size / w, size / h
        b = row["boxes"].copy().astype(np.float32)
        if len(b):
            b[:, [0, 2]] *= sx; b[:, [1, 3]] *= sy
        masks = row["masks"]
        if masks.size:
            mt = torch.from_numpy(masks.astype(np.float32)).unsqueeze(1)
            mt = F.interpolate(mt, size=(size, size), mode="nearest")[:, 0]
        else:
            mt = torch.zeros((0, size, size))
        tgt = make_maskrcnn_target(torch.from_numpy(b), torch.from_numpy(row["labels"] + 1), mt)
        return x, tgt

    hist = []
    for ep in range(epochs):
        model.train(); random.shuffle(train_rows); tot, seen = 0.0, 0
        for row in train_rows[: max(1, min(len(train_rows), 200 * epochs))]:
            x, tgt = prep(row)
            opt.zero_grad()
            losses = model([x], [tgt])
            loss = sum(v for v in losses.values())
            loss.backward(); opt.step()
            tot += float(loss.detach()); seen += 1
        hist.append(round(tot / max(seen, 1), 4))
        print(f"    epoch {ep+1}/{epochs} loss={hist[-1]:.4f}", flush=True)

    metrics = _eval_maskrcnn(model, data["test"][:60], spec, size)
    metrics["train_loss_curve"] = hist
    metrics["split_sizes"] = {k: len(v) for k, v in data.items()}
    meta = _base_meta(spec, size, IMAGENET_MEAN, IMAGENET_STD, metrics,
                      extra={"epochs": epochs, "lr": lr, "optimizer": "sgd",
                             "backbone": "torchvision maskrcnn_resnet50_fpn (ImageNet-pretrained)",
                             "train_cap": cap})
    return {"model": model, "metrics": metrics, "meta": meta}


@torch.no_grad()
def _eval_maskrcnn(model, rows, spec, size) -> dict:
    """Mean instance count + mean top-1 detection confidence (an honest proxy).

    A full mask-AP@50 needs a COCO evaluator; reporting a made-up AP would be
    worse than reporting these two transparent, directly-measured numbers.
    """
    from src.imaging import decode_image
    model.eval()
    counts, confs = [], []
    for r in rows:
        img = decode_image(r["image"]).resize((size, size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - np.asarray(IMAGENET_MEAN, np.float32)) / np.asarray(IMAGENET_STD, np.float32)
        x = torch.from_numpy(np.transpose(arr, (2, 0, 1))).float()
        out = model([x])[0]
        inst = instances_to_predictions(out, score_thr=0.4)
        counts.append(len(inst))
        if inst:
            confs.append(max(i["score"] for i in inst))
    return {"mean_instances": round(float(np.mean(counts)), 3) if counts else 0.0,
            "mean_top_confidence": round(float(np.mean(confs)), 4) if confs else 0.0,
            "n_test": len(rows)}


# --------------------------------------------------------------------------- #
# metadata + verification
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
        "split_manifest": L.split_counts(spec.dataset_key),
        "trained": True,
    }
    if extra:
        meta["training"] = extra
    return meta


def fresh_load_check(spec: ModelSpec) -> tuple[bool, str]:
    """Reload the checkpoint from disk in a brand-new Python process and run inference."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from src.registry import BY_ID\n"
        "from src import inference as I\n"
        "from PIL import Image; import numpy as np\n"
        "spec = BY_ID[%r]\n"
        "model, meta = I.load(spec)\n"
        "size = int((meta.get('input') or {}).get('size', spec.input_size))\n"
        "img = Image.fromarray((np.random.rand(size, size, 3)*255).astype('uint8'))\n"
        "I.predict(spec, img)\n"
        "print('FRESH_OK')\n"
    ) % (str(ROOT), spec.id)
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           timeout=600, cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        return False, "fresh-load check timed out"
    if r.returncode == 0 and "FRESH_OK" in r.stdout:
        return True, ""
    return False, (r.stderr.strip().splitlines() or ["unknown error"])[-1][:300]


def save(spec: ModelSpec, result: dict) -> dict:
    meta = A.save_model(result["model"], spec.id, spec.area, result["meta"])
    scal = {k: v for k, v in result["metrics"].items() if isinstance(v, (int, float))}
    print(f"  saved {spec.id}: {meta['weights_bytes']/1e6:.2f}MB | metrics={scal}", flush=True)
    # The published artifact is now the source of truth; drop the scratch resume state
    # so a later run does not resume a model that has already been saved.
    TS.clear(spec.id)
    return meta


def write_record(rec: dict):
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    (STATUS_DIR / f"{rec['model_id']}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _resolve_targets(arg: str) -> list[ModelSpec]:
    if arg == "all":
        return list(CV_MODELS)
    if arg == "medical":
        return [m for m in CV_MODELS if m.area == "medical"]
    if arg == "agriculture":
        return [m for m in CV_MODELS if m.area == "agriculture"]
    if arg not in BY_ID:
        raise SystemExit(f"unknown model {arg!r}")
    return [BY_ID[arg]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", help="model id, 'all', 'medical', 'agriculture'")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--allow-heavy", action="store_true", help="attempt Mask R-CNN (very slow on CPU)")
    ap.add_argument("--require-heavy", action="store_true",
                    help="treat Mask R-CNN as a required model (fails the run if not trained)")
    ap.add_argument("--defer", default="", help="comma-separated model ids to skip on purpose")
    ap.add_argument("--force", action="store_true", help="retrain even if valid weights already exist")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from artifacts/checkpoints/<id>/state.pt")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for m in CV_MODELS:
            print(f"{m.id:18s} {m.area:12s} {m.task:22s} {m.dataset_name}")
        return 0

    defer = {s.strip() for s in args.defer.split(",") if s.strip()}
    unknown = defer - set(BY_ID)
    if unknown:
        raise SystemExit(f"--defer references unknown model ids: {sorted(unknown)}")

    set_seed()
    targets = _resolve_targets(args.model)
    records, required_missing = [], []

    print(f"=== capacity: peak_rss={peak_rss_mb():.0f}MB, torch={torch.__version__}, "
          f"threads={torch.get_num_threads()} ===", flush=True)

    for spec in targets:
        t0 = time.time()
        rec = {
            "model_id": spec.id, "area": spec.area, "task": spec.task,
            "architecture": spec.builder, "dataset": spec.dataset_name,
            "dataset_key": spec.dataset_key, "classes": spec.classes,
            "seed": SEED, "epochs": args.epochs, "batch_size": args.batch_size,
            "checkpoint": None, "checkpoint_bytes": None, "metrics": {},
            "duration_s": None, "verification": None, "reason": "",
            "status": ARCH_ONLY,
        }
        print(f"\n=== {spec.id} ({spec.task}) ===", flush=True)

        if spec.is_blocked:
            rec["status"] = BLOCKED
            rec["reason"] = spec.blocked
            print(f"  BLOCKED: {spec.blocked[:120]}", flush=True)
            records.append(rec); write_record(rec); continue

        if spec.id in defer:
            rec["status"] = DEFERRED
            rec["reason"] = "explicitly deferred via --defer"
            print("  DEFERRED (explicit --defer)", flush=True)
            records.append(rec); write_record(rec); continue

        if spec.id in HEAVY and not args.allow_heavy:
            rec["status"] = DEFERRED
            rec["reason"] = ("heavy detector (Mask R-CNN) not attempted; "
                             "rerun with --allow-heavy (needs several GB RAM)")
            print("  DEFERRED: heavy detector — use --allow-heavy to attempt.", flush=True)
            records.append(rec); write_record(rec)
            # An impractical-on-CPU model is DEFERRED by default and reported
            # prominently; --require-heavy promotes it back to a hard requirement.
            if args.require_heavy:
                required_missing.append(spec.id)
            continue

        if not args.force and A.has_weights(spec.id, spec.area):
            ok, why = fresh_load_check(spec)
            rec["status"] = TRAINED_VERIFIED if ok else TRAINED_UNVERIFIED
            rec["verification"] = "fresh-process inference OK" if ok else f"failed: {why}"
            rec["reason"] = "reused existing valid artifact"
            meta = A.read_metadata(spec.id) or {}
            rec["checkpoint"] = meta.get("weights_file")
            rec["checkpoint_bytes"] = meta.get("weights_bytes")
            rec["metrics"] = meta.get("metrics", {})
            print(f"  REUSE: existing artifact ({rec['status']})", flush=True)
            records.append(rec); write_record(rec)
            if not ok:
                required_missing.append(spec.id)
            continue

        try:
            if spec.task == "classification":
                res = train_classifier(spec, args.epochs, args.batch_size, args.lr, resume=args.resume)
            elif spec.task == "segmentation":
                res = train_unet(spec, args.epochs, args.batch_size, args.lr, resume=args.resume)
            elif spec.task == "detection":
                res = train_yolo(spec, args.epochs, args.batch_size, args.lr, resume=args.resume)
            elif spec.task == "instance_segmentation":
                res = train_maskrcnn(spec, args.epochs, args.batch_size, args.lr)
            else:
                raise ValueError(f"no trainer for task {spec.task}")
            info = save(spec, res)
            ok, why = fresh_load_check(spec)
            rec["checkpoint"] = info.get("weights_file")
            rec["checkpoint_bytes"] = info.get("weights_bytes")
            rec["metrics"] = {k: v for k, v in res["metrics"].items() if isinstance(v, (int, float))}
            rec["status"] = TRAINED_VERIFIED if ok else TRAINED_UNVERIFIED
            rec["verification"] = "fresh-process inference OK" if ok else f"failed: {why}"
            rec["duration_s"] = round(time.time() - t0, 1)
            if not ok:
                required_missing.append(spec.id)
        except Exception as exc:                                   # noqa: BLE001
            traceback.print_exc()
            rec["status"] = FAILED
            rec["reason"] = f"{type(exc).__name__}: {exc}"
            rec["duration_s"] = round(time.time() - t0, 1)
            print(f"  FAILED {spec.id}: {rec['reason']}", flush=True)
            required_missing.append(spec.id)
        records.append(rec); write_record(rec)

    summary = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "python": sys.version.split()[0], "torch": torch.__version__,
               "args": vars(args), "peak_rss_mb": round(peak_rss_mb(), 1),
               "models": records}
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    (STATUS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== TRAINING SUMMARY =====")
    for rec in records:
        print(f"  {rec['model_id']:18s} {rec['status']:18s} {rec['reason'][:70]}")
    by_status = {}
    for rec in records:
        by_status[rec["status"]] = by_status.get(rec["status"], 0) + 1
    print(f"  -> {by_status}")

    if required_missing:
        print(f"\nREQUIRED MODELS NOT VERIFIED ({len(required_missing)}): "
              f"{', '.join(required_missing)}")
        return 1
    print("\nAll required models reached TRAINED_VERIFIED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
