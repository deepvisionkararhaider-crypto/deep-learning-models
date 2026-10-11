#!/usr/bin/env python3
"""Offline training for the original 20-model laboratory.

    python scripts/train_lab.py --model all
    python scripts/train_lab.py --model lab02
    python scripts/train_lab.py --list

Each model gets a task-appropriate objective (cross-entropy, reconstruction,
adversarial, language-modelling, Dice+BCE, detection, contrastive, diffusion,
graph CE) — no shared generic loss. Writes artifacts with area="lab".
"""
from __future__ import annotations

import argparse
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
from src.datasets import lab_datasets as D           # noqa: E402
from src.models import lab20 as M                    # noqa: E402
from src.models.detection import (encode_targets, yolo_loss, decode_predictions,  # noqa: E402
                                  box_iou, xywh_to_xyxy)
from src.models.segmentation import bce_dice_loss, dice_coef, iou_score  # noqa: E402
from src.registry import LAB_BY_INDEX, ModelSpec    # noqa: E402

SEED = 42
MNIST_TRAIN, MNIST_TEST = 8000, 2000


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def _meta(spec: ModelSpec, metrics: dict, extra: dict) -> dict:
    return {"name": spec.name, "area": "lab", "task": spec.task, "framework": "PyTorch",
            "architecture": spec.builder, "dataset": spec.dataset_name,
            "dataset_key": spec.dataset_key, "classes": spec.classes,
            "n_classes": len(spec.classes),
            "input": {"size": spec.input_size, "channels": spec.in_ch, "mean": [0.5, 0.5, 0.5],
                      "std": [0.5, 0.5, 0.5], "desc": spec.input_desc},
            "outputs": spec.outputs, "metric_names": spec.metric_names, "metrics": metrics,
            "training": extra, "notes": spec.notes, "trained": True}


def _acc(model, x, y, bs=512):
    model.eval(); correct = 0
    with torch.inference_mode():
        for i in range(0, len(x), bs):
            out = model(x[i:i + bs])
            if isinstance(out, tuple):
                out = out[0]
            correct += int((out.argmax(1) == y[i:i + bs]).sum())
    return correct / max(len(x), 1)


def _macro_f1(y_true, y_pred, n):
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# --------------------------------------------------------------------------- #
# trainers
# --------------------------------------------------------------------------- #
def train_mlp(spec):
    x_tr, y_tr, x_te, y_te, scaler, names = D.breast_cancer()
    model = M.MLP(30, 2); opt = torch.optim.Adam(model.parameters(), 1e-3)
    for ep in range(60):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(x_tr), y_tr); loss.backward(); opt.step()
    with torch.inference_mode():
        pred = model(x_te).argmax(1).numpy()
    acc = float((pred == y_te.numpy()).mean())
    m = {"accuracy": round(acc, 4), "f1_binary": round(_macro_f1(y_te.numpy(), pred, 2), 4)}
    meta = _meta(spec, m, {"epochs": 60, "optimizer": "adam"})
    meta["scaler"] = scaler; meta["classes"] = names; meta["input"]["features"] = 30
    return model, m, meta


def _mnist_image_model(spec, cls):
    x_tr, y_tr = D.mnist("train", MNIST_TRAIN)
    x_te, y_te = D.mnist("test", MNIST_TEST)
    model = cls(**spec.build_kwargs); opt = torch.optim.Adam(model.parameters(), 1e-3)
    for ep in range(6):
        model.train(); perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), 128):
            idx = perm[i:i + 128]; opt.zero_grad()
            loss = F.cross_entropy(model(x_tr[idx]), y_tr[idx]); loss.backward(); opt.step()
    acc = _acc(model, x_te, y_te)
    with torch.inference_mode():
        pred = torch.cat([model(x_te[i:i + 512]).argmax(1) for i in range(0, len(x_te), 512)]).numpy()
    m = {"accuracy": round(acc, 4), "f1_macro": round(_macro_f1(y_te.numpy(), pred, 10), 4)}
    return model, m, _meta(spec, m, {"epochs": 6, "batch_size": 128, "train_n": MNIST_TRAIN})


def train_sequence(spec, cls):
    x, y = D.har()
    x = x[:6000]; y = y[:6000]
    n = int(len(x) * 0.8); x_tr, y_tr, x_te, y_te = x[:n], y[:n], x[n:], y[n:]
    model = cls(**spec.build_kwargs); opt = torch.optim.Adam(model.parameters(), 1e-3)
    for ep in range(8):
        model.train(); perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), 128):
            idx = perm[i:i + 128]; opt.zero_grad()
            loss = F.cross_entropy(model(x_tr[idx]), y_tr[idx]); loss.backward(); opt.step()
    acc = _acc(model, x_te, y_te)
    m = {"accuracy": round(acc, 4)}
    return model, m, _meta(spec, m, {"epochs": 8, "batch_size": 128, "train_n": len(x_tr)})


def train_autoencoder(spec):
    x_tr, _ = D.mnist("train", MNIST_TRAIN)
    x_te, _ = D.mnist("test", MNIST_TEST)
    model = M.Autoencoder(); opt = torch.optim.Adam(model.parameters(), 1e-3)
    for ep in range(8):
        model.train(); perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), 128):
            idx = perm[i:i + 128]; opt.zero_grad()
            loss = F.mse_loss(model(x_tr[idx]), x_tr[idx]); loss.backward(); opt.step()
    model.eval()
    with torch.inference_mode():
        mse = float(F.mse_loss(model(x_te), x_te))
    m = {"recon_mse": round(mse, 5)}
    return model, m, _meta(spec, m, {"epochs": 8, "loss": "mse"})


def train_vae(spec):
    x_tr, _ = D.mnist("train", MNIST_TRAIN)
    x_te, _ = D.mnist("test", MNIST_TEST)
    model = M.VAE(); opt = torch.optim.Adam(model.parameters(), 1e-3)
    for ep in range(8):
        model.train(); perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), 128):
            idx = perm[i:i + 128]; opt.zero_grad()
            rec, mu, lv = model(x_tr[idx])
            loss = F.binary_cross_entropy(rec, x_tr[idx]) + 1e-3 * torch.mean(mu.pow(2) + lv.exp() - lv - 1)
            loss.backward(); opt.step()
    model.eval()
    with torch.inference_mode():
        rec, mu, lv = model(x_te)
        bce = float(F.binary_cross_entropy(rec, x_te)); kl = float(torch.mean(mu.pow(2) + lv.exp() - lv - 1))
    m = {"recon_bce": round(bce, 5), "kl": round(kl, 4)}
    return model, m, _meta(spec, m, {"epochs": 8, "loss": "bce + beta*kl"})


def train_gan(spec):
    x_tr, _ = D.mnist("train", MNIST_TRAIN)
    x_te, y_te = D.mnist("test", MNIST_TEST)
    G = M.GANGenerator(); Dm = M.GANDiscriminator()
    og = torch.optim.Adam(G.parameters(), 2e-4); od = torch.optim.Adam(Dm.parameters(), 2e-4)
    for ep in range(8):
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), 128):
            real = x_tr[perm[i:i + 128]]; z = torch.randn(len(real), 64)
            od.zero_grad()
            fake = G(z).detach()
            ld = (F.binary_cross_entropy_with_logits(Dm(real), torch.ones(len(real), 1))
                  + F.binary_cross_entropy_with_logits(Dm(fake), torch.zeros(len(fake), 1)))
            ld.backward(); od.step()
            og.zero_grad()
            lg = F.binary_cross_entropy_with_logits(Dm(G(z)), torch.ones(len(real), 1))
            lg.backward(); og.step()
    Dm.eval()
    with torch.inference_mode():
        score = torch.sigmoid(Dm(x_te)).mean().item()
    m = {"real_score_mean": round(float(score), 4), "accuracy": round(float(score), 4)}
    meta = _meta(spec, m, {"epochs": 8, "note": "discriminator artifact; generator trained jointly"})
    meta["classes"] = ["generated", "real"]
    return Dm, m, meta


def train_text(spec, cls):
    X, y, cats = D.newsgroups(limit=3000, length=64)
    n = int(len(X) * 0.8); Xtr, ytr, Xte, yte = X[:n], y[:n], X[n:], y[n:]
    model = cls(**spec.build_kwargs); opt = torch.optim.Adam(model.parameters(), 1e-3)
    for ep in range(4):
        model.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]; opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx]); loss.backward(); opt.step()
    acc = _acc(model, Xte, yte)
    m = {"accuracy": round(acc, 4)}
    meta = _meta(spec, m, {"epochs": 4, "batch_size": 64, "train_n": len(Xtr), "length": 64,
                           "vocab": spec.build_kwargs.get("vocab", 4096)})
    return model, m, meta


def train_lm(spec, cls):
    X = D.newsgroups_lm(limit=2000, length=96)
    model = cls(**spec.build_kwargs); opt = torch.optim.Adam(model.parameters(), 1e-3)
    for ep in range(4):
        model.train(); perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]; seq = X[idx]; opt.zero_grad()
            out = model(seq[:, :-1])
            loss = F.cross_entropy(out.reshape(-1, out.shape[-1]), seq[:, 1:].reshape(-1))
            loss.backward(); opt.step()
    model.eval()
    with torch.inference_mode():
        out = model(X[:64, :-1])
        ppl = float(torch.exp(F.cross_entropy(out.reshape(-1, out.shape[-1]), X[:64, 1:].reshape(-1))))
    m = {"perplexity": round(ppl, 2)}
    meta = _meta(spec, m, {"epochs": 4, "batch_size": 64, "length": 96})
    return model, m, meta


def train_seg_mnist(spec):
    x_tr, m_tr, _ = D.mnist_segmentation("train", MNIST_TRAIN)
    x_te, m_te, _ = D.mnist_segmentation("test", MNIST_TEST)
    model = M.UNetSmall(); opt = torch.optim.Adam(model.parameters(), 1e-3)
    for ep in range(6):
        model.train(); perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), 128):
            idx = perm[i:i + 128]; opt.zero_grad()
            loss = bce_dice_loss(model(x_tr[idx]), m_tr[idx]); loss.backward(); opt.step()
    model.eval()
    with torch.inference_mode():
        logits = model(x_te)
        d = dice_coef(logits, m_te); iou = iou_score(logits, m_te)
    m = {"dice": round(d, 4), "iou": round(iou, 4)}
    return model, m, _meta(spec, m, {"epochs": 6, "loss": "bce_dice"})


def train_det_mnist(spec):
    grid = spec.build_kwargs.get("grid", 7)
    x_tr, b_tr, y_tr = D.mnist_detection("train", MNIST_TRAIN)
    x_te, b_te, y_te = D.mnist_detection("test", MNIST_TEST)
    model = M.YOLOLite(**spec.build_kwargs); opt = torch.optim.Adam(model.parameters(), 1e-3)
    n_classes = spec.build_kwargs["n_classes"]
    for ep in range(8):
        model.train(); perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), 128):
            idx = perm[i:i + 128]
            # mnist_detection returns normalised xywh -> encode_targets wants xyxy
            boxes_xyxy = [xywh_to_xyxy(b_tr[j][None]) for j in idx]
            tgt = encode_targets(boxes_xyxy, [y_tr[j][None] for j in idx], grid, n_classes)
            opt.zero_grad(); loss = yolo_loss(model(x_tr[idx]), tgt); loss.backward(); opt.step()
    model.eval()
    with torch.inference_mode():
        preds = [decode_predictions(model(x_te[i:i + 128])[k].cpu(), conf_thr=0.3)
                 for k, i in enumerate(range(0, len(x_te), 128))]
    tp = sum(len(p) for p in preds); aps = []
    for k, i in enumerate(range(0, len(x_te), 128)):
        for j, b in enumerate(b_tr[i:i + 128]):
            gt = xywh_to_xyxy(b[None])                       # (1,4)
            pr = torch.tensor([[q["x1"], q["y1"], q["x2"], q["y2"]] for q in preds[k]])
            if pr.numel() and gt.numel():
                iou = box_iou(pr, gt)                        # (K,1)
                aps.append(float(iou.max(0).values.item() >= 0.5))
    m = {"mAP@50": round(float(np.mean(aps)) if aps else 0.0, 4), "detections": tp}
    return model, m, _meta(spec, m, {"epochs": 8, "grid": grid})


def train_siamese(spec):
    x_tr, y_tr = D.mnist("train", 4000)
    x_te, y_te = D.mnist("test", 1000)
    model = M.Siamese(); opt = torch.optim.Adam(model.parameters(), 1e-3)

    def pair_loss(xa, xb, same):
        za, zb = model(xa, xb); dist = (za - zb).pow(2).sum(1).sqrt()
        return (same * dist.pow(2) + (1 - same) * F.relu(1 - dist).pow(2)).mean()

    for ep in range(6):
        model.train(); perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), 128):
            idx = perm[i:i + 128]; a = x_tr[idx]; b = torch.roll(a, 1, 0)
            same = (y_tr[idx] == torch.roll(y_tr[idx], 1, 0)).float()
            opt.zero_grad(); loss = pair_loss(a, b, same); loss.backward(); opt.step()
    model.eval()
    with torch.inference_mode():
        za, zb = model(x_te, torch.roll(x_te, 1, 0))
        dist = (za - zb).pow(2).sum(1).sqrt()
        same = (y_te == torch.roll(y_te, 1, 0)).float()
        acc = float(((dist < 0.5).float() == same).float().mean())
        thr = float(dist.median())
    m = {"pair_accuracy": round(acc, 4)}
    meta = _meta(spec, m, {"epochs": 6, "loss": "contrastive"}); meta["threshold"] = thr
    return model, m, meta


def train_diffusion(spec):
    x_tr, _, x_te, _, scaler, _ = D.breast_cancer()
    model = M.DiffusionMLP(); opt = torch.optim.Adam(model.parameters(), 1e-3)
    for ep in range(400):
        t = torch.rand(len(x_tr), 1); noise = torch.randn_like(x_tr) * t
        opt.zero_grad(); loss = F.mse_loss(model(x_tr + noise, t), noise); loss.backward(); opt.step()
    model.eval()
    with torch.inference_mode():
        t = torch.full((len(x_te), 1), 0.5); noise = torch.randn_like(x_te) * t
        err = float(F.mse_loss(model(x_te + noise, t), noise))
    m = {"denoise_mse": round(err, 5)}
    meta = _meta(spec, m, {"steps": 400, "loss": "noise_prediction_mse"})
    meta["scaler"] = scaler; meta["input"]["features"] = 30
    return model, m, meta


def train_gnn(spec):
    x, y, a = D.cora()
    model = M.GNN(); opt = torch.optim.Adam(model.parameters(), 1e-2, weight_decay=5e-4)
    n = len(x); tr = torch.arange(0, int(n * 0.6)); te = torch.arange(int(n * 0.8), n)
    for ep in range(60):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(x, a)[tr], y[tr]); loss.backward(); opt.step()
    model.eval()
    with torch.inference_mode():
        acc = float((model(x, a)[te].argmax(1) == y[te]).float().mean())
    m = {"accuracy": round(acc, 4)}
    return model, m, _meta(spec, m, {"epochs": 60, "train_n": int(len(tr)), "test_n": int(len(te))})


# --------------------------------------------------------------------------- #
TRAINERS = {
    1: train_mlp, 2: lambda s: _mnist_image_model(s, M.CNN), 3: lambda s: train_sequence(s, M.RNN),
    4: lambda s: train_sequence(s, M.LSTM), 5: lambda s: train_sequence(s, M.GRU),
    6: train_autoencoder, 7: train_vae, 8: train_gan,
    9: lambda s: train_text(s, M.Transformer), 10: lambda s: train_text(s, M.TinyBERT),
    11: lambda s: train_lm(s, M.TinyGPT), 12: lambda s: _mnist_image_model(s, M.ResNet),
    13: lambda s: _mnist_image_model(s, M.DenseNet), 14: lambda s: _mnist_image_model(s, M.ViT),
    15: train_seg_mnist, 16: train_det_mnist, 17: train_siamese,
    18: lambda s: train_lm(s, M.Seq2Seq), 19: train_diffusion, 20: train_gnn,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for i, sp in LAB_BY_INDEX.items():
            print(f"lab{i:02d}  {sp.task:22s} {sp.dataset_name}")
        return

    idxs = list(range(1, 21)) if args.model == "all" else [int(args.model.replace("lab", ""))]
    set_seed()
    summary = []
    for i in idxs:
        spec = LAB_BY_INDEX[i]
        print(f"\n=== lab{i:02d} {spec.name} ({spec.task}) ===", flush=True)
        t0 = time.time()
        try:
            model, metrics, meta = TRAINERS[i](spec)
            info = A.save_model(model, spec.id, "lab", meta)
            print(f"  saved {spec.id} {info['weights_bytes']/1e6:.2f}MB "
                  f"metrics={ {k: v for k, v in metrics.items()} } ({time.time()-t0:.0f}s)", flush=True)
            summary.append((spec.id, "ok"))
        except Exception as exc:  # noqa: BLE001
            import traceback; traceback.print_exc()
            summary.append((spec.id, f"FAILED {type(exc).__name__}: {exc}"))
    print("\n===== LAB SUMMARY =====")
    for mid, st in summary:
        print(f"  {mid}  {st}")


if __name__ == "__main__":
    main()
