"""Train every model ONCE on real data and export deployable inference artifacts.

Usage:
    python tools/build_artifacts.py             # build all models
    python tools/build_artifacts.py --only 1    # build a single model id

Outputs, per model id N (placed inside the matching project folder when known):
    <folder>/model/model.pt        torch state_dict
    <folder>/model/preprocess.pkl  preprocessing / metadata pickle
    <folder>/model/meta.json       human-readable metadata

Artifacts are written to a staging directory (.build/artifacts/N) and copied
into the folder by tools/finalize_artifacts.py so a partial build never breaks
the repository.

The inference applications (backend/) reproduce EXACTLY the preprocessing the
training loop used here, so the deployed model is the same model trained here.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import struct
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_breast_cancer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_zoo import MODEL_SPECS, SPEC_BY_ID, train_one  # noqa: E402
import real_datasets as rd  # noqa: E402

torch.set_num_threads(2)
SEED = 7
STAGE = ROOT / ".build" / "artifacts"
STAGE.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / ".artifacts_cache"
CACHE.mkdir(parents=True, exist_ok=True)

CORA_CLASS_NAMES = [
    "Case_Based", "Genetic_Algorithms", "Neural_Networks", "Probabilistic_Methods",
    "Reinforcement_Learning", "Rule_Learning", "Theory",
]


# --------------------------------------------------------------------------- #
# generic helpers
# --------------------------------------------------------------------------- #
def seed_all(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=-1, keepdims=True)


def train_classifier(model, x, y, epochs, lr=2e-3, wd=1e-4, batch=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n = len(x)
    for _ in range(epochs):
        model.train()
        if batch and batch < n:
            perm = torch.randperm(n)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                F.cross_entropy(model(x[idx]), y[idx]).backward()
                opt.step()
        else:
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, n, 128):                     # batched eval (low memory)
            idx = slice(i, min(i + 128, n))
            pred = model(x[idx]).argmax(1)
            correct += int((pred == y[idx]).sum())
    return correct / n


def staging(model_id: int):
    d = STAGE / str(model_id) / "model"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_artifacts(model_id: int, model, preprocess: dict, meta: dict) -> None:
    d = staging(model_id)
    torch.save(model.state_dict(), d / "model.pt")
    import pickle
    with open(d / "preprocess.pkl", "wb") as f:
        pickle.dump(preprocess, f, protocol=4)
    with open(d / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  -> artifacts/{model_id}: "
          f"{[p.name for p in sorted(d.iterdir())]}", flush=True)


# --------------------------------------------------------------------------- #
# dataset loaders
# --------------------------------------------------------------------------- #
MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
AGNEWS_URL = ("https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/"
              "master/data/ag_news_csv/train.csv")
AGNEWS_CLASSES = ["World", "Sports", "Business", "Sci/Tech"]


def _idx_images(raw: bytes) -> np.ndarray:
    with gzip.open(io.BytesIO(raw)) as g:
        _, n, rows, cols = struct.unpack(">IIII", g.read(16))
        return np.frombuffer(g.read(), dtype=np.uint8).reshape(n, rows, cols)


def _idx_labels(raw: bytes) -> np.ndarray:
    with gzip.open(io.BytesIO(raw)) as g:
        _, n = struct.unpack(">II", g.read(8))
        return np.frombuffer(g.read(), dtype=np.uint8)


def load_mnist():
    """Real 28x28 MNIST (LeCun). Cached locally as npz."""
    cache = CACHE / "mnist.npz"
    if cache.exists():
        d = np.load(cache)
        return d["xtr"], d["ytr"], d["xte"], d["yte"]
    print("  downloading real MNIST (28x28)...", flush=True)
    files = {
        "xtr": "train-images-idx3-ubyte.gz", "ytr": "train-labels-idx1-ubyte.gz",
        "xte": "t10k-images-idx3-ubyte.gz", "yte": "t10k-labels-idx1-ubyte.gz",
    }
    out = {}
    for k, name in files.items():
        raw = urllib.request.urlopen(MNIST_BASE + name, timeout=120).read()
        out[k] = _idx_images(raw) if k.startswith("x") else _idx_labels(raw)
    np.savez_compressed(cache, **out)
    return out["xtr"], out["ytr"], out["xte"], out["yte"]


def load_har_robust():
    """Robust UCI HAR loader. The upstream ``real_datasets.har()`` assumes a
    ``UCI_HAR_Dataset`` root, but the official zip extracts 'UCI HAR Dataset'
    (with spaces), so it can raise FileNotFoundError. This version searches for
    the train files wherever they extracted to."""
    cache = CACHE / "har"
    xtr_f, ytr_f = cache / "X_train.txt", cache / "y_train.txt"
    if not (xtr_f.exists() and ytr_f.exists()):
        cache.mkdir(parents=True, exist_ok=True)
        z = CACHE / "har.zip"
        if not z.exists():
            print("  downloading UCI HAR ...", flush=True)
            urllib.request.urlretrieve(
                "https://archive.ics.uci.edu/static/public/240/"
                "human+activity+recognition+using+smartphones.zip", z)
        # The official archive is a zip that itself contains a nested zip
        # ('UCI HAR Dataset.zip'). Unwrap any nesting until X_train.txt appears.
        pending, depth = [z], 0
        while pending and depth < 4:
            cur = pending.pop()
            with zipfile.ZipFile(cur) as f:
                f.extractall(cache)
            if list(cache.rglob("X_train.txt")):
                break
            pending = list(cache.rglob("*.zip")) + [p for p in cache.rglob("*.zip")]
            depth += 1
        hits = list(cache.rglob("X_train.txt"))
        if not hits:
            raise FileNotFoundError("X_train.txt not found after extracting HAR archive")
        src = hits[0].parent
        for name in ("X_train.txt", "y_train.txt"):
            (cache / name).write_bytes((src / name).read_bytes())
    x = np.loadtxt(xtr_f, dtype="float32")
    y = np.loadtxt(ytr_f, dtype="int64") - 1
    return (torch.from_numpy(x[:, :558].reshape(-1, 93, 6)), torch.from_numpy(y),
            ["walking", "upstairs", "downstairs", "sitting", "standing", "laying"])


def synth_segmentation(n=600, size=64, seed=42):
    np.random.seed(seed)
    imgs = np.zeros((n, size, size, 1), dtype=np.float32)
    masks = np.zeros((n, size, size, 1), dtype=np.float32)
    for i in range(n):
        imgs[i, :, :, 0] = np.random.rand(size, size) * 0.2
        for _ in range(np.random.randint(1, 4)):
            cx = np.random.randint(10, size - 10)
            cy = np.random.randint(10, size - 10)
            r = np.random.randint(5, 15)
            yg, xg = np.ogrid[:size, :size]
            circle = (xg - cx) ** 2 + (yg - cy) ** 2 <= r ** 2
            imgs[i, circle, 0] = np.random.uniform(0.6, 1.0)
            masks[i, circle, 0] = 1.0
    return torch.from_numpy(imgs), torch.from_numpy(masks)


def synth_detection(n=1000, img=64, grid=8, n_classes=3, max_obj=3, seed=42):
    """Exactly the generator used by 16_yolo/model.py."""
    rng = np.random.RandomState(seed)

    def draw(image, cx, cy, r, shape_id, value):
        yy, xx = np.ogrid[:img, :img]
        if shape_id == 0:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        elif shape_id == 1:
            mask = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
        else:
            x1, y1, x2, y2, x3, y3 = cx, cy - r, cx - r, cy + r, cx + r, cy + r
            def side(ax, ay, bx, by):
                return (xx - bx) * (ay - by) - (ax - bx) * (yy - by)
            d1, d2, d3 = side(x1, y1, x2, y2), side(x2, y2, x3, y3), side(x3, y3, x1, y1)
            mask = ~(((d1 < 0) | (d2 < 0) | (d3 < 0)) & ((d1 > 0) | (d2 > 0) | (d3 > 0)))
        image[mask] = value
        return image

    images = np.zeros((n, img, img, 1), dtype=np.float32)
    targets = np.zeros((n, grid, grid, 1 + 4 + n_classes), dtype=np.float32)
    for i in range(n):
        im = rng.rand(img, img).astype(np.float32) * 0.15
        occupied = []
        for _ in range(rng.randint(1, max_obj + 1)):
            for _try in range(20):
                r = rng.randint(4, 8)
                cx = rng.randint(r + 1, img - r - 1)
                cy = rng.randint(r + 1, img - r - 1)
                if all((cx - ox) ** 2 + (cy - oy) ** 2 > (r + orr + 4) ** 2
                       for ox, oy, orr in occupied):
                    break
            else:
                continue
            shape_id = rng.randint(0, n_classes)
            im = draw(im, cx, cy, r, shape_id, float(rng.uniform(0.7, 1.0)))
            occupied.append((cx, cy, r))
            cxx, cyy = cx, cy
            col = min(int(cxx / img * grid), grid - 1)
            row = min(int(cyy / img * grid), grid - 1)
            targets[i, row, col, 0] = 1.0
            targets[i, row, col, 1] = cxx / img * grid - col
            targets[i, row, col, 2] = cyy / img * grid - row
            targets[i, row, col, 3] = 2 * r / img
            targets[i, row, col, 4] = 2 * r / img
            targets[i, row, col, 5 + int(shape_id)] = 1.0
        images[i, :, :, 0] = im
    return torch.from_numpy(images), torch.from_numpy(targets)


_TEXT_CORPUS = {"texts": [], "labels": [], "cats": [], "source": ""}


def _load_text_corpus(cap=6000):
    """Return (texts, labels, class_names) from a real 4-class news dataset."""
    if _TEXT_CORPUS["texts"]:
        return _TEXT_CORPUS["texts"], _TEXT_CORPUS["labels"], _TEXT_CORPUS["cats"]
    try:
        tr, y, _, _, cats = rd.newsgroups()
        texts, labels, source = list(tr), [int(v) for v in y], "20 Newsgroups"
    except Exception as e:  # noqa: BLE001
        print(f"  20 Newsgroups unavailable ({type(e).__name__}); using AG News", flush=True)
        cache = CACHE / "agnews.csv"
        if not cache.exists():
            cache.write_bytes(urllib.request.urlopen(AGNEWS_URL, timeout=180).read())
        import csv
        texts, labels = [], []
        with open(cache, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 3:
                    labels.append(int(row[0]) - 1)
                    texts.append(f"{row[1]} {row[2]}")
        cats, source = list(AGNEWS_CLASSES), "AG News"
    texts, labels = texts[:cap], labels[:cap]
    _TEXT_CORPUS.update(texts=texts, labels=labels, cats=list(cats), source=source)
    print(f"  text dataset: {source} ({len(texts)} docs, {len(cats)} classes)", flush=True)
    return texts, labels, list(cats)


def text_docs():
    return _load_text_corpus()[0]


def load_text_data(cap=4000):
    texts, labels, cats = _load_text_corpus()
    texts, labels = texts[:cap], labels[:cap]
    x = torch.tensor([_hash_ids(t) for t in texts], dtype=torch.long)
    return x, torch.tensor(labels, dtype=torch.long), cats


def _hash_ids(text, length=32, vocab=1024):
    words = text.lower().split()[:length]
    ids = [1 + int(hashlib.md5(w.encode("utf-8", "ignore")).hexdigest(), 16) % (vocab - 2)
           for w in words]
    return ids + [0] * (length - len(ids))


def reverse_vocab(corpus, vocab_size=1024, length=32):
    w2i, i2w = {}, {}
    for doc in corpus:
        for w in doc.lower().split()[:length]:
            h = 1 + int(hashlib.md5(w.encode("utf-8", "ignore")).hexdigest(), 16) % (vocab_size - 2)
            w2i.setdefault(w, h)
            i2w.setdefault(h, w)
    return w2i, i2w
# --------------------------------------------------------------------------- #
# per-model builders  (each returns model, preprocess dict, meta dict)
# --------------------------------------------------------------------------- #
def build_01():
    from model_zoo import MLP
    seed_all()
    d = load_breast_cancer(as_frame=True)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(d.data.values)
    x = torch.from_numpy(scaler.transform(d.data.values).astype("float32"))
    y = torch.from_numpy(d.target.values.astype("int64"))
    model = MLP()
    metric = train_classifier(model, x, y, epochs=80, lr=2e-3)
    pre = {
        "kind": "tabular", "n_features": 30,
        "feature_names": list(d.feature_names),
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "class_names": list(d.target_names),
    }
    meta = {
        "id": 1, "name": "ANN / MLP", "architecture": "Multi-Layer Perceptron",
        "framework": "PyTorch (mirrors the TensorFlow/Keras training script in 01_ann)",
        "task": "Binary classification — malignant vs benign",
        "dataset": "Breast Cancer Wisconsin (Diagnostic), UCI / scikit-learn",
        "input": "30 numeric cell-nucleus features", "n_classes": 2,
        "metric_name": "train accuracy", "metric_value": round(metric, 4),
    }
    return model, pre, meta


def build_02():
    from model_zoo import CNN
    seed_all()
    xtr, ytr, xte, yte = load_mnist()
    cap = 20000                                        # low-memory CPU budget
    x = torch.from_numpy(xtr[:cap].reshape(-1, 1, 28, 28).astype("float32") / 255.0)
    y = torch.from_numpy(ytr[:cap].astype("int64"))
    model = CNN()
    metric = train_classifier(model, x, y, epochs=5, lr=1e-3, batch=128)
    pre = {"kind": "image", "size": 28, "channels": 1, "normalize": "divide by 255",
           "class_names": [str(i) for i in range(10)]}
    meta = {"id": 2, "name": "CNN", "architecture": "Convolutional Neural Network",
            "framework": "PyTorch (mirrors 02_cnn TF/Keras script)",
            "task": "10-class handwritten-digit classification",
            "dataset": "MNIST (LeCun) — the dataset named in 02_cnn", "input": "28x28 grayscale image",
            "n_classes": 10, "metric_name": "train accuracy", "metric_value": round(metric, 4)}
    return model, pre, meta


def _har_model(model_id, cls_name, epochs=25):
    from model_zoo import CNN, ResNet, DenseNet, ViT
    seed_all()
    x, y, names = load_har_robust()
    from model_zoo import RNN, LSTM, GRU
    cls = {"RNN": RNN, "LSTM": LSTM, "GRU": GRU}[cls_name]
    model = cls()
    metric = train_classifier(model, x, y, epochs=epochs, lr=2e-3, batch=256)
    pre = {"kind": "sequence", "length": 93, "channels": 6, "class_names": names}
    return model, pre, metric


def build_03():
    model, pre, metric = _har_model(3, "RNN")
    meta = {"id": 3, "name": "RNN", "architecture": "Recurrent Neural Network",
            "framework": "PyTorch (mirrors 03_rnn)", "task": "6-class activity recognition",
            "dataset": "UCI Human Activity Recognition Using Smartphones", "input": "93 x 6 sensor window",
            "n_classes": 6, "metric_name": "train accuracy", "metric_value": round(metric, 4)}
    return model, pre, meta


def build_04():
    model, pre, metric = _har_model(4, "LSTM")
    meta = {"id": 4, "name": "LSTM", "architecture": "Long Short-Term Memory network",
            "framework": "PyTorch (mirrors 04_lstm)", "task": "6-class activity recognition",
            "dataset": "UCI Human Activity Recognition Using Smartphones", "input": "93 x 6 sensor window",
            "n_classes": 6, "metric_name": "train accuracy", "metric_value": round(metric, 4)}
    return model, pre, meta


def build_05():
    model, pre, metric = _har_model(5, "GRU")
    meta = {"id": 5, "name": "GRU", "architecture": "Gated Recurrent Unit network",
            "framework": "PyTorch (mirrors 05_gru)", "task": "6-class activity recognition",
            "dataset": "UCI Human Activity Recognition Using Smartphones", "input": "93 x 6 sensor window",
            "n_classes": 6, "metric_name": "train accuracy", "metric_value": round(metric, 4)}
    return model, pre, meta


def build_06():
    from model_zoo import Autoencoder
    seed_all()
    xtr, _, _, _ = load_mnist()
    x = torch.from_numpy(xtr[:6000].astype("float32") / 255.0)
    x = F.adaptive_avg_pool2d(
        torch.from_numpy(xtr[:6000].reshape(-1, 1, 28, 28).astype("float32") / 255.0), (8, 8))
    model = Autoencoder()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(40):
        opt.zero_grad()
        F.mse_loss(model(x), x).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        rec = model(x)
    mse = float(F.mse_loss(rec, x))
    pre = {"kind": "image", "size": 8, "channels": 1, "ossize": 28,
           "note": "image resized to 28x28 then avg-pooled to 8x8, scaled to [0,1]"}
    meta = {"id": 6, "name": "Autoencoder", "architecture": "Convolutional autoencoder (64->12->64)",
            "framework": "PyTorch (mirrors 06_autoencoder)", "task": "Image reconstruction / anomaly score",
            "dataset": "MNIST (unnormalised digits)", "input": "28x28 grayscale image",
            "n_classes": 0, "metric_name": "reconstruction MSE", "metric_value": round(mse, 4)}
    return model, pre, meta


def build_07():
    from model_zoo import VAE
    seed_all()
    xtr, _, _, _ = load_mnist()
    x = F.adaptive_avg_pool2d(torch.from_numpy(xtr[:6000].reshape(-1, 1, 28, 28).astype("float32") / 255.0), (8, 8))
    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(40):
        opt.zero_grad()
        rec, mu, lv = model(x)
        loss = F.binary_cross_entropy(rec, x) + 1e-4 * torch.mean(mu.pow(2) + lv.exp() - lv - 1)
        loss.backward()
        opt.step()
    model.eval()
    pre = {"kind": "image", "size": 8, "channels": 1, "ossize": 28,
           "note": "28x28 -> avg-pool 8x8 -> [0,1]; sampling in latent space"}
    meta = {"id": 7, "name": "VAE", "architecture": "Variational autoencoder",
            "framework": "PyTorch (mirrors 07_vae)", "task": "Reconstruction + latent generation",
            "dataset": "MNIST", "input": "28x28 grayscale image", "n_classes": 0,
            "metric_name": "train loss", "metric_value": round(float(loss), 4)}
    return model, pre, meta


def build_08():
    from model_zoo import GANGenerator, GANDiscriminator
    seed_all()
    xtr, _, _, _ = load_mnist()
    x = F.adaptive_avg_pool2d(torch.from_numpy(xtr[:6000].reshape(-1, 1, 28, 28).astype("float32") / 255.0), (8, 8))
    G, D = GANGenerator(), GANDiscriminator()
    og = torch.optim.Adam(G.parameters(), lr=2e-3)
    od = torch.optim.Adam(D.parameters(), lr=2e-3)
    for _ in range(60):
        z = torch.randn(len(x), 32)
        fake = G(z).detach()
        od.zero_grad()
        F.binary_cross_entropy_with_logits(D(x), torch.ones(len(x), 1)).backward()
        F.binary_cross_entropy_with_logits(D(fake), torch.zeros(len(x), 1)).backward()
        od.step()
        og.zero_grad()
        F.binary_cross_entropy_with_logits(D(G(z)), torch.ones(len(x), 1)).backward()
        og.step()
    G.eval(); D.eval()
    with torch.no_grad():
        acc = float(((torch.sigmoid(D(x)) > 0.5).float().mean()))
    pre = {"kind": "image", "size": 8, "channels": 1, "ossize": 28,
           "note": "28x28 -> avg-pool 8x8 -> [0,1]"}
    meta = {"id": 8, "name": "GAN", "architecture": "GAN discriminator (+generator shipped)",
            "framework": "PyTorch (mirrors 08_gan)", "task": "Real vs generated digit classification",
            "dataset": "MNIST", "input": "28x28 grayscale image", "n_classes": 2,
            "metric_name": "discriminator accuracy on real digits", "metric_value": round(acc, 4)}
    return D, pre, meta


def _text_data(cap=4000):
    return load_text_data(cap)


def build_09():
    from model_zoo import Transformer
    seed_all()
    x, y, cats = _text_data()
    model = Transformer()
    metric = train_classifier(model, x, y, epochs=12, lr=1e-3, batch=256)
    pre = {"kind": "text", "length": 32, "vocab_size": 1024, "tokenizer": "sha1-hash",
           "class_names": cats}
    meta = {"id": 9, "name": "Transformer", "architecture": "Transformer encoder (attention)",
            "framework": "PyTorch (mirrors 09_transformer)", "task": "4-class topic classification",
            "dataset": "20 Newsgroups (4 categories)", "input": "free text", "n_classes": 4,
            "metric_name": "train accuracy", "metric_value": round(metric, 4)}
    return model, pre, meta


def build_10():
    from model_zoo import TinyBERT
    seed_all()
    x, y, cats = _text_data()
    model = TinyBERT()
    metric = train_classifier(model, x, y, epochs=12, lr=1e-3, batch=256)
    pre = {"kind": "text", "length": 32, "vocab_size": 1024, "tokenizer": "sha1-hash",
           "class_names": cats}
    meta = {"id": 10, "name": "BERT-style", "architecture": "Bidirectional Transformer encoder",
            "framework": "PyTorch (mirrors 10_bert DistilBERT script; a small in-repo encoder is "
                         "trained here so the demo stays CPU/free-tier friendly)",
            "task": "4-class topic classification", "dataset": "20 Newsgroups (4 categories)",
            "input": "free text", "n_classes": 4, "metric_name": "train accuracy",
            "metric_value": round(metric, 4)}
    return model, pre, meta


def build_11():
    from model_zoo import TinyGPT
    seed_all()
    x, _, _ = _text_data()
    model = TinyGPT()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(15):
        idx = torch.randperm(len(x))[:256]
        opt.zero_grad()
        out = model(x[idx][:, :-1])
        F.cross_entropy(out.reshape(-1, 1024), x[idx][:, 1:].reshape(-1)).backward()
        opt.step()
    model.eval()
    tr_docs = text_docs()
    w2i, i2w = reverse_vocab(tr_docs)
    pre = {"kind": "text_gen", "length": 32, "vocab_size": 1024, "tokenizer": "sha1-hash",
           "id_to_word": {str(k): v for k, v in list(i2w.items())[:2000]}}
    meta = {"id": 11, "name": "GPT-style", "architecture": "Causal (masked) Transformer LM head",
            "framework": "PyTorch (mirrors 11_gpt GPT-2 script; small in-repo LM trained here)",
            "task": "Autoregressive next-token text generation", "dataset": "20 Newsgroups corpus",
            "input": "free text prompt", "n_classes": 0,
            "metric_name": "vocab size", "metric_value": 1024}
    return model, pre, meta
def _mnist_cls(model, epochs=6, cap=12000, batch=128):
    seed_all()
    xtr, ytr, _, _ = load_mnist()
    x = torch.from_numpy(xtr[:cap].reshape(-1, 1, 28, 28).astype("float32") / 255.0)
    y = torch.from_numpy(ytr[:cap].astype("int64"))
    return model, x, y, train_classifier(model, x, y, epochs=epochs, lr=1e-3, batch=batch)


def build_12():
    from model_zoo import ResNet
    model, x, y, metric = _mnist_cls(ResNet())
    pre = {"kind": "image", "size": 28, "channels": 1, "normalize": "divide by 255",
           "class_names": [str(i) for i in range(10)]}
    meta = {"id": 12, "name": "ResNet", "architecture": "Residual network",
            "framework": "PyTorch (mirrors 12_resnet)", "task": "10-class digit classification",
            "dataset": "MNIST (named as the image task in 12_resnet)", "input": "28x28 grayscale image",
            "n_classes": 10, "metric_name": "train accuracy", "metric_value": round(metric, 4)}
    return model, pre, meta


def build_13():
    from model_zoo import DenseNet
    model, x, y, metric = _mnist_cls(DenseNet())
    pre = {"kind": "image", "size": 28, "channels": 1, "normalize": "divide by 255",
           "class_names": [str(i) for i in range(10)]}
    meta = {"id": 13, "name": "DenseNet", "architecture": "Dense connectivity network",
            "framework": "PyTorch (mirrors 13_densenet)", "task": "10-class digit classification",
            "dataset": "MNIST", "input": "28x28 grayscale image", "n_classes": 10,
            "metric_name": "train accuracy", "metric_value": round(metric, 4)}
    return model, pre, meta


def build_14():
    from model_zoo import ViT
    model, x, y, metric = _mnist_cls(ViT(), epochs=6, cap=6000, batch=64)
    pre = {"kind": "image", "size": 28, "channels": 1, "normalize": "divide by 255",
           "class_names": [str(i) for i in range(10)]}
    meta = {"id": 14, "name": "ViT", "architecture": "Vision Transformer (patch embeddings)",
            "framework": "PyTorch (mirrors 14_vit)", "task": "10-class digit classification",
            "dataset": "MNIST", "input": "28x28 grayscale image", "n_classes": 10,
            "metric_name": "train accuracy", "metric_value": round(metric, 4)}
    return model, pre, meta


def build_15():
    from model_zoo import UNet
    seed_all()
    x, mask = synth_segmentation()
    x, mask = x.permute(0, 3, 1, 2).contiguous(), mask.permute(0, 3, 1, 2).contiguous()
    xtr, mtr = x[:500], mask[:500]
    xte, mte = x[500:], mask[500:]
    model = UNet()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(15):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(model(xtr), mtr).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        iou = []
        for i in range(len(xte)):
            p = (torch.sigmoid(model(xte[i:i + 1])) > 0.5)
            t = mte[i:i + 1] > 0.5
            inter = (p & t).sum().float()
            union = (p | t).sum().float()
            iou.append(float(inter / (union + 1e-8)))
    mean_iou = float(np.mean(iou))
    pre = {"kind": "image", "size": 64, "channels": 1, "normalize": "divide by 255",
           "note": "binary foreground mask at 0.5 threshold"}
    meta = {"id": 15, "name": "U-Net", "architecture": "Encoder-decoder with skip connections",
            "framework": "PyTorch (mirrors 15_unet)", "task": "Binary image segmentation",
            "dataset": "Synthetic circle-segmentation dataset (same generator as 15_unet)",
            "input": "64x64 grayscale image", "n_classes": 0,
            "metric_name": "test mean IoU", "metric_value": round(mean_iou, 4)}
    return model, pre, meta


def build_16():
    from model_zoo import YOLOLite
    seed_all()
    x, t = synth_detection()
    x = x.permute(0, 3, 1, 2).contiguous()          # (N,H,W,1) -> (N,1,H,W)
    xtr, ttr = x[:800], t[:800]
    xte, tte = x[800:], t[800:]
    model = YOLOLite()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(25):
        idx = torch.randperm(len(xtr))[:128]
        opt.zero_grad()
        out = model(xtr[idx])
        labels = ttr[idx]
        obj = labels[..., 0:1]
        box_loss = F.mse_loss(torch.sigmoid(out[..., 1:5]) * obj, labels[..., 1:5] * obj)
        cls_loss = F.binary_cross_entropy_with_logits(out[..., 5:], labels[..., 5:])
        (box_loss + cls_loss).backward()
        opt.step()
    model.eval()
    pre = {"kind": "image", "size": 64, "channels": 1, "normalize": "divide by 255",
           "grid": 8, "class_names": ["circle", "square", "triangle"],
           "note": "single grid head: [obj, cxd, cyd, w, h, class logits]"}

    def decode(img, conf=0.5):
        with torch.no_grad():
            out = model(img[None])[0]
        prob = torch.sigmoid(out[..., 0])
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
                boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2,
                              float(prob[r, c]), cls))
        return boxes

    # metric: are there detections where objects exist (recall@0.3)
    hit = 0
    with torch.no_grad():
        for i in range(64):
            boxes = decode(xte[i], conf=0.3)
            gt = int((tte[i, ..., 0] > 0.5).sum())
            if gt == 0 or len(boxes) > 0:
                hit += 1
    metric = hit / 64
    meta = {"id": 16, "name": "YOLO-style", "architecture": "Single-stage detection head (8x8 grid)",
            "framework": "PyTorch (mirrors 16_yolo)", "task": "Object detection + classification",
            "dataset": "Synthetic multi-shape dataset (same generator as 16_yolo)",
            "input": "64x64 grayscale image", "n_classes": 3,
            "metric_name": "detection recall (test, conf>=0.3)", "metric_value": round(metric, 4)}
    return model, pre, meta


def build_17():
    from model_zoo import Siamese
    seed_all()
    xtr, ytr, _, _ = load_mnist()
    x28 = torch.from_numpy(xtr[:5000].reshape(-1, 1, 28, 28).astype("float32") / 255.0)
    x = F.adaptive_avg_pool2d(x28, (8, 8))          # twin encoder expects 8x8=64
    y = torch.from_numpy(ytr[:5000].astype("int64"))
    model = Siamese()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(40):
        idx = torch.randperm(len(x))[:256]
        a, b = x[idx], torch.roll(x[idx], 1, 0)
        same = (y[idx] == torch.roll(y[idx], 1, 0)).float()
        opt.zero_grad()
        za, zb = model(a, b)
        dist = (za - zb).pow(2).sum(1).sqrt()
        loss = (same * dist.pow(2) + (1 - same) * F.relu(1 - dist).pow(2)).mean()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        idx = torch.randperm(len(x))[:1000]
        a, b = x[idx], torch.roll(x[idx], 1, 0)
        same = (y[idx] == torch.roll(y[idx], 1, 0))
        za, zb = model(a, b)
        pred = ((za - zb).pow(2).sum(1).sqrt() < 0.5)
        metric = float((pred == same).float().mean())
    pre = {"kind": "pair", "size": 28, "channels": 1, "ossize": 28, "normalize": "divide by 255",
           "threshold": 0.5, "class_names": ["different", "same"]}
    meta = {"id": 17, "name": "Siamese Network", "architecture": "Twin encoder + L2 distance head",
            "framework": "PyTorch", "task": "Same/different digit verification (metric learning)",
            "dataset": "MNIST", "input": "two 28x28 grayscale images", "n_classes": 2,
            "metric_name": "verification accuracy", "metric_value": round(metric, 4)}
    return model, pre, meta


def build_18():
    from model_zoo import Seq2Seq
    seed_all()
    x, _, _ = _text_data()
    model = Seq2Seq()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(15):
        idx = torch.randperm(len(x))[:256]
        opt.zero_grad()
        out = model(x[idx][:, :-1])
        F.cross_entropy(out.reshape(-1, 1024), x[idx][:, 1:].reshape(-1)).backward()
        opt.step()
    model.eval()
    docs = text_docs()
    w2i, i2w = reverse_vocab(docs)
    i2w = {int(k): v for k, v in i2w.items()}
    w2i = {}
    for doc in docs:
        for w in doc.lower().split()[:32]:
            h = 1 + int(hashlib.md5(w.encode("utf-8", "ignore")).hexdigest(), 16) % 1022
            w2i.setdefault(w, h)
    pre = {"kind": "seq2seq", "length": 32, "vocab_size": 1024,
           "word_to_id": {k: int(v) for k, v in list(w2i.items())[:6000]},
           "id_to_word": {str(k): v for k, v in list(i2w.items())[:6000]},
           "class_names": []}
    meta = {"id": 18, "name": "Seq2Seq", "architecture": "Encoder-decoder (noisy reconstruction)",
            "framework": "PyTorch", "task": "Token-level reconstruction of a text window",
            "dataset": "20 Newsgroups corpus", "input": "free text", "n_classes": 0,
            "metric_name": "vocab size", "metric_value": 1024}
    return model, pre, meta


def build_19():
    from model_zoo import DiffusionMLP
    seed_all()
    d = load_breast_cancer(as_frame=True)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(d.data.values)
    x = torch.from_numpy(scaler.transform(d.data.values).astype("float32"))
    model = DiffusionMLP()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(400):
        t = torch.rand(len(x), 1)
        noise = torch.randn_like(x[:, :30]) * t
        opt.zero_grad()
        F.mse_loss(model(x[:, :30] + noise, t), noise).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        t = torch.full((len(x), 1), 0.5)
        noise = torch.randn_like(x[:, :30]) * t
        mse = float(F.mse_loss(model(x[:, :30] + noise, t), noise))
    pre = {"kind": "tabular", "n_features": 30,
           "feature_names": list(d.feature_names),
           "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
           "preset_rows": {"benign-like": scaler.transform(d.data.values[d.target == 1][:1])[0].tolist(),
                           "malignant-like": scaler.transform(d.data.values[d.target == 0][:1])[0].tolist()},
           "class_names": ["malignant", "benign"]}
    meta = {"id": 19, "name": "Diffusion MLP", "architecture": "Noise-conditioned denoising MLP",
            "framework": "PyTorch", "task": "Tabular denoising (reconstruct a clean feature row)",
            "dataset": "Breast Cancer Wisconsin (Diagnostic)", "input": "30 numeric features",
            "n_classes": 0, "metric_name": "denoise MSE", "metric_value": round(mse, 4)}
    return model, pre, meta


def build_20():
    from model_zoo import GNN
    seed_all()
    x, y, a = rd.cora()
    n = min(512, x.shape[0])
    x, y, a = x[:n], y[:n], a[:n, :n]
    model = GNN()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(30):
        opt.zero_grad()
        F.cross_entropy(model(x, a), y).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(x, a)
        metric = float((logits.argmax(1) == y).float().mean())
    pre = {"kind": "graph", "n_features": 1433, "n_classes": 7, "class_names": CORA_CLASS_NAMES,
           "preset_rows": {"row0": x[0].tolist(), "row1": x[1].tolist()}}
    meta = {"id": 20, "name": "GNN", "architecture": "Graph convolution (message passing)",
            "framework": "PyTorch", "task": "Cora 7-class node classification",
            "dataset": "Cora citation network (Planetoid)", "input": "1433-dim node feature vector",
            "n_classes": 7, "metric_name": "train accuracy", "metric_value": round(metric, 4)}
    return model, pre, meta


BUILDERS = {
    1: build_01, 2: build_02, 3: build_03, 4: build_04, 5: build_05, 6: build_06, 7: build_07,
    8: build_08, 9: build_09, 10: build_10, 11: build_11, 12: build_12, 13: build_13, 14: build_14,
    15: build_15, 16: build_16, 17: build_17, 18: build_18, 19: build_19, 20: build_20,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=0)
    args = ap.parse_args()
    ids = [args.only] if args.only else sorted(BUILDERS)
    manifest = {}
    for mid in ids:
        name = SPEC_BY_ID[mid][1]
        if (STAGE / str(mid) / "model" / "model.pt").exists():
            print(f"[{mid:02d}] {name} already built — skipping (resume)", flush=True)
            manifest[mid] = {"name": name, "resumed": True}
            continue
        t = time.time()
        print(f"[{mid:02d}] training {name} ...", flush=True)
        try:
            model, pre, meta = BUILDERS[mid]()
            write_artifacts(mid, model, pre, meta)
            dt = time.time() - t
            manifest[mid] = {"name": name, "ok": True, "seconds": round(dt, 1),
                             "metric": meta["metric_name"], "value": meta["metric_value"]}
            print(f"[{mid:02d}] done in {dt:.1f}s | "
                  f"{meta['metric_name']}={meta['metric_value']}", flush=True)
        except Exception as e:  # noqa: BLE001 — a single failure must not abort the rest
            import traceback
            traceback.print_exc()
            manifest[mid] = {"name": name, "ok": False, "error": f"{type(e).__name__}: {e}"}
            print(f"[{mid:02d}] FAILED: {type(e).__name__}: {e}", flush=True)
    (STAGE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ok = [k for k, v in manifest.items() if v.get("ok") or v.get("resumed")]
    print(f"ALL DONE — {len(ok)}/{len(ids)} ready", json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
