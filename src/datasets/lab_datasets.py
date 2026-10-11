"""Real public datasets for the 20-model laboratory.

Every loader downloads from a public, citable source and caches under ``.cache``:

* MNIST          — torchvision (LeCun et al.)
* UCI HAR        — UCI ML repository (Anguita et al.)
* 20 Newsgroups  — HuggingFace ``SetFit/20_newsgroups`` parquet mirror
* Cora           — Planetoid (Yang et al.) raw files
* Breast Cancer  — scikit-learn bundled copy of the UCI WDBC dataset

Only the offline trainer imports the heavy loaders; the Streamlit runtime never
calls them.
"""
from __future__ import annotations

import io
import json
import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache" / "lab"
CACHE.mkdir(parents=True, exist_ok=True)


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
    return dest


# --------------------------------------------------------------------------- #
# MNIST (and derived segmentation / detection targets)
# --------------------------------------------------------------------------- #
def mnist(split: str = "train", limit: int | None = None):
    from torchvision import datasets, transforms

    ds = datasets.MNIST(str(CACHE / "torchvision"), train=(split == "train"), download=True,
                        transform=transforms.ToTensor())
    xs, ys = [], []
    n = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n):
        x, y = ds[i]
        xs.append(x)
        ys.append(int(y))
    return torch.stack(xs), torch.tensor(ys)


def mnist_segmentation(split: str = "train", limit: int = 4000):
    """Foreground masks derived from the digit intensity (threshold > 0.15)."""
    x, y = mnist(split, limit)
    return x, (x > 0.15).float(), y


def mnist_detection(split: str = "train", limit: int = 4000):
    """Bounding boxes from the ink extent of each digit (normalised cx,cy,w,h)."""
    x, y = mnist(split, limit)
    boxes = []
    for im in x:
        ink = (im[0] > 0.2)
        if ink.any():
            ys, xs = ink.nonzero(as_tuple=True)
            x1, x2 = xs.min().item() / 27, xs.max().item() / 27
            y1, y2 = ys.min().item() / 27, ys.max().item() / 27
            boxes.append([(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 0.05), max(y2 - y1, 0.05)])
        else:
            boxes.append([0.5, 0.5, 0.5, 0.5])
    return x, torch.tensor(boxes, dtype=torch.float32), y


# --------------------------------------------------------------------------- #
# UCI HAR
# --------------------------------------------------------------------------- #
HAR_URL = ("https://archive.ics.uci.edu/static/public/240/"
           "human+activity+recognition+using+smartphones.zip")
HAR_CLASSES = ["walking", "walking_upstairs", "walking_downstairs", "sitting", "standing", "laying"]


def har(limit: int | None = None):
    """(N,128,9) inertial signals -> 6 activity classes."""
    z = CACHE / "har.zip"
    root = CACHE / "UCI_HAR_Dataset"
    expected = root / "train" / "X_train.txt"
    # GitHub runners may retain an incomplete extraction from a failed download.
    # Re-download/extract unless the expected training file is actually present.
    if not expected.is_file():
        CACHE.mkdir(parents=True, exist_ok=True)
        if not z.is_file() or z.stat().st_size < 1_000_000:
            z.unlink(missing_ok=True)
            _download(HAR_URL, z)
        if not zipfile.is_zipfile(z):
            z.unlink(missing_ok=True)
            _download(HAR_URL, z)
        if not zipfile.is_zipfile(z):
            raise RuntimeError(f"Downloaded HAR archive is invalid: {z}")
        with zipfile.ZipFile(z) as archive:
            bad = archive.testzip()
            if bad:
                raise RuntimeError(f"Corrupt HAR archive member: {bad}")
            archive.extractall(CACHE)
        inner = CACHE / "UCI HAR Dataset"
        if (inner / "train" / "X_train.txt").is_file():
            if root.exists():
                import shutil
                shutil.rmtree(root)
            inner.rename(root)
        if not expected.is_file():
            raise FileNotFoundError(
                f"HAR extraction did not produce {expected}; inspect {z} and the archive layout."
            )
    x = np.loadtxt(expected, dtype=np.float32)
    y = np.loadtxt(root / "train" / "y_train.txt", dtype=np.int64) - 1
    # first 9 columns are the raw body-acceleration triaxial means; reshape to 128 steps
    x = x[:, :9].reshape(-1, 128, 9) if x.shape[1] >= 9 else x[:, :9]
    if limit:
        x, y = x[:limit], y[:limit]
    return torch.from_numpy(x), torch.from_numpy(y)


# --------------------------------------------------------------------------- #
# 20 Newsgroups (4 categories)
# --------------------------------------------------------------------------- #
NEWS_CATEGORIES = ["comp.graphics", "sci.space", "rec.sport.baseball", "talk.politics.misc"]
_NEWS_PARQUET = ("https://huggingface.co/datasets/SetFit/20_newsgroups/resolve/"
                 "refs%2Fconvert%2Fparquet/default/train/0000.parquet")


def _char_vocab():
    chars = "\n !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz"
    return {c: i + 2 for i, c in enumerate(chars)}   # 0=pad, 1=unk


def encode_chars(text: str, length: int = 96) -> list[int]:
    v = _char_vocab()
    ids = [v.get(ch, 1) for ch in text[:length].lower()]
    return ids + [0] * (length - len(ids))


def newsgroups(limit: int = 4000, length: int = 96):
    """Returns (X ids (N,L), y labels, class names).

    ``SetFit/20_newsgroups`` stores the numeric ``label`` as a *string* and the
    human-readable category in ``label_text``; we filter on ``label_text``.
    """
    import pyarrow.parquet as pq

    fp = CACHE / "newsgroups_train.parquet"
    _download(_NEWS_PARQUET, fp)
    table = pq.read_table(fp, columns=["text", "label_text"])
    d = table.to_pydict()
    want = set(NEWS_CATEGORIES)
    idx = [i for i, c in enumerate(d["label_text"]) if c in want]
    random.Random(42).shuffle(idx)
    idx = idx[:limit]
    if not idx:
        raise RuntimeError("newsgroups: no rows matched the target categories")
    X = torch.tensor([encode_chars(d["text"][i], length) for i in idx], dtype=torch.long)
    remap = {c: k for k, c in enumerate(NEWS_CATEGORIES)}
    y = torch.tensor([remap[d["label_text"][i]] for i in idx], dtype=torch.long)
    return X, y, NEWS_CATEGORIES


def newsgroups_lm(limit: int = 2000, length: int = 96):
    """Char-level corpus for the GPT / Seq2Seq models."""
    X, _, _ = newsgroups(limit=limit, length=length)
    return X


# --------------------------------------------------------------------------- #
# Cora
# --------------------------------------------------------------------------- #
CORA_FILES = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
_CORA_BASE = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/ind.cora.{f}"


def cora():
    """(features, labels, normalised adjacency)."""
    import pickle as pkl

    from scipy import sparse

    paths = {}
    for f in CORA_FILES:
        p = CACHE / f"ind.cora.{f}"
        _download(_CORA_BASE.format(f=f), p)
        paths[f] = p

    def read(p):
        with open(p, "rb") as fh:
            return pkl.load(fh, encoding="latin1")

    allx, ally, tx, ty, graph = (read(paths["allx"]), read(paths["ally"]),
                                 read(paths["tx"]), read(paths["ty"]), read(paths["graph"]))
    x = sparse.vstack((allx, tx)).toarray().astype("float32")
    y = np.vstack((ally, ty)).argmax(1).astype("int64")
    n = len(x)
    A = np.eye(n, dtype="float32")
    for i, ns in graph.items():
        for j in ns:
            A[i, j] = 1
            A[j, i] = 1
    A /= np.maximum(A.sum(1, keepdims=True), 1)
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(A)


# --------------------------------------------------------------------------- #
# Breast Cancer Wisconsin (scikit-learn bundled, no network needed)
# --------------------------------------------------------------------------- #
def breast_cancer():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    d = load_breast_cancer()
    x_tr, x_te, y_tr, y_te = train_test_split(d.data, d.target, test_size=0.2,
                                              random_state=42, stratify=d.target)
    sc = StandardScaler().fit(x_tr)
    x_tr = sc.transform(x_tr).astype("float32")
    x_te = sc.transform(x_te).astype("float32")
    return (torch.from_numpy(x_tr), torch.from_numpy(y_tr.astype("int64")),
            torch.from_numpy(x_te), torch.from_numpy(y_te.astype("int64")),
            {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()},
            list(d.target_names))
