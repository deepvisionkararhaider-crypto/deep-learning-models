"""Real public dataset loaders.

All datasets are fetched from the HuggingFace *datasets-server* parquet mirror
(no ``datasets``/``pyarrow``-heavy runtime dependency is needed for inference; the
loaders below are used only by the offline trainer and the CI data-contract test).

Datasets actually used (see DATASETS.md for licences):

======================  =========================================================  ================
key                     HuggingFace dataset                                        task
======================  =========================================================  ================
brain_mri_cls           PranomVignesh/MRI-Images-of-Brain-Tumor                   4-class MRI
plant_cls               Dipra444/plant-leaf-disease-jharkhand                     25-class leaf
brain_det               mmenendezg/brain-tumor-object-detection                   boxes (COCO)
cotton_det              Francesco/cotton-plant-disease                            boxes (COCO)
plant_seg               Mauregato/leaf_disease_segmentation                       binary masks
brain_tumor_instseg     dwb2023/brain-tumor-image-dataset-semantic-segmentation   COCO polygons
======================  =========================================================  ================
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from PIL import Image

from ..imaging import decode_image

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache" / "parquet"
MANIFEST = ROOT / ".cache" / "parquet_manifest.json"


# --------------------------------------------------------------------------- #
# raw parquet access
# --------------------------------------------------------------------------- #
def _manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def files_for(key: str) -> list[Path]:
    return sorted(CACHE.glob(f"{key}/*.parquet"))


def available(key: str) -> bool:
    return bool(files_for(key))


def _rows(key: str, columns: list[str] | None = None) -> Iterator[dict]:
    import pyarrow.parquet as pq

    for fp in files_for(key):
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(batch_size=64, columns=columns):
            d = batch.to_pydict()
            n = len(next(iter(d.values())))
            for i in range(n):
                yield {c: d[c][i] for c in d}


_IMG_FETCH_CACHE: dict[str, bytes] = {}


def _resolve_hf_path(path: str) -> str | None:
    """``hf://datasets/REPO@REV/FILE`` -> a public resolve URL (path-only datasets)."""
    if not path.startswith("hf://"):
        return None
    body = path[len("hf://"):]
    kind, _, rest = body.partition("/")
    if kind != "datasets":
        return None
    repo, _, tail = rest.partition("/")
    rev, sep, filepart = tail.partition("/")
    if not sep:
        return None
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{filepart}"


def _img_bytes(cell: Any) -> bytes:
    """Parquet image column is {'bytes': b'..', 'path': '..'}.

    Some datasets store only a path (``hf://datasets/...``); those are fetched on
    demand from the HuggingFace resolve endpoint and cached in memory.
    """
    if isinstance(cell, dict):
        b = cell.get("bytes")
        if b is not None:
            return bytes(b)
        path = cell.get("path")
        if path and path != "None":
            url = _resolve_hf_path(path)
            if url is None:
                raise ValueError(f"cannot resolve image path {path!r}")
            if url in _IMG_FETCH_CACHE:
                return _IMG_FETCH_CACHE[url]
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            _IMG_FETCH_CACHE[url] = data
            return data
        raise ValueError("image cell has neither bytes nor a usable path")
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    raise ValueError(f"unsupported image cell type {type(cell)}")


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
BRAIN_MRI_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
PLANT_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Corn___Cercospora_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight", "Grape___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]


def _collect_classification(key: str, split: str | None, limit: int | None,
                            seed: int) -> list[tuple[bytes, int]]:
    out: list[tuple[bytes, int]] = []
    for r in _rows(key, ["image", "label"]):
        out.append((_img_bytes(r["image"]), int(r["label"])))
        if limit and len(out) >= limit:
            break
    rng = random.Random(seed)
    rng.shuffle(out)
    return out


def stratified_split(items: list[tuple[bytes, int]], fractions=(0.7, 0.15, 0.15),
                     seed: int = 42) -> dict[str, list[tuple[bytes, int]]]:
    """Class-stratified train/val/test split with a fixed seed (saved by callers)."""
    rng = random.Random(seed)
    by_class: dict[int, list[tuple[bytes, int]]] = {}
    for it in items:
        by_class.setdefault(it[1], []).append(it)
    splits = {"train": [], "val": [], "test": []}
    for _c, group in by_class.items():
        group = group[:]
        rng.shuffle(group)
        n = len(group)
        n_tr = int(n * fractions[0])
        n_va = int(n * fractions[1])
        splits["train"] += group[:n_tr]
        splits["val"] += group[n_tr:n_tr + n_va]
        splits["test"] += group[n_tr + n_va:]
    for k in splits:
        rng.shuffle(splits[k])
    return splits


def load_brain_mri(limit: int | None = None, seed: int = 42) -> dict[str, list[tuple[bytes, int]]]:
    """4-class brain MRI. Uses the dataset's own train/val/test split (patient-safe)."""
    if not available("brain_mri_cls"):
        raise FileNotFoundError("brain_mri_cls not cached; run scripts/_fetch_data.py brain_mri_cls")
    return {
        "train": _collect_classification("brain_mri_cls", "train", limit, seed),
        "val": _collect_classification("brain_mri_cls", "validation", limit, seed),
        "test": _collect_classification("brain_mri_cls", "test", limit, seed),
    }


def load_plant_cls(limit: int | None = None, seed: int = 42) -> dict[str, list[tuple[bytes, int]]]:
    """25-class plant leaf disease. Single split -> stratified split (seed=42)."""
    if not available("plant_cls"):
        raise FileNotFoundError("plant_cls not cached; run scripts/_fetch_data.py plant_cls")
    items = _collect_classification("plant_cls", None, limit, seed)
    return stratified_split(items, seed=seed)


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def _objects_to_boxes(obj: Any, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """COCO objects dict -> (xyxy pixel boxes, labels)."""
    if not isinstance(obj, dict) or not obj.get("bbox"):
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
    bboxes = np.asarray(obj["bbox"], dtype=np.float32).reshape(-1, 4)   # x,y,w,h
    labels = obj.get("label", obj.get("category"))
    labels = np.asarray(labels, dtype=np.int64).reshape(-1) if labels is not None else np.zeros(len(bboxes), np.int64)
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 0] + bboxes[:, 2]
    y2 = bboxes[:, 1] + bboxes[:, 3]
    xyxy = np.stack([x1, y1, x2, y2], axis=1)
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height)
    return xyxy, labels


def load_detection(key: str, limit: int | None = None) -> dict[str, list[dict]]:
    """Return {split: [ {image, boxes(N,4 px), labels(N,), width, height} ]}."""
    if not available(key):
        raise FileNotFoundError(f"{key} not cached; run scripts/_fetch_data.py {key}")
    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    files = files_for(key)
    for i, fp in enumerate(files):
        split = "train" if i == len(files) - 1 or "train" in fp.name else ("test" if "test" in fp.name else "val")
        if "test" in fp.name:
            split = "test"
        elif "validation" in fp.name or "val" in fp.name:
            split = "val"
        else:
            split = "train"
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(fp)
        cols = ["image", "objects"] + (["width", "height"] if "width" in pf.schema_arrow.names else [])
        for batch in pf.iter_batches(batch_size=32, columns=cols):
            d = batch.to_pydict()
            for j in range(len(d["image"])):
                b = _img_bytes(d["image"][j])
                img = decode_image(b)
                w, h = (d["width"][j], d["height"][j]) if "width" in d else img.size
                xyxy, labels = _objects_to_boxes(d["objects"][j], w, h)
                out[split].append({"image": b, "boxes": xyxy, "labels": labels,
                                   "width": int(w), "height": int(h)})
                if limit and len(out[split]) >= limit:
                    break
    return out


# --------------------------------------------------------------------------- #
# segmentation
# --------------------------------------------------------------------------- #
def load_plant_seg(limit: int | None = None, seed: int = 42) -> dict[str, list[tuple[bytes, bytes]]]:
    """Binary leaf-disease segmentation: returns {split: [(image_bytes, mask_png_bytes)]}."""
    if not available("plant_seg"):
        raise FileNotFoundError("plant_seg not cached; run scripts/_fetch_data.py plant_seg")
    items: list[tuple[bytes, bytes]] = []
    for r in _rows("plant_seg", ["image", "mask"]):
        items.append((_img_bytes(r["image"]), _img_bytes(r["mask"])))
        if limit and len(items) >= limit:
            break
    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
    return {"train": items[: int(n * 0.8)], "val": items[int(n * 0.8): int(n * 0.9)],
            "test": items[int(n * 0.9):]}


def mask_array(mask_bytes: bytes, size: int) -> np.ndarray:
    """Decode a mask PNG into a binary (size, size) float32 array."""
    m = Image.open(io.BytesIO(mask_bytes)).convert("L")
    m = m.resize((size, size), Image.NEAREST)
    return (np.asarray(m, dtype=np.float32) > 127).astype(np.float32)


# --------------------------------------------------------------------------- #
# torch Dataset wrappers
# --------------------------------------------------------------------------- #
class ImageClassificationDataset:
    """Torch Dataset over (bytes, label) pairs using the shared transform."""

    def __init__(self, items: list[tuple[bytes, int]], transform: Callable, classes: list[str]):
        self.items = items
        self.transform = transform
        self.classes = classes

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        b, y = self.items[idx]
        return self.transform(decode_image(b)), int(y)


class SegmentationDataset:
    def __init__(self, items: list[tuple[bytes, bytes]], size: int,
                 mean, std, train: bool = False):
        from ..imaging import make_transform
        self.items = items
        self.size = size
        self.transform = make_transform(size, mean, std, train=train, hflip=True)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_b, mask_b = self.items[idx]
        x = self.transform(decode_image(img_b))
        m = mask_array(mask_b, self.size)
        return x, m[None, :, :]      # (1,H,W)
