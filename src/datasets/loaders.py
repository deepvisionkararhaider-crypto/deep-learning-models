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
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from PIL import Image

from ..imaging import decode_image

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache" / "parquet"
MANIFEST = ROOT / ".cache" / "parquet_manifest.json"
# Path-only image columns (e.g. plant_cls) are fetched from the HF resolve
# endpoint on demand. Cache them on disk so a multi-epoch training run does not
# re-download the same JPEG hundreds of times (and so a resumed run is cheap).
IMG_CACHE = ROOT / ".cache" / "images"


# --------------------------------------------------------------------------- #
# raw parquet access
# --------------------------------------------------------------------------- #
def _manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


# The downloaded files are named ``<split>-<index>.parquet`` from the HuggingFace
# datasets-server split names, which are not spelled consistently across datasets
# (train / validation / valid / val / test). Canonicalise them so a loader can ask
# for one split and be sure it never sees rows from another (that would leak the
# test set into training and make every reported metric meaningless).
_SPLIT_ALIASES = {"train": "train", "training": "train",
                  "validation": "validation", "valid": "validation", "val": "validation",
                  "test": "test"}


def split_of(fp: Path) -> str:
    """Canonical split for a cached parquet file (``valid-2.parquet`` -> ``validation``)."""
    return _SPLIT_ALIASES.get(fp.stem.split("-")[0].lower(), "train")


def files_for(key: str, split: str | None = None) -> list[Path]:
    files = sorted(CACHE.glob(f"{key}/*.parquet"))
    if split is None:
        return files
    want = _SPLIT_ALIASES.get(split.lower(), split.lower())
    return [f for f in files if split_of(f) == want]


def available(key: str) -> bool:
    return bool(files_for(key))


def _rows(key: str, columns: list[str] | None = None, split: str | None = None) -> Iterator[dict]:
    import pyarrow.parquet as pq

    files = files_for(key, split)
    if not files:
        raise FileNotFoundError(
            f"no cached parquet for {key!r} split={split!r}; run scripts/_fetch_data.py {key}"
        )
    for fp in files:
        try:
            pf = pq.ParquetFile(fp)
        except Exception as exc:                       # noqa: BLE001
            raise RuntimeError(f"corrupt or incomplete parquet file {fp.name}: {exc}") from exc
        for batch in pf.iter_batches(batch_size=64, columns=columns):
            d = batch.to_pydict()
            n = len(next(iter(d.values())))
            for i in range(n):
                yield {c: d[c][i] for c in d}


_IMG_FETCH_CACHE: dict[str, bytes] = {}


def _fetch_image(url: str, attempts: int = 4) -> bytes:
    """Download one image with a disk cache + retries.

    The HF resolve endpoint occasionally returns 5xx/timeouts under load; without
    retries a single flaky request would abort an entire training epoch. Cached
    bytes are written atomically so an interrupted run cannot leave a truncated
    file that later reads as a corrupt image.
    """
    import hashlib
    import time
    import urllib.request

    IMG_CACHE.mkdir(parents=True, exist_ok=True)
    fp = IMG_CACHE / hashlib.sha1(url.encode("utf-8")).hexdigest()
    if fp.is_file() and fp.stat().st_size > 0:
        return fp.read_bytes()
    last: Exception | None = None
    for a in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if not data:
                raise RuntimeError("empty image response")
            tmp = fp.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(fp)
            return data
        except Exception as exc:                       # noqa: BLE001
            last = exc
            time.sleep(1.5 * (a + 1))
    raise RuntimeError(f"failed to fetch image after {attempts} attempts: "
                       f"{type(last).__name__}: {last}")


def _resolve_hf_path(path: str) -> str | None:
    """``hf://datasets/<owner>/<name>@<rev>/<file path>`` -> a public resolve URL.

    File paths in these datasets contain spaces and other characters that are
    illegal in a URL, so the file component is percent-encoded (keeping ``/``).
    """
    if not path.startswith("hf://"):
        return None
    body = path[len("hf://"):]
    kind, _, rest = body.partition("/")
    if kind != "datasets":
        return None
    parts = rest.split("/")
    if len(parts) < 3:
        return None
    owner, name = parts[0], parts[1]
    rev = "main"
    if "@" in name:
        name, _, rev = name.partition("@")
    repo = f"{owner}/{name}"
    filepart = "/".join(parts[2:])
    if not rev:
        rev = "main"
    return (f"https://huggingface.co/datasets/{repo}/resolve/"
            f"{urllib.parse.quote(rev, safe='')}/{urllib.parse.quote(filepart)}")


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
            data = _fetch_image(url)
            _IMG_FETCH_CACHE[url] = data
            return data
        raise ValueError("image cell has neither bytes nor a usable path")
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    raise ValueError(f"unsupported image cell type {type(cell)}")


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
# Class names are taken from each dataset's own feature metadata (HuggingFace
# ``ClassLabel`` names), not invented. Where a dataset ships un-named numeric
# labels we say so explicitly instead of guessing a clinical meaning.
BRAIN_MRI_CLASSES = ["glioma", "meningioma", "no-tumor", "pituitary"]   # PranomVignesh/MRI-Images-of-Brain-Tumor
PLANT_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Corn___Cercospora_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight", "Grape___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]

# Detection / instance-segmentation class names, taken verbatim from each dataset's
# own ClassLabel feature metadata. brain_det ships un-named numeric labels (no
# ClassLabel), so its classes are labelled explicitly as numeric ids.
BRAIN_DET_CLASSES = ["label_0", "label_1"]                      # mmenendezg/brain-tumor-object-detection
COTTON_DET_CLASSES = ["cotton-plant-disease", "dc"]             # Francesco/cotton-plant-disease
BRAIN_INSTSEG_CLASSES = ["Tumor", "0"]                          # dwb2023/... category ids 1,2


def _collect_classification(key: str, split: str | None, limit: int | None,
                            seed: int) -> list[tuple[bytes, int]]:
    """Collect (image_bytes, label) rows for ONE split.

    ``split`` must be honoured: reading every cached parquet file regardless of
    the requested split would hand the trainer the test rows as well, so every
    reported metric would be contaminated by train/test leakage.
    """
    out: list[tuple[bytes, int]] = []
    for r in _rows(key, ["image", "label"], split=split):
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


def load_detection(key: str, limit: int | None = None,
                   train_limit: int | None = None) -> dict[str, list[dict]]:
    """Return {split: [ {image, boxes(N,4 px xyxy), labels(N,), width, height} ]}.

    Split assignment is derived from the cached file name via the canonical
    :func:`split_of` (``validation``/``valid``/``val`` all map to ``val``). The
    previous implementation guessed the split with ad-hoc substring checks, which
    could route a validation shard into training and contaminate every metric.
    """
    if not available(key):
        raise FileNotFoundError(f"{key} not cached; run scripts/_fetch_data.py {key}")
    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    import pyarrow.parquet as pq
    for fp in files_for(key):
        canon = split_of(fp)                       # train | validation | test
        split = "val" if canon == "validation" else canon
        if split not in out:
            split = "train"
        cap = train_limit if (split == "train" and train_limit) else limit
        try:
            pf = pq.ParquetFile(fp)
        except Exception as exc:                   # noqa: BLE001
            raise RuntimeError(f"corrupt or incomplete parquet file {fp.name}: {exc}") from exc
        names = pf.schema_arrow.names
        if "image" not in names or "objects" not in names:
            raise RuntimeError(f"{fp.name} lacks required columns (image, objects); got {names}")
        cols = ["image", "objects"] + (["width", "height"] if "width" in names else [])
        for batch in pf.iter_batches(batch_size=32, columns=cols):
            d = batch.to_pydict()
            for j in range(len(d["image"])):
                b = _img_bytes(d["image"][j])
                img = decode_image(b)
                w, h = (d["width"][j], d["height"][j]) if "width" in d else img.size
                xyxy, labels = _objects_to_boxes(d["objects"][j], w, h)
                out[split].append({"image": b, "boxes": xyxy, "labels": labels,
                                   "width": int(w), "height": int(h)})
                if cap and len(out[split]) >= cap:
                    break
            if cap and len(out[split]) >= cap:
                break
    return out


# --------------------------------------------------------------------------- #
# segmentation
# --------------------------------------------------------------------------- #
def load_plant_seg(limit: int | None = None, seed: int = 42) -> dict[str, list[tuple[bytes, bytes]]]:
    """Leaf-disease segmentation: returns {split: [(image_bytes, mask_png_bytes)]}.

    Uses the dataset's own train/validation/test split. The masks are **class-index
    encoded** (0 = background, 38 = disease), not a 0/255 image mask, so they are
    thresholded with ``> 0`` by :func:`mask_array`.
    """
    if not available("plant_seg"):
        raise FileNotFoundError("plant_seg not cached; run scripts/_fetch_data.py plant_seg")
    out: dict[str, list[tuple[bytes, bytes]]] = {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        for r in _rows("plant_seg", ["image", "mask"], split=split):
            out[split].append((_img_bytes(r["image"]), _img_bytes(r["mask"])))
            if limit and len(out[split]) >= limit:
                break
    for k in out:
        random.Random(seed).shuffle(out[k])
    return out


def mask_array(mask_bytes: bytes, size: int) -> np.ndarray:
    """Decode a segmentation mask into a binary (size, size) float32 array.

    Handles both conventions seen in these datasets: a 0/255 image mask (``> 127``)
    and a class-index mask where the foreground is any non-zero label (``> 0``).
    """
    m = Image.open(io.BytesIO(mask_bytes)).convert("L")
    m = m.resize((size, size), Image.NEAREST)
    arr = np.asarray(m, dtype=np.float32)
    if arr.max() > 127:            # 0/255-style mask
        return (arr > 127).astype(np.float32)
    return (arr > 0).astype(np.float32)


# --------------------------------------------------------------------------- #
# instance segmentation (COCO polygons)
# --------------------------------------------------------------------------- #
# Category ids present in dwb2023/brain-tumor-image-dataset-semantic-segmentation.
# The registry declares two classes; map the source ids onto 0-based indices.
INSTSEG_BRAIN_CLASSES = ["glioma", "meningioma"]


def _polygons_to_mask(segmentation: Any, width: int, height: int) -> np.ndarray:
    """Rasterise COCO polygon list(s) into a (H, W) uint8 mask."""
    from PIL import ImageDraw

    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    if not isinstance(segmentation, (list, tuple)):
        return np.zeros((height, width), np.uint8)
    polys = segmentation
    if polys and not isinstance(polys[0], (list, tuple)):
        polys = [polys]                      # single flat polygon -> [[x0,y0,...]]
    for poly in polys:
        try:
            pts = [(float(poly[k]), float(poly[k + 1])) for k in range(0, len(poly) - 1, 2)]
        except Exception:                    # noqa: BLE001
            continue
        if len(pts) >= 3:
            draw.polygon(pts, fill=1)
    return np.asarray(canvas, dtype=np.uint8)


def load_instance_segmentation(key: str = "brain_instseg", limit: int | None = None
                               ) -> dict[str, list[dict]]:
    """Return {split: [ {image, boxes(N,4 px xyxy), labels(N,), masks(N,H,W), width, height} ]}.

    Reads genuine COCO polygon annotations and rasterises them into per-instance
    masks. Annotations are grouped by source image (``file_name``); labels are
    remapped from the dataset's category ids to 0-based class indices.
    """
    if not available(key):
        raise FileNotFoundError(f"{key} not cached; run scripts/_fetch_data.py {key}")
    cols = ["file_name", "image", "category_id", "bbox", "segmentation", "width", "height"]
    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        grouped: dict[str, dict] = {}
        order: list[str] = []
        for r in _rows(key, cols, split=split):
            fn = r["file_name"]
            if fn not in grouped:
                b = _img_bytes(r["image"])
                img = decode_image(b)
                w = int(r.get("width") or img.size[0])
                h = int(r.get("height") or img.size[1])
                grouped[fn] = {"image": b, "boxes": [], "labels": [], "masks": [],
                               "width": w, "height": h}
                order.append(fn)
            g = grouped[fn]
            bb = r["bbox"]
            if bb is None or len(bb) != 4:
                continue                                   # annotation without a box -> skip
            x, y, bw, bh = [float(v) for v in bb]
            if bw <= 0 or bh <= 0:
                continue                                   # degenerate box -> skip
            mask = _polygons_to_mask(r["segmentation"], g["width"], g["height"])
            if mask.max() == 0:
                continue                                   # polygon failed to rasterise -> skip
            g["boxes"].append([x, y, x + bw, y + bh])
            g["labels"].append(int(r["category_id"]) - 1)     # 1-based -> 0-based
            g["masks"].append(mask)
            if limit and len(order) >= limit:
                break
        for fn in order:
            g = grouped[fn]
            g["boxes"] = np.asarray(g["boxes"], dtype=np.float32).reshape(-1, 4)
            g["labels"] = np.asarray(g["labels"], dtype=np.int64).reshape(-1)
            g["masks"] = (np.stack(g["masks"]) if g["masks"]
                          else np.zeros((0, g["height"], g["width"]), np.uint8))
            out[split].append(g)
    return out


def load_brain_seg(limit: int | None = None) -> dict[str, list[tuple[bytes, bytes]]]:
    """Binary brain-tumour segmentation masks derived from the COCO polygons.

    The brain-tumour dataset is only published with per-instance polygons, so the
    binary U-Net target is produced by unioning every instance polygon of an image
    into one foreground mask. The image is re-encoded to PNG bytes so the existing
    :class:`SegmentationDataset` can consume it unchanged.
    """
    inst = load_instance_segmentation("brain_instseg", limit=limit)
    out: dict[str, list[tuple[bytes, bytes]]] = {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        for g in inst[split]:
            union = (g["masks"].sum(0) > 0).astype(np.uint8) if g["masks"].size else \
                np.zeros((g["height"], g["width"]), np.uint8)
            mask_img = Image.fromarray(union * 255)
            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")
            out[split].append((g["image"], buf.getvalue()))
    return out


def split_counts(key: str) -> dict[str, int]:
    """Row counts per canonical split from parquet metadata (no image decode).

    Used to write an auditable split manifest into every artifact so a reviewer
    can confirm the train/val/test sizes without re-running the pipeline.
    """
    import pyarrow.parquet as pq

    counts: dict[str, int] = {}
    for fp in files_for(key):
        canon = split_of(fp)
        try:
            n = pq.ParquetFile(fp).metadata.num_rows
        except Exception:                              # noqa: BLE001
            n = 0
        counts[canon] = counts.get(canon, 0) + int(n)
    return counts


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
