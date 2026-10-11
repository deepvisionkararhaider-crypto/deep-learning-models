"""Image preprocessing shared by the training pipeline and the Streamlit runtime.

The single rule that prevents train/inference skew: **both** paths call
``to_tensor`` / ``make_transform`` from this module, and the exact normalisation
constants are stored inside each artifact's metadata (``mean``/``std``/``size``).
At inference time we read those constants back out of the artifact instead of
re-deriving them, so a model trained with one normalisation can never be fed a
different one.

No PIL/numpy -> torch magic happens anywhere else in the codebase.
"""
from __future__ import annotations

import io
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image

# ImageNet statistics: the default for transfer-learned backbones.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def decode_image(data: bytes | bytearray | str, mode: str = "RGB") -> Image.Image:
    """Decode raw bytes (parquet ``image.bytes``) or a path into a PIL image."""
    if isinstance(data, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(data))).convert(mode)
    return Image.open(data).convert(mode)


def load_image(obj, mode: str = "RGB") -> Image.Image:
    """Accept a PIL image, a numpy array, raw bytes or a file-like object."""
    if isinstance(obj, Image.Image):
        return obj.convert(mode)
    if isinstance(obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(obj))).convert(mode)
    if isinstance(obj, np.ndarray):
        arr = obj
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype("uint8")
        return Image.fromarray(arr).convert(mode)
    return Image.open(obj).convert(mode)


def resize(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.BILINEAR)


def normalize_array(arr: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """arr: HxWxC float in [0,1] -> normalised float32 HxWxC."""
    m = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
    s = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
    return ((arr - m) / s).astype(np.float32)


def to_tensor(img: Image.Image, size: int, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """PIL image -> (1, C, size, size) normalised tensor. The one true path."""
    img = img.convert("RGB")
    if size and img.size != (size, size):
        img = resize(img, size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = normalize_array(arr, mean, std)
    chw = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def denormalize(tensor: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """(1,C,H,W) normalised tensor -> HxWxC uint8 for display."""
    arr = tensor.detach().cpu().numpy()[0]
    arr = np.transpose(arr, (1, 2, 0))
    m = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
    s = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
    arr = arr * s + m
    return (np.clip(arr, 0, 1) * 255).astype("uint8")


def mask_to_pil(mask: np.ndarray, size: int | None = None) -> Image.Image:
    """Binary/soft mask (HxW float in [0,1]) -> grayscale PIL for overlay."""
    m = (np.clip(mask, 0, 1) * 255).astype("uint8")
    img = Image.fromarray(m)
    if size:
        img = img.resize((size, size), Image.NEAREST)
    return img


def overlay_mask(base: Image.Image, mask: np.ndarray, color=(255, 64, 64), alpha: float = 0.45) -> Image.Image:
    """Blend a red mask over the base image (both resized to the mask's HxW)."""
    base = base.convert("RGB")
    h, w = mask.shape
    base = base.resize((w, h), Image.BILINEAR)
    b = np.asarray(base, dtype=np.float32)
    m = np.clip(mask, 0, 1)[..., None]
    col = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    blended = b * (1 - alpha * m) + col * (alpha * m)
    return Image.fromarray(blended.astype("uint8"))


def draw_boxes(base: Image.Image, boxes: Iterable[dict], class_names: Sequence[str] | None = None,
               palette: Sequence[tuple[int, int, int]] | None = None) -> Image.Image:
    """Draw detection boxes + labels onto a copy of the image."""
    from PIL import ImageDraw

    out = base.convert("RGB").copy()
    drw = ImageDraw.Draw(out)
    default = [(255, 59, 48), (52, 199, 89), (0, 122, 255), (255, 149, 0),
               (175, 82, 222), (255, 214, 10), (90, 200, 250)]
    pal = list(palette) if palette else default
    for b in boxes:
        c = pal[int(b.get("class", 0)) % len(pal)]
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        drw.rectangle([x1, y1, x2, y2], outline=c, width=3)
        label = str(b.get("label", b.get("class", "")))
        if b.get("score") is not None:
            label = f"{label} {float(b['score']):.2f}"
        if label:
            tw = drw.textlength(label)
            drw.rectangle([x1, max(0, y1 - 14), x1 + tw + 6, y1], fill=c)
            drw.text((x1 + 3, max(0, y1 - 13)), label, fill=(255, 255, 255))
    return out


def make_transform(size: int, mean: Sequence[float] = IMAGENET_MEAN,
                   std: Sequence[float] = IMAGENET_STD, train: bool = False,
                   hflip: bool = True):
    """Return a torchvision transform consistent with :func:`to_tensor`.

    Only geometric augmentation is added for training; photometric statistics are
    never changed, so the normalisation matches inference exactly.
    """
    from torchvision import transforms

    ops = []
    if train:
        ops.append(transforms.Resize((size, size)))
        if hflip:
            ops.append(transforms.RandomHorizontalFlip())
        ops.append(transforms.RandomRotation(12))
        ops.append(transforms.ColorJitter(0.15, 0.15, 0.1))
    else:
        ops.append(transforms.Resize((size, size)))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=list(mean), std=list(std)))
    return transforms.Compose(ops)
