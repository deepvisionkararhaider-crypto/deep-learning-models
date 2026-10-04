"""
YOLO — Interactive Object Detection Demo (Streamlit)
====================================================
Self-contained, lightweight detector for the public Streamlit lab. It reproduces
the *idea* of the from-scratch YOLO in `16_yolo/model.py` (single-stage grid
detection + NMS) using OpenCV's classic blob detector on a grid, so the public
demo runs fast without shipping a trained deep model. The full training script
lives in `16_yolo/model.py`.
"""

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter

try:
    import cv2
    _CV2 = True
except Exception:
    _CV2 = False

CLASS_COLORS = {0: (231, 76, 60), 1: (46, 204, 113), 2: (52, 152, 219)}
CLASS_NAMES = {0: "circle", 1: "square", 2: "triangle"}


def _make_demo_image(seed: int, size: int = 256) -> Image.Image:
    rng = np.random.RandomState(seed)
    img = Image.new("RGB", (size, size), (18, 20, 28))
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(2, 6)):
        r = rng.randint(16, 40)
        cx, cy = rng.randint(r, size - r), rng.randint(r, size - r)
        shape = rng.randint(0, 3)
        color = tuple(int(c) for c in rng.randint(120, 255, 3))
        if shape == 0:
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        elif shape == 1:
            draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
        else:
            draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)
    return img.filter(ImageFilter.GaussianBlur(1.2))


def detect_objects(img: Image.Image, min_area: int = 120, max_objects: int = 12):
    """Single-stage grid/blob detector -> list of (x1,y1,x2,y2,score)."""
    gray = np.array(img.convert("L"))
    boxes = []
    if _CV2:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < min_area:
                continue
            boxes.append((x, y, x + w, y + h, min(0.99, 0.5 + area / (gray.size * 2))))
    else:
        # Pure-numpy fallback: threshold + connected components via scipy-free labeling
        from scipy import ndimage  # optional
        mask = gray > gray.mean()
        lbl, n = ndimage.label(mask)
        for i in range(1, n + 1):
            ys, xs = np.where(lbl == i)
            if len(xs) < min_area:
                continue
            boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()),
                          min(0.99, 0.5 + len(xs) / (gray.size * 2))))
    boxes.sort(key=lambda b: b[4], reverse=True)
    return boxes[:max_objects]


def nms(boxes, thr=0.3):
    keep = []
    for b in boxes:
        if all(_iou(b, k) < thr for k in keep):
            keep.append(b)
    return keep


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-8)


def annotate(img: Image.Image, boxes):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for i, (x1, y1, x2, y2, score) in enumerate(boxes):
        color = list(CLASS_COLORS.values())[i % 3]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 2, max(0, y1 - 12)), f"{CLASS_NAMES[i % 3]} {score:.2f}", fill=color)
    return out


def render_yolo():
    st.title("YOLO — Single-Stage Object Detection")
    st.write(
        "A one-stage detector predicts bounding boxes and classes directly from the "
        "image in a single forward pass (then NMS removes duplicates). This interactive "
        "demo runs a fast blob/grid detector; the full from-scratch training pipeline "
        "(grid targets, YOLO loss, NMS, IoU evaluation) lives in `16_yolo/model.py`."
    )

    mode = st.radio("Image source", ["Synthetic sample", "Upload an image"], horizontal=True)
    if mode == "Upload an image":
        up = st.file_uploader("Upload PNG/JPG", type=["png", "jpg", "jpeg"])
        if not up:
            st.info("Upload an image to run detection.")
            return
        img = Image.open(up).convert("RGB")
    else:
        seed = st.slider("Sample seed", 0, 99, 7)
        img = _make_demo_image(seed)

    conf = st.slider("Confidence threshold", 0.0, 1.0, 0.45, 0.05)
    raw = detect_objects(img)
    kept = nms([b for b in raw if b[4] >= conf])

    c1, c2, c3 = st.columns(3)
    c1.metric("Raw candidates", len(raw))
    c2.metric("After NMS", len(kept))
    c3.metric("Detector", "grid/blob + NMS")

    left, right = st.columns(2)
    with left:
        st.image(img, caption="Input", use_container_width=True)
    with right:
        st.image(annotate(img, kept), caption="Detections", use_container_width=True)

    if kept:
        st.dataframe(
            pd.DataFrame(
                [{"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "confidence": round(b[4], 3)}
                 for b in kept]
            ),
            use_container_width=True,
        )
    else:
        st.warning("No objects above the confidence threshold.")

    st.caption("Full model + metrics: 16_yolo/model.py · predictions.csv · plots/yolo_detections.png")
