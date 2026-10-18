"""Dynamic prediction forms generated from model metadata (§8 / §9).

Each model declares its own input contract (modality, task, size, classes,
limits). The UI renders the *correct* control for that contract automatically —
an image uploader for vision models, a text area for text models, a feature form
for tabular models, a sequence box for sensor models. A generic "one input for
everything" form is explicitly avoided, and a model whose schema is missing is
reported as **not ready** rather than being given invented fields.

Every form function returns a ``(payload, submitted)`` tuple:

* ``payload`` is a dict of validated inputs (or ``None`` when invalid/unsubmitted)
* ``submitted`` is True only when the visitor pressed the Predict button

Validation is done on the bytes/values, never on the filename or the browser
MIME type.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from ..registry import ModelSpec

#: Guard rails (documented, not magic): upload and free-text limits.
MAX_IMAGE_MB = 15.0
MAX_TEXT_CHARS = 20_000
MAX_SEQUENCE_VALUES = 20_000

IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"]


def form_kind(spec: ModelSpec, entry: dict[str, Any] | None = None) -> str:
    """Which UI form this model needs, from its manifest/registry contract."""
    e = entry or {}
    modality = (e.get("input") or {}).get("modality") or e.get("modality")
    if spec.area == "lab":
        from ..lab_inference import INPUT_KIND
        return INPUT_KIND.get(spec.id, "image")
    if modality == "image" or spec.task in ("classification", "segmentation",
                                            "detection", "instance_segmentation"):
        return "image"
    return modality or "image"


def schema(spec: ModelSpec, entry: dict[str, Any] | None = None) -> dict[str, Any]:
    """The declared input contract, used for help text and validation."""
    e = entry or {}
    inp = e.get("input") or {}
    return {
        "modality": inp.get("modality") or e.get("modality") or ("image" if spec.in_ch else "features"),
        "task": spec.task,
        "size": int(inp.get("size") or spec.input_size or 224),
        "channels": int(inp.get("channels") or spec.in_ch or 3),
        "formats": IMAGE_TYPES if spec.in_ch else [],
        "classes": e.get("classes") or spec.classes,
        "max_image_mb": MAX_IMAGE_MB,
        "max_text_chars": MAX_TEXT_CHARS,
        "feature_names": (e.get("input") or {}).get("feature_names"),
        "limitations": e.get("limitations") or spec.notes,
    }


# --------------------------------------------------------------------------- #
# image
# --------------------------------------------------------------------------- #
def image_form(spec: ModelSpec, *, key: str, allow_pairs: bool = False) -> tuple[dict | None, bool]:
    """Drag-and-drop image uploader with preview, size and decode validation."""
    s = schema(spec)
    st.caption(f"Accepted formats: {', '.join(s['formats'])} · max {s['max_image_mb']:.0f} MB · "
               f"resized to {s['size']}×{s['size']} at inference")

    if allow_pairs:
        c1, c2 = st.columns(2)
        a = c1.file_uploader("Image A", type=IMAGE_TYPES, key=f"{key}_a")
        b = c2.file_uploader("Image B", type=IMAGE_TYPES, key=f"{key}_b")
        if a and b:
            _preview_pair(a, b)
        submitted = st.button("Compare", type="primary", key=f"{key}_go", disabled=not (a and b))
        if not submitted:
            return None, False
        ok_a, err_a = _validate_image(a)
        ok_b, err_b = _validate_image(b)
        if not (ok_a and ok_b):
            st.error(err_a or err_b)
            return None, True
        return {"image": a.getvalue(), "image_b": b.getvalue()}, True

    up = st.file_uploader("Upload an image", type=IMAGE_TYPES, key=key)
    if up is None:
        st.info("Upload an image to run inference. The image is processed in this session only "
                "and is never stored.")
        return None, False

    ok, err = _validate_image(up)
    if not ok:
        st.error(err)
        return None, False
    st.image(up.getvalue(), caption="Preview", width=260)

    submitted = st.button("Predict", type="primary", key=f"{key}_go")
    if not submitted:
        return None, False
    return {"image": up.getvalue()}, True


def _validate_image(upload) -> tuple[bool, str]:
    from ..services.prediction_service import image_bytes_ok
    return image_bytes_ok(upload.getvalue(), max_mb=MAX_IMAGE_MB)


def _preview_pair(a, b):
    c1, c2 = st.columns(2)
    c1.image(a.getvalue(), caption="Image A", use_container_width=True)
    c2.image(b.getvalue(), caption="Image B", use_container_width=True)


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #
def text_form(spec: ModelSpec, *, key: str, example: str = "",
              label: str = "Text") -> tuple[dict | None, bool]:
    txt = st.text_area(label, value=example, height=140, key=key,
                       help=f"Max {MAX_TEXT_CHARS:,} characters.")
    submitted = st.button("Predict", type="primary", key=f"{key}_go")
    if not submitted:
        return None, False
    if not txt or not txt.strip():
        st.error("Please enter some text before predicting.")
        return None, True
    if len(txt) > MAX_TEXT_CHARS:
        st.error(f"Text is too long ({len(txt):,} chars); the limit is {MAX_TEXT_CHARS:,}.")
        return None, True
    return {"text": txt.strip()}, True


# --------------------------------------------------------------------------- #
# tabular / sequence
# --------------------------------------------------------------------------- #
def _parse_numbers(raw: str) -> list[float]:
    vals: list[float] = []
    for chunk in raw.replace("\n", ",").replace(" ", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals.append(float(chunk))
    return vals


def tabular_form(spec: ModelSpec, *, key: str, n_features: int,
                 example: str | None = None) -> tuple[dict | None, bool]:
    """A feature form driven by the model's declared feature count.

    Features are entered as a validated vector; ordering is preserved exactly as
    the trainer expected it (no re-ordering, no invented column names).
    """
    st.caption(f"This model expects exactly **{n_features}** numeric features, in the same order "
               "used during training.")
    default = example if example is not None else ",".join(["0.0"] * min(n_features, 30))
    raw = st.text_area(f"{n_features} comma-separated values", default, height=110, key=key)
    submitted = st.button("Predict", type="primary", key=f"{key}_go")
    if not submitted:
        return None, False
    try:
        vals = _parse_numbers(raw)
    except ValueError:
        st.error("Could not parse those values — use comma- or space-separated numbers.")
        return None, True
    if len(vals) != n_features:
        st.error(f"Expected {n_features} values but received {len(vals)}.")
        return None, True
    return {"values": vals}, True


def sequence_form(spec: ModelSpec, *, key: str, length: int, channels: int) -> tuple[dict | None, bool]:
    n = length * channels
    st.caption(f"Paste **{length}×{channels} = {n}** numeric values (whitespace or comma separated), "
               "or upload a single-column CSV.")
    csv = st.file_uploader("…or upload a CSV", type=["csv"], key=f"{key}_csv")
    raw = ""
    if csv is not None:
        try:
            text = csv.getvalue().decode("utf-8", errors="strict")
            raw = ",".join(text.replace("\n", ",").split(","))
            st.success(f"Loaded {len(_parse_numbers(raw))} values from {csv.name}")
        except Exception as exc:                                    # noqa: BLE001
            st.error(f"Could not read that CSV: {exc}")
            return None, True
    else:
        raw = st.text_area("sequence values", "", height=130, key=key)

    submitted = st.button("Predict", type="primary", key=f"{key}_go")
    if not submitted:
        return None, False
    try:
        vals = _parse_numbers(raw)
    except ValueError:
        st.error("Could not parse those values — expected numbers only.")
        return None, True
    if len(vals) != n:
        st.error(f"Expected {n} values ({length}×{channels}) but received {len(vals)}.")
        return None, True
    return {"values": vals}, True


def noise_slider(key: str) -> float:
    return st.slider("Noise level", 0.0, 1.0, 0.5, 0.05, key=key)


def build(spec: ModelSpec, entry: dict[str, Any] | None = None, *, key: str,
          example: str | None = None) -> tuple[dict | None, bool]:
    """Render the correct form for a model and return its validated payload.

    Raises :class:`ValueError` when the model's schema is unknown — the caller
    must then mark it not-ready instead of inventing input fields.
    """
    kind = form_kind(spec, entry)
    if kind == "image":
        return image_form(spec, key=key)
    if kind == "image_pair":
        return image_form(spec, key=key, allow_pairs=True)
    if kind == "text":
        return text_form(spec, key=key, label="Text to classify",
                         example=example or "The space shuttle launched successfully into orbit.")
    if kind == "text_prompt":
        return text_form(spec, key=key, label="Prompt",
                         example=example or "the mission control team reported")
    if kind == "tabular":
        n = int((entry or {}).get("input", {}).get("features")
                or spec.build_kwargs.get("in_features") or spec.build_kwargs.get("dim") or 30)
        return tabular_form(spec, key=key, n_features=n, example=example)
    if kind == "graph":
        n = int(spec.build_kwargs.get("in_features") or 1433)
        return tabular_form(spec, key=key, n_features=n, example=example)
    if kind == "tabular_denoise":
        n = int(spec.build_kwargs.get("dim") or 30)
        payload, submitted = tabular_form(spec, key=key, n_features=n, example=example)
        if payload is not None:
            payload["noise"] = noise_slider(f"{key}_noise")
        return payload, submitted
    if kind == "sequence":
        length = int(spec.build_kwargs.get("length") or 128)
        channels = int(spec.build_kwargs.get("feat") or 9)
        return sequence_form(spec, key=key, length=length, channels=channels)
    raise ValueError(f"no form is defined for input kind {kind!r} (model {spec.id})")


__all__ = ["form_kind", "schema", "build", "image_form", "text_form", "tabular_form",
           "sequence_form", "MAX_IMAGE_MB", "MAX_TEXT_CHARS", "IMAGE_TYPES"]
