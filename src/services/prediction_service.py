"""Uniform, cached prediction calls.

Every task goes through :func:`run`, which returns a normalised envelope::

    {"ok": bool, "kind": str, "result": {...}, "error": str|None, "model_id": str}

Normalising here means the UI never has to guess what a predictor returned, and a
failure is always a value (never an exception escaping into a Streamlit traceback).

Predictions are **not** cached by data value: images and text are visitor-specific
and caching them would grow memory without bound. Only the *model* is cached (see
:mod:`src.services.model_loader`).
"""
from __future__ import annotations

from typing import Any

from ..registry import ModelSpec


def _envelope(spec: ModelSpec, kind: str, result: dict | None, error: str | None) -> dict[str, Any]:
    return {"ok": error is None, "kind": kind, "result": result or {},
            "error": error, "model_id": spec.id}


def run(spec: ModelSpec, *, image=None, **kwargs) -> dict[str, Any]:
    """Route to the right predictor for the spec's task and normalise the result."""
    from .. import lab_inference as LI

    try:
        if spec.area == "lab":
            kind, result = _run_lab(spec, LI, image=image, **kwargs)
        else:
            from .. import inference as I
            result = I.predict(spec, image, **kwargs)
            kind = result.get("kind", spec.task)
        return _envelope(spec, kind, result, None)
    except Exception as exc:                                        # noqa: BLE001
        return _envelope(spec, spec.task, None, f"{type(exc).__name__}: {exc}")


def _run_lab(spec: ModelSpec, LI, *, image=None, text: str | None = None,
             values: list[float] | None = None, noise: float | None = None,
             image_b=None, **_) -> tuple[str, dict]:
    """Per-architecture dispatch for the 20 laboratory models.

    The input kind comes from :data:`src.lab_inference.INPUT_KIND`, so the model's
    real contract decides which predictor runs — no guessing from the model name.
    """
    kind = LI.INPUT_KIND.get(spec.id, "image")

    if kind == "tabular":
        return kind, LI.predict_tabular(spec, values or [])
    if kind == "tabular_denoise":
        return kind, LI.denoise_tabular(spec, values or [], float(noise if noise is not None else 0.5))
    if kind == "graph":
        return kind, LI.gnn_predict(spec, values or [])
    if kind == "sequence":
        return kind, LI.predict_sequence(spec, values or [])
    if kind == "text":
        return kind, LI.text_classify(spec, text or "")
    if kind == "text_prompt":
        return kind, LI.generate_text(spec, text or "")
    if kind == "image_pair":
        return kind, LI.siamese_pair(spec, image, image_b)

    # single-image lab models, by their actual inference function
    if spec.id in ("lab02", "lab12", "lab13", "lab14"):
        return kind, LI.predict_image_class(spec, image)
    if spec.id in ("lab06", "lab07"):
        return kind, LI.reconstruct(spec, image)
    if spec.id == "lab08":
        return kind, LI.gan_score(spec, image)
    if spec.id == "lab15":
        return kind, LI.segmentation_28(spec, image)
    if spec.id == "lab16":
        return kind, LI.detection_28(spec, image)
    raise ValueError(f"no lab predictor wired for {spec.id} (input kind {kind!r})")


def image_bytes_ok(data: bytes, *, max_mb: float = 15.0) -> tuple[bool, str]:
    """Cheap upload validation: size and a real decode attempt.

    Never trust the filename or the browser-reported MIME type — the bytes are
    decoded, and a failure is reported rather than passed to the model.
    """
    if not data:
        return False, "The uploaded file is empty."
    if len(data) > max_mb * 1024 * 1024:
        return False, f"Image is larger than the {max_mb:.0f} MB limit."
    from ..imaging import load_image
    try:
        load_image(data)
    except Exception as exc:                                        # noqa: BLE001
        return False, f"That file is not a readable image ({type(exc).__name__})."
    return True, ""


__all__ = ["run", "image_bytes_ok"]
