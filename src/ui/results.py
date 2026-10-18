"""Result rendering for every task kind — one place, consistent output.

Consumes the normalised envelope from
:func:`src.services.prediction_service.run`::

    {"ok": bool, "kind": str, "result": {...}, "error": str|None, "model_id": str}

Confidence is always presented as a **model score**, never as a calibrated
probability, and every medical/agricultural result carries its disclaimer.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from ..registry import ModelSpec
from .common import (confusion_matrix_view, metric_card, model_disclaimer, show_metrics,
                     show_provenance)


def render(spec: ModelSpec, envelope: dict, meta: dict | None = None) -> None:
    """Render a prediction envelope, or an honest error."""
    if not envelope.get("ok"):
        st.error(f"Inference failed — {envelope.get('error')}")
        st.caption("This usually means the checkpoint is missing or does not match the current "
                   "architecture. Retrain with "
                   f"`python scripts/train_models.py --model {spec.id}` and republish.")
        return

    kind = envelope.get("kind") or spec.task
    res = envelope.get("result") or {}
    ms = res.get("inference_ms")
    st.success(f"Done{f' in {ms:.0f} ms' if ms else ''}.")

    if kind == "classification":
        _classification(res, spec)
    elif kind == "segmentation":
        _segmentation(res)
    elif kind == "detection":
        _detection(res, spec)
    elif kind == "instance_segmentation":
        _instances(res, spec)
    elif kind in ("tabular", "tabular_denoise", "graph", "sequence"):
        _vector_task(kind, res)
    elif kind in ("text", "text_prompt"):
        _text_task(kind, res)
    elif kind == "image_pair":
        _pair(res)
    elif kind == "reconstruction":
        _reconstruction(res)
    else:
        st.json(res)

    st.divider()
    _evaluation(spec, meta, res)


# --------------------------------------------------------------------------- #
# per-task renderers
# --------------------------------------------------------------------------- #
def _classification(res: dict, spec: ModelSpec) -> None:
    probs = res.get("probabilities") or {}
    top = res.get("top_k") or []
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Predicted class", str(res.get("predicted", "—")))
    with c2:
        conf = res.get("confidence")
        metric_card("Model score", f"{conf:.1%}" if isinstance(conf, float) else "—",
                    "not a calibrated probability")
    with c3:
        metric_card("Classes", str(len(probs) or len(spec.classes)))
    if probs:
        ordered = dict(sorted(probs.items(), key=lambda kv: -kv[1]))
        st.markdown("**Score breakdown**")
        st.bar_chart(ordered)
        if len(top) > 1:
            with st.expander("Top-k predictions", expanded=False):
                for t in top:
                    st.write(f"{t['label']} — {t['prob']:.4f}")
    st.caption("Confidence is an uncalibrated model score; it is not the probability that the "
               "prediction is correct.")


def _segmentation(res: dict) -> None:
    mask = res.get("mask")
    overlay = res.get("overlay")
    c1, c2, c3 = st.columns(3)
    with c1:
        if res.get("display_image") is not None:
            st.image(res["display_image"], caption="Input", use_container_width=True)
    with c2:
        if mask is not None:
            st.image((np.asarray(mask) * 255).astype("uint8"), caption="Predicted mask",
                     clamp=True, use_container_width=True)
            st.download_button("Download mask (PNG)", _png_bytes(mask), "mask.png", "image/png")
    with c3:
        if overlay is not None:
            st.image(overlay, caption="Overlay", use_container_width=True)
    metric_card("Foreground ratio", f"{res.get('foreground_ratio', 0):.2%}")


def _detection(res: dict, spec: ModelSpec) -> None:
    from ..imaging import draw_boxes
    boxes = res.get("boxes") or []
    c1, c2 = st.columns(2)
    with c1:
        if res.get("display_image") is not None:
            st.image(res["display_image"], caption="Input", use_container_width=True)
    with c2:
        if res.get("display_image") is not None:
            annotated = draw_boxes(res["display_image"].copy(), boxes, spec.classes)
            st.image(annotated, caption=f"{len(boxes)} detection(s)", use_container_width=True)
    if boxes:
        st.dataframe([{k: (round(v, 1) if isinstance(v, float) else v) for k, v in b.items()}
                      for b in boxes], use_container_width=True, hide_index=True)
    else:
        st.info("No objects passed the confidence threshold. Try a clearer image or lower the "
                "threshold.")


def _instances(res: dict, spec: ModelSpec) -> None:
    from ..imaging import draw_boxes
    inst = res.get("instances") or []
    if res.get("display_image") is not None:
        annotated = draw_boxes(res["display_image"].copy(), inst, spec.classes)
        c1, c2 = st.columns(2)
        c1.image(res["display_image"], caption="Input", use_container_width=True)
        c2.image(annotated, caption=f"{len(inst)} instance(s)", use_container_width=True)
    if inst:
        st.dataframe([{k: (round(v, 1) if isinstance(v, float) else v)
                       for k, v in i.items() if k != "mask"} for i in inst],
                     use_container_width=True, hide_index=True)
        masks = [i for i in inst if "mask" in i]
        if masks:
            cols = st.columns(min(4, len(masks)))
            for col, i in zip(cols * 4, masks):
                col.image((np.asarray(i["mask"]) * 255).astype("uint8"),
                          caption=f"{i['label']} ({i['score']:.2f})", clamp=True,
                          use_container_width=True)
    else:
        st.info("No instances were detected above the score threshold.")


def _vector_task(kind: str, res: dict) -> None:
    if kind == "tabular_denoise":
        metric_card("MSE to input", str(res.get("mse_to_input", res.get("mse", "—"))))
    if "probabilities" in res:
        st.bar_chart(res["probabilities"])
    for k in ("predicted", "label", "confidence", "class_index"):
        if k in res:
            st.write(f"**{k}:** {res[k]}")
    rest = {k: v for k, v in res.items()
            if k not in ("probabilities", "predicted", "label", "confidence")}
    if rest:
        st.json(rest)


def _text_task(kind: str, res: dict) -> None:
    if "probabilities" in res:
        st.bar_chart(res["probabilities"])
    for k in ("predicted", "label", "confidence", "perplexity"):
        if k in res:
            st.write(f"**{k}:** {res[k]}")
    if res.get("continuation") or res.get("generated"):
        st.code(res.get("continuation") or res.get("generated"))
    rest = {k: v for k, v in res.items()
            if k not in ("probabilities", "predicted", "label", "confidence", "continuation",
                         "generated")}
    if rest:
        st.json(rest)


def _pair(res: dict) -> None:
    for k in ("similar", "distance", "pair_accuracy", "probability"):
        if k in res:
            st.write(f"**{k}:** {res[k]}")
    st.json({k: v for k, v in res.items() if k not in ("similar", "distance", "probability")})


def _reconstruction(res: dict) -> None:
    c1, c2 = st.columns(2)
    if res.get("input") is not None:
        c1.image((np.asarray(res["input"]) * 255).astype("uint8"), caption="Input", clamp=True)
    if res.get("reconstruction") is not None:
        c2.image((np.asarray(res["reconstruction"]) * 255).astype("uint8"),
                 caption="Reconstruction", clamp=True)
    if "mse" in res:
        metric_card("Reconstruction MSE", str(res["mse"]))


def _evaluation(spec: ModelSpec, meta: dict | None, res: dict) -> None:
    st.subheader("Model evaluation (held-out test data)")
    show_metrics((meta or {}).get("metrics", {}))
    confusion_matrix_view((meta or {}).get("metrics", {}), spec.classes)
    model_disclaimer(spec.area)
    with st.expander("Dataset & model limitations", expanded=False):
        st.markdown(_limitations_md(spec))
    show_provenance(meta or {})


def _limitations_md(spec: ModelSpec) -> str:
    lines = [f"- **Dataset:** {spec.dataset_name} (`{spec.dataset_key}`). Class balance, "
             "acquisition protocol and population biases from the source data are inherited."]
    if spec.notes:
        lines.append(f"- **Model notes:** {spec.notes}")
    lines += [
        "- **Splits:** the source dataset's own splits are used where provided; otherwise a "
        "fixed-seed stratified split is applied and recorded in the artifact metadata.",
        "- **Preprocessing** (resize, normalisation) is fixed at training time and stored in the "
        "artifact, so inference preprocessing is identical.",
    ]
    if spec.area == "medical":
        lines.append("- **Not clinically validated.** No external cohort, no prospective testing, "
                     "no regulatory review.")
    if spec.area == "agriculture":
        lines.append("- **Field conditions vary.** Performance on images unlike the training "
                     "distribution (lighting, device, growth stage) is not guaranteed.")
    return "\n".join(lines)


def _png_bytes(mask) -> bytes:
    import io
    from PIL import Image
    arr = (np.asarray(mask) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()
