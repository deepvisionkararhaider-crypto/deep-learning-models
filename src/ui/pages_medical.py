"""Medical AI page: brain-tumour classification, segmentation, detection & instances."""
from __future__ import annotations

import numpy as np
import streamlit as st

from .. import inference as I
from ..imaging import draw_boxes, load_image
from ..registry import MEDICAL
from .common import (badge, get_model, hero, inject_css, medical_disclaimer, metric_card,
                     show_metrics, show_provenance)


TASK_LABEL = {
    "classification": "Tumour classification",
    "segmentation": "Tumour segmentation (U-Net)",
    "detection": "Tumour detection (YOLO-style)",
    "instance_segmentation": "Tumour instance segmentation (Mask R-CNN)",
}


def render():
    inject_css()
    hero("Medical AI — Brain Tumour Analysis",
         "Upload a brain-MRI slice and run one of the trained medical models. Educational/research "
         "prototype only.", "🩺")
    medical_disclaimer()

    tasks = ["classification", "segmentation", "detection", "instance_segmentation"]
    task = st.radio("Choose an analysis task", tasks,
                    format_func=lambda t: TASK_LABEL[t], horizontal=True)
    candidates = [m for m in MEDICAL if m.task == task]

    ready = [m for m in candidates if I.is_ready(m)]
    st.markdown("**Available models for this task**")
    for m in candidates:
        ok = I.is_ready(m)
        st.markdown(f"{badge(ok)} &nbsp; **{m.name}** — {m.dataset_name}", unsafe_allow_html=True)
    if not ready:
        st.warning("No trained weights are available for this task yet. "
                   "Train one offline, e.g. `python scripts/train_models.py --model brain_densenet`, "
                   "then reload this page. Missing weights are shown honestly rather than faked.")
        return

    model_name = st.selectbox("Select model", [m.name for m in ready])
    spec = next(m for m in ready if m.name == model_name)
    st.caption(f"**Dataset:** {spec.dataset_name}  ·  **Input:** {spec.input_desc}  ·  "
               f"**Output:** {spec.outputs}")

    up = st.file_uploader("Upload an MRI image (PNG/JPG)", type=["png", "jpg", "jpeg", "bmp", "webp"],
                          key="med_upload")
    if up is None:
        st.info("Upload an MRI image to run inference. Nothing is uploaded to a server — inference "
                "runs in this Streamlit session and the image is discarded when you close the tab.")
        return

    try:
        img = load_image(up.getvalue())
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read that image: {exc}")
        return

    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(img, caption="Uploaded MRI", use_container_width=True)

    if st.button("🔎 Analyze MRI", type="primary"):
        try:
            with st.spinner("Loading model and running inference…"):
                model, meta = get_model(spec)
                result = I.predict(spec, img)
            _render_result(spec, result, meta, img)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Inference failed: {type(exc).__name__}: {exc}")
            st.caption("This usually means the stored weights do not match the architecture. "
                       "Retrain with `python scripts/train_models.py --model " + spec.id + "`.")


def _render_result(spec, result, meta, img):
    kind = result["kind"]
    st.success(f"Analysis complete in {result.get('inference_ms', 0):.0f} ms.")

    if kind == "classification":
        top = result["top_k"][0]
        a, b, c = st.columns(3)
        with a:
            metric_card("Predicted class", top["label"])
        with b:
            metric_card("Confidence", f"{top['prob']:.1%}")
        with c:
            metric_card("Classes", str(len(result["probabilities"])))
        st.bar_chart({k: v for k, v in sorted(result["probabilities"].items(),
                                              key=lambda kv: -kv[1])})
        st.caption("Confidence is a model score, not a probability that the diagnosis is correct. "
                   "Calibration was not verified on external data.")

    elif kind == "segmentation":
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(img, caption="Input", use_container_width=True)
        with c2:
            st.image((result["mask"] * 255).astype("uint8"), caption="Predicted mask",
                     clamp=True, use_container_width=True)
        with c3:
            st.image(result["overlay"], caption="Overlay", use_container_width=True)
        metric_card("Tumour-pixel ratio", f"{result['foreground_ratio']:.2%}")

    elif kind == "detection":
        disp = result["display_image"].copy()
        annotated = draw_boxes(disp, result["boxes"], spec.classes)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(img, caption="Input", use_container_width=True)
        with c2:
            st.image(annotated, caption=f"{result['count']} detection(s)", use_container_width=True)
        if result["boxes"]:
            st.dataframe([{k: (round(v, 1) if isinstance(v, float) else v)
                           for k, v in b.items()} for b in result["boxes"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("No regions passed the confidence threshold. Try a clearer slice or lower the threshold.")

    elif kind == "instance_segmentation":
        annotated = draw_boxes(result["display_image"], result["instances"], spec.classes)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(img, caption="Input", use_container_width=True)
        with c2:
            st.image(annotated, caption=f"{result['count']} instance(s)", use_container_width=True)
        for inst in result["instances"]:
            if "mask" in inst:
                st.image((inst["mask"] * 255).astype("uint8"),
                         caption=f"{inst['label']} (score {inst['score']:.2f})", clamp=True, width=180)

    st.divider()
    st.subheader("Model evaluation (measured on held-out test data)")
    show_metrics((meta or {}).get("metrics", {}))
    _limitations(spec)
    show_provenance(meta)


def _limitations(spec):
    with st.expander("Dataset & model limitations", expanded=False):
        st.markdown(
            f"- **Dataset:** {spec.dataset_name}. Class balance, scanner/protocol and population "
            "biases from the source data are inherited by the model.\n"
            "- **Split:** where the source dataset ships patient/subject-disjoint splits we use them; "
            "otherwise a fixed-seed split is used and recorded in the artifact metadata.\n"
            "- **Not clinically validated.** No external cohort, no prospective testing, no regulatory "
            "review.\n"
            "- **Preprocessing** (resize + normalisation) is fixed at training time and recorded in the "
            "artifact, so it is identical at inference.\n"
        )
