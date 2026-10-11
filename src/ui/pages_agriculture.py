"""Smart Agriculture AI page: leaf-disease classification, segmentation & cotton detection."""
from __future__ import annotations

import streamlit as st

from .. import inference as I
from ..imaging import draw_boxes, load_image
from ..registry import AGRI
from .common import (agri_disclaimer, badge, get_model, hero, inject_css, metric_card,
                     show_metrics, show_provenance)

TASK_LABEL = {
    "classification": "Leaf disease classification",
    "segmentation": "Leaf disease segmentation (U-Net)",
    "detection": "Cotton disease detection (YOLO-style)",
    "instance_segmentation": "Cotton disease instances (Mask R-CNN)",
}

CLASS_NOTES = {
    "healthy": "No disease detected by the model. Healthy leaves should still be monitored.",
    "Apple___Apple_scab": "Apple scab — fungal (Venturia inaequalis); favour resistant cultivars and sanitation.",
    "Apple___Black_rot": "Apple black rot — fungal; remove mummified fruit and cankers.",
    "Apple___Cedar_apple_rust": "Cedar-apple rust — needs both hosts; manage nearby junipers.",
    "Corn___Cercospora_leaf_spot": "Gray leaf spot — fungal; rotate crops and use resistant hybrids.",
    "Corn___Common_rust": "Common rust — fungal; scout early, use resistant hybrids.",
    "Corn___Northern_Leaf_Blight": "Northern leaf blight — fungal; residue management helps.",
    "Potato___Early_blight": "Early blight (Alternaria) — fungal; avoid overhead irrigation.",
    "Potato___Late_blight": "Late blight (Phytophthora) — aggressive; act quickly, destroy infected plants.",
    "Tomato___Bacterial_spot": "Bacterial spot — copper sprays and clean seed help.",
    "Tomato___Early_blight": "Early blight — fungal; stake plants and remove lower leaves.",
    "Tomato___Late_blight": "Late blight — serious; remove and destroy infected plants.",
    "Tomato___Leaf_Mold": "Leaf mold — improve airflow and reduce humidity.",
    "Tomato___Septoria_leaf_spot": "Septoria leaf spot — remove affected leaves, mulch.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Yellow leaf curl virus — whitefly-vectored; manage whiteflies.",
    "Tomato___Tomato_mosaic_virus": "Mosaic virus — viral, no cure; remove plants, disinfect tools.",
}


def render():
    inject_css()
    hero("Smart Agriculture AI — Crop Disease & Pest Analysis",
         "Upload a leaf or crop image and run a trained model for disease classification, "
         "segmentation or detection.", "🌿")
    agri_disclaimer()

    tasks = ["classification", "segmentation", "detection", "instance_segmentation"]
    task = st.radio("Choose a task", tasks, format_func=lambda t: TASK_LABEL[t], horizontal=True)
    candidates = [m for m in AGRI if m.task == task]
    ready = [m for m in candidates if I.is_ready(m)]

    st.markdown("**Available models for this task**")
    for m in candidates:
        st.markdown(f"{badge(I.is_ready(m))} &nbsp; **{m.name}** — {m.dataset_name}",
                    unsafe_allow_html=True)
    if not ready:
        st.warning("No trained weights available for this task yet. Train offline, e.g. "
                   "`python scripts/train_models.py --model plant_densenet`, then reload. "
                   "Unavailable models are shown honestly rather than faked.")
        return

    model_name = st.selectbox("Select model", [m.name for m in ready])
    spec = next(m for m in ready if m.name == model_name)
    st.caption(f"**Dataset:** {spec.dataset_name}  ·  **Input:** {spec.input_desc}  ·  "
               f"**Output:** {spec.outputs}")

    up = st.file_uploader("Upload a leaf / crop image", type=["png", "jpg", "jpeg", "bmp", "webp"],
                          key="agri_upload")
    if up is None:
        st.info("Upload a leaf image. Inference runs locally in this session; the image is not stored.")
        return

    try:
        img = load_image(up.getvalue())
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read that image: {exc}")
        return

    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(img, caption="Uploaded image", use_container_width=True)

    if st.button("🌿 Analyze Crop", type="primary"):
        try:
            with st.spinner("Running inference…"):
                model, meta = get_model(spec)
                result = I.predict(spec, img)
            _render_result(spec, result, meta, img)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Inference failed: {type(exc).__name__}: {exc}")


def _render_result(spec, result, meta, img):
    kind = result["kind"]
    st.success(f"Analysis complete in {result.get('inference_ms', 0):.0f} ms.")

    if kind == "classification":
        top = result["top_k"][0]
        a, b, c = st.columns(3)
        with a:
            metric_card("Predicted class", top["label"].replace("___", " / "))
        with b:
            metric_card("Confidence", f"{top['prob']:.1%}")
        with c:
            metric_card("Candidate classes", str(len(result["probabilities"])))
        ordered = sorted(result["probabilities"].items(), key=lambda kv: -kv[1])
        st.bar_chart({k.replace("___", " / "): v for k, v in ordered[:10]})
        st.markdown("**Top-5 predictions**")
        st.dataframe([{"class": t["label"].replace("___", " / "), "probability": round(t["prob"], 4)}
                      for t in result["top_k"]], use_container_width=True, hide_index=True)
        note = CLASS_NOTES.get(top["label"])
        if note:
            st.info(f"About **{top['label'].replace('___', ' / ')}**: {note}")

    elif kind == "segmentation":
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(img, caption="Input", use_container_width=True)
        with c2:
            st.image((result["mask"] * 255).astype("uint8"), caption="Predicted mask",
                     clamp=True, use_container_width=True)
        with c3:
            st.image(result["overlay"], caption="Overlay", use_container_width=True)
        metric_card("Diseased-pixel ratio", f"{result['foreground_ratio']:.2%}")

    elif kind == "detection":
        annotated = draw_boxes(result["display_image"], result["boxes"], spec.classes)
        c1, c2 = st.columns(2)
        with c1:
            st.image(img, caption="Input", use_container_width=True)
        with c2:
            st.image(annotated, caption=f"{result['count']} detection(s)", use_container_width=True)
        if result["boxes"]:
            st.dataframe([{"x1": round(b["x1"], 1), "y1": round(b["y1"], 1), "x2": round(b["x2"], 1),
                           "y2": round(b["y2"], 1), "score": round(b["score"], 3), "class": b["label"]}
                          for b in result["boxes"]], use_container_width=True, hide_index=True)
        else:
            st.info("No regions exceeded the confidence threshold.")

    elif kind == "instance_segmentation":
        annotated = draw_boxes(result["display_image"], result["instances"], spec.classes)
        c1, c2 = st.columns(2)
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
    with st.expander("Dataset & model limitations", expanded=False):
        st.markdown(
            f"- **Dataset:** {spec.dataset_name}. Field conditions, camera types and geographic "
            "coverage of the source data bound how well this generalises.\n"
            "- Lab-condition (PlantVillage-style) images are easier than real field photos; expect "
            "lower accuracy on cluttered backgrounds.\n"
            "- **Decision support only** — confirm with an agronomist before acting.\n"
        )
    show_provenance(meta)
