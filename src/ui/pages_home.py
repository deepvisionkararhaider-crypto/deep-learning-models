"""Home dashboard: real, artifact-derived platform status (no invented numbers)."""
from __future__ import annotations

import streamlit as st

from .. import artifacts as A
from ..registry import ALL_MODELS, CV_MODELS, LAB_BY_INDEX
from .common import GITHUB, badge, hero, inject_css, metric_card


def _ready(spec) -> bool:
    return A.has_weights(spec.id, spec.area)


def render():
    inject_css()
    hero("Deep Learning · Medical AI · Smart Agriculture",
         "A single Streamlit platform: a 20-model deep-learning laboratory plus six computer-vision "
         "model types applied to brain-MRI and plant-disease analysis. Inference only — models are "
         "trained offline and loaded from committed artifacts.", "🧠")

    cv_ready = sum(1 for m in CV_MODELS if _ready(m))
    lab_ready = sum(1 for m in LAB_BY_INDEX.values() if _ready(m))
    total_ready = cv_ready + lab_ready

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Registered models", str(len(ALL_MODELS)))
    with c2:
        metric_card("With trained weights", str(total_ready))
    with c3:
        metric_card("Medical / Agri models", f"{cv_ready} / {len(CV_MODELS)}")
    with c4:
        metric_card("Training at prediction time", "Never")

    st.caption(
        "Counts are read live from `artifacts/` — a model is only shown as *trained* when its "
        "weights file exists and passes the load + inference smoke test."
    )

    st.divider()
    st.subheader("Three areas of the platform")

    a, b, c = st.columns(3)
    with a:
        st.markdown("### 🩺 Medical AI")
        st.write("Brain-MRI **classification** (CNN, DenseNet121, ViT), **segmentation** (U-Net) "
                 "and **detection / instance segmentation** (YOLO-style, Mask R-CNN).")
        st.page_link("app.py", label="Open Medical AI →", icon="🩺") if False else None
        st.caption("Navigate with the sidebar → Medical AI.")
    with b:
        st.markdown("### 🌿 Smart Agriculture")
        st.write("Plant-leaf **disease classification** (25 classes), **leaf-disease segmentation**, "
                 "and **cotton disease detection**.")
        st.caption("Navigate with the sidebar → Agriculture AI.")
    with c:
        st.markdown("### 🧪 20-Model Laboratory")
        st.write("The original catalogue — MLP, CNN, RNN/LSTM/GRU, autoencoder, VAE, GAN, "
                 "Transformer, BERT, GPT, ResNet, DenseNet, ViT, U-Net, YOLO, Siamese, Seq2Seq, "
                 "diffusion, GNN — each with its own real dataset and IO contract.")
        st.caption("Navigate with the sidebar → Model Laboratory.")

    st.divider()
    st.subheader("Model readiness (live)")

    st.markdown("**Computer-vision models**")
    _readiness_table(CV_MODELS)
    st.markdown("**20-model laboratory**")
    _readiness_table(list(LAB_BY_INDEX.values()))

    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Supported tasks")
        st.markdown(
            "- Image classification (4-class brain MRI, 25-class plant leaf)\n"
            "- Binary & multi-class semantic segmentation (U-Net, Dice+BCE loss)\n"
            "- Object detection with genuine bounding boxes (YOLO-style)\n"
            "- Instance segmentation with genuine masks (Mask R-CNN)\n"
            "- Tabular / sequence / text / graph classification & generation (laboratory)\n"
        )
    with col2:
        st.subheader("Links")
        st.link_button("⭐ GitHub repository", GITHUB, use_container_width=True)
        st.link_button("📚 Dataset documentation", f"{GITHUB}/blob/main/DATASETS.md", use_container_width=True)


def _readiness_table(specs):
    rows = []
    for m in specs:
        ready = _ready(m)
        meta = A.read_metadata(m.id) or {}
        acc = None
        for k in ("accuracy", "f1_macro", "dice", "mAP@50", "pair_accuracy", "recon_mse", "perplexity"):
            if k in (meta.get("metrics") or {}):
                acc = f"{k}={meta['metrics'][k]}"
                break
        rows.append({"model": m.name, "task": m.task, "dataset": m.dataset_name,
                     "status": "✔ trained" if ready else "✖ training required", "metric": acc or "—"})
    st.dataframe(rows, use_container_width=True, hide_index=True)
