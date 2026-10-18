"""Home — landing page driven entirely by the manifest (no hardcoded counts)."""
from __future__ import annotations

import streamlit as st

from ..services import manifest_service as MS
from ..registry import ALL_MODELS
from .common import GITHUB, hf_url, hero, inject_css, metric_card, status_badge
from .catalog import status_legend


def render() -> None:
    inject_css()
    hero("Deep Learning Platform · Medical AI · Smart Agriculture",
         "A 20-model deep-learning laboratory plus six computer-vision architectures applied to "
         "brain-MRI and plant-disease analysis. Trained checkpoints are hosted on Hugging Face and "
         "loaded automatically — you never train anything to get a prediction.", "🧠")

    c = MS.counts()
    entries = MS.entries()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Registered models", str(c["models"]),
                    f"{c['laboratory']} lab · {c['medical']} medical · {c['agriculture']} agri")
    with c2:
        metric_card("Ready for inference", str(c["ready"]), "TRAINED_VERIFIED")
    with c3:
        metric_card("Architectures", str(len({s.builder for s in ALL_MODELS})), "distinct builders")
    with c4:
        metric_card("Training at prediction time", "Never", "inference only")

    st.caption("Counts are read live from `models/manifest.json`, which is generated from the "
               "registry plus verified artifacts — never hardcoded.")

    st.divider()
    st.subheader("Start here")
    a, b, ccol = st.columns(3)
    with a:
        st.markdown("#### 🖼️ Run a prediction")
        st.write("Pick a ready model, upload an image or paste text, and get a result with the "
                 "model's measured metrics and limitations.")
        st.caption("Sidebar → **Image Prediction** or **Text & Tabular Prediction**.")
    with b:
        st.markdown("#### 🔎 Explore the catalogue")
        st.write("Filter all 32 models by domain, task and status; every card shows its real "
                 "contract, checkpoint size and primary metric.")
        st.caption("Sidebar → **Explore Models**.")
    with ccol:
        st.markdown("#### 📖 Understand the model")
        st.write("Read a model's architecture, dataset, split sizes, metrics and known limitations "
                 "before you trust an output.")
        st.caption("Sidebar → **Model Details**.")

    st.divider()
    st.subheader("How it works")
    st.markdown(
        "1. **Source code** lives in GitHub. 2. **Training** runs offline (or in GitHub Actions) on "
        "real public datasets. 3. **Checkpoints** are published to the **Hugging Face Hub** with a "
        "SHA-256, a pinned revision and a model card. 4. This app **downloads only the model you "
        "select**, verifies the checksum, and runs inference on CPU."
    )
    st.info("Visitors never train a model, never download a dataset, and never need an account. "
            "If a model has no verified checkpoint it is shown as *training required* — never "
            "faked with random weights.")

    st.divider()
    st.subheader("Platform coverage")
    x, y, z = st.columns(3)
    with x:
        st.markdown("### 🩺 Medical AI")
        st.write("Brain-MRI **classification** (CNN, DenseNet121, ViT), **segmentation** (U-Net), "
                 "**detection** (YOLO-style) and **instance segmentation** (Mask R-CNN).")
        st.caption("Research prototype — not a medical device.")
    with y:
        st.markdown("### 🌿 Smart Agriculture")
        st.write("Plant-leaf **disease classification** (25 classes), **leaf-disease "
                 "segmentation**, and **cotton disease detection**.")
        st.caption("Decision support only.")
    with z:
        st.markdown("### 🧪 20-Model Laboratory")
        st.write("MLP, CNN, RNN/LSTM/GRU, autoencoder, VAE, GAN, Transformer, BERT-style, "
                 "GPT-style, ResNet, DenseNet, ViT, U-Net, YOLO, Siamese, Seq2Seq, diffusion, GNN — "
                 "each with its own real dataset and I/O contract.")
        st.caption("Catalogue → **Deep Learning Labs**.")

    st.divider()
    st.subheader("Model status (live from the manifest)")
    rows = []
    for spec in ALL_MODELS:
        e = entries.get(spec.id, {})
        rows.append({"id": spec.id, "model": spec.name, "domain": e.get("domain", spec.area),
                     "task": spec.task, "status": e.get("status", "ARCH_ONLY"),
                     "checkpoint": f"{(e.get('weights_bytes') or 0)/1e6:.1f} MB"
                                   if e.get("weights_bytes") else "—"})
    st.dataframe(rows, use_container_width=True, hide_index=True)
    status_legend()

    st.divider()
    st.subheader("Source & artifacts")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.link_button("⭐ GitHub repository", GITHUB, use_container_width=True)
    with c2:
        st.link_button("📚 Dataset documentation", f"{GITHUB}/blob/main/DATASETS.md",
                       use_container_width=True)
    with c3:
        url = hf_url()
        if url:
            st.link_button("🤗 Hugging Face models", url, use_container_width=True)
        else:
            st.button("🤗 Hugging Face models", disabled=True, use_container_width=True,
                      help="HF_MODEL_REPO is not configured for this deployment.")
