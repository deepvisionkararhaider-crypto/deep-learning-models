"""Streamlit Community Cloud entry point.

Set the app's **main file path** to ``app.py`` (or ``streamlit_app.py``, which
simply runs this module). The whole platform — landing page, model catalogue,
image prediction, text/tabular prediction, Medical AI, Agriculture AI, the
20-model laboratory, model details, evaluation, datasets and about — is reachable
from the sidebar.

Inference only:

* this app **never trains** (no ``torch.optim``, no backward pass);
* it **never downloads a dataset**;
* a checkpoint is downloaded **only** when a visitor selects that model, then
  cached by ``huggingface_hub`` and ``st.cache_resource``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep the CPU thread count predictable on small Community Cloud instances.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

st.set_page_config(page_title="Deep Learning Platform · Medical AI · Smart Agriculture",
                   page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

from src.services import manifest_service as MS  # noqa: E402
from src.ui import (pages_agriculture, pages_explore, pages_home, pages_image,  # noqa: E402
                    pages_info, pages_lab, pages_medical, pages_text)
from src.ui.common import GITHUB, hf_url, inject_css  # noqa: E402

PAGES = {
    "🏠 Home": pages_home.render,
    "🔎 Explore Models": pages_explore.render,
    "🖼️ Image Prediction": pages_image.render,
    "⌨️ Text & Tabular Prediction": pages_text.render,
    "🩺 Medical AI": pages_medical.render,
    "🌿 Agriculture AI": pages_agriculture.render,
    "🧪 Deep Learning Labs": pages_lab.render,
    "📄 Model Details": pages_info.render_model_details,
    "📊 Model Evaluation": pages_info.render_evaluation,
    "🗂️ Dataset Explorer": pages_info.render_datasets,
    "ℹ️ About & Documentation": pages_info.render_about,
}


def _sidebar() -> str:
    with st.sidebar:
        st.markdown("# 🧠 AI Platform")
        st.caption("Deep learning · Medical AI · Smart Agriculture")
        choice = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

        st.divider()
        try:
            counts = MS.counts()
            st.caption(f"**{counts['ready']} / {counts['models']}** models ready for inference")
        except Exception:                                           # noqa: BLE001
            st.caption("Manifest unavailable")

        st.link_button("⭐ GitHub", GITHUB, use_container_width=True)
        url = hf_url()
        if url:
            st.link_button("🤗 Model hub", url, use_container_width=True)

        with st.expander("Configuration", expanded=False):
            from src import hf_hub
            st.write({
                "HF_MODEL_REPO": hf_hub.model_repo() if hf_hub.is_configured() else "not set",
                "cache": str(hf_hub.DEFAULT_CACHE),
                "trains models": "never",
            })
        st.caption("Inference only. Models are trained offline and served from the Hugging Face Hub.")
    return choice


def main() -> None:
    inject_css()
    choice = _sidebar()
    PAGES[choice]()


# Streamlit executes this module top-to-bottom on every rerun, so call main()
# unconditionally (a __main__ guard would never fire under `streamlit run`).
main()
