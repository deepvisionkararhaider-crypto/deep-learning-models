"""Streamlit Community Cloud entry point.

Deploy by setting the app's main file path to ``app.py``. The whole platform —
home dashboard, Medical AI, Agriculture AI, the 20-model laboratory, evaluation,
dataset explorer and about — is reachable from the sidebar.

Inference only: this app never trains and never downloads datasets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Deep Learning · Medical AI · Smart Agriculture",
                   page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")

from src.ui import (pages_agriculture, pages_home, pages_info,  # noqa: E402
                    pages_lab, pages_medical)
from src.ui.common import GITHUB  # noqa: E402

PAGES = {
    "🏠 Home": pages_home.render,
    "🩺 Medical AI": pages_medical.render,
    "🌿 Agriculture AI": pages_agriculture.render,
    "🧪 Model Laboratory (20)": pages_lab.render,
    "📊 Model Evaluation": pages_info.render_evaluation,
    "🗂️ Dataset Explorer": pages_info.render_datasets,
    "ℹ️ About & Limitations": pages_info.render_about,
}


def main():
    with st.sidebar:
        st.markdown("# 🧠 AI Platform")
        st.caption("Deep learning · Medical AI · Smart Agriculture")
        choice = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()
        st.link_button("⭐ GitHub", GITHUB, use_container_width=True)
        st.caption("Inference only — models are trained offline and shipped as artifacts.")

    PAGES[choice]()


if __name__ == "__main__":
    main()
else:
    main()
