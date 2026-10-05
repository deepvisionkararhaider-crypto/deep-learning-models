"""AI MODEL LAB — portfolio dashboard for all 20 model demos.

This is the main Streamlit entry point for the repository. It lists every model
with its task, dataset, framework and reported metric, and links to each
model's own public demo URL.

Individual app URLs are derived (never invented) from two optional values:

    STREAMLIT_APP_BASE   e.g. https://<your-app-slug>.streamlit.app
    GITHUB_REPOSITORY    e.g. owner/deep-learning-models   (optional, for source links)

Set STREAMLIT_APP_BASE once you have created your apps on Streamlit Community
Cloud; if it is unset the dashboard shows how to run each app locally instead
of showing a link that would not resolve.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="AI Model Lab · 20 Deep-Learning Models",
                   page_icon="🧠", layout="wide")

import app_shared as A  # noqa: E402

REPO = os.environ.get("GITHUB_REPOSITORY", "deepvisionkararhaider-crypto/deep-learning-models")
GITHUB = f"https://github.com/{REPO}"
APP_BASE = os.environ.get("STREAMLIT_APP_BASE", "").rstrip("/")

CATEGORIES = {
    "Tabular": [1, 19],
    "Vision": [2, 12, 13, 14, 15, 16, 17],
    "Sequence": [3, 4, 5],
    "Natural language": [9, 10, 11, 18],
    "Generative": [6, 7, 8],
    "Graph": [20],
}


def _meta(mid: int) -> dict:
    p = HERE / A.MODELS[mid]["folder"] / "model" / "meta.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _url(mid: int) -> str | None:
    if not APP_BASE:
        return None
    return f"{APP_BASE}/{A.MODELS[mid]['folder']}"


def _ready(mid: int) -> bool:
    return (HERE / A.MODELS[mid]["folder"] / "model" / "model.pt").exists()


def _card(mid: int) -> None:
    info, meta = A.MODELS[mid], _meta(mid)
    with st.container(border=True):
        st.markdown(f"#### {mid:02d} · {info['name']}")
        st.caption(info["task"])
        st.markdown(
            f"- **Dataset:** {info['dataset']}\n"
            f"- **Framework:** {info['framework']}\n"
            f"- **Input:** {info['input']}"
        )
        if meta.get("metric_name"):
            st.metric(meta["metric_name"], meta.get("metric_value", "—"))
        url = _url(mid)
        if url:
            st.link_button("Open live demo", url, use_container_width=True)
        else:
            st.code(f"streamlit run {info['folder']}/app.py", language="bash")
        st.caption(f"Folder: `{info['folder']}/`")


def main() -> None:
    st.title("🧠 AI MODEL LAB")
    st.caption("Twenty real deep-learning models — trained once, committed as artifacts, "
               "and served through real inference. Enter new data, get a real prediction.")

    ready = sum(1 for m in A.MODELS if _ready(m))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Models", len(A.MODELS))
    c2.metric("Artifacts ready", f"{ready}/{len(A.MODELS)}")
    c3.metric("Runtime", "CPU")
    c4.metric("Retraining on visit", "Never")

    if not APP_BASE:
        st.info(
            "Public demo links appear here once **STREAMLIT_APP_BASE** is set. "
            "Deploy each model as its own Streamlit Community Cloud app "
            "(see `DEPLOY_STREAMLIT.md`) and set the secret/env var to your app base URL."
        )
    st.link_button("View source on GitHub", GITHUB)

    st.divider()
    for category, ids in CATEGORIES.items():
        st.subheader(category)
        cols = st.columns(3)
        for i, mid in enumerate(ids):
            with cols[i % 3]:
                _card(mid)

    st.divider()
    st.subheader("How the inference path works")
    st.markdown(
        "```\n"
        "Streamlit UI  →  app_shared.predict(model_id, payload)\n"
        "              →  load model/model.pt (real trained weights)\n"
        "              →  apply the training-time preprocessing\n"
        "              →  model forward pass  →  real prediction\n"
        "```\n"
        "Every folder also ships a FastAPI service (`backend/main.py`, `POST /predict`) "
        "that shares the same inference layer."
    )
    st.caption("Educational portfolio. Metrics are train/eval figures from the reproducible "
               "training pipeline and are not leaderboard claims.")


if __name__ == "__main__":
    main()
