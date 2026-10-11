"""Shared Streamlit UI helpers: styling, status badges, safe model loading."""
from __future__ import annotations

import time

import numpy as np
import streamlit as st

from .. import artifacts as A
from ..registry import ModelSpec

GITHUB = "https://github.com/deepvisionkararhaider-crypto/deep-learning-models"

CSS = """
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px;}
  .app-hero {
      background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 55%, #0e7490 100%);
      border-radius: 16px; padding: 1.6rem 1.8rem; color: #f8fafc; margin-bottom: 1.2rem;
  }
  .app-hero h1 {margin: 0 0 .3rem 0; font-size: 1.85rem; color:#fff;}
  .app-hero p {margin: 0; opacity: .9; font-size: .98rem;}
  .badge {display:inline-block; padding:.16rem .55rem; border-radius:999px; font-size:.74rem;
          font-weight:600; letter-spacing:.02em;}
  .badge-ready {background:#dcfce7; color:#166534; border:1px solid #86efac;}
  .badge-missing {background:#fee2e2; color:#991b1b; border:1px solid #fca5a5;}
  .badge-info {background:#e0e7ff; color:#3730a3; border:1px solid #a5b4fc;}
  .metric-card {background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:.9rem 1rem;}
  .metric-card .k {font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; color:#64748b;}
  .metric-card .v {font-size:1.45rem; font-weight:700; color:#0f172a;}
  .disclaimer {background:#fffbeb; border-left:4px solid #f59e0b; padding:.7rem .9rem;
               border-radius:6px; font-size:.87rem; color:#78350f;}
  .disclaimer-med {background:#fef2f2; border-left:4px solid #dc2626; padding:.7rem .9rem;
               border-radius:6px; font-size:.87rem; color:#7f1d1d;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, icon: str = "🧠"):
    st.markdown(
        f'<div class="app-hero"><h1>{icon} {title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def badge(ready: bool, text_ready: str = "trained weights", text_missing: str = "training required") -> str:
    cls = "badge-ready" if ready else "badge-missing"
    return f'<span class="badge {cls}">{"✔ " if ready else "✖ "}{text_ready if ready else text_missing}</span>'


def metric_card(label: str, value: str):
    st.markdown(f'<div class="metric-card"><div class="k">{label}</div><div class="v">{value}</div></div>',
                unsafe_allow_html=True)


def medical_disclaimer():
    st.markdown(
        '<div class="disclaimer-med"><b>Research prototype — not a medical device.</b> '
        'Outputs are educational and may be wrong. They are <b>not</b> a diagnosis and must never '
        'drive treatment. Always consult a qualified clinician and verify with the original imaging.</div>',
        unsafe_allow_html=True,
    )


def agri_disclaimer():
    st.markdown(
        '<div class="disclaimer"><b>Decision support only.</b> Predictions may be incorrect and do not '
        'guarantee a diagnosis or outcome. Verify important findings with an agronomist or extension '
        'service, and consider image quality and dataset coverage limits.</div>',
        unsafe_allow_html=True,
    )


def status_line(spec: ModelSpec) -> dict:
    return {
        "ready": A.has_weights(spec.id, spec.area),
        "meta": A.read_metadata(spec.id),
    }


@st.cache_resource(show_spinner=False)
def _cached_load(model_id: str, area: str):
    """Cache a loaded (eval-mode) model for the process lifetime."""
    from ..inference import load
    from ..lab_inference import load as lab_load
    from ..registry import BY_ID
    spec = BY_ID[model_id]
    loader = lab_load if spec.area == "lab" else load
    return loader(spec)


def get_model(spec: ModelSpec):
    return _cached_load(spec.id, spec.area)


def show_metrics(metrics: dict, exclude=("confusion_matrix", "train_loss_curve", "split_sizes")):
    """Render a metrics dict as st.metric cards (skipping non-scalars)."""
    scalars = {k: v for k, v in (metrics or {}).items()
               if isinstance(v, (int, float)) and k not in exclude}
    if not scalars:
        st.info("No evaluation metrics recorded for this model yet.")
        return
    cols = st.columns(min(4, len(scalars)))
    for col, (k, v) in zip(cols * 10, scalars.items()):
        col.metric(k, f"{v:.4f}" if isinstance(v, float) else str(v))


def show_provenance(meta: dict):
    if not meta:
        return
    with st.expander("Provenance & artifact details", expanded=False):
        st.write({
            "artifact_version": meta.get("artifact_version"),
            "dataset": meta.get("dataset"),
            "trained_at": meta.get("saved_at"),
            "torch": meta.get("torch"),
            "python": meta.get("python"),
            "weights_file": meta.get("weights_file"),
            "weights_MB": round((meta.get("weights_bytes") or 0) / 1e6, 3),
            "git_commit": (meta.get("git_commit") or "")[:12] or None,
        })
        if meta.get("notes"):
            st.caption(meta["notes"])


def confusion_matrix_view(metrics: dict, classes: list[str]):
    cm = (metrics or {}).get("confusion_matrix")
    if not cm:
        return
    try:
        import pandas as pd
        labels = classes[: len(cm)]
        st.markdown("**Confusion matrix (test set)**")
        st.dataframe(pd.DataFrame(cm, index=labels, columns=labels), use_container_width=True)
    except Exception:
        st.write(cm)


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t0) * 1000
