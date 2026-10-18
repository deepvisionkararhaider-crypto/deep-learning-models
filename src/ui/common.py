"""Shared Streamlit design system: theme, badges, cards, disclaimers, provenance.

Kept deliberately small and dependency-free (Streamlit-native components plus one
compact CSS block) so it renders reliably on Streamlit Community Cloud.
"""
from __future__ import annotations

import streamlit as st

from .. import artifacts as A
from ..registry import ModelSpec

GITHUB = "https://github.com/deepvisionkararhaider-crypto/deep-learning-models"

#: status -> (css class, glyph, short label)
STATUS_STYLE: dict[str, tuple[str, str, str]] = {
    "TRAINED_VERIFIED": ("badge-ready", "✔", "verified"),
    "TRAINED_UNVERIFIED": ("badge-warn", "◐", "unverified"),
    "ARCH_ONLY": ("badge-missing", "✖", "training required"),
    "BLOCKED": ("badge-blocked", "⊘", "blocked"),
    "DEFERRED": ("badge-info", "⏸", "deferred"),
    "FAILED": ("badge-missing", "!", "failed"),
}

CSS = """
<style>
  .block-container {padding-top: 2.0rem; padding-bottom: 3rem; max-width: 1240px;}
  .app-hero {background: linear-gradient(118deg,#0b1220 0%,#1e3a8a 58%,#0e7490 100%);
      border-radius:14px; padding:1.5rem 1.7rem; color:#f8fafc; margin-bottom:1.1rem;}
  .app-hero h1 {margin:0 0 .3rem 0; font-size:1.8rem; color:#fff; letter-spacing:-.01em;}
  .app-hero p {margin:0; opacity:.92; font-size:.97rem; max-width:70ch;}
  .badge {display:inline-block; padding:.15rem .55rem; border-radius:999px; font-size:.73rem;
      font-weight:600; letter-spacing:.02em; white-space:nowrap;}
  .badge-ready {background:#dcfce7;color:#166534;border:1px solid #86efac;}
  .badge-warn {background:#fef3c7;color:#92400e;border:1px solid #fcd34d;}
  .badge-missing {background:#fee2e2;color:#991b1b;border:1px solid #fca5a5;}
  .badge-blocked {background:#e5e7eb;color:#374151;border:1px solid #9ca3af;}
  .badge-info {background:#e0e7ff;color:#3730a3;border:1px solid #a5b4fc;}
  .metric-card {background:#f8fafc;border:1px solid #e2e8f0;border-radius:11px;padding:.85rem 1rem;}
  .metric-card .k {font-size:.71rem;text-transform:uppercase;letter-spacing:.06em;color:#64748b;}
  .metric-card .v {font-size:1.4rem;font-weight:700;color:#0f172a;line-height:1.25;}
  .metric-card .s {font-size:.74rem;color:#64748b;}
  .disclaimer {background:#fffbeb;border-left:4px solid #f59e0b;padding:.7rem .9rem;
      border-radius:6px;font-size:.87rem;color:#78350f;margin:.4rem 0;}
  .disclaimer-med {background:#fef2f2;border-left:4px solid #dc2626;padding:.7rem .9rem;
      border-radius:6px;font-size:.87rem;color:#7f1d1d;margin:.4rem 0;}
  .model-card {border:1px solid #e2e8f0;border-radius:12px;padding:.85rem 1rem;background:#fff;
      margin-bottom:.5rem;}
  .model-card h4 {margin:0 0 .15rem 0;font-size:1rem;color:#0f172a;}
  .model-card .sub {color:#64748b;font-size:.82rem;}
  .chip {display:inline-block;padding:.1rem .45rem;border-radius:6px;font-size:.7rem;
      background:#eef2ff;color:#3730a3;margin-right:.3rem;font-weight:600;}
  .chip-dom {background:#ecfdf5;color:#065f46;}
  .chip-lab {background:#f5f3ff;color:#5b21b6;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, icon: str = "🧠") -> None:
    st.markdown(f'<div class="app-hero"><h1>{icon} {title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def status_badge(status: str) -> str:
    cls, glyph, label = STATUS_STYLE.get(status, ("badge-info", "•", status.lower()))
    return f'<span class="badge {cls}">{glyph} {label}</span>'


def badge(ready: bool, text_ready: str = "trained", text_missing: str = "training required") -> str:
    """Backwards-compatible boolean badge (prefer :func:`status_badge`)."""
    return status_badge("TRAINED_VERIFIED" if ready else "ARCH_ONLY")


def domain_chip(area: str) -> str:
    cls = "chip chip-lab" if area == "lab" else "chip chip-dom"
    label = {"medical": "medical", "agriculture": "agriculture", "lab": "laboratory"}[area]
    return f'<span class="{cls}">{label}</span>'


def task_chip(task: str) -> str:
    return f'<span class="chip">{task.replace("_", " ")}</span>'


def metric_card(label: str, value: str, sub: str = "") -> None:
    extra = f'<div class="s">{sub}</div>' if sub else ""
    st.markdown(f'<div class="metric-card"><div class="k">{label}</div>'
                f'<div class="v">{value}</div>{extra}</div>', unsafe_allow_html=True)


def medical_disclaimer() -> None:
    st.markdown(
        '<div class="disclaimer-med"><b>Research prototype — not a medical device.</b> Outputs are '
        'educational and can be wrong. They are <b>not</b> a diagnosis and must never drive treatment. '
        'Always consult a qualified clinician and verify against the original imaging.</div>',
        unsafe_allow_html=True)


def agri_disclaimer() -> None:
    st.markdown(
        '<div class="disclaimer"><b>Decision support only.</b> Predictions can be wrong and do not '
        'guarantee a diagnosis or outcome. Verify important findings with an agronomist or extension '
        'service, and consider image quality and the dataset\u2019s coverage limits.</div>',
        unsafe_allow_html=True)


def model_disclaimer(area: str) -> None:
    if area == "medical":
        medical_disclaimer()
    elif area == "agriculture":
        agri_disclaimer()
    else:
        st.caption("Educational demonstration. Predictions are statistical estimates and can be wrong.")


def hf_url() -> str | None:
    from .. import hf_hub
    try:
        return f"https://huggingface.co/{hf_hub.model_repo()}"
    except Exception:                                               # noqa: BLE001
        return None


def show_metrics(metrics: dict, exclude=("confusion_matrix", "train_loss_curve", "split_sizes",
                                         "val", "n_test")) -> None:
    scalars = {k: v for k, v in (metrics or {}).items()
               if isinstance(v, (int, float)) and k not in exclude}
    if not scalars:
        st.info("No evaluation metrics recorded for this model yet.")
        return
    cols = st.columns(min(4, len(scalars)))
    for col, (k, v) in zip(cols * 10, scalars.items()):
        col.metric(k, f"{v:.4f}" if isinstance(v, float) else str(v))


def show_provenance(meta: dict) -> None:
    if not meta:
        return
    with st.expander("Provenance & artifact details", expanded=False):
        st.write({
            "artifact_version": meta.get("artifact_version"),
            "dataset": meta.get("dataset"),
            "dataset_key": meta.get("dataset_key"),
            "trained_at": meta.get("saved_at"),
            "torch": meta.get("torch"),
            "python": meta.get("python"),
            "weights_file": meta.get("weights_file"),
            "weights_MB": round((meta.get("weights_bytes") or 0) / 1e6, 3),
            "weights_sha256": (meta.get("weights_sha256") or "")[:16] or None,
            "split_manifest": meta.get("split_manifest"),
            "training": meta.get("training"),
            "git_commit": (meta.get("git_commit") or "")[:12] or None,
        })
        if meta.get("notes"):
            st.caption(meta["notes"])


def confusion_matrix_view(metrics: dict, classes: list[str]) -> None:
    cm = (metrics or {}).get("confusion_matrix")
    if not cm:
        return
    try:
        import pandas as pd
        labels = classes[: len(cm)]
        st.markdown("**Confusion matrix (test set)**")
        st.dataframe(pd.DataFrame(cm, index=labels, columns=labels), use_container_width=True)
    except Exception:                                               # noqa: BLE001
        st.write(cm)


def status_line(spec: ModelSpec) -> dict:
    return {"ready": A.has_weights(spec.id, spec.area), "meta": A.read_metadata(spec.id)}
